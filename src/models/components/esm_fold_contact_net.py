from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from esm.esmfold.v1.trunk import FoldingTrunk, FoldingTrunkConfig
except ModuleNotFoundError as e:  # pragma: no cover - dependency guard
    raise ModuleNotFoundError(
        "esm.esmfold.v1.trunk requires openfold. Please install openfold to use ESMFoldContactNet."
    ) from e


class RelativePositionEmbedding(nn.Module):
    """Relative position embedding with clipped distance and far bucket."""

    def __init__(self, max_distance: int, dim: int) -> None:
        super().__init__()
        self.max_distance = max_distance
        # distances -max..max plus one far bucket
        num_buckets = (2 * max_distance + 1) + 1
        self.embedding = nn.Embedding(num_buckets, dim)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        dist = idx[None, :] - idx[:, None]  # L x L
        dist_clipped = dist.clamp(-self.max_distance, self.max_distance)
        far_mask = dist.abs() > self.max_distance
        # shift to positive
        dist_indices = dist_clipped + self.max_distance
        dist_indices = dist_indices + 1  # reserve 0 for far bucket
        dist_indices[far_mask] = 0
        emb = self.embedding(dist_indices)
        return emb.to(dtype)


class PairMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ESMFoldContactNet(nn.Module):
    """Model combining ESM2 embeddings, folding trunk, and contact/CRF heads."""

    def __init__(
        self,
        input_dim: int = 5120,
        projection_dim: int = 1024,
        model_dim: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.3,
        pair_dim: int = 128,
        pair_mlp_dim: int = 256,
        sequence_head_width: int = 32,
        pairwise_head_width: int = 32,
        relative_distance_bins: int = 32,
        num_classes: int = 10,
        residue_vocab_size: int = 27,
        residue_embed_dim: int = 128,
        max_recycles: int = 0,
        chunk_size: Optional[int] = 64,
    ) -> None:
        super().__init__()
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")
        if pair_dim % pairwise_head_width != 0:
            raise ValueError(
                f"pair_dim ({pair_dim}) must be divisible by pairwise_head_width ({pairwise_head_width})."
            )

        # Residue embedding + gated residual fusion (from ESMTransformerResNet)
        self.residue_embedding = nn.Embedding(
            num_embeddings=residue_vocab_size,
            embedding_dim=residue_embed_dim,
            padding_idx=0,
        )
        self.residue_proj = nn.Linear(residue_embed_dim, input_dim)
        self.gate_linear = nn.Linear(input_dim * 2, 1)

        # Sequence projection to folding trunk dimension
        self.input_proj = nn.Linear(input_dim, projection_dim)
        self.input_norm = nn.LayerNorm(projection_dim)
        if projection_dim != model_dim:
            self.model_proj = nn.Linear(projection_dim, model_dim)
        else:
            self.model_proj = nn.Identity()
        self.input_dropout = nn.Dropout(dropout)

        # Pair initialization
        self.pair_proj = nn.Linear(model_dim, pair_dim)
        self.pair_mlp = PairMLP(pair_dim, pair_mlp_dim)
        self.relative_pos_emb = RelativePositionEmbedding(relative_distance_bins, pair_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)

        # Folding trunk
        trunk_cfg = FoldingTrunkConfig(
            num_blocks=num_layers,
            sequence_state_dim=model_dim,
            pairwise_state_dim=pair_dim,
            sequence_head_width=sequence_head_width,
            pairwise_head_width=pairwise_head_width,
            dropout=dropout,
            max_recycles=max_recycles,
            chunk_size=chunk_size,
        )
        self.trunk = FoldingTrunk(**trunk_cfg.__dict__)

        # Heads
        self.seq_norm = nn.LayerNorm(model_dim)
        self.seq_head = nn.Linear(model_dim, num_classes)

        self.pair_norm_out = nn.LayerNorm(pair_dim)
        self.contact_head = nn.Linear(pair_dim, 1)

    def forward(
        self,
        embeddings: torch.Tensor,
        residue_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param embeddings: (B, L, input_dim) ESM2 token embeddings.
        :param residue_ids: (B, L) residue index tensor.
        :param valid_mask: (B, L) boolean mask for valid residues (structure), optional.
        :return: seq_logits (B, L, C), contact_logits (B, L, L), mask (B, L) used in trunk.
        """
        if embeddings.dim() != 3 or residue_ids.dim() != 2:
            raise ValueError("Invalid input shapes for ESMFoldContactNet.")

        padding_mask = (embeddings.abs().sum(dim=-1) == 0)
        trunk_mask = ~padding_mask
        if valid_mask is not None:
            trunk_mask = trunk_mask & valid_mask
        trunk_mask_float = trunk_mask.float()

        # Residue embedding + gated residual
        residue_emb = self.residue_embedding(residue_ids)
        residue_emb = self.residue_proj(residue_emb)
        concat = torch.cat([embeddings, residue_emb], dim=-1)
        gate = torch.sigmoid(self.gate_linear(concat))
        seq = embeddings + gate * residue_emb

        seq = self.input_proj(seq)
        seq = self.input_norm(seq)
        seq = self.model_proj(seq)
        seq = self.input_dropout(seq)

        # Pair init: symmetric pair features + relative position
        p = self.pair_proj(seq)  # (B, L, pair_dim)
        pair_sym = p[:, :, None, :] + p[:, None, :, :]
        pair_sym = self.pair_mlp(pair_sym)

        L = seq.size(1)
        rel = self.relative_pos_emb(L, device=seq.device, dtype=seq.dtype)
        pair_init = pair_sym + rel
        pair_init = self.pair_norm(pair_init)

        residx = torch.arange(L, device=seq.device).long().unsqueeze(0).expand(seq.size(0), L)
        true_aa = torch.zeros_like(residx)

        trunk_out = self.trunk(seq, pair_init, true_aa, residx, trunk_mask_float)
        seq_rep = trunk_out["s_s"]
        pair_rep = trunk_out["s_z"]

        seq_logits = self.seq_head(self.seq_norm(seq_rep))

        pair_rep = self.pair_norm_out(pair_rep)
        contact_logits = self.contact_head(pair_rep).squeeze(-1)  # (B, L, L)
        # enforce symmetry
        contact_logits = 0.5 * (contact_logits + contact_logits.transpose(-1, -2))

        return seq_logits, contact_logits, trunk_mask

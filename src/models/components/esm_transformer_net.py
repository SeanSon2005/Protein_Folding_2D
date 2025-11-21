from typing import Optional, Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Generates rotary positional encodings for attention heads."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even.")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.sin().to(dtype), emb.cos().to(dtype)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings.")

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.rotary = RotaryEmbedding(self.head_dim, theta=rope_theta)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        sin, cos = self.rotary(seq_len, device=x.device, dtype=x.dtype)
        q, k = apply_rotary_pos_emb(q, k, sin, cos)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if padding_mask is not None:
            key_mask = padding_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(key_mask, torch.finfo(attn_scores.dtype).min)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        context = self.dropout(self.out_proj(context))
        x = residual + context

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.ffn_dropout(x)
        x = residual + x

        return x


class ESMTransformerNet(nn.Module):
    """Transformer classifier operating on pre-computed ESM embeddings."""

    def __init__(
        self,
        input_dim: int = 5120,
        projection_dim: int = 1024,
        model_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")

        self.input_proj = nn.Linear(input_dim, projection_dim)
        self.input_norm = nn.LayerNorm(projection_dim)
        if projection_dim != model_dim:
            self.model_proj = nn.Linear(projection_dim, model_dim)
        else:
            self.model_proj = nn.Identity()

        self.input_dropout = nn.Dropout(dropout)
        hidden_dim = ffn_dim or model_dim * 4
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    ffn_dim=hidden_dim,
                    dropout=dropout,
                    rope_theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, seq_len, input_dim) tensor of embeddings.
        :return: (batch, seq_len, num_classes) logits.
        """
        if x.dim() != 3:
            raise ValueError("Input tensor must be of shape (batch, seq_len, features).")

        padding_mask = (x.abs().sum(dim=-1) == 0)

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.model_proj(x)
        x = self.input_dropout(x)

        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)

        x = self.final_norm(x)
        logits = self.classifier(x)
        return logits

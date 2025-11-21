import torch
import torch.nn as nn

from src.data.constants import INPUT_ALPHABET


class ESMAndClassificationHeadNet(nn.Module):
    def __init__(
        self,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        num_classes: int = 10,
        num_finetune_layers: int = 0,
    ):
        super().__init__()
        
        # Load ESM2 model using torch.hub
        self.esm_model, self.esm_alphabet = torch.hub.load("facebookresearch/esm:main", esm_model_name)
        
        # Freeze ESM2 layers, then selectively unfreeze from the top of the encoder stack
        self._freeze_encoder()
        self.num_finetune_layers = max(0, num_finetune_layers)
        if self.num_finetune_layers > 0:
            self._unfreeze_last_layers(self.num_finetune_layers)
        self.finetune_encoder = self.num_finetune_layers > 0
        
        # Classifier Head
        # Input: 1280 (for t33), Output: num_classes
        embed_dim = self.esm_model.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes) 
        )
        
        # Mapping from DataModule indices to ESM indices
        self.register_buffer("mapping", self._create_mapping())
        
        self.cls_idx = self.esm_alphabet.cls_idx
        self.eos_idx = self.esm_alphabet.eos_idx
        self.pad_idx = self.esm_alphabet.padding_idx
        self.esm_layers = self.esm_model.num_layers

    def _freeze_encoder(self) -> None:
        for param in self.esm_model.parameters():
            param.requires_grad = False

    def _unfreeze_last_layers(self, num_layers: int) -> None:
        layers = None
        encoder = getattr(self.esm_model, "encoder", None)
        if encoder is not None and hasattr(encoder, "layers"):
            layers = encoder.layers
        elif hasattr(self.esm_model, "layers"):
            layers = self.esm_model.layers
        if layers is None:
            return
        num_layers = min(num_layers, len(layers))
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        # Ensure layer norm/head parameters of the encoder are trainable when fine-tuning
        if encoder is not None and hasattr(encoder, "layer_norm"):
            for param in encoder.layer_norm.parameters():
                param.requires_grad = True
        if hasattr(self.esm_model, "contact_head") and self.esm_model.contact_head is not None:
            for param in self.esm_model.contact_head.parameters():
                param.requires_grad = True

    def _create_mapping(self):
        # DataModule Vocab:
        # 0: <PAD>
        # 1..N: Sorted canonical amino acids + "X" for unknowns
        datamodule_vocab = {c: i + 1 for i, c in enumerate(INPUT_ALPHABET)}
        datamodule_vocab["<PAD>"] = 0
        
        # ESM Alphabet
        esm_vocab = self.esm_alphabet.to_dict()
        
        # Create mapping tensor: mapping[datamodule_idx] = esm_idx
        mapping = torch.zeros(len(datamodule_vocab), dtype=torch.long)
        
        for char, idx in datamodule_vocab.items():
            if char == "<PAD>":
                esm_idx = esm_vocab["<pad>"]
            else:
                # If char not in esm, use <unk>
                esm_idx = esm_vocab.get(char, esm_vocab.get("<unk>"))
            mapping[idx] = esm_idx
            
        return mapping

    def forward(self, x):
        # x shape: (B, L) with indices 0..20
        B, L = x.shape
        device = x.device
        
        # Map to ESM indices
        esm_indices = self.mapping[x] # (B, L)
        
        # Calculate lengths (excluding padding)
        # 0 is the padding index in our input vocab
        lengths = (x != 0).sum(dim=1)
        
        # Create new tensor for ESM input: (B, L+2)
        # Initialize with ESM padding index
        esm_input = torch.full((B, L + 2), self.pad_idx, device=device, dtype=torch.long)
        
        # Set CLS token at start
        esm_input[:, 0] = self.cls_idx
        
        # Fill sequences and add EOS
        for i in range(B):
            l = lengths[i]
            esm_input[i, 1:l+1] = esm_indices[i, :l]
            esm_input[i, l+1] = self.eos_idx
            
        # Pass to ESM (enable gradients only when encoder is partially unfrozen and in training mode)
        enable_grads = self.finetune_encoder and self.training
        with torch.set_grad_enabled(enable_grads):
            results = self.esm_model(esm_input, repr_layers=[self.esm_layers], return_contacts=False)
        
        # Extract representations: (B, L+2, 1280)
        representations = results["representations"][self.esm_layers]
        
        # We need to map back to (B, L, 1280)
        # Removing CLS and EOS, and keeping padding where it was
        out = torch.zeros((B, L, representations.shape[-1]), device=device, dtype=representations.dtype)
        
        for i in range(B):
            l = lengths[i]
            # Take 1..l+1 (skipping CLS, taking sequence)
            out[i, :l] = representations[i, 1:l+1]
            
        # Pass through classifier
        logits = self.classifier(out) # (B, L, num_classes)
        
        return logits

"""CRF wrapper module for protein sequence labeling.

This module provides a wrapper around pytorch-crf that handles:
- Padding mask conversion (padding index 0 is excluded from CRF states)
- Label shifting (model labels 1..C map to CRF states 0..C-1)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchcrf import CRF


class ProteinCRF(nn.Module):
    """Linear-chain CRF wrapper for protein sequence labeling.
    
    This wrapper handles the padding convention where label 0 is padding
    and should not participate in CRF transitions. The actual class labels
    are 1 to num_classes, which get mapped to CRF states 0 to num_classes-1.
    
    Args:
        num_tags: Number of actual class labels (excluding padding).
                  If your labels are 0 (pad), 1, 2, ..., C, then num_tags = C.
        batch_first: Whether the batch dimension is first. Default: True.
    """
    
    def __init__(self, num_tags: int, batch_first: bool = True) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.crf = CRF(num_tags=num_tags, batch_first=batch_first)
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute the negative log-likelihood loss.
        
        Args:
            emissions: Emission scores from the model. Shape: (B, L, num_tags).
                       These should already be sliced to exclude the padding class
                       (i.e., emissions for classes 1..C, not 0..C).
            tags: Target labels with padding index 0. Shape: (B, L).
                  Labels should be in range [0, num_tags] where 0 is padding.
            mask: Optional boolean mask. If None, will be computed from tags != 0.
                  Shape: (B, L). True for valid positions, False for padding.
            reduction: Reduction method. One of "mean", "sum", "none".
        
        Returns:
            Negative log-likelihood loss (negated because pytorch-crf returns
            log-likelihood, but we want loss for minimization).
        """
        if mask is None:
            mask = tags != 0
        
        # Shift tags: label 1 -> CRF state 0, label 2 -> CRF state 1, etc.
        # For masked (padding) positions, we use 0 as placeholder (won't affect loss)
        shifted_tags = (tags - 1).clamp(min=0)
        
        # pytorch-crf's forward returns log-likelihood, negate for NLL loss
        log_likelihood = self.crf(emissions, shifted_tags, mask=mask, reduction=reduction)
        return -log_likelihood
    
    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Find the most likely tag sequence using Viterbi algorithm.
        
        Args:
            emissions: Emission scores from the model. Shape: (B, L, num_tags).
            mask: Optional boolean mask. Shape: (B, L).
                  True for valid positions, False for padding.
        
        Returns:
            Tensor of predicted labels shifted back to original space (1..C).
            Padding positions are filled with 0. Shape: (B, L).
        """
        # Viterbi decode returns list of lists
        best_paths: List[List[int]] = self.crf.decode(emissions, mask=mask)
        
        batch_size = emissions.size(0)
        seq_len = emissions.size(1)
        device = emissions.device
        
        # Efficiently convert list of lists to tensor
        # Pad all paths to seq_len and create tensor in one operation
        # This avoids creating individual tensors per sample (major GPU bottleneck)
        padded_paths = []
        for path in best_paths:
            path_len = len(path)
            if path_len < seq_len:
                # Pad with -1 (will become 0 after +1 shift)
                padded_paths.append(path + [-1] * (seq_len - path_len))
            else:
                padded_paths.append(path[:seq_len])
        
        # Single tensor creation + single GPU transfer
        preds = torch.tensor(padded_paths, dtype=torch.long, device=device) + 1
        # -1 + 1 = 0, so padding positions are correctly 0
        
        return preds
    
    @property
    def transitions(self) -> torch.Tensor:
        """Access the transition matrix."""
        return self.crf.transitions
    
    @property
    def start_transitions(self) -> Optional[torch.Tensor]:
        """Access start transition scores."""
        return self.crf.start_transitions
    
    @property
    def end_transitions(self) -> Optional[torch.Tensor]:
        """Access end transition scores."""
        return self.crf.end_transitions

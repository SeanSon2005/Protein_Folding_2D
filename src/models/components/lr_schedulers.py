import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    """Linear warmup followed by cosine annealing measured in optimizer steps."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive for WarmupCosineLR")
        self.total_steps = total_steps

        if warmup_steps is None:
            warmup_steps = int(total_steps * warmup_ratio)
        self.warmup_steps = max(0, min(warmup_steps, total_steps))
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = min(self.last_epoch + 1, self.total_steps)

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            warmup_progress = step / max(1, self.warmup_steps)
            return [base_lr * warmup_progress for base_lr in self.base_lrs]

        if self.total_steps == self.warmup_steps:
            return [self.eta_min for _ in self.base_lrs]

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs
        ]

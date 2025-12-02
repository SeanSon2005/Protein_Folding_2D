from typing import Any, Dict, Optional, Tuple
from copy import deepcopy

from hydra.utils import instantiate, get_class
from omegaconf import DictConfig
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class ProteinLitModule(LightningModule):
    """LightningModule for Protein Folding classification."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[DictConfig],
        compile: bool,
        num_classes: int = 10,
    ) -> None:
        """Initialize a `ProteinLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Number of output classes produced by `net`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        # Ignore padding index 0
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # metric objects for calculating and averaging accuracy across batches
        # ignore padding index 0
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=0)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=0)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=0)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self._scheduler_cfg = scheduler
        self._scheduler_callable = None
        self.scheduler_interval = "epoch"
        self.scheduler_monitor = "val/loss"
        self.scheduler_frequency = 1
        self.scheduler_requires_total_steps = False
        self.scheduler_target_cfg: Optional[DictConfig] = None
        self.scheduler_target_class: Optional[type] = None
        self.scheduler_init_kwargs: Dict[str, Any] = {}

        if isinstance(scheduler, (DictConfig, dict)):
            cfg = scheduler
            target_cfg = cfg.get("target")
            target_class_path = cfg.get("target_class") or cfg.get("class_path")
            if target_cfg is not None:
                self.scheduler_target_cfg = target_cfg
            elif target_class_path is not None:
                self.scheduler_target_class = get_class(target_class_path)

            self.scheduler_interval = scheduler.get("interval", self.scheduler_interval)
            self.scheduler_monitor = scheduler.get("monitor", self.scheduler_monitor)
            self.scheduler_frequency = scheduler.get("frequency", self.scheduler_frequency)
            self.scheduler_requires_total_steps = scheduler.get(
                "requires_total_steps", self.scheduler_requires_total_steps
            )

            meta_keys = {
                "target",
                "target_class",
                "class_path",
                "interval",
                "frequency",
                "monitor",
                "requires_total_steps",
            }
            for key, value in scheduler.items():
                if key not in meta_keys:
                    self.scheduler_init_kwargs[key] = value
        else:
            self._scheduler_callable = scheduler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of protein sequences.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of sequences and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        # logits: (B, L, C)
        # y: (B, L)
        # Permute logits for CrossEntropyLoss: (B, C, L)
        loss = self.criterion(logits.permute(0, 2, 1), y)
        preds = torch.argmax(logits, dim=2)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of sequences and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of sequences and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of sequences and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler_cfg = self.scheduler_target_cfg
        scheduler_callable = self._scheduler_callable
        if (
            scheduler_cfg is not None
            or scheduler_callable is not None
            or self.scheduler_target_class is not None
        ):
            if scheduler_cfg is not None:
                scheduler_kwargs = {}
                if self.scheduler_requires_total_steps:
                    scheduler_kwargs["total_steps"] = self.trainer.estimated_stepping_batches
                scheduler_cfg_instance = scheduler_cfg
                if isinstance(scheduler_cfg, (DictConfig, dict)):
                    scheduler_cfg_instance = deepcopy(scheduler_cfg)
                for meta_key in ("interval", "frequency", "monitor", "requires_total_steps"):
                    if isinstance(scheduler_cfg_instance, dict):
                        scheduler_cfg_instance.pop(meta_key, None)
                    elif isinstance(scheduler_cfg_instance, DictConfig):
                        if meta_key in scheduler_cfg_instance:
                            del scheduler_cfg_instance[meta_key]
                scheduler = instantiate(scheduler_cfg_instance, optimizer=optimizer, **scheduler_kwargs)
            elif self.scheduler_target_class is not None:
                scheduler_kwargs = dict(self.scheduler_init_kwargs)
                if self.scheduler_requires_total_steps:
                    scheduler_kwargs["total_steps"] = self.trainer.estimated_stepping_batches
                scheduler = self.scheduler_target_class(optimizer, **scheduler_kwargs)
            else:
                scheduler = scheduler_callable(optimizer=optimizer)
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": self.scheduler_interval,
                "frequency": self.scheduler_frequency,
            }
            if self.scheduler_monitor:
                lr_scheduler_config["monitor"] = self.scheduler_monitor
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
        return {"optimizer": optimizer}
    
class ProteinCRFLitModule(ProteinLitModule):
    """LightningModule for Protein Folding classification with CRF layer.
    
    Uses a linear-chain CRF on top of the emission scores from the neural network.
    Training uses CRF negative log-likelihood loss.
    Inference uses Viterbi decoding to find the best label sequence.
    
    Note: The network `net` should output emissions of shape (B, L, num_classes - 1)
    since padding (index 0) is not a valid CRF state. Alternatively, if net outputs
    (B, L, num_classes), set exclude_padding_class=True to slice off the padding dim.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[DictConfig],
        compile: bool,
        num_classes: int = 10,
        exclude_padding_class: bool = True,
    ) -> None:
        """Initialize a `ProteinCRFLitModule`.

        :param net: The model to train (outputs emission scores).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Total number of classes including padding (index 0).
        :param exclude_padding_class: If True, slice emissions to exclude padding class.
        """
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            num_classes=num_classes,
        )
        
        self.exclude_padding_class = exclude_padding_class
        
        # Number of actual tags for CRF (excluding padding)
        num_tags = num_classes - 1
        
        # Import here to avoid circular imports
        from src.models.components.crf import ProteinCRF
        
        # Initialize CRF layer
        self.crf = ProteinCRF(num_tags=num_tags, batch_first=True)
        
        # Override criterion - we'll use CRF loss instead
        # Keep parent's criterion for potential comparison, but mark it unused
        self._ce_criterion = self.criterion
        self.criterion = None  # CRF loss is computed via self.crf

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], decode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor 
                      of sequences and target labels.
        :param decode: If True, run Viterbi decoding for predictions. If False,
                       use argmax on emissions (faster, for training metrics).

        :return: A tuple containing (in order):
            - A tensor of CRF negative log-likelihood loss.
            - A tensor of predictions (Viterbi-decoded if decode=True, else argmax).
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        # logits: (B, L, C) where C = num_classes
        
        # Extract emissions for CRF (exclude padding class if needed)
        if self.exclude_padding_class:
            # Slice off index 0 (padding) -> emissions for classes 1..C
            emissions = logits[:, :, 1:]
        else:
            emissions = logits
        
        # Create mask: True for non-padding positions
        mask = (y != 0)
        
        # CRF negative log-likelihood loss
        loss = self.crf(emissions, y, mask=mask, reduction="mean")
        
        if decode:
            # Viterbi decoding for predictions (slower but exact)
            preds = self.crf.decode(emissions, mask=mask)
        else:
            # Fast approximation: argmax on emissions (ignores transitions)
            # Shift +1 to match label space (CRF state 0 -> label 1)
            preds = torch.argmax(emissions, dim=2) + 1
            # Zero out padding positions
            preds = preds * mask.long()
        
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Uses fast argmax predictions instead of Viterbi for training metrics.
        """
        loss, preds, targets = self.model_step(batch, decode=False)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Uses Viterbi decoding for accurate validation metrics.
        """
        loss, preds, targets = self.model_step(batch, decode=True)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Uses Viterbi decoding for accurate test metrics.
        """
        loss, preds, targets = self.model_step(batch, decode=True)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)


class ProteinResLitModule(ProteinLitModule):
    """LightningModule for Protein Folding with residue embeddings.
    
    Handles 3-tuple batches: (embeddings, residue_ids, targets) for models
    like ESMTransformerResNet that use both ESM embeddings and residue sequences.
    """

    def forward(self, x: torch.Tensor, residue_ids: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of ESM embeddings.
        :param residue_ids: A tensor of residue (amino acid) indices.
        :return: A tensor of logits.
        """
        return self.net(x, residue_ids)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a 3-tuple) containing ESM embeddings,
                      residue IDs, and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, residue_ids, y = batch
        logits = self.forward(x, residue_ids)
        # logits: (B, L, C)
        # y: (B, L)
        # Permute logits for CrossEntropyLoss: (B, C, L)
        loss = self.criterion(logits.permute(0, 2, 1), y)
        preds = torch.argmax(logits, dim=2)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)


class ProteinResCRFLitModule(ProteinResLitModule):
    """LightningModule for Protein Folding with residue embeddings and CRF layer.
    
    Handles 3-tuple batches: (embeddings, residue_ids, targets) for models
    like ESMTransformerResNet, with a CRF layer for sequence labeling.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[DictConfig],
        compile: bool,
        num_classes: int = 10,
        exclude_padding_class: bool = True,
    ) -> None:
        """Initialize a `ProteinResCRFLitModule`.

        :param net: The model to train (outputs emission scores).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Total number of classes including padding (index 0).
        :param exclude_padding_class: If True, slice emissions to exclude padding class.
        """
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            num_classes=num_classes,
        )
        
        self.exclude_padding_class = exclude_padding_class
        
        # Number of actual tags for CRF (excluding padding)
        num_tags = num_classes - 1
        
        # Import here to avoid circular imports
        from src.models.components.crf import ProteinCRF
        
        # Initialize CRF layer
        self.crf = ProteinCRF(num_tags=num_tags, batch_first=True)
        
        # Override criterion - we'll use CRF loss instead
        self._ce_criterion = self.criterion
        self.criterion = None  # CRF loss is computed via self.crf

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], decode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a 3-tuple) containing ESM embeddings,
                      residue IDs, and target labels.
        :param decode: If True, run Viterbi decoding for predictions.

        :return: A tuple containing (in order):
            - A tensor of CRF negative log-likelihood loss.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, residue_ids, y = batch
        logits = self.forward(x, residue_ids)
        # logits: (B, L, C) where C = num_classes
        
        # Extract emissions for CRF (exclude padding class if needed)
        if self.exclude_padding_class:
            emissions = logits[:, :, 1:]
        else:
            emissions = logits
        
        # Create mask: True for non-padding positions
        mask = (y != 0)
        
        # CRF negative log-likelihood loss
        loss = self.crf(emissions, y, mask=mask, reduction="mean")
        
        if decode:
            preds = self.crf.decode(emissions, mask=mask)
        else:
            preds = torch.argmax(emissions, dim=2) + 1
            preds = preds * mask.long()
        
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step using fast argmax predictions."""
        loss, preds, targets = self.model_step(batch, decode=False)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step using Viterbi decoding."""
        loss, preds, targets = self.model_step(batch, decode=True)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step using Viterbi decoding."""
        loss, preds, targets = self.model_step(batch, decode=True)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)


class ProteinContactModule(ProteinLitModule):
    """LightningModule for joint sequence (CRF) + contact prediction."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[DictConfig],
        compile: bool,
        num_classes: int = 10,
        exclude_padding_class: bool = True,
        contact_weight: float = 1.0,
    ) -> None:
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            num_classes=num_classes,
        )
        self.exclude_padding_class = exclude_padding_class
        self.contact_weight = contact_weight

        num_tags = num_classes - 1
        from src.models.components.crf import ProteinCRF

        self.crf = ProteinCRF(num_tags=num_tags, batch_first=True)
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

        # separate trackers
        self.train_seq_loss = MeanMetric()
        self.train_contact_loss = MeanMetric()
        self.val_seq_loss = MeanMetric()
        self.val_contact_loss = MeanMetric()
        self.test_seq_loss = MeanMetric()
        self.test_contact_loss = MeanMetric()

    def forward(
        self,
        embeddings: torch.Tensor,
        residue_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        return self.net(embeddings, residue_ids, valid_mask)

    def _contact_loss(
        self,
        contact_logits: torch.Tensor,
        contact_targets: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_raw = self.bce(contact_logits, contact_targets)
        masked_loss = loss_raw * contact_mask
        denom = contact_mask.sum().clamp(min=1)
        return masked_loss.sum() / denom

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        decode: bool = True,
    ):
        embeddings, residue_ids, targets, contact_map, valid_mask = batch
        seq_logits, contact_logits, trunk_mask = self.forward(embeddings, residue_ids, valid_mask)

        # Sequence loss via CRF
        emissions = seq_logits[:, :, 1:] if self.exclude_padding_class else seq_logits
        mask_seq = targets != 0
        seq_loss = self.crf(emissions, targets, mask=mask_seq, reduction="mean")
        if decode:
            preds = self.crf.decode(emissions, mask=mask_seq)
        else:
            preds = torch.argmax(emissions, dim=2) + (1 if self.exclude_padding_class else 0)
            preds = preds * mask_seq.long()

        # Contact loss (mask-aware BCE)
        # contact_mask combines padding mask and valid structure mask if provided
        if valid_mask is not None:
            contact_valid = mask_seq & valid_mask
        else:
            contact_valid = mask_seq
        contact_mask = contact_valid.unsqueeze(1) & contact_valid.unsqueeze(2)
        contact_loss = self._contact_loss(contact_logits, contact_map, contact_mask)

        total_loss = seq_loss + self.contact_weight * contact_loss
        return total_loss, seq_loss.detach(), contact_loss.detach(), preds, targets

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, seq_loss, contact_loss, preds, targets = self.model_step(batch, decode=False)

        self.train_loss(loss)
        self.train_seq_loss(seq_loss)
        self.train_contact_loss(contact_loss)
        self.train_acc(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/seq_loss", self.train_seq_loss, on_step=False, on_epoch=True)
        self.log("train/contact_loss", self.train_contact_loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss, seq_loss, contact_loss, preds, targets = self.model_step(batch, decode=True)

        self.val_loss(loss)
        self.val_seq_loss(seq_loss)
        self.val_contact_loss(contact_loss)
        self.val_acc(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/seq_loss", self.val_seq_loss, on_step=False, on_epoch=True)
        self.log("val/contact_loss", self.val_contact_loss, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss, seq_loss, contact_loss, preds, targets = self.model_step(batch, decode=True)

        self.test_loss(loss)
        self.test_seq_loss(seq_loss)
        self.test_contact_loss(contact_loss)
        self.test_acc(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/seq_loss", self.test_seq_loss, on_step=False, on_epoch=True)
        self.log("test/contact_loss", self.test_contact_loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

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


if __name__ == "__main__":
    _ = ProteinLitModule(None, None, None, None)

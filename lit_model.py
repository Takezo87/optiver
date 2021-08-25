# import argparse
import pytorch_lightning as pl
import torch

# from util.core import *


# OPTIMIZER = "AdamW"
OPTIMIZER = "AdamW"
LR = 1e-5
LOSS = "mse_loss"
ONE_CYCLE_TOTAL_STEPS = None
PRETRAINED_OU = '/home/johannes/google-drive/experiments/models/pretrained/bets_historic_20210302_over_bets_processed_12c_2yoverunder_80_110_110_tst_tsbert_100_points.pth'


class Accuracy(pl.metrics.Accuracy):
    """Accuracy Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Metrics in Pytorch-lightning 1.2+ versions expect preds to be between 0 and 1 else fails with the ValueError:
        "The `preds` should be probabilities, but values were detected outside of [0,1] range."
        This is being tracked as a bug in https://github.com/PyTorchLightning/metrics/issues/60.
        This method just hacks around it by normalizing preds before passing it in.
        Normalized preds are not necessary for accuracy computation as we just care about argmax().
        """
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.nn.functional.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)

def rmspe(y_pred, y_true):
    return (torch.sqrt(torch.mean(torch.square((y_true - y_pred) / y_true))))

class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args = None):
        super().__init__()
        self.model = model.float()
        self.loss_fn = rmspe
        self.optimizer_class = getattr(torch.optim, OPTIMIZER)
        self.lr = LR
        self.weight_decay = 1e-2




    def forward(self, *x):
        return self.model(*x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # x, y = batch[:-1], batch[-1]
        # x, y = batch
        # logits = self(*[batch[k] for k in self.x_keys])
        logits = self(batch[0])
        # loss = self.loss_fn(logits, batch['y'])
        loss = self.loss_fn(logits, batch[1])
        self.log("train_loss", loss)
        # self.train_acc(logits, y)
        # self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        print(batch)
        # logits = self(*[batch[k] for k in self.x_keys])
        logits = self(batch[0])
        # print(logits)
        loss = self.loss_fn(logits, batch[1])
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        print('configure_optimizers')
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr,
              weight_decay=self.weight_decay)
        return {"optimizer": optimizer}

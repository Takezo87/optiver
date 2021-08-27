# import argparse
import pytorch_lightning as pl
import torch

# from util.core import *


# OPTIMIZER = "AdamW"
OPTIMIZER = "AdamW"
LR = 3e-4
LOSS = "mse_loss"
ONE_CYCLE_TOTAL_STEPS = None



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
        logits = self(batch['x'])
        # loss = self.loss_fn(logits, batch['y'])
        loss = self.loss_fn(logits, batch['y'])
        self.log("train_loss", loss)
        # self.train_acc(logits, y)
        # self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        print(batch)
        # logits = self(*[batch[k] for k in self.x_keys])
        logits = self(batch['x'])
        # print(logits)
        loss = self.loss_fn(logits, batch['y'])
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        print('configure_optimizers')
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr,
              weight_decay=self.weight_decay)
        return {"optimizer": optimizer}

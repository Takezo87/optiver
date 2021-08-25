import pytorch_lightning as pl
from dataset import OpDataset

from torch.utils.data import DataLoader

BATCH_SIZE=8


class DM(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__()

    def setup(self, stage=None):
        self.ds = OpDataset()

    def train_dataloader(self):
        dl = DataLoader(self.ds, batch_size=BATCH_SIZE)
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.ds, batch_size=BATCH_SIZE)
        return dl


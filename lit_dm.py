from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from dataset import OpDataset

from torch.utils.data import DataLoader

BATCH_SIZE=64


class DM(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__()
        self.dir = './data/'
        

    def setup(self, stage=None):
        self.df = pd.read_csv(Path(self.dir)/'train.csv')
        train, valid = train_test_split(self.df) 
        self.ds_train = OpDataset(train)
        self.ds_valid = OpDataset(valid)


    def train_dataloader(self):
        dl = DataLoader(self.ds_train, batch_size=BATCH_SIZE)
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.ds_valid, batch_size=BATCH_SIZE)
        return dl


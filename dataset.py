from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from transforms import Standardize

class OpDataset(Dataset):
    
    def __init__(self, df, dir='./data/'):
        self.dir = dir
        self.train = df
        self.book_train_files = list((Path(self.dir)/'book_train.parquet/').iterdir())
        self.standardize = Standardize()
        self.means = np.array([
                0.99969482421875, 1.000321388244629, 0.9995064735412598, 1.0005191564559937,
                769.990177708821, 766.7345672818379, 959.3416027831918, 928.2202512713748])
        
        self.stds = np.array([
                0.0036880988627672195, 0.003687119111418724, 0.0037009266670793295, 0.0036990800872445107,
                5354.051690318169, 4954.947103063445, 6683.816183660414, 5735.299917793827])
                # 0.003689893218043926, 0.00370745215558702, 6.618708642293018e-07, 1.2508970015188411e-06
            # ])   
        
    def __len__(self):
        return self.train.shape[0]
    
    def __getitem__(self, i):
        row = self.train.iloc[i]
        stock_id, time_id, target = int(row.stock_id), int(row.time_id), row.target
        # stock_id, time_id, target = row.stock_id.values, row.time_id.values, row.target.values
        # print(stock_id, time_id, target)
        fn_parquet = f'{self.dir}book_train.parquet/stock_id={stock_id}'
        # print(fn_parquet)
        data = pd.read_parquet(fn_parquet)
        bucket = data.loc[data.time_id==time_id]
        bucket.set_index('seconds_in_bucket', inplace=True)
        bucket=bucket.reindex(list(range(0,600)), fill_value=0)

        sequences = (bucket[bucket.columns[1:]].values - self.means)/self.stds
        return {'x':torch.tensor(sequences, dtype=torch.float).permute(-1,0), 'y':torch.tensor(target)}
        

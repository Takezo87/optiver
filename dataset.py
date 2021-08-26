from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class OpDataset(Dataset):
    
    def __init__(self, df, dir='./data/'):
        self.dir = dir
        self.train = df
        self.book_train_files = list((Path(self.dir)/'book_train.parquet/').iterdir())
        
    def __len__(self):
        return self.train.shape[0]
    
    def __getitem__(self, i):
        row = self.train.iloc[i]
        stock_id, time_id, target = int(row.stock_id), int(row.time_id), row.target
        print(stock_id, time_id, target)
        fn_parquet = f'{self.dir}book_train.parquet/stock_id={stock_id}'
        print(fn_parquet)
        data = pd.read_parquet(fn_parquet)
        bucket = data.loc[data.time_id==time_id]
        bucket.set_index('seconds_in_bucket', inplace=True)
        bucket=bucket.reindex(list(range(0,600)), fill_value=0)
        return torch.tensor(bucket.values, dtype=torch.float).permute(-1,0), torch.tensor(target)
        

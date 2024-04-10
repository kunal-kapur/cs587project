import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class MovieLensDataSet(Dataset):
    def __init__(self, filename) -> None:
        self.df = pd.read_csv(filename)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.df.iloc[idx, 1:]


# class MovielensDataLoader(DataLoader):
#     def __init__(self):
#         super().__init__()
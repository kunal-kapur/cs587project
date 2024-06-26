import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class BooksDataset:

    def __init__(self, path):
        df = pd.read_csv(path)
        ratings = df[['Book-Rating']].map(lambda x: (((x + 3)//3) - 1))
        df = df.drop(['Book-Rating'], axis=1)

        self.categories_list = []
        # could also do max. Encoded from values 0-len
        for column in df.columns[1:7]:
            self.categories_list.append(len(df[column].unique()))
        
        self.categorical = torch.tensor(df.iloc[:, 1:7].values)
        self.numerical = torch.tensor(df.iloc[:, 7:].values)

        self.ratings = torch.tensor(ratings.values).squeeze()
        # print(self.ratings.shape)
        # self.ratings = torch.nn.functional.one_hot(self.ratings, num_classes=4)
    
    def __len__(self):
        return len(self.ratings)

    def get_category_list(self):
        return self.categories_list.copy()



    def __getitem__(self, idx):

        return self.categorical[idx], self.numerical[idx].float(), self.ratings[idx]
    

class AvazuDataSet:

    def __init__(self, path):
        df = pd.read_csv(path)
        ratings = df[['click']]
        df = df.drop(['click'], axis=1)

        self.categories_list = []
        # could also do max. Encoded from values 0-len
        for column in df.columns[1:3]:
            self.categories_list.append(len(df[column].unique()))
        
        self.categorical = torch.tensor(df.iloc[:, 1:len(df.columns)].values)
        self.numerical = torch.tensor(df.iloc[:, len(df.columns):].values)

        self.ratings = torch.tensor(ratings.values).squeeze()
    
    def __len__(self):
        return len(self.ratings)

    def get_category_list(self):
        return self.categories_list.copy()



    def __getitem__(self, idx):

        return self.categorical[idx], self.numerical[idx].float(), self.ratings[idx]
        
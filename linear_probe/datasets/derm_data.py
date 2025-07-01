# import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import torch

def load_data_balanced(df, data_percent=0.5, label_col='label'):
    df_train = df.query("split == 'train'").sample(frac=1, random_state=42).reset_index(drop=True)

    def sample_at_least_one(g):
        n = max(1, int(len(g) * data_percent))
        n = min(n, len(g))
        return g.sample(n=n, random_state=42)

    df_bal = (
        df_train.groupby(label_col, group_keys=False)
                .apply(sample_at_least_one)
    )

    df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_bal


class Derm_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, root, image_key='filename', train=False,val=False,test=False,transforms=None, data_percent=1):
        """
		Class initialization
		Args:
			df (pd.DataFrame): DataFrame with data description
			train (bool): flag of whether a training dataset is being initialized or testing one
			transforms: image transformation method to be applied
			meta_features (list): list of features with meta information, such as sex and age

		"""
        if train==True:
            self.df = df[df['split']=='train']
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            # half_rows = int(len(self.df) * data_percent)
            # self.df=self.df.head(half_rows)
            if data_percent < 1:
                self.df = load_data_balanced(df, data_percent=data_percent, label_col='label')

        elif val==True:
            self.df = df[df['split'] == 'val']
        elif test==True:
            self.df = df[df['split'] == 'test']
        self.transforms = transforms
        self.root=root
        self.image_key = image_key

    def __getitem__(self, index):
        im_path = self.df.iloc[index][self.image_key]
        
        x = Image.open(im_path).convert('RGB')

        if self.transforms:
            x = self.transforms(x)
            y = self.df.iloc[index]['label']
        return x,y,im_path

    def __len__(self):
        return len(self.df)

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import librosa
import os
import pandas as pd



class LoadDataset(Dataset):
    def __init__(self, ds_path, mode, esc10_only=False):
        meta_path = os.path.join(ds_path, "meta", "esc50.csv")
        if mode == "train":
            fold = ["1", "2", "3", "4"]
        elif mode == "test":
            fold = ["5"]
        else:
            raise ValueError('Incorrect mode, must be train or test')
        df = pd.read_csv(meta_path)
        df = df[df["fold"].isin(fold)]
        if esc10_only:
            df = df[df["esc10"] == True]
            d = dict(zip(df["target"].unique(), range(10)))
            df['target'] = df['target'].map(lambda x: d[x])
        self.files = [os.path.join(ds_path, "audio", x) for x in df["filename"]]
        self.targets = [x for x in df["target"]]

    def __getitem__(self, index):
        im, lb = self.pull_item(index)
        return im, lb

    def __len__(self):
        return len(self.files)

    def pull_item(self, index):
        x, _ = librosa.load(self.files[index])
        return torch.FloatTensor(x).unsqueeze(0), torch.LongTensor([self.targets[index]])


if __name__ == "__main__":
    loader = LoadDataset("/home/rauf/workspace/data/ESC-50", "train", True)
    for i in loader:
        print (i)
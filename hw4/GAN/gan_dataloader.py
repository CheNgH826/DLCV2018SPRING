from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from skimage import io

class FaceDataset(Dataset):
    def __init__(self, img_dir, attr_csv, transfrom=None):
        self.img_dir = img_dir
        self.attr = pd.read_csv(attr_csv)
        self.transfrom = transfrom
    def __len__(self):
        return len(self.attr)
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.attr.iloc[idx, 0])
        img = io.imread(img_name)
        if self.transfrom:
            img = self.transfrom(img)
        return img
            
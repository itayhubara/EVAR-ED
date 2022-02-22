from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR='/home/ihubara/experiments/DevinProject/data/data_evar.npz'

class EVARDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, filename=ROOT_DIR,train_val='train', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        npzfile = np.load(filename)
        if train_val=='train':
            self.features = npzfile["features_train"]
            self.y = npzfile["target_train"]
        else:
            self.features = npzfile["features_val"]
            self.y = npzfile["target_val"]
        npzfile=None
        self.filename = filename
        self.transform = transform


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = self.features[idx]
        y = self.y[idx]
        return features,y #,idx

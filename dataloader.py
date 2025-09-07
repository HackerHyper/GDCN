import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import random
import sklearn



class NGs():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'NGs.mat')
        self.Y = data['Y'].astype(np.int32)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 500
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()



def load_data(dataset):
    if  dataset == "NGs":
        dataset = NGs('./data/')
        dims = [2000, 2000, 2000]
        view = 3
        data_size = 500
        class_num = 5
    
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

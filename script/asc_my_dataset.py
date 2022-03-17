"""
データセットを作成するプログラム
1サンプルずつ、.pickle形式で保存してるはずだから、それを1つずつ読み込む
"""



import pickle
import glob
import torch
import numpy as np
from tensorflow.python.ops.gen_linalg_ops import self_adjoint_eig
from torch.utils.data import Dataset
from torchvision import transforms





class AscMyDataset(Dataset):
    def __init__(self, transform, dataset_name, feature_name, data_type):
        self.top_path = '../output/' + dataset_name + '/' + feature_name + '/' + data_type + '/'
        self.transform = transform

    def __getitem__(self, index):
        path = self.top_path + 'feature_' + str(index) + '.pickle'
        with open(path, 'rb') as f:
            image = pickle.load(f)
        if self.transform is not None:
            image = self.transform(image)
        with open(self.top_path + 'label.pickle', 'rb') as f:
            labels = pickle.load(f)
        classes = np.unique(labels)
        classes = {v: i for i, v in enumerate(sorted(classes))}
        return image, classes[labels[index]]

    def __len__(self):
        return len(glob.glob(self.top_path + 'feature_*.pickle'))
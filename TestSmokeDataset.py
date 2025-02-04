from PIL import Image
#from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader

import skimage


class SmokeDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_fns = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_fns['data'])

    def __getitem__(self, idx):
        data_fn = self.data_fns['data'][idx]
        truth_fn = self.data_fns['truth'][idx]
        data_img = skimage.io.imread(data_fn, plugin='tifffile')
        truth_img = skimage.io.imread(truth_fn, plugin='tifffile')
        data_tensor = self.transform(data_img)#.unsqueeze_(0)
        truth_tensor = self.transform(truth_img)#.unsqueeze_(0)
        truth_tensor = truth_tensor.type(torch.float)

        return data_tensor, truth_tensor, truth_fn

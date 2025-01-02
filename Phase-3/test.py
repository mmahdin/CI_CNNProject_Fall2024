import pickle
import torch.optim as optim
import os
from ph3 import *
import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
import time
import numpy as np

# for plotting
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


if torch.cuda.is_available:
    print('Good to go!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    print('Please set GPU via Edit -> Notebook Settings.')

# data type and device for torch.tensor
to_float = {'dtype': torch.float, 'device': 'cpu'}
to_float_cuda = {'dtype': torch.float, 'device': 'cuda'}
to_double = {'dtype': torch.double, 'device': 'cpu'}
to_double_cuda = {'dtype': torch.double, 'device': 'cuda'}
to_long = {'dtype': torch.long, 'device': 'cpu'}
to_long_cuda = {'dtype': torch.long, 'device': 'cuda'}


# Configuration
image_size = (299, 299)
flicker = '8k'
data_dict_path = {'8k': "./dataset/image_captioning_dataset.pt",
                  '30k': "./dataset/flicker30k.pt"}
captions_path = {'8k': "./dataset/archive/captions.txt",
                 '30k': "./dataset/captions.csv"}
data_path = {'8k': "./dataset/archive/images/", '30k': "./dataset/flicker30k/"}

data_dict = load_data(data_dict_path, captions_path,
                      data_path, image_size, flicker=flicker)

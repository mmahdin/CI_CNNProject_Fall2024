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

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def main(checkpoint_path, image_path):
    # data type and device for torch.tensor
    to_float = {'dtype': torch.float, 'device': 'cpu'}
    to_float_cuda = {'dtype': torch.float, 'device': 'cuda'}
    to_double = {'dtype': torch.double, 'device': 'cpu'}
    to_double_cuda = {'dtype': torch.double, 'device': 'cuda'}
    to_long = {'dtype': torch.long, 'device': 'cpu'}
    to_long_cuda = {'dtype': torch.long, 'device': 'cuda'}

    vocab = read_captions_and_build_vocab(
        "./dataset/archive/captions.txt", min_freq=1)

    # create the image captioning model
    rnn_model = CaptioningRNN(
        cell_type='attention',
        word_to_idx=vocab.token_to_idx,
        token_to_idx=vocab.idx_to_token,
        input_dim=1280,
        hidden_dim=512,
        wordvec_dim=256,
        ignore_index=0,
        **to_float_cuda)

    checkpoint = torch.load(checkpoint_path)
    model = model.to('cuda')
    model.load_state_dict(checkpoint['model_state'])

    image = process_images_batch(
        [read_images(image_path, (299, 299))], (299, 299))
    generated_captions, attn_weights_all = model.sample(image)
    generated_captions = decode_captions(
        generated_captions, vocab.idx_to_token)
    print(generated_captions)

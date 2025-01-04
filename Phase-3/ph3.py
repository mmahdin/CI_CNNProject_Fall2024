import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time
import random
import cv2
import string
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
import re
from transformers import BertModel, BertTokenizer
from concurrent.futures import ThreadPoolExecutor
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

N_P = 10

#########################################################################
#                           FeatureExtractor                            #
#########################################################################


class FeatureExtractor(object):
    """
    Image feature extraction with EfficientNet.
    """

    def __init__(self, pooling=False, verbose=False, device='cuda', dtype=torch.float32):
        import torch
        import torch.nn as nn
        from torchvision import transforms, models
        from torchvision.models import EfficientNet_B0_Weights

        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        self.device, self.dtype = device, dtype
        self.pooling = pooling

        # Load EfficientNet_B0 with pretrained weights
        self.efficientnet = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1).to(device)

        # Remove the classifier head to retain spatial features
        self.efficientnet = nn.Sequential(
            *list(self.efficientnet.children())[:-2])

        # If pooling is applied, add global average pooling
        if pooling:
            self.efficientnet.add_module(
                'GlobalAvgPool', nn.AdaptiveAvgPool2d((1, 1)))

        self.efficientnet.eval()

    def extract_feature(self, img, verbose=False):
        """
        Inputs:
        - img: Batch of resized images, of shape N x 3 x H x W (e.g., 224 x 224)

        Outputs:
        - feat: Image feature, of shape N x 1280 (if pooling is applied) or N x 1280 x 7 x 7 (if not applied)
        """
        import math
        import torch.nn.functional as F
        import torch

        num_img = img.shape[0]

        # Preprocess each image in the batch
        img_prepro = []
        for i in range(num_img):
            img_prepro.append(self.preprocess(
                img[i].type(self.dtype).div(255.)))
        img_prepro = torch.stack(img_prepro).to(self.device)

        with torch.no_grad():
            feat = []
            process_batch = 500
            for b in range(math.ceil(num_img / process_batch)):
                # Pass the batch through the model
                output = self.efficientnet(
                    img_prepro[b * process_batch:(b + 1) * process_batch])
                feat.append(output)
            feat = torch.cat(feat)

            # If pooling is applied, flatten the output
            if self.pooling:
                feat = feat.view(feat.size(0), -1)  # N x 1280

            # Add L2 normalization
            feat = F.normalize(feat.view(feat.size(0), -1),
                               p=2, dim=1).view_as(feat)

        if verbose:
            print('Output feature shape: ', feat.shape)

        return feat

#########################################################################
#                                 DATA                                  #
#########################################################################


def get_augmentation_pipeline(image_size):
    return A.Compose([
        A.HorizontalFlip(p=0.8),
        # A.RandomBrightnessContrast(p=0.7),
        # A.HueSaturationValue(hue_shift_limit=20,
        #                      sat_shift_limit=30, val_shift_limit=50, p=0.5),
        A.Rotate(limit=40, p=0.8),  # Random rotation within ±50 degrees
        # Crop to a fixed size
        A.RandomCrop(width=int(image_size[0]*0.9),
                     height=int(image_size[1]*0.9), p=0.7),
        # Apply Gaussian blur, ensure blur_limit >= 3
        # A.GaussianBlur(blur_limit=3, p=0.5),
        A.CoarseDropout(max_holes=30, max_height=5, max_width=5,
                        min_holes=20, p=0.8),  # Coarse dropout
        A.Resize(*image_size),
        ToTensorV2()
    ])


def process_images_batch(batch_images, image_size, augment=False, num_augmented=5):
    processed_images = []

    for img_np in batch_images:
        processed_image = process_images(
            img_np, image_size, augment=augment, num_augmented=num_augmented
        )
        processed_images.append(processed_image)

    # If augment is True, processed_image tensors have shape (num_augmented, C, H, W),
    # so the final tensor should stack into (batch_size, num_augmented, C, H, W).
    return torch.stack(processed_images)


def process_images(img_np, image_size, augment=False, num_augmented=5):
    images = []
    augmentation_pipeline = get_augmentation_pipeline(
        image_size) if augment else None

    original_img = torch.tensor(img_np).permute(2, 0, 1).float()
    if not augment:
        return original_img

    images.append(original_img)

    # Generate augmented versions of the image
    if augment and augmentation_pipeline:
        for _ in range(num_augmented):
            augmented_img = augmentation_pipeline(image=img_np)["image"]
            images.append(augmented_img)

    return torch.stack(images) if images else torch.empty(0)


def read_images(image_paths, image_size):
    """
    Reads and processes multiple images from a list of paths.

    Args:
        image_paths (list): List of file paths to images.
        image_size (tuple): Desired size (width, height) for resizing images.

    Returns:
        list: List of numpy arrays representing the processed images.
    """
    def process_single_image(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(image_size, Image.LANCZOS)
            img_np = np.array(img)
            return img_np
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

    with ThreadPoolExecutor() as executor:
        if isinstance(image_paths, str):
            results = list(executor.map(process_single_image, [image_paths]))
        else:
            results = list(executor.map(process_single_image, image_paths))

    # Filter out None values in case of errors
    if isinstance(image_paths, str):
        return results[0]
    return [img for img in results if img is not None]


class Vocabulary:
    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = []
        self.counter = Counter()

    def build_vocab(self, captions, min_freq=1):
        self.counter.update(
            word for caption_list in captions for caption in caption_list for word in caption.split())
        for token, freq in self.counter.items():
            if freq >= min_freq:
                self.idx_to_token.append(token)
        self.idx_to_token = ["<pad>", "<start>",
                             "<end>", "<unk>"] + sorted(self.idx_to_token)
        self.token_to_idx = {token: idx for idx,
                             token in enumerate(self.idx_to_token)}

    def numericalize(self, caption):
        tokens = ["<start>"] + caption.split() + ["<end>"]
        return [self.token_to_idx.get(token, self.token_to_idx["<unk>"]) for token in tokens]


def process_captions(captions_list, vocab, max_caption_length):
    caption_tensors = []
    for caption_list in captions_list:
        processed_captions = [
            torch.tensor(vocab.numericalize(caption), dtype=torch.long)
            for caption in caption_list
        ]
        # Pad all captions to max_caption_length
        padded_captions = torch.stack([
            torch.cat([caption, torch.tensor(
                [vocab.token_to_idx["<pad>"]] * (max_caption_length - len(caption)))])
            if len(caption) < max_caption_length else caption[:max_caption_length]
            for caption in processed_captions
        ])
        caption_tensors.append(padded_captions)
    return torch.stack(caption_tensors)


def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = " ".join(text.split())
    return text


def read_captions_and_build_vocab(file_path="./dataset/archive/captions.txt", min_freq=1):
    """
    Reads a text file with image captions and builds a Vocabulary object.

    Args:
        file_path (str): Path to the text file.
        min_freq (int): Minimum frequency for a word to be included in the vocabulary.

    Returns:
        Vocabulary: An instance of the Vocabulary class with the built vocabulary.
    """
    captions = []

    # Read the file and extract captions
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            _, caption = line.strip().split(',', 1)
            captions.append(caption)

    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocab([captions], min_freq=min_freq)

    return vocab


def load_data(file_path, captions_path, data_path, image_size, test_size=0.066667, flicker='8k', min_freq=4):
    if os.path.exists(file_path):
        dataset = torch.load(file_path)
        print("Dataset loaded successfully.")
        return dataset
    else:
        print(
            f"The file '{file_path}' does not exist. Creating a new dataset...")
        if flicker == '30k':
            captions_data = []
            with open(captions_path, 'r') as file:
                for line in file:
                    parts = line.strip().split('|')
                    if len(parts) == 3:
                        image, _, caption = parts
                        captions_data.append(
                            (os.path.join(data_path, image), caption.strip()))
        else:
            captions_data = []
            with open(captions_path, "r") as f:
                for line in f:
                    try:
                        image_id, caption = line.strip().split(",", 1)
                        captions_data.append(
                            (os.path.join(data_path, image_id), caption.strip()))
                    except ValueError:
                        continue
        # captions_data = captions_data[:10000]
        # Convert to DataFrame
        captions_df = pd.DataFrame(captions_data, columns=["image", "caption"])

        if flicker != 'coco':
            # Group captions by image
            grouped_captions = captions_df.groupby(
                "image")["caption"].apply(list).reset_index()

            grouped_captions["caption"] = grouped_captions["caption"].apply(
                lambda captions: [preprocess_text(
                    caption) for caption in captions]
            )

            # Filter images with exactly 5 captions
            grouped_captions = grouped_captions[grouped_captions["caption"].apply(
                len) == 5]
        else:
            grouped_captions = captions_df
            grouped_captions["caption"] = grouped_captions["caption"].apply(
                lambda captions: [preprocess_text(captions)]
            )

        train_df, val_df = train_test_split(
            grouped_captions, test_size=test_size, random_state=42)

        vocab = read_captions_and_build_vocab(min_freq=min_freq)

        # Find the maximum caption length
        max_caption_length = max(
            len(vocab.numericalize(caption))
            for caption_list in train_df["caption"]
            for caption in caption_list
        )

        # Process training and validation data
        train_images = train_df["image"].tolist()
        print(len(train_images))
        val_images = val_df["image"].tolist()

        # Convert captions to numerical form
        train_captions = process_captions(
            train_df["caption"], vocab, max_caption_length)
        val_captions = process_captions(
            val_df["caption"], vocab, max_caption_length)

        # Build the dataset dictionary
        dataset = {
            "train_images": train_images,
            "val_images": val_images,
            "train_captions": train_captions,
            "val_captions": val_captions,
            "vocab": {
                "idx_to_token": vocab.idx_to_token,
                "token_to_idx": vocab.token_to_idx,
            },
            "max_caption_length": max_caption_length,
        }

        # # Save the dataset for future use
        # torch.save(dataset, file_path)
        # print("Dataset created successfully.")
        return dataset


def decode_captions(captions, idx_to_word):
    """
    Decoding caption indexes into words.
    Inputs:
    - captions: Caption indexes in a tensor of shape (Nx)T.
    - idx_to_word: Mapping from the vocab index to word.

    Outputs:
    - decoded: A sentence (or a list of N sentences).
    """
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<pad>':
                words.append(word)
            if word == '<end>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


#########################################################################
#                               Simple RNN                              #
#########################################################################

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases, of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    next_h = torch.tanh(x.mm(Wx) + prev_h.mm(Wh) + b)
    cache = (x, Wx, prev_h, Wh, b, next_h)

    return next_h, cache


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases, of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None

    N, T, D = x.shape
    H = h0.shape[1]
    h = torch.zeros((N, T, H), dtype=h0.dtype, device=h0.device)
    prev_h = h0
    cache = []
    for i in range(T):
        next_h, cache_h = rnn_step_forward(x[:, i, :], prev_h, Wx, Wh, b)
        prev_h = next_h
        h[:, i, :] = prev_h
        cache.append(cache_h)

    return h, cache


class RNN(nn.Module):
    """
    A single-layer vanilla RNN module.

    Arguments for initialization:
    - input_size: Input size, denoted as D before
    - hidden_size: Hidden size, denoted as H before
    """

    def __init__(self, input_size, hidden_size, device='cpu',
                 dtype=torch.float32):
        """
        Initialize a RNN.
        Model parameters to initialize:
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases, of shape (H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = Parameter(torch.randn(input_size, hidden_size,
                                        device=device, dtype=dtype).div(math.sqrt(input_size)))
        self.Wh = Parameter(torch.randn(hidden_size, hidden_size,
                                        device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.b = Parameter(torch.zeros(hidden_size,
                           device=device, dtype=dtype))

    def forward(self, x, h0):
        """
        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - h0: Initial hidden state, of shape (N, H)

        Outputs:
        - hn: The hidden state output
        """
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        """
        Inputs:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)

        Outputs:
        - next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


#########################################################################
#                                   LSTM                                #
#########################################################################

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b, attn=None, Wattn=None):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    - attn and Wattn are for Attention LSTM only, indicate the attention input and
      embedding weights for the attention input

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    """
    next_h, next_c = None, None

    N, H = prev_h.shape
    a = None
    if attn is None:  # regular lstm
        a = x.mm(Wx) + prev_h.mm(Wh) + b
    else:
        a = x.mm(Wx) + prev_h.mm(Wh) + b + attn.mm(Wattn)
    i = torch.sigmoid(a[:, 0:H])
    f = torch.sigmoid(a[:, H:2*H])
    o = torch.sigmoid(a[:, 2*H:3*H])
    g = torch.tanh(a[:, 3*H:4*H])
    next_c = f * prev_c + i * g
    next_h = o * torch.tanh(next_c)

    return next_h, next_c


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    """
    h = None
    # we provide the intial cell state c0 here for you!
    c0 = torch.zeros_like(h0)

    N, T, D = x.shape
    _, H = h0.shape
    h = torch.zeros((N, T, H), dtype=h0.dtype, device=h0.device)
    prev_h = h0
    prev_c = c0
    for i in range(T):
        next_h, next_c = lstm_step_forward(
            x[:, i, :], prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c
        h[:, i, :] = prev_h

    return h


class LSTM(nn.Module):
    """
    This is our single-layer, uni-directional LSTM module.

    Arguments for initialization:
    - input_size: Input size, denoted as D before
    - hidden_size: Hidden size, denoted as H before
    """

    def __init__(self, input_size, hidden_size, device='cpu',
                 dtype=torch.float32):
        """
        Initialize a LSTM.
        Model parameters to initialize:
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases, of shape (4H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = Parameter(torch.randn(input_size, hidden_size*4,
                                        device=device, dtype=dtype).div(math.sqrt(input_size)))
        self.Wh = Parameter(torch.randn(hidden_size, hidden_size*4,
                                        device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.b = Parameter(torch.zeros(hidden_size*4,
                           device=device, dtype=dtype))

    def forward(self, x, h0):
        """
        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - h0: Initial hidden state, of shape (N, H)

        Outputs:
        - hn: The hidden state output
        """
        hn = lstm_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h, prev_c):
        """
        Inputs:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)
        - prev_c: The previous cell state, of shape (N, H)

        Outputs:
        - next_h: The next hidden state, of shape (N, H)
        - next_c: The next cell state, of shape (N, H)
        """
        next_h, next_c = lstm_step_forward(
            x, prev_h, prev_c, self.Wx, self.Wh, self.b)
        return next_h, next_c


class LSTMCellWithLNAndDropout(torch.nn.Module):
    def __init__(self, D, H, use_ln=False, dropout=0.0):
        super().__init__()
        self.H = H
        self.use_ln = use_ln
        self.dropout = dropout

        self.Wx = torch.nn.Parameter(torch.randn(D, 4 * H) * 0.01)
        self.Wh = torch.nn.Parameter(torch.randn(H, 4 * H) * 0.01)
        self.b = torch.nn.Parameter(torch.zeros(4 * H))

        # Optional attention weights
        self.Wattn = None

        # Layer normalization layers (optional)
        if use_ln:
            self.ln = torch.nn.LayerNorm(4 * H)

        # Dropout layer (optional)
        self.do = torch.nn.Dropout(dropout)

    def forward(self, x, prev_h, prev_c, attn=None):
        """
        Forward pass for a single timestep of an LSTM with optional LN and DO.
        """
        a = x.mm(self.Wx) + prev_h.mm(self.Wh) + self.b
        if attn is not None and self.Wattn is not None:
            a += attn.mm(self.Wattn)

        # Apply layer normalization if enabled
        if self.use_ln:
            a = self.ln(a)

        i = torch.sigmoid(a[:, :self.H])
        f = torch.sigmoid(a[:, self.H:2*self.H])
        o = torch.sigmoid(a[:, 2*self.H:3*self.H])
        g = torch.tanh(a[:, 3*self.H:])

        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)

        # Apply dropout to the hidden state if enabled
        if self.dropout > 0.0:
            next_h = self.do(next_h)

        return next_h, next_c

#########################################################################
#                               Attention LSTM                          #
#########################################################################


def dot_product_attention(prev_h, A):
    """
    A simple scaled dot-product attention layer.
    Inputs:
    - prev_h: The LSTM hidden state from the previous time step, of shape (N, H)
    - A: **Projected** CNN feature activation, of shape (N, H, 4, 4),
         where H is the LSTM hidden state size

    Outputs:
    - attn: Attention embedding output, of shape (N, H)
    - attn_weights: Attention weights, of shape (N, 4, 4)

    """
    N, H, D_a, _ = A.shape

    attn, attn_weights = None, None

    from math import sqrt
    h_tilt = prev_h.reshape(N, 1, H)
    A_tilt = A.reshape(N, H, -1)
    Matt = (torch.bmm(h_tilt, A_tilt).div(sqrt(H))
            ).reshape(N, -1, 1)  # N * 16 * 1
    Matt_tilt = F.softmax(Matt, dim=1)  # probability
    attn = torch.bmm(A_tilt, Matt_tilt).reshape(N, H)
    attn_weights = Matt_tilt.reshape(N, N_P, N_P)

    return attn, attn_weights


def attention_forward(x, A, lstm_cell):
    """
    Inputs:
    - x: Input data, of shape (N, T, D)
    - A: **Projected** activation map, of shape (N, H, 4, 4)
    - lstm_cell: Instance of LSTMCellWithLNAndDropout, initialized with required parameters

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    """
    h = None

    h0 = A.mean(dim=(2, 3))  # Initial hidden state, of shape (N, H)
    c0 = h0  # Initial cell state, of shape (N, H)

    N, T, D = x.shape
    _, H = h0.shape
    h = torch.zeros((N, T, H), dtype=h0.dtype, device=h0.device)
    prev_h = h0
    prev_c = c0
    for i in range(T):
        attn, attn_weights = dot_product_attention(prev_h, A)
        # Use LSTMCellWithLNAndDropout
        next_h, next_c = lstm_cell(x[:, i, :], prev_h, prev_c, attn=attn)
        prev_h = next_h
        prev_c = next_c
        h[:, i, :] = prev_h

    return h


class AttentionLSTM(nn.Module):
    """
    Single-layer, uni-directional Attention module.

    Arguments for initialization:
    - input_size: Input size, denoted as D before
    - hidden_size: Hidden size, denoted as H before
    """

    def __init__(self, input_size, hidden_size, use_ln=False, dropout=0.0, device='cpu', dtype=torch.float32):
        """
        Initialize the Attention LSTM.

        Parameters:
        - input_size: Size of the input (D).
        - hidden_size: Size of the hidden state (H).
        - use_ln: Whether to use Layer Normalization (default: False).
        - dropout: Dropout probability (default: 0.0).
        - device: Device to initialize tensors on.
        - dtype: Data type for tensors.
        """
        super().__init__()

        # Initialize LSTMCellWithLNAndDropout
        self.lstm_cell = LSTMCellWithLNAndDropout(
            input_size, hidden_size, use_ln=use_ln, dropout=dropout
        ).to(device=device, dtype=dtype)

    def forward(self, x, A):
        """  
        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D).
        - A: The projected CNN feature activation, of shape (N, H, 4, 4).

        Outputs:
        - h: Hidden states for all timesteps, of shape (N, T, H).
        """
        h = attention_forward(x, A, self.lstm_cell)
        return h

    def step_forward(self, x, prev_h, prev_c, attn):
        """
        Inputs:
        - x: Input data for one time step, of shape (N, D).
        - prev_h: The previous hidden state, of shape (N, H).
        - prev_c: The previous cell state, of shape (N, H).
        - attn: The attention embedding, of shape (N, H).

        Outputs:
        - next_h: The next hidden state, of shape (N, H).
        - next_c: The next cell state, of shape (N, H).
        """
        # Use LSTMCellWithLNAndDropout for step-wise computation
        next_h, next_c = self.lstm_cell(x, prev_h, prev_c, attn=attn)
        return next_h, next_c


def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return


#########################################################################
#                               CaptioningRNN                           #
#########################################################################


class CaptioningRNN(nn.Module):
    """
    A CaptioningRNN produces captions from images using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.

    You will implement the `__init__` method for model initialization and
    the `forward` method first, then come back for the `sample` method later.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', device='cpu', p=0.3,
                 ignore_index=None, dtype=torch.float32):
        """
        Construct a new CaptioningRNN instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        super().__init__()
        if cell_type not in {'rnn', 'lstm', 'attention'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<pad>']
        self._start = word_to_idx.get('<start>', None)
        self._end = word_to_idx.get('<end>', None)
        self.ignore_index = ignore_index

        self.wordvec_dim = wordvec_dim
        self.feat_extract = None
        self.affine = None
        self.rnn = None
        self.affine_l = nn.Linear(input_dim, hidden_dim).to(
            device=device, dtype=dtype)
        self.affine = nn.Sequential(
            self.affine_l,
            nn.Dropout(p=p)  # Adding dropout with a probability of 0.5
        )

        if cell_type == 'rnn' or cell_type == 'lstm':
            self.feat_extract = FeatureExtractor(
                pooling=True, device=device, dtype=dtype)
            if cell_type == 'rnn':
                self.rnn = RNN(wordvec_dim, hidden_dim,
                               device=device, dtype=dtype)
            else:
                self.rnn = LSTM(wordvec_dim, hidden_dim,
                                device=device, dtype=dtype)
        elif cell_type == 'attention':
            self.feat_extract = FeatureExtractor(
                pooling=False, device=device, dtype=dtype)
            self.rnn = AttentionLSTM(
                wordvec_dim, hidden_dim, use_ln=True, dropout=0.3, device=device, dtype=dtype)
        else:
            raise ValueError
        nn.init.kaiming_normal_(self.affine_l.weight)
        nn.init.zeros_(self.affine_l.bias)
        self.word_embed = WordEmbedding(
            vocab_size, wordvec_dim, device=device, dtype=dtype)
        self.temporal_affine_l = nn.Linear(
            hidden_dim, vocab_size).to(device=device, dtype=dtype)
        self.temporal_affine = nn.Sequential(
            self.temporal_affine_l,
            nn.Dropout(p=0.0)  # Adding dropout with a probability of 0.5
        )
        nn.init.kaiming_normal_(self.temporal_affine_l.weight)
        nn.init.zeros_(self.temporal_affine_l.bias)

    def forward(self, images, captions):
        """
        Compute training-time loss for the RNN. We input images and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss. The backward part will be done by torch.autograd.

        Inputs:
        - images: Input images, of shape (N, 3, 112, 112)
        - captions: Ground-truth captions; an integer array of shape (N, T + 1) where
          each element is in the range 0 <= y[i, t] < V

        Outputs:
        - loss: A scalar loss
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]  # N x T
        captions_out = captions[:, 1:]

        loss = 0.0

        feature = self.feat_extract.extract_feature(images)  # N x 1280

        if self.cell_type == 'attention':
            # make it N * 4 * 4 * input_dim
            feature = feature.permute(0, 2, 3, 1)

        h0 = self.affine(feature)  # N x hidden_dim

        if self.cell_type == 'attention':
            h0 = h0.permute(0, 3, 1, 2)  # permute back (N, H, 4, 4)

        x = self.word_embed(captions_in)  # N x T x wordvec_dim
        h = self.rnn(x, h0)  # N x T x H
        score = self.temporal_affine(h)  # N x T x V
        loss = temporal_softmax_loss(
            score, captions_out, ignore_index=self._null)

        return loss

    def sample(self, images, max_length=15):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - images: Input images, of shape (N, 3, 112, 112)
        - max_length: Maximum length T of generated captions

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == 'attention':
            attn_weights_all = images.new(
                N, max_length, N_P, N_P).fill_(0).float()

        feature = self.feat_extract.extract_feature(images)
        A = None
        if self.cell_type == 'attention':
            # make it N * 4 * 4 * input_dim
            feature = feature.permute(0, 2, 3, 1)
        prev_h = self.affine(feature)
        prev_c = torch.zeros_like(prev_h)
        if self.cell_type == 'attention':
            A = prev_h.permute(0, 3, 1, 2)  # permute back
            prev_h = A.mean(dim=(2, 3))
            prev_c = A.mean(dim=(2, 3))

        x = torch.ones((N, self.wordvec_dim), dtype=prev_h.dtype,
                       device=prev_h.device) * self.word_embed(self._start).reshape(1, -1)
        for i in range(max_length):
            next_h = None
            if self.cell_type == 'rnn':
                next_h = self.rnn.step_forward(x, prev_h)
            elif self.cell_type == 'lstm':
                next_h, prev_c = self.rnn.step_forward(x, prev_h, prev_c)
            else:
                attn, attn_weights = dot_product_attention(prev_h, A)
                attn_weights_all[:, i] = attn_weights
                next_h, prev_c = self.rnn.step_forward(x, prev_h, prev_c, attn)

            score = self.temporal_affine(next_h)
            # loss = temporal_softmax_loss(score, captions_out, ignore_index=self._null)
            max_idx = torch.argmax(score, dim=1)
            captions[:, i] = max_idx
            x = self.word_embed(max_idx)
            # print(x.shape)
            prev_h = next_h

        if self.cell_type == 'attention':
            return captions, attn_weights_all.cpu()
        else:
            return captions


def temporal_softmax_loss(x, y, ignore_index=None):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, *summing* the loss over all timesteps and *averaging* across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional ignore_index argument
    tells us which elements in the caption should not contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V

    Returns a tuple of:
    - loss: Scalar giving loss
    """
    loss = None

    N, T, V = x.shape
    x_flat = x.reshape(N*T, V)
    y_flat = y.reshape(N*T)
    loss = F.cross_entropy(
        x_flat, y_flat, ignore_index=ignore_index, reduction='sum')
    loss = loss / N

    return loss


class WordEmbedding(nn.Module):
    """
    Simplified version of torch.nn.Embedding.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    """

    def __init__(self, vocab_size, embed_size,
                 device='cpu', dtype=torch.float32):
        super().__init__()

        # Register parameters
        self.W_embed = Parameter(torch.randn(vocab_size, embed_size,
                                             device=device, dtype=dtype).div(math.sqrt(vocab_size)))

    def forward(self, x):
        out = None
        out = self.W_embed[x]

        return out


#########################################################################
#                                  Train                                #
#########################################################################


def train_captioning_model(
    rnn_decoder, optimizer, data_dict,
    device='cuda', dtype=torch.float32, epochs=1, batch_size=256,
    scheduler=None, val_perc=0.5, image_size=(256, 256), lr=None, weight_decay=None,
    verbose=True, checkpoint_path='./models/captioning_checkpoint.pth',
    history_path='./history/captioning_train_history.pkl', vocab_size=None
):

    total_images = len(data_dict["train_images"])
    num_batches = math.ceil(total_images // batch_size)

    total_images_val = len(data_dict["val_images"])*val_perc
    val_batch_size = 32
    num_batches_val = math.ceil(total_images_val // val_batch_size)

    rnn_decoder = rnn_decoder.to(device)

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    start_epoch = 0

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        rnn_decoder.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # Change the learning rate
        if lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if weight_decay:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = weight_decay
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
        train_loss_history = checkpoint['train_loss_history']
        val_loss_history = checkpoint['val_loss_history']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        start_time = time.time()  # Start time for epoch

        # Training phase
        rnn_decoder.train()
        epoch_loss = 0.0
        for j in range(num_batches):
            images = read_images(
                data_dict['train_images'][batch_size*j:batch_size*(j+1)], image_size)
            num_aug = 0
            images_torch = process_images_batch(
                images, image_size=image_size, augment=True, num_augmented=num_aug).to(device=device, dtype=dtype)
            captions = data_dict['train_captions'][batch_size *
                                                   j:batch_size*(j+1)]
            bc_loss = 0
            for ag in range(num_aug+1):
                cap_idx = random.randint(0, captions.shape[1])-1
                caption = captions[:, cap_idx, :].to(device=device)
                loss = rnn_decoder(images_torch[:, ag, :, :], caption)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bc_loss += loss.item()
                epoch_loss += loss.item()

            if verbose and j % 20 == 0:
                if num_aug != 0:
                    print(
                        f"  Batch {j+1}/{num_batches}, lr = {optimizer.param_groups[0]['lr']}, Loss = {bc_loss/(num_aug):.4f}")
                else:
                    print(
                        f"  Batch {j+1}/{num_batches}, lr = {optimizer.param_groups[0]['lr']}, Loss = {bc_loss:.4f}")
        if scheduler:
            scheduler.step()

        if num_aug != 0:
            avg_loss = epoch_loss / (num_aug*num_batches)
        else:
            avg_loss = epoch_loss / (num_batches)
        train_loss_history.append(avg_loss)

        print(f"  Training Loss: {avg_loss:.4f}")

        # Validation phase
        rnn_decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for j in range(num_batches_val):
                images = read_images(
                    data_dict['val_images'][val_batch_size * j:val_batch_size*(j+1)], image_size)
                images_torch = process_images_batch(
                    images, image_size=image_size, augment=False).to(device=device, dtype=dtype)
                captions = data_dict['val_captions'][val_batch_size *
                                                     j:val_batch_size*(j+1)]
                for cap_idx in range(captions.shape[1]):
                    caption = captions[:, cap_idx, :].to(device=device)

                    loss = rnn_decoder(images_torch, caption)

                    val_loss += loss.item()

        avg_val_loss = val_loss / (num_batches_val*captions.shape[1])
        val_loss_history.append(avg_val_loss)

        print(f"  Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint if validation loss improves
        best_val_loss = avg_val_loss
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': rnn_decoder.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, checkpoint_path)

        # Print time spent for the epoch
        end_time = time.time()
        elapsed_time = end_time - start_time

    print("Training complete!")
    return train_loss_history, val_loss_history


def attention_visualizer(img, attn_weights, token):
    """
    Visuailze the attended regions on a single frame from a single query word.
    Inputs:
    - img: Image tensor input, of shape (3, H, W)
    - attn_weights: Attention weight tensor, on the final activation map
    - token: The token string you want to display above the image

    Outputs:
    - img_output: Image tensor output, of shape (3, H+25, W)

    """
    C, H, W = img.shape
    assert C == 3, 'We only support image with three color channels!'

    # Resize the attention weights to match the image dimensions
    attn_weights_resized = cv2.resize(
        attn_weights.data.numpy().copy(), (W, H), interpolation=cv2.INTER_NEAREST)

    # Normalize the resized attention weights to range [0, 1]
    attn_weights_normalized = (attn_weights_resized - attn_weights_resized.min()) / (
        attn_weights_resized.max() - attn_weights_resized.min())

    # Convert the normalized attention weights to a 3D array by duplicating across the color channels
    attn_weights_3d = np.stack([attn_weights_normalized] * 3, axis=-1)

    # Combine the image and attention map
    img_copy = img.float().div(255.).permute(1, 2, 0).numpy()[
        :, :, ::-1].copy()  # Convert to BGR for cv2
    masked_img = cv2.addWeighted(attn_weights_3d, 0.5, img_copy, 0.5, 0)
    img_copy = np.concatenate((np.zeros((50, W, 3)), masked_img), axis=0)

    # Add text
    cv2.putText(img_copy, '%s' % (token), (10, 40),
                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), thickness=2)

    return img_copy

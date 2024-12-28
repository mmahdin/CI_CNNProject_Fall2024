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
from torch.nn.utils.rnn import pad_sequence
import random

#########################################################################
#                           FeatureExtractor                            #
#########################################################################


class FeatureExtractor(object):
    """
    Image feature extraction with MobileNet.
    """

    def __init__(self, pooling=False, verbose=False, device='cuda', dtype=torch.float32):
        from torchvision import transforms, models
        from torchvision.models import MobileNet_V2_Weights

        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        self.device, self.dtype = device, dtype

        # Use the recommended 'weights' parameter
        self.mobilenet = models.mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(device)

        # Remove the last classifier
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])

        # average pooling
        if pooling:
            # input: N x 1280 x 4 x 4
            self.mobilenet.add_module('LastAvgPool', nn.AvgPool2d(4, 4))

        self.mobilenet.eval()

    def extract_mobilenet_feature(self, img, verbose=False):
        """
        Inputs:
        - img: Batch of resized images, of shape N x 3 x 112 x 112

        Outputs:
        - feat: Image feature, of shape N x 1280 (pooled) or N x 1280 x 4 x 4
        """
        num_img = img.shape[0]

        img_prepro = []
        for i in range(num_img):
            img_prepro.append(self.preprocess(
                img[i].type(self.dtype).div(255.)))
        img_prepro = torch.stack(img_prepro).to(self.device)

        with torch.no_grad():
            feat = []
            process_batch = 500
            for b in range(math.ceil(num_img/process_batch)):
                feat.append(self.mobilenet(img_prepro[b*process_batch:(b+1)*process_batch]
                                           ).squeeze(-1).squeeze(-1))  # forward and squeeze
            feat = torch.cat(feat)

            # add l2 normalization
            F.normalize(feat, p=2, dim=1)

        if verbose:
            print('Output feature shape: ', feat.shape)

        return feat

#########################################################################
#                                 DATA                                  #
#########################################################################


def process_images(image_list, data_path, image_size):
    images = []
    for image_name in image_list:
        img_path = os.path.join(data_path, image_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image '{img_path}' not found. Skipping.")
            continue
        img = Image.open(img_path).convert("RGB")
        img = img.resize(image_size)
        img_tensor = torch.tensor(np.array(img)).permute(
            2, 0, 1).float() / 255.0  # Normalize to [0, 1]
        images.append(img_tensor)
    return torch.stack(images) if images else torch.empty(0)


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


def load_data(file_path, captions_path, data_path, image_size):
    if os.path.exists(file_path):
        dataset = torch.load(file_path)
        print("Dataset loaded successfully.")
        return dataset
    else:
        print(
            f"The file '{file_path}' does not exist. Creating a new dataset...")

        # Load and parse captions
        captions_data = []
        with open(captions_path, "r") as f:
            for line in f:
                try:
                    image_id, caption = line.strip().split(",", 1)
                    captions_data.append((image_id, caption.strip()))
                except ValueError:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue

        # Convert to DataFrame
        captions_df = pd.DataFrame(captions_data, columns=["image", "caption"])

        # Group captions by image
        grouped_captions = captions_df.groupby(
            "image")["caption"].apply(list).reset_index()

        # Filter images with exactly 5 captions
        grouped_captions = grouped_captions[grouped_captions["caption"].apply(
            len) == 5]

        # Split into train and validation sets
        train_df, val_df = train_test_split(
            grouped_captions, test_size=0.2, random_state=42)

        vocab = Vocabulary()
        vocab.build_vocab(train_df["caption"].tolist())

        # Find the maximum caption length
        max_caption_length = max(
            len(vocab.numericalize(caption))
            for caption_list in train_df["caption"]
            for caption in caption_list
        )

        # Process training and validation data
        train_images = process_images(
            train_df["image"].tolist(), data_path, image_size)
        val_images = process_images(
            val_df["image"].tolist(), data_path, image_size)

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

        # Save the dataset for future use
        torch.save(dataset, file_path)
        print("Dataset created successfully.")
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
    attn_weights = Matt_tilt.reshape(N, 4, 4)

    return attn, attn_weights


def attention_forward(x, A, Wx, Wh, Wattn, b):
    """
    Inputs:
    - x: Input data, of shape (N, T, D)
    - A: **Projected** activation map, of shape (N, H, 4, 4)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - Wattn: Weights for attention-to-hidden connections, of shape (H, 4H)
    - b: Biases, of shape (4H,)

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
        next_h, next_c = lstm_step_forward(
            x[:, i, :], prev_h, prev_c, Wx, Wh, b, attn=attn, Wattn=Wattn)
        prev_h = next_h
        prev_c = next_c
        h[:, i, :] = prev_h

    return h


class AttentionLSTM(nn.Module):
    """
    This is our single-layer, uni-directional Attention module.

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
        - Wattn: Weights for attention-to-hidden connections, of shape (H, 4H)
        - b: Biases, of shape (4H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = Parameter(torch.randn(input_size, hidden_size*4,
                                        device=device, dtype=dtype).div(math.sqrt(input_size)))
        self.Wh = Parameter(torch.randn(hidden_size, hidden_size*4,
                                        device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.Wattn = Parameter(torch.randn(hidden_size, hidden_size*4,
                                           device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.b = Parameter(torch.zeros(hidden_size*4,
                           device=device, dtype=dtype))

    def forward(self, x, A):
        """  
        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - A: The projected CNN feature activation, of shape (N, H, 4, 4)

        Outputs:
        - hn: The hidden state output
        """
        hn = attention_forward(x, A, self.Wx, self.Wh, self.Wattn, self.b)
        return hn

    def step_forward(self, x, prev_h, prev_c, attn):
        """
        Inputs:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)
        - prev_c: The previous cell state, of shape (N, H)
        - attn: The attention embedding, of shape (N, H)

        Outputs:
        - next_h: The next hidden state, of shape (N, H)
        - next_c: The next cell state, of shape (N, H)
        """
        next_h, next_c = lstm_step_forward(x, prev_h, prev_c, self.Wx, self.Wh,
                                           self.b, attn=attn, Wattn=self.Wattn)
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
                 hidden_dim=128, cell_type='rnn', device='cpu',
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

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        self.ignore_index = ignore_index

        self.wordvec_dim = wordvec_dim
        self.feat_extract = None
        self.affine = None
        self.rnn = None
        self.affine = nn.Linear(input_dim, hidden_dim).to(
            device=device, dtype=dtype)
        if cell_type == 'rnn' or cell_type == 'lstm':
            self.feat_extract = FeatureExtractor(
                pooling=True, device=device, dtype=dtype)
            # self.affine = nn.Linear(input_dim, hidden_dim).to(device=device, dtype=dtype)
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
                wordvec_dim, hidden_dim, device=device, dtype=dtype)
        else:
            raise ValueError
        nn.init.kaiming_normal_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        self.word_embed = WordEmbedding(
            vocab_size, wordvec_dim, device=device, dtype=dtype)
        self.temporal_affine = nn.Linear(
            hidden_dim, vocab_size).to(device=device, dtype=dtype)
        nn.init.kaiming_normal_(self.temporal_affine.weight)
        nn.init.zeros_(self.temporal_affine.bias)

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

        feature = self.feat_extract.extract_mobilenet_feature(
            images)  # N x 1280
        if self.cell_type == 'attention':
            # make it N * 4 * 4 * input_dim
            feature = feature.permute(0, 2, 3, 1)

        h0 = self.affine(feature)  # N x hidden_dim

        if self.cell_type == 'attention':
            h0 = h0.permute(0, 3, 1, 2)  # permute back (N, H, 4, 4)

        x = self.word_embed(captions_in)  # N x T x wordvec_dim
        h = self.rnn(x, h0)  # N x T x H
        score = self.temporal_affine(h)
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
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()

        feature = self.feat_extract.extract_mobilenet_feature(images)
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

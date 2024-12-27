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

# Vocabulary


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

# Caption processing


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

# Load data


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

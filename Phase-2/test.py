import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from os import listdir
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from torch.nn import CrossEntropyLoss

from sklearn.model_selection import train_test_split

from PIL import Image
from PIL import Image, ImageEnhance

from ph2 import *

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2


base_path = "./dataset/archive/"
folder = listdir(base_path)
len(folder)

total_images = 0
for n in range(len(folder)):
    patient_id = folder[n]
    for c in [0, 1]:
        patient_path = base_path + patient_id
        class_path = patient_path + "/" + str(c) + "/"
        subfiles = listdir(class_path)
        total_images += len(subfiles)

data = pd.DataFrame(index=np.arange(0, total_images),
                    columns=["patient_id", "path", "target"])

k = 0
for n in range(len(folder)):
    patient_id = folder[n]
    patient_path = base_path + patient_id
    for c in [0, 1]:
        class_path = patient_path + "/" + str(c) + "/"
        subfiles = listdir(class_path)
        for m in range(len(subfiles)):
            image_path = subfiles[m]
            data.iloc[k]["path"] = class_path + image_path
            data.iloc[k]["target"] = c
            data.iloc[k]["patient_id"] = patient_id
            k += 1

image_cache = {}

for idx, row in data.iterrows():
    image_path = row["path"]
    image = Image.open(image_path).convert("RGB")
    image_cache[image_path] = image

dest_dir = "./dataset/augmented/"
augmented = listdir(dest_dir)


for i in augmented:
    image_path = dest_dir + i
    image = Image.open(image_path).convert("RGB")
    image_cache[image_path] = image


def extract_coords(df, column="path"):
    """Extract x and y coordinates from the file path."""
    coords = df[column].str.extract(r"x(?P<x>\d+)_y(?P<y>\d+)")
    df["x"] = coords["x"].astype(int)
    df["y"] = coords["y"].astype(int)
    return df


def get_cancer_dataframe(patient_id, cancer_id):
    """Create a DataFrame for a specific cancer ID."""
    path = f"{base_path}/{patient_id}/{cancer_id}"
    files = listdir(path)

    dataframe = pd.DataFrame(files, columns=["path"])
    dataframe["path"] = path + "/" + dataframe["path"]
    dataframe["target"] = int(cancer_id)

    # Extract x and y coordinates
    dataframe = extract_coords(dataframe, column="path")
    return dataframe


def get_patient_dataframe(patient_id):
    """Combine DataFrames for both cancer ID 0 and 1 for a patient."""
    df_0 = get_cancer_dataframe(patient_id, "0")
    df_1 = get_cancer_dataframe(patient_id, "1")

    # Combine dataframes
    patient_df = pd.concat([df_0, df_1], ignore_index=True)
    return patient_df


BATCH_SIZE = 256
NUM_CLASSES = 2

torch.manual_seed(0)
np.random.seed(0)


patients = data.patient_id.unique()

train_ids, sub_test_ids = train_test_split(patients,
                                           test_size=0.3,
                                           random_state=0)
test_ids, dev_ids = train_test_split(
    sub_test_ids, test_size=0.5, random_state=0)

print(len(train_ids), len(dev_ids), len(test_ids))


train_df = data.loc[data.patient_id.isin(train_ids), :].copy()
test_df = data.loc[data.patient_id.isin(test_ids), :].copy()
dev_df = data.loc[data.patient_id.isin(dev_ids), :].copy()

train_df = extract_coords(train_df)
test_df = extract_coords(test_df)
dev_df = extract_coords(dev_df)


# Define source and destination directories
os.makedirs(dest_dir, exist_ok=True)

# Define Albumentations augmentation pipeline
# Define Albumentations augmentation pipeline
augmentation_pipeline = A.Compose([
    # Slightly adjust brightness and contrast to maintain medical information
    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    # Elastic Transform: Reduce distortion to avoid excessive deformation of features
    # A.ElasticTransform(alpha=30, sigma=30 * 0.05, alpha_affine=30 * 0.03, p=0.5),
    # Random Resized Crop: Ensure the crop covers most of the important tissue area
    A.RandomResizedCrop(height=50, width=50, scale=(0.9, 1.0), p=0.5),
    # CoarseDropout: Reduce the size and number of holes to avoid losing crucial regions
    # A.CoarseDropout(max_holes=3, max_height=3, max_width=3, p=0.5),
    # Horizontal Flip: Augment by flipping images horizontally
    A.HorizontalFlip(p=0.2),
    # Vertical Flip: Augment by flipping images vertically
    A.VerticalFlip(p=0.2),
    # Resize to maintain consistency
    A.Resize(height=50, width=50)
])


# Function to apply augmentations
def augment_image_albumentations(image):
    # Convert PIL image to numpy array
    image_np = np.array(image)
    augmented = augmentation_pipeline(image=image_np)
    return Image.fromarray(augmented["image"])


# Find class distributions
class_counts = train_df['target'].value_counts()
minority_class = class_counts.idxmin()  # Class with fewer samples
majority_class = class_counts.idxmax()  # Class with more samples

# Filter rows for minority class
minority_data = train_df[train_df['target'] == minority_class]
majority_count = class_counts[majority_class]

# List to store new rows for the augmented dataset
new_rows = []

# Perform augmentation until the dataset is balanced
while len(minority_data) + len(new_rows) < majority_count:
    for index, row in minority_data.iterrows():
        # Check if balance is achieved
        if len(minority_data) + len(new_rows) >= majority_count:
            break

        image_path = row['path']
        patient_id = row['patient_id']
        x, y = row['x'], row['y']

        # Apply augmentations
        new_image_name = os.path.basename(
            "aug_" + image_path.split('/')[-1])  # Augmented name
        new_image_path = os.path.join(dest_dir, new_image_name)

        # Skip if already augmented
        if os.path.exists(new_image_path):
            new_rows.append({
                'patient_id': patient_id,
                'path': new_image_path,
                'target': minority_class,
                'x': x,
                'y': y
            })
            continue

        try:
            # Load image
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        # Augment image using Albumentations
        augmented_image = augment_image_albumentations(image)
        augmented_image.save(new_image_path)

        # Add new row to the dataset
        new_rows.append({
            'patient_id': patient_id,
            'path': new_image_path,
            'target': minority_class,
            'x': x,
            'y': y
        })

    if len(minority_data) + len(new_rows) >= majority_count:
        break

# Append new rows to the dataset
train_df = pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True)

# Check class distribution after augmentation
print(
    f"Class distribution after augmentation:\n{train_df['target'].value_counts()}")


def my_transform(key="train", plot=False):
    train_sequence = [
        transforms.Resize((50, 50)),
        # Data Augmentations for Training
        transforms.RandomHorizontalFlip(p=0.1),  # Horizontal flip
        transforms.RandomRotation(10),  # Reduced rotation to ±15 degrees
        transforms.RandomResizedCrop(size=(50, 50), scale=(
            0.95, 1.0)),  # Slightly reduced cropping scale
    ]

    val_sequence = [
        transforms.Resize((50, 50))  # Only resizing for validation
    ]

    # Convert to tensor and normalize for both train and validation
    if not plot:
        train_sequence.extend([
            transforms.ToTensor(),
            # Normalize to ImageNet stats
            transforms.Normalize([0.7854, 0.6031, 0.7135], [
                                 0.0953, 0.1400, 0.1035])
        ])
        val_sequence.extend([
            transforms.ToTensor(),
            # Normalize to ImageNet stats
            transforms.Normalize([0.7854, 0.6031, 0.7135], [
                                 0.0953, 0.1400, 0.1035])
        ])

    # Define transformations for train and val
    data_transforms = {
        'train': transforms.Compose(train_sequence),
        'val': transforms.Compose(val_sequence)
    }

    return data_transforms[key]


# Update the Dataset class to use the image cache
class BreastCancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.states = df
        self.transform = transform
        self.image_cache = image_cache  # Use the preloaded images

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        patient_id = self.states.patient_id.values[idx]
        x_coord = self.states.x.values[idx]
        y_coord = self.states.y.values[idx]
        image_path = self.states.path.values[idx]

        # Retrieve the preloaded image
        image = self.image_cache[image_path]

        if self.transform:
            image = self.transform(image)

        if "target" in self.states.columns.values:
            target = int(self.states.target.values[idx])
        else:
            target = None

        return {
            "image": image,
            "label": target,
            "patient_id": patient_id,
            "x": x_coord,
            "y": y_coord
        }


train_dataset = BreastCancerDataset(
    train_df, transform=my_transform(key="train"))
dev_dataset = BreastCancerDataset(dev_df, transform=my_transform(key="val"))
test_dataset = BreastCancerDataset(test_df, transform=my_transform(key="val"))


train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dev_dataloader = DataLoader(
    dev_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

dataloaders = {"train": train_dataloader,
               "dev": dev_dataloader, "test": test_dataloader}
print(len(dataloaders["train"]), len(
    dataloaders["dev"]), len(dataloaders["test"]))


# Define the ResNet structure for your task
networks = {
    'resnet18_light': {
        'block': ResidualBlock,
        'stage_args': [
            (32, 64, 3, False)
        ],
        'dropout': True,  # Enable dropout
        'p': 0.5  # Dropout probability
    }
}


def get_resnet(name):
    return ResNet(**networks[name])


to_float = torch.float
to_long = torch.long
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


##################################################################################
#                                       MAIN                                     #
##################################################################################
lr = 0.0005
weight_decay = 1e-3
epochs = 50

# Model and file paths
name = 'resnet18_light'
version = 3
checkpoint_path = f'./checkpoint/{name}_{version}_checkpoint.pth'
model_path = f'./models/{name}_{version}_checkpoint.pth'
history_path = f'./history/{name}_{version}.pth'

min_lr = 1e-6
max_lr = 0.006
max_iterations = len(dataloaders["train"])/2


if os.path.exists(checkpoint_path):
    # Resume training from checkpoint
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model = get_resnet(name).to(device)
    model.load_state_dict(checkpoint['model_state'])

    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.SGD(model.fc.parameters(), min_lr)
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    scheduler = CyclicLR(optimizer=optimizer,
                         base_lr=min_lr,
                         max_lr=max_lr,
                         step_size_up=max_iterations,
                         step_size_down=max_iterations,
                         mode="triangular")

    if checkpoint['scheduler_state'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state'])

    start_epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    train_metrics_history = checkpoint['train_history']
    val_metrics_history = checkpoint['val_history']
    lr_history = checkpoint['lr_history']

    print(f"Training will resume from epoch {start_epoch}.\n")

else:
    # Start new training
    print(f"Training new model: {name}\n")

    # Initialize model and optimizer
    model = get_resnet(name).to(device)
    model.apply(initialize_weights)

    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.SGD(model.fc.parameters(), min_lr)

    # Define scheduler
    scheduler = CyclicLR(optimizer=optimizer,
                         base_lr=min_lr,
                         max_lr=max_lr,
                         step_size_up=max_iterations,
                         step_size_down=max_iterations,
                         mode="triangular")

    # Initialize metrics and state
    start_epoch = 0
    best_val_acc = 0.0
    train_metrics_history = {'loss': [], 'accuracy': [],
                             'precision': [], 'recall': [], 'f1': []}
    val_metrics_history = {'accuracy': [],
                           'precision': [], 'recall': [], 'f1': []}
    lr_history = []

# Train model
train_metrics_history, val_metrics_history, lr_history = train_model(
    model, optimizer, train_dataloader, dev_dataloader,
    device=device, dtype=torch.float32, epochs=epochs,
    scheduler=scheduler, verbose=True,
    checkpoint_path=checkpoint_path,
    history_path=history_path
)

# Save final model and history after training completes
torch.save(model.state_dict(), model_path)
print(f"Final model saved at: {model_path}")

with open(history_path, 'wb') as f:
    pickle.dump((train_metrics_history, val_metrics_history, lr_history), f)
print(f"Training history saved at: {history_path}")

# Evaluate model on the test set
test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(
    test_dataloader, model, device=device)
print(f"Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
      f"Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")

# Plot metrics
plot_all_metrics(train_metrics_history, val_metrics_history)
plot_learning_rate(lr_history)
plot_loss(train_metrics_history['loss'])

# cifar10_loader.py

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_CIFAR10(data_dir, batch_size=128, validation_split=0.1, num_workers=2):
    """
    Downloads the CIFAR-10 dataset, applies transformations, and returns data loaders.

    Args:
        data_dir (str): Directory to download/store the CIFAR-10 dataset.
        batch_size (int): Batch size for data loaders.
        validation_split (float): Fraction of training data to use as validation.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
        # Randomly crop the image with padding
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                             0.247, 0.243, 0.261])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # Normalize as above
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    # Split training dataset into training and validation
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

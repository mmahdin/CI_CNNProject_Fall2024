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
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
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

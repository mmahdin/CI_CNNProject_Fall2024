import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import torch.nn.init as init
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import pandas as pd
import numpy as np
import os
################################################################################
# ResNet for CIFAR-10
################################################################################


class PlainBlock(nn.Module):
    def __init__(self, Cin, Cout, downsample=False, dropout=False, p=0.5):
        super().__init__()
        stride = 2 if downsample else 1
        layers = [
            nn.Conv2d(Cin, Cout, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(Cout),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(p))
        layers.extend([
            nn.Conv2d(Cout, Cout, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(Cout)
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, Cin, Cout, downsample=False, dropout=False, p=0.5):
        super().__init__()
        self.block = PlainBlock(Cin, Cout, downsample, dropout, p)
        stride = 2 if downsample else 1
        self.shortcut = nn.Sequential(
            nn.Conv2d(Cin, Cout, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(Cout)
        ) if downsample or Cin != Cout else nn.Identity()

    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x))


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, Cin, Cout, downsample=False, dropout=False, p=0.5):
        super().__init__()
        stride = 2 if downsample else 1
        mid_channels = Cout // 4
        layers = [
            nn.Conv2d(Cin, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(p))
        layers.extend([
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, Cout, kernel_size=1, bias=False),
            nn.BatchNorm2d(Cout)
        ])
        self.block = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(
            nn.Conv2d(Cin, Cout, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(Cout)
        ) if downsample or Cin != Cout else nn.Identity()

    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x))


class ResNetStem(nn.Module):
    def __init__(self, Cin=3, Cout=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(Cin, Cout, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(Cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResNetStage(nn.Module):
    def __init__(self, Cin, Cout, num_blocks, downsample=True, block=ResidualBlock, dropout=False, p=0.5):
        super().__init__()
        blocks = [block(Cin, Cout, downsample, dropout, p)]
        for _ in range(num_blocks - 1):
            blocks.append(block(Cout, Cout, dropout=dropout, p=p))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    def __init__(self, stage_args, Cin=3, block=ResidualBlock, num_classes=10, dropout=False, p=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            ResNetStem(Cin=Cin, Cout=stage_args[0][0]),
            *[ResNetStage(*args, block=block, dropout=dropout, p=p) for args in stage_args]
        )
        self.fc = nn.Linear(stage_args[-1][1], num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.fc(x)


########################### Functions ###########################


def reset_seed(number):
    random.seed(number)
    torch.manual_seed(number)
    return


def adjust_learning_rate(optimizer, lrd, epoch, schedule):
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * lrd
            param_group['lr'] = new_lr
            print(f'Learning rate updated from {old_lr:.6f} to {new_lr:.6f}')


def calculate_metrics(loader, model, device='cpu', dtype=torch.float32):
    model.eval()
    num_correct = 0
    num_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            images = data['image'].to(device=device, dtype=dtype)
            labels = data['label'].to(device=device)

            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            num_correct += (preds == labels).sum().item()
            num_samples += labels.size(0)

    accuracy = float(num_correct) / num_samples
    precision = precision_score(
        all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds,
                          average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1


def check_accuracy(loader, model, device='cpu', dtype=torch.float32):
    """
    Check model accuracy on the given dataset loader.

    Inputs:
    - loader: DataLoader containing the dataset to evaluate on.
    - model: PyTorch model to evaluate.
    - device: Device to perform computations on ('cpu' or 'cuda').
    - dtype: Data type for input tensors.

    Returns:
    - acc: Computed accuracy as a float.
    """
    model.eval()  # Set the model to evaluation mode
    num_correct, num_samples = 0, 0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for data in loader:
            images = data['image'].to(device=device, dtype=dtype)
            labels = data['label'].to(device=device)

            # Forward pass
            outputs = model(images)
            # Get the index of the max log-probability
            _, preds = outputs.max(1)

            # Calculate the number of correct predictions
            num_correct += (preds == labels).sum().item()
            num_samples += labels.size(0)

    acc = float(num_correct) / num_samples
    print(f'Accuracy: {num_correct} / {num_samples} ({100.0 * acc:.2f}%)')
    return acc


def train_model(
    model, optimizer, loader_train, loader_val,
    device='cuda', dtype=torch.float32, epochs=1,
    scheduler=None, learning_rate_decay=0.1, schedule=[],
    verbose=True, checkpoint_path='./models/checkpoint.pth',
    history_path='./history/train_history.pkl'
):
    model = model.to(device)
    train_metrics_history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
    }
    val_metrics_history = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': []
    }
    lr_history = []
    best_val_acc = 0.0
    start_epoch = 0

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
        train_metrics_history = checkpoint['train_history']
        val_metrics_history = checkpoint['val_history']
        lr_history = checkpoint['lr_history']
        best_val_acc = checkpoint['best_val_acc']
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        epoch_loss = 0.0
        num_correct = 0
        num_samples = 0
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(loader_train):
            x = batch["image"].to(device=device, dtype=dtype)
            y = batch["label"].to(device=device)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = scores.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            num_correct += (preds == y).sum().item()
            num_samples += y.size(0)
            epoch_loss += loss.item()

            if verbose and batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}, Loss = {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader_train)
        train_metrics_history['loss'].append(avg_loss)
        train_accuracy = float(num_correct) / num_samples
        train_precision = precision_score(
            all_labels, all_preds, average='weighted', zero_division=0)
        train_recall = recall_score(
            all_labels, all_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(all_labels, all_preds,
                            average='weighted', zero_division=0)

        train_metrics_history['accuracy'].append(train_accuracy)
        train_metrics_history['precision'].append(train_precision)
        train_metrics_history['recall'].append(train_recall)
        train_metrics_history['f1'].append(train_f1)

        print(f"  Training Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
              f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}")

        # Validation phase
        val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(
            loader_val, model, device=device, dtype=dtype)
        val_metrics_history['accuracy'].append(val_accuracy)
        val_metrics_history['precision'].append(val_precision)
        val_metrics_history['recall'].append(val_recall)
        val_metrics_history['f1'].append(val_f1)

        print(f"  Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")

        # Update learning rate
        if scheduler:
            scheduler.step()
        else:
            adjust_learning_rate(
                optimizer, learning_rate_decay, epoch, schedule)

        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'train_history': train_metrics_history,
            'val_history': val_metrics_history,
            'lr_history': lr_history,
            'best_val_acc': best_val_acc
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved at epoch {epoch + 1}")

    print("Training complete!")
    return train_metrics_history, val_metrics_history, lr_history


def plot_metrics(metrics_history_train, metrics_history_val, metric_name):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history_train, label=f'Train {metric_name}', marker='o')
    plt.plot(metrics_history_val,
             label=f'Validation {metric_name}', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot for all metrics


def plot_all_metrics(train_metrics, val_metrics):
    for metric in train_metrics.keys():
        if metric in val_metrics:
            plot_metrics(train_metrics[metric], val_metrics[metric], metric)


def plot_val_train_acc(train_acc_history, val_acc_history):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_history, label='Training Accuracy', marker='o')
    plt.plot(val_acc_history, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_learning_rate(lr_history):
    plt.figure(figsize=(8, 6))
    plt.plot(lr_history, marker='o', linestyle='-', color='b')
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()


def plot_loss(loss_history):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, marker='o', linestyle='-', color='b')
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def initialize_weights(m):
    """
    Initialize weights of the model.
    Applies Kaiming initialization to Conv2D layers and Xavier initialization to Linear layers.
    """
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

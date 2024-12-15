import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import torch.nn.init as init

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
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return


def adjust_learning_rate(optimizer, lrd, epoch, schedule):
    """
    Adjusts the learning rate by multiplying it with lrd at specified epochs.

    Inputs:
    - optimizer: PyTorch optimizer object.
    - lrd: Learning rate decay factor.
    - epoch: Current epoch number.
    - schedule: List of epochs to decay the learning rate.

    Returns: None (updates optimizer's learning rate in place).
    """
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * lrd
            param_group['lr'] = new_lr
            print(f'Learning rate updated from {old_lr:.6f} to {new_lr:.6f}')


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
    verbose=True, model_path='./models/best.pth'
):
    """
    Train a PyTorch model using the provided data loaders and optimizer.

    Inputs:
    - model: PyTorch model to train.
    - optimizer: Optimizer for updating model weights.
    - loader_train: DataLoader for training data.
    - loader_val: DataLoader for validation data.
    - device: Computation device ('cpu' or 'cuda').
    - dtype: Data type for tensors (default: torch.float32).
    - epochs: Number of epochs for training.
    - scheduler: Learning rate scheduler (optional).
    - learning_rate_decay: Decay factor for learning rate (if no scheduler).
    - schedule: Epochs to adjust learning rate manually (if no scheduler).
    - verbose: Print progress information during training.
    - model_path: Path to save the best model checkpoint.

    Returns:
    - train_acc_history: List of training accuracies for each epoch.
    - val_acc_history: List of validation accuracies for each epoch.
    - lr_history: List of learning rates for each epoch.
    - train_loss_history: List of average training losses for each epoch.
    """
    model = model.to(device)
    train_acc_history = []
    val_acc_history = []
    lr_history = []
    train_loss_history = []

    best_val_acc = 0.0  # Track the best validation accuracy

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        num_correct = 0
        num_samples = 0
        num_batches = 0

        for batch_idx, batch in enumerate(loader_train):
            x = batch["image"].to(device=device, dtype=dtype)
            y = batch["label"].to(device=device)

            # Forward pass
            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            _, preds = scores.max(1)
            num_correct += (preds == y).sum().item()
            num_samples += y.size(0)
            epoch_loss += loss.item()
            num_batches += 1

            if verbose and batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}, Loss = {loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        train_loss_history.append(avg_loss)

        # Training accuracy
        train_acc = num_correct / num_samples
        train_acc_history.append(train_acc)

        print(f"  Training Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation phase
        val_acc = check_accuracy(loader_val, model, device=device, dtype=dtype)
        val_acc_history.append(val_acc)

        # Update learning rate
        if scheduler:
            scheduler.step()
        else:
            adjust_learning_rate(
                optimizer, learning_rate_decay, epoch, schedule)

        # Record learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(
                f"  Best model saved with Validation Accuracy: {val_acc:.4f}")

    print("Training complete!")
    return train_acc_history, val_acc_history, lr_history, train_loss_history


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

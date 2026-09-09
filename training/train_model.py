"""
NETRADRISHTI Model Training Pipeline
Trains the NetraNet convolutional neural network on partitioned fundus data.
Adheres to strict scientific validation (No test data leakage into training/val).
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, MODELS_DIR, MODEL_WEIGHTS_PATH, NUM_CLASSES
from models.dr_net import NetraNet, get_preprocessing_transforms


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training NetraNet on device: {device}")

    # Transforms: Data augmentation on training, standard normalization on validation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = get_preprocessing_transforms()

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "validation"

    if not train_dir.exists() or not val_dir.exists():
        print("[ERROR] Train or validation directory missing. Running dataset generator first...")
        from training.prepare_sample_dataset import generate_dataset_and_demo_cases
        generate_dataset_and_demo_cases()

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"Dataset classes detected: {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Initialize model
    model = NetraNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    num_epochs = 8
    best_val_acc = 0.0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("-" * 65)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<10}")
    print("-" * 65)

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        print(f"{epoch:<8} | {epoch_train_loss:<12.4f} | {epoch_train_acc:<11.1f}% | {epoch_val_loss:<12.4f} | {epoch_val_acc:<9.1f}%")

        # Save checkpoint
        if epoch_val_acc >= best_val_acc or epoch == num_epochs:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_acc': epoch_val_acc,
                'classes': train_dataset.classes
            }, MODEL_WEIGHTS_PATH)

    print("-" * 65)
    print(f"[SUCCESS] Best Model Checkpoint successfully saved to:")
    print(f"          {MODEL_WEIGHTS_PATH}")


if __name__ == "__main__":
    train()

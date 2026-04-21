#!/usr/bin/env python3
"""
TRAIN CNN — 4 class classification
Input:  64x64 RGB image
Output: one of 4 classes
  0 = go straight
  1 = steer left  (line is on left, robot drifted right)
  2 = steer right (line is on right, robot drifted left)
  3 = spin in place (no line visible)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# ── Dataset ───────────────────────────────────────────────────────────────────
CLASS_DIRS = {
    0: os.path.expanduser("~/line_follower_ml/data/class_0_center"),
    1: os.path.expanduser("~/line_follower_ml/data/class_1_drift_right"),
    2: os.path.expanduser("~/line_follower_ml/data/class_2_drift_left"),
    3: os.path.expanduser("~/line_follower_ml/data/class_3_no_line"),
}

class LineDataset(Dataset):
    def __init__(self):
        self.samples = []   # list of (filepath, class_label)

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std= [0.5, 0.5, 0.5])
        ])

        for label, folder in CLASS_DIRS.items():
            if not os.path.exists(folder):
                print(f"WARNING: folder missing: {folder}")
                continue
            files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            for f in files:
                self.samples.append((os.path.join(folder, f), label))

        print(f"\nDataset summary:")
        for label, folder in CLASS_DIRS.items():
            count = sum(1 for _, l in self.samples if l == label)
            print(f"  Class {label}: {count} images")
        print(f"  TOTAL: {len(self.samples)} images\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert('RGB')
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# ── CNN Model ─────────────────────────────────────────────────────────────────
class LineCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 64→32

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 32→16

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 16→8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)   # 4 output neurons
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x   # raw logits — CrossEntropyLoss handles softmax internally


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = LineDataset()
    if len(dataset) == 0:
        print("ERROR: No images found! Did you run collect_data.py first?")
        return

    # 80% train, 20% validation split
    val_size   = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

    model     = LineCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    model_path = os.path.expanduser("~/line_follower_ml/model.pth")
    best_val_acc = 0.0
    EPOCHS = 40

    print(f"Starting training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds    = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = 100.0 * train_correct / train_total
        val_acc   = 100.0 * val_correct   / val_total
        scheduler.step()

        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Loss: {train_loss/len(train_loader):.3f} | "
              f"Train acc: {train_acc:.1f}% | "
              f"Val acc: {val_acc:.1f}%", end="")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  <- SAVED (best)")
        else:
            print()

    print(f"\nTraining done! Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    train()

import os
import sklearn as sl
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import functions as fn


train_data = os.path.expanduser('~/desktop/Steel-Image-Fault-Detection/images/train_images')
train_labels = os.path.expanduser('~/desktop/Steel-Image-Fault-Detection/images/train.csv')

data = fn.build_training_dataset(train_labels, train_data)

#fn.show_sample(data, idx=1)
# -------------------------
# Dataset
# -------------------------
class SteelDataset(Dataset):
    def __init__(self, data, image_size=(256, 1600)):
        self.data = data
        self.image_size = image_size

        self.img_tf = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask, class_ids = self.data[idx]

        image = Image.open(img_path).convert("L")
        image = self.img_tf(image)

        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


# -------------------------
# U-Net
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

# =========================
# 2. DICE LOSS
# =========================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)

        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=preds.shape[1]
        ).permute(0, 3, 1, 2).float()

        intersection = (preds * targets_one_hot).sum(dim=(2,3))
        union = preds.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

# =========================
# 3. UPDATED TRAIN FUNCTION (uses Dice Loss)
# =========================


import time

def train(model, loader, device, epochs=5, lr=1e-3, optimizer_name="Adam"):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = DiceLoss()

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    end_time = time.time()

    return {
        "runtime_sec": end_time - start_time
    }

# -------------------------
# Usage
# -------------------------
def run_training(data):
    dataset = SteelDataset(data)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, num_classes=5)

    train(model, loader, device)


# =========================
# 1. MODEL SUMMARY / PARAMETERS
# =========================
def model_summary(model):
    total_params = 0
    trainable_params = 0

    print("\n--- MODEL LAYERS ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()}")
            trainable_params += param.numel()
        total_params += param.numel()

    print("\n--- SUMMARY ---")
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")

# =========================
# 4. LOGGING FUNCTION
# =========================
import pandas as pd

def log_run(
    model,
    runtime,
    epochs,
    batch_size,
    lr,
    optimizer_name,
    loss_name,
    val_score=None,
    notes=""
):
    summary_lines = []

    for name, param in model.named_parameters():
        summary_lines.append(f"{name}:{param.numel()}")

    log = {
        "model_summary": " | ".join(summary_lines),
        "runtime_sec": runtime,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": optimizer_name,
        "loss_function": loss_name,
        "val_score": val_score,
        "notes": notes
    }

    df = pd.DataFrame([log])

    file_exists = os.path.exists("training_logs.csv")
    df.to_csv("training_logs.csv", mode="a", header=not file_exists, index=False)

    return df

def run_training_pipeline(data, epochs=5, batch_size=2, lr=1e-3):
    dataset = SteelDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, num_classes=5)

    model_summary(model)

    result = train(
        model,
        loader,
        device,
        epochs=epochs,
        lr=lr
    )

    log_run(
        model=model,
        runtime=result["runtime_sec"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer_name="Adam",
        loss_name="DiceLoss",
        val_score=None,
        notes=""
    )

    return model

if __name__ == "__main__":
    model = run_training_pipeline(data, epochs=5, batch_size=2, lr=1e-3)

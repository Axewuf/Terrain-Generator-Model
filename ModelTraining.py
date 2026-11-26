from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from Model import TerrainSR
from ImageProcessing import TerrainDataset


# Configuration
ROOT = Path(__file__).resolve().parent

DEM_DIR = ROOT / "SierraNevada_DEM"
OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
PATCHES_PER_IMAGE = 20

# Only accept DEMs with 100% valid elevation data
MIN_VALID_COVERAGE = 1.0


# Validate DEM coverage
def is_valid_dem(path, min_coverage=1.0):

    with rasterio.open(path) as src:

        data = src.read(1, masked=True)

        valid = (
            ~np.ma.getmaskarray(data)
            & np.isfinite(data.data)
        )

        coverage = valid.mean()

        return coverage >= min_coverage


# Find DEM files
files = sorted(
    DEM_DIR.glob(
        "USGS_OPR_CA_SierraNevada_B22_be_*.tif"
    )
)

if not files:
    raise RuntimeError(f"No DEM files found in {DEM_DIR}")

print(f"Found {len(files)} DEM files")


# Filter incomplete DEMs
valid_files = []

for i, path in enumerate(files):

    if is_valid_dem(path, MIN_VALID_COVERAGE):
        valid_files.append(path)
    else:
        print(f"Skipping incomplete DEM: {path.name}")

print(f"Usable DEM files: {len(valid_files)}")
print(f"Discarded DEM files: {len(files) - len(valid_files)}")

files = valid_files

if len(files) < 2:
    raise RuntimeError(
        "Not enough valid DEM files for training and validation"
    )


# Reproducible file-level split
generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(files), generator=generator
).tolist()

split = int(len(files) * 0.9)

train_files = [files[i] for i in indices[:split]]
val_files = [files[i] for i in indices[split:]]

print(f"Training DEMs: {len(train_files)}")
print(f"Validation DEMs: {len(val_files)}")


# Datasets
train_dataset = TerrainDataset(
    train_files,
    patches_per_image=PATCHES_PER_IMAGE
)

val_dataset = TerrainDataset(
    val_files,
    patches_per_image=5
)


# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Device selection
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Training on: {device}")


# Model
model = TerrainSR().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()


# Training
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for lr, hr in train_loader:

        lr = lr.to(device)
        hr = hr.to(device)

        optimizer.zero_grad()

        prediction = model(lr)
        loss = criterion(prediction, hr)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(device)
            hr = hr.to(device)

            prediction = model(lr)

            val_loss += criterion(
                prediction, hr
            ).item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train: {train_loss:.6f} | "
        f"Validation: {val_loss:.6f}"
    )


# Save model
model_path = OUTPUT_DIR / "TerrainSR.pth"

torch.save(
    model.cpu().state_dict(),
    model_path
)

print(f"Model saved to {model_path}")

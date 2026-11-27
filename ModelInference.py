from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from rasterio.windows import Window
from rasterio.transform import Affine

from Model import TerrainSR


# Configuration
ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "output" / "TerrainSR.pth"

INPUT_PATH = (
    ROOT / "SierraNevada_DEM"
    / "USGS_OPR_CA_SierraNevada_B22_be_11SKA7089.tif"
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_DIR = OUTPUT_DIR / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

HR_SIZE = 256
SCALE_FACTOR = 2
LR_SIZE = HR_SIZE // SCALE_FACTOR


# Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# Load trained model
model = TerrainSR().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()


# Read original 256x256 elevation patch
with rasterio.open(INPUT_PATH) as src:

    if src.width < HR_SIZE or src.height < HR_SIZE:
        raise ValueError("DEM is smaller than the requested patch.")

    x = (src.width - HR_SIZE) // 2
    y = (src.height - HR_SIZE) // 2

    window = Window(x, y, HR_SIZE, HR_SIZE)

    patch = src.read(
        1,
        window=window,
        masked=True
    )

    if np.ma.getmaskarray(patch).any():
        raise ValueError(
            "Selected patch contains NoData pixels."
        )

    original = np.asarray(
        patch, dtype=np.float32
    )

    if not np.isfinite(original).all():
        raise ValueError("Invalid elevation values")

    hr_transform = src.window_transform(window)
    profile = src.profile.copy()


# Convert original to tensor (raw elevations)
hr_tensor = torch.from_numpy(
    original.copy()
).unsqueeze(0).unsqueeze(0)


# Downsample to 128x128 (bicubic)
lr_tensor = F.interpolate(
    hr_tensor,
    size=(LR_SIZE, LR_SIZE),
    mode="bicubic",
    align_corners=False
)


# Preserve raw LR elevations
lr = lr_tensor.squeeze().numpy().copy()


# Calculate normalization parameters from LR
minimum = lr_tensor.min()
maximum = lr_tensor.max()

elevation_range = (maximum - minimum).clamp(min=1e-6)


# Normalize LR tensor
lr_normalized = (
    lr_tensor - minimum
) / elevation_range


# Run super-resolution
with torch.inference_mode():

    prediction = model(
        lr_normalized.to(device)
    )


# Denormalize prediction to actual elevation values
sr = (
    prediction.squeeze().cpu().numpy()
    * elevation_range.item()
    + minimum.item()
).astype(np.float32)


# GeoTIFF saving function
def save_geotiff(path, data, transform):

    output_profile = profile.copy()

    output_profile.update(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        transform=transform,
        nodata=None
    )

    with rasterio.open(
        path, "w", **output_profile
    ) as dst:

        dst.write(data, 1)


# Save original HR image
save_geotiff(
    OUTPUT_DIR / "Terrain_Original_256.tif",
    original,
    hr_transform
)


# Save downsampled LR input
lr_transform = hr_transform * Affine.scale(
    SCALE_FACTOR,
    SCALE_FACTOR
)

save_geotiff(
    OUTPUT_DIR / "Terrain_Input_128.tif",
    lr,
    lr_transform
)


# Save super-resolved output
save_geotiff(
    OUTPUT_DIR / "Terrain_SR_256.tif",
    sr,
    hr_transform
)


# Export raw Float32 binary files (little-endian)
original.astype("<f4", copy=False).tofile(
    EXPORT_DIR / "Terrain_Original_256.bin"
)

lr.astype("<f4", copy=False).tofile(
    EXPORT_DIR / "Terrain_Input_128.bin"
)

sr.astype("<f4", copy=False).tofile(
    EXPORT_DIR / "Terrain_SR_256.bin"
)


# Print elevation statistics
print("\nElevation statistics:")

for name, data in [
    ("Original HR", original),
    ("Input LR", lr),
    ("Output SR", sr)
]:
    print(f"\n{name}:")
    print(f" - Shape: {data.shape}")
    print(f" - Data type: {data.dtype}")
    print(f" - Minimum: {data.min():.4f} m")
    print(f" - Maximum: {data.max():.4f} m")
    print(f" - Mean: {data.mean():.4f} m")
    print(f" - Std: {data.std():.4f} m")


# Print saved files
print("\nSaved:")
print(" - Terrain_Original_256.tif")
print(" - Terrain_Input_128.tif")
print(" - Terrain_SR_256.tif")
print(" - export/Terrain_Original_256.bin")
print(" - export/Terrain_Input_128.bin")
print(" - export/Terrain_SR_256.bin")
import random
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from rasterio.windows import Window
from torch.utils.data import Dataset


class TerrainDataset(Dataset):

    def __init__(
        self,
        file_paths,
        patches_per_image=20,
        patch_size=256
    ):
        self.file_paths = list(file_paths)
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size

    def __len__(self):
        return len(self.file_paths) * self.patches_per_image

    def __getitem__(self, index):

        size = self.patch_size

        # Try different DEM files if necessary
        for file_attempt in range(len(self.file_paths)):

            file_index = index // self.patches_per_image
            path = self.file_paths[file_index]

            with rasterio.open(path) as src:

                if src.width < size or src.height < size:
                    index = random.randrange(len(self))
                    continue

                for attempt in range(100):

                    x = random.randint(0, src.width - size)
                    y = random.randint(0, src.height - size)

                    patch = src.read(
                        1,
                        window=Window(x, y, size, size),
                        masked=True
                    )

                    if np.ma.getmaskarray(patch).any():
                        continue

                    heights = np.asarray(
                        patch, dtype=np.float32
                    )

                    if np.isfinite(heights).all():
                        break

                else:
                    print(f"Skipping DEM: {path.name}")

                    index = random.randrange(len(self))
                    continue

            # HR tensor (raw elevations)
            hr = torch.from_numpy(
                heights.copy()
            ).unsqueeze(0)

            # LR tensor (raw elevations)
            lr = F.interpolate(
                hr.unsqueeze(0),
                size=(size // 2, size // 2),
                mode="bicubic",
                align_corners=False
            ).squeeze(0)

            # Calculate normalization parameters from LR
            minimum = lr.min()
            maximum = lr.max()

            elevation_range = (maximum - minimum).clamp(min=1e-6)

            # Normalize both tensors using LR statistics
            lr = (lr - minimum) / elevation_range
            hr = (hr - minimum) / elevation_range

            return lr, hr

        raise RuntimeError("Could not find a valid DEM patch")

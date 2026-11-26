import rasterio
import numpy as np

with rasterio.open("output/Terrain_Input_128.tif") as src:
    elevation = src.read(1).astype(np.float32)

    print("Width:", src.width)
    print("Height:", src.height)
    print("Data type:", elevation.dtype)

    elevation.tofile("output/export/Terrain_Input_128.bin")
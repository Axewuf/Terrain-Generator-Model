
from pathlib import Path

import torch
import coremltools as ct
import numpy as np

from Model import TerrainSR


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"

# Load trained weights
model = TerrainSR()

model.load_state_dict(
    torch.load(
        OUTPUT_DIR / "TerrainSR.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()


# Trace the PyTorch model
example_input = torch.randn(1, 1, 128, 128)

traced_model = torch.jit.trace(
    model,
    example_input
)


# Convert to Core ML
mlmodel = ct.convert(
    traced_model,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS13,
    compute_precision=ct.precision.FLOAT16,

    inputs=[
        ct.TensorType(
            name="elevation",
            shape=(1, 1, 128, 128),
            dtype=np.float32
        )
    ],

    outputs=[
        ct.TensorType(
            name="superResolved",
            dtype=np.float32
        )
    ]
)


# Save Core ML model
output_path = OUTPUT_DIR / "TerrainSR.mlpackage"

mlmodel.save(str(output_path))

print(f"Core ML model saved to {output_path}")


# Verify prediction against PyTorch
test_input = np.random.default_rng(42).random(
    (1, 1, 128, 128)
).astype(np.float32)

with torch.no_grad():
    pytorch_output = model(
        torch.from_numpy(test_input)
    ).numpy()

coreml_output = mlmodel.predict({
    "elevation": test_input
})["superResolved"]

difference = np.max(
    np.abs(pytorch_output - coreml_output)
)

print(f"Maximum absolute difference: {difference:.6f}")
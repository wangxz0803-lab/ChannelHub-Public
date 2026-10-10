"""Create a finite, known-amplitude custom transmit grid for custom-x.json."""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(123)
bits = rng.integers(0, 2, (28, 1, 288, 2))
x = ((1 - 2 * bits[..., 0]) + 1j * (1 - 2 * bits[..., 1])) / np.sqrt(2) * np.sqrt(0.001)
output = Path("outputs/custom-x.npy")
output.parent.mkdir(exist_ok=True)
if output.exists():
    raise FileExistsError("Choose a new filename to preserve the existing waveform")
np.save(output, x.astype(np.complex128), allow_pickle=False)
print(f"Saved {output}: X[time=28, tx_port=1, subcarrier=288], sqrt(W)/RE")

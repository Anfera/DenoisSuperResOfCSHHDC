import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import BLUE_NOISE_PATH

def create_mask(
    input_shape: tuple[int, ...],
    ratio: float,
    mask_type: str = "blue_noise",
    device: str = "cpu",
    blue_noise_path: str | Path = BLUE_NOISE_PATH,
) -> torch.Tensor:
    """
    Create a sampling mask for the acquisition model.

    Args:
        input_shape: Shape of the target tensor (expects [..., H, W]).
        ratio: Fraction of elements to sample (0-1].
        mask_type: One of {"random", "blue_noise", "bayer"}.
        device: Target device for the mask.
        blue_noise_path: Path to the blue-noise TIFF mask.

    Returns:
        A boolean mask tensor on the requested device.
    """
    if ratio > 1:
        raise ValueError("Ratio must be less than or equal to 1.")

    blue_noise_path = Path(blue_noise_path)
    mask_size = (round(input_shape[-2]), input_shape[-1])

    if mask_type == "random":
        mask = torch.rand(mask_size, device=device) < ratio
    elif mask_type == "blue_noise":
        blue_noise_img = Image.open(blue_noise_path).convert("L")
        blue_noise = np.array(blue_noise_img, dtype=np.float32) / 255.0
        blue_noise = blue_noise[:mask_size[0], :mask_size[1]]
        mask = torch.tensor(blue_noise, device=device) < ratio
    elif mask_type == "bayer":
        bayer = np.array(
            [
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5],
            ],
            dtype=np.float32,
        )
        bayer = bayer / 16.0
        mask_rows, mask_cols = mask_size
        tile_rows = int(math.ceil(mask_rows / bayer.shape[0]))
        tile_cols = int(math.ceil(mask_cols / bayer.shape[1]))
        tiled = np.tile(bayer, (tile_rows, tile_cols))[:mask_rows, :mask_cols]
        mask = torch.tensor(tiled, device=device) < ratio
    else:
        raise ValueError("Invalid mask_type. Choose among 'random', 'blue_noise', or 'bayer'.")

    return mask

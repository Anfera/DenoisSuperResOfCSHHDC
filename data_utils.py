import os
from typing import Tuple

import numpy as np
import torch

from config import EPSILON, RESOLUTION


def load_test_data(
    resolution: int = RESOLUTION,
    factor: int = 3,
    device: str = "cpu",
    data_dir: str = "TestCube",
    ground_truth_file: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess the test cube and ground truth tensors.

    Args:
        resolution: Spatial resolution factor of the acquisition.
        factor: Upsampling factor to apply to the loaded tensors.
        device: Device to place tensors on.
        data_dir: Directory containing the `.npy` files.
        ground_truth_file: Optional override for the GT file name.

    Returns:
        Tuple of (input_data, ground_truth) tensors on the requested device.
    """
    input_path = os.path.join(data_dir, f"input{resolution}.npy")
    gt_filename = ground_truth_file or f"gt{resolution}.npy"
    gt_path = os.path.join(data_dir, gt_filename)

    # Fall back to the provided gt2.npy if a resolution-specific GT file is absent.
    if not os.path.exists(gt_path):
        gt_path = os.path.join(data_dir, "gt2.npy")

    input_data = torch.from_numpy(np.load(input_path)).float().to(device)
    ground_truth = torch.from_numpy(np.swapaxes(np.load(gt_path), -1, -2)).float().to(device)

    input_data = input_data[:, :32 * factor, :16 * factor]
    ground_truth = ground_truth[:, :(96 // resolution) * factor, :(96 // resolution) * factor]

    ground_truth = ground_truth / (ground_truth.max(dim=0)[0] + EPSILON)
    return input_data, ground_truth

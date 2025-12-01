import os
import torch
import numpy as np

# Reproducibility
SEED = 42

# Data and model dimensions
RESOLUTION = 2
FACTOR = 3
IMAGE_SIZE = 96 // RESOLUTION

# Sampling and reconstruction
MASK_PERCENTAGE = 100
MASK_TYPE = "blue_noise"
MASK_RATIO = MASK_PERCENTAGE / 100
SNR_DB = 10.0
SAMPLING_TIMESTEPS = 250
PHOTONS = 20

# Training / optimization
LR_MULTIPLIER = 0.010
EPSILON = 1e-6
RECON_THRESHOLD = 0.05

# Paths and runtime settings
DEVICE = "cuda"
RESULT_DIR = "resultCubes"
INTERMEDIATE_DIR = "intermediateCubesTest"
CHECKPOINT_DIR = "results"
CHECKPOINT_TEMPLATE = "model{resolution}.pt"


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_path(resolution: int = RESOLUTION) -> str:
    """Return the expected checkpoint path for a given resolution."""
    filename = CHECKPOINT_TEMPLATE.format(resolution=resolution)
    return os.path.join(CHECKPOINT_DIR, filename)

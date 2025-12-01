from pathlib import Path

import numpy as np
import torch

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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "resultCubes"
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediateCubesTest"
CHECKPOINT_DIR = PROJECT_ROOT / "results"
CHECKPOINT_TEMPLATE = "model{resolution}.pt"
DATA_DIR = PROJECT_ROOT / "data" / "TestCube"
BLUE_NOISE_PATH = PROJECT_ROOT / "assets" / "Mblue.tiff"


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_path(resolution: int = RESOLUTION) -> str:
    """Return the expected checkpoint path for a given resolution."""
    filename = CHECKPOINT_TEMPLATE.format(resolution=resolution)
    return str(CHECKPOINT_DIR / filename)

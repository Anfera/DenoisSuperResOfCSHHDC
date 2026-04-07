from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Data and model dimensions
# ---------------------------------------------------------------------------
RESOLUTION = 2
FACTOR = 3
IMAGE_SIZE = 96 // RESOLUTION  # 48

# ---------------------------------------------------------------------------
# Physics forward model  (LidarForwardImagingModel)
# ---------------------------------------------------------------------------
INPUT_RES_M = (2.0, 2.0)        # Input pixel size (dy, dx) in metres
OUTPUT_RES_M = (3.0, 6.0)       # Output pixel size (dy, dx) in metres
FOOTPRINT_DIAMETER_M = 10.0     # 1/e² laser beam diameter in metres
BACKGROUND_RATE = 0.1           # b: background photon rate per pixel
READOUT_NOISE = 0.5             # eta: Gaussian readout noise standard deviation
REF_ALTITUDE = 500.0            # Reference flight altitude in metres
REF_PHOTON_COUNT = 20.0         # Target mean photon count at reference altitude

# ---------------------------------------------------------------------------
# Sampling mask
# ---------------------------------------------------------------------------
MASK_PERCENTAGE = 100
MASK_TYPE = "blue_noise"         # Options: "blue_noise", "random", "bayer"
MASK_RATIO = MASK_PERCENTAGE / 100

# ---------------------------------------------------------------------------
# Diffusion sampling
# ---------------------------------------------------------------------------
SAMPLING_TIMESTEPS = 250

# ---------------------------------------------------------------------------
# DPS optimisation
# ---------------------------------------------------------------------------
LR_MULTIPLIER = 50000            # Gradient descent step scale (tuned for physics model)
EPSILON = 1e-6
RECON_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Paths and runtime settings
# ---------------------------------------------------------------------------
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

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


class CubeDataset(Dataset):
    """Dataset loader for pre-rendered cubes used during training."""

    def __init__(self, resolution: int, root: str | Path = "./Dataset"):
        preferred_root = Path(root) / f"DataOut{resolution}meter"
        fallback_roots = list(Path(root).glob(f"DataOut{resolution}*"))

        self.data_root = preferred_root
        for candidate in [preferred_root] + fallback_roots:
            if candidate.exists():
                self.data_root = candidate
                break

        self.cubes = sorted(self.data_root.glob("*.npy")) if self.data_root.exists() else []

        self.augmentations = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

    def __len__(self) -> int:
        return len(self.cubes)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if not self.cubes:
            raise FileNotFoundError(f"No cubes found under {self.data_root}.")

        cube = np.load(self.cubes[idx])
        cube = torch.from_numpy(cube).float()
        cube = self.augmentations(cube)

        return cube * 2 - 1

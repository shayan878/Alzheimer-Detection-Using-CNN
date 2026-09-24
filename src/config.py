from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass(frozen=True)
class Config:
    data_dir: Path
    image_size: int = 64
    augmentation_repeats: int = 4
    batch_size: int = 64
    test_size: float = 0.20
    random_state: int = 42
    learning_rate: float = 0.001
    epochs: int = 50
    num_classes: int = 2
    class_names: tuple[str, ...] = ("AD", "MCI")

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

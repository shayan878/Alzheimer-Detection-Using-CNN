from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ADNMIDataset(Dataset):
    """
    Loads original AD/MCI images and creates a fixed number of augmented
    samples per original image.

    The training dataset returned by __getitem__ contains only (image, label).
    Original and augmented tensors are exposed separately for visualization.
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 64,
        augmentation_repeats: int = 4,
        class_names: tuple[str, ...] = ("AD", "MCI"),
    ):
        self.data_dir = Path(data_dir)
        self.class_names = class_names
        self.augmentation_repeats = augmentation_repeats

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.augmentation = transforms.Compose(
            [
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.2,
                ),
            ]
        )

        self.img_data: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.labels_orig: list[int] = []
        self.orig_img: list[torch.Tensor] = []
        self.aug_img: list[torch.Tensor] = []

        self._load_data()

    def _load_data(self) -> None:
        for label, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file():
                    continue

                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image_tensor = self.transform(image)

                    self.img_data.append(image_tensor)
                    self.orig_img.append(image_tensor)
                    self.labels.append(label)
                    self.labels_orig.append(label)

                    for _ in range(self.augmentation_repeats):
                        augmented = self.augmentation(image)
                        augmented_tensor = self.transform(augmented)
                        self.img_data.append(augmented_tensor)
                        self.aug_img.append(augmented_tensor)
                        self.labels.append(label)

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.img_data[idx], self.labels[idx]

    @property
    def original_count(self) -> int:
        return len(self.orig_img)

    @property
    def augmented_count(self) -> int:
        return len(self.aug_img)

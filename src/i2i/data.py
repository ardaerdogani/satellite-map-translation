from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if hasattr(Image, "Resampling"):
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
else:
    RESAMPLE_BICUBIC = Image.BICUBIC


def list_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    )


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    rgb = image.convert("RGB")
    # Use a writable, contiguous copy to avoid torch.from_numpy readonly warnings.
    array = np.array(rgb, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return tensor


def normalize_01_to_m11(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2.0 - 1.0


def denormalize_m11_to_01(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor + 1.0) / 2.0


def _resize(image: Image.Image, image_size: int) -> Image.Image:
    return image.resize((image_size, image_size), RESAMPLE_BICUBIC)


class PairedImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        in_domain: str,
        out_domain: str,
        image_size: int = 256,
        hflip: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.in_domain = in_domain
        self.out_domain = out_domain
        self.image_size = image_size
        self.hflip = hflip

        in_dir = self.root / split / in_domain
        out_dir = self.root / split / out_domain
        in_files = {path.stem: path for path in list_image_files(in_dir)}
        out_files = {path.stem: path for path in list_image_files(out_dir)}

        common_ids = sorted(set(in_files.keys()) & set(out_files.keys()))
        if not common_ids:
            raise RuntimeError(
                f"No paired files found between {in_dir} and {out_dir}. "
                "Ensure filenames match across domains."
            )

        self.pairs = [(sample_id, in_files[sample_id], out_files[sample_id]) for sample_id in common_ids]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id, in_path, out_path = self.pairs[index]
        with Image.open(in_path) as in_image, Image.open(out_path) as out_image:
            in_image = _resize(in_image, self.image_size)
            out_image = _resize(out_image, self.image_size)

            if self.hflip and random.random() < 0.5:
                in_image = in_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                out_image = out_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            in_tensor = normalize_01_to_m11(_pil_to_tensor(in_image))
            out_tensor = normalize_01_to_m11(_pil_to_tensor(out_image))

        return {
            "A": in_tensor,
            "B": out_tensor,
            "id": sample_id,
            "A_path": str(in_path),
            "B_path": str(out_path),
        }


class UnpairedImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        domain_a: str,
        domain_b: str,
        image_size: int = 256,
        hflip: bool = True,
        random_pair: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.domain_a = domain_a
        self.domain_b = domain_b
        self.image_size = image_size
        self.hflip = hflip
        self.random_pair = random_pair

        self.files_a = list_image_files(self.root / split / domain_a)
        self.files_b = list_image_files(self.root / split / domain_b)
        if not self.files_a or not self.files_b:
            raise RuntimeError(
                f"Missing images for unpaired training in {self.root / split / domain_a} "
                f"or {self.root / split / domain_b}"
            )

    def __len__(self) -> int:
        return max(len(self.files_a), len(self.files_b))

    def _load_one(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = _resize(image, self.image_size)
            if self.hflip and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return normalize_01_to_m11(_pil_to_tensor(image))

    def __getitem__(self, index: int) -> dict[str, Any]:
        path_a = self.files_a[index % len(self.files_a)]
        if self.random_pair:
            path_b = self.files_b[random.randrange(len(self.files_b))]
        else:
            path_b = self.files_b[index % len(self.files_b)]

        return {
            "A": self._load_one(path_a),
            "B": self._load_one(path_b),
            "id_A": path_a.stem,
            "id_B": path_b.stem,
            "A_path": str(path_a),
            "B_path": str(path_b),
        }


class SingleDomainDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        directory: str | Path,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.image_size = image_size
        self.files = list_image_files(self.directory)
        if not self.files:
            raise RuntimeError(f"No images found in {self.directory}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.files[index]
        with Image.open(image_path) as image:
            image = _resize(image, self.image_size)
            image_tensor = normalize_01_to_m11(_pil_to_tensor(image))
        return {"image": image_tensor, "id": image_path.stem, "path": str(image_path)}

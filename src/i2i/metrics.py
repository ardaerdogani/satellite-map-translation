from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _list_images(directory: Path) -> dict[str, Path]:
    files = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        files[path.stem] = path
    return files


def _load_rgb_01(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def compute_paired_psnr_ssim(
    pred_dir: str | Path,
    target_dir: str | Path,
    max_items: int | None = None,
) -> dict[str, Any]:
    pred_root = Path(pred_dir)
    target_root = Path(target_dir)
    pred_files = _list_images(pred_root)
    target_files = _list_images(target_root)

    common = sorted(set(pred_files.keys()) & set(target_files.keys()))
    if not common:
        raise RuntimeError(f"No overlapping filenames between {pred_root} and {target_root}.")
    if max_items is not None:
        common = common[:max_items]

    psnr_scores: list[float] = []
    ssim_scores: list[float] = []
    for sample_id in common:
        pred = _load_rgb_01(pred_files[sample_id])
        target = _load_rgb_01(target_files[sample_id])

        if pred.shape != target.shape:
            raise RuntimeError(
                f"Shape mismatch for {sample_id}: pred={pred.shape}, target={target.shape}. "
                "Use same preprocessing resolution for generated and target images."
            )

        psnr_scores.append(float(peak_signal_noise_ratio(target, pred, data_range=1.0)))
        ssim_scores.append(
            float(
                structural_similarity(
                    target,
                    pred,
                    channel_axis=2,
                    data_range=1.0,
                )
            )
        )

    return {
        "num_pairs": len(common),
        "psnr": float(np.mean(psnr_scores)),
        "ssim": float(np.mean(ssim_scores)),
    }


def compute_fid(
    real_dir: str | Path,
    fake_dir: str | Path,
    batch_size: int = 16,
    device: str = "cpu",
    max_items: int | None = None,
) -> float:
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    real_root = Path(real_dir)
    fake_root = Path(fake_dir)
    real_files = _list_images(real_root)
    fake_files = _list_images(fake_root)
    common = sorted(set(real_files.keys()) & set(fake_files.keys()))
    if not common:
        raise RuntimeError(f"No overlapping filenames between {real_root} and {fake_root}.")
    if max_items is not None:
        common = common[:max_items]

    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def load_batch(paths: list[Path]) -> torch.Tensor:
        batch = []
        for path in paths:
            array = _load_rgb_01(path)
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            batch.append(tensor)
        return torch.stack(batch, dim=0).to(device)

    for start in range(0, len(common), batch_size):
        ids = common[start : start + batch_size]
        real_batch = load_batch([real_files[sample_id] for sample_id in ids])
        fake_batch = load_batch([fake_files[sample_id] for sample_id in ids])
        metric.update(real_batch, real=True)
        metric.update(fake_batch, real=False)

    return float(metric.compute().item())


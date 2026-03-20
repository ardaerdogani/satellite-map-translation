from __future__ import annotations

import gc
import json
import random
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _cuda_mem_get_info(index: int) -> tuple[int, int]:
    with torch.cuda.device(index):
        return torch.cuda.mem_get_info()


def best_available_cuda_device(exclude_index: int | None = None) -> torch.device:
    """Pick the visible CUDA device with the most free memory."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    candidates: list[tuple[int, int, int]] = []
    for index in range(torch.cuda.device_count()):
        if exclude_index is not None and index == exclude_index:
            continue
        try:
            free_bytes, total_bytes = _cuda_mem_get_info(index)
        except Exception:
            props = torch.cuda.get_device_properties(index)
            total_bytes = int(props.total_memory)
            reserved_bytes = int(torch.cuda.memory_reserved(index))
            free_bytes = max(0, total_bytes - reserved_bytes)
        candidates.append((free_bytes, total_bytes, index))

    if not candidates:
        raise RuntimeError("No CUDA devices are available after applying exclusions.")

    _, _, best_index = max(candidates)
    return torch.device(f"cuda:{best_index}")


def resolve_device(device_arg: str) -> torch.device:
    """Resolve runtime device with Apple Silicon aware auto mode."""
    normalized = device_arg.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return best_available_cuda_device()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps but MPS is not available in this PyTorch build.")
        return torch.device("mps")

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return best_available_cuda_device()

    if normalized == "cpu":
        return torch.device("cpu")

    # Allow explicit forms like cuda:0
    return torch.device(device_arg)


def configure_runtime_for_device(device: torch.device) -> None:
    """Runtime tweaks for faster/stabler execution per backend."""
    # Helps matmul kernels where supported.
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if device.type == "mps":
        # Fallback unsupported ops to CPU instead of hard failure.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def autocast_context(
    device: torch.device,
    *,
    enabled: bool,
    dtype: torch.dtype = torch.float16,
):
    """Return an autocast context when CUDA AMP is enabled, otherwise a no-op context."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and "cuda" in message and "out of memory" in message


def move_modules_to_device(
    modules: dict[str, torch.nn.Module],
    device: torch.device,
    *,
    allow_cpu_fallback: bool = False,
) -> tuple[dict[str, torch.nn.Module], torch.device, bool]:
    """Move one or more modules to a device, optionally falling back to CPU on CUDA OOM."""
    try:
        return {name: module.to(device) for name, module in modules.items()}, device, False
    except RuntimeError as exc:
        if device.type != "cuda" or not _is_cuda_oom_error(exc):
            raise

        gc.collect()
        torch.cuda.empty_cache()

        current_index = device.index if device.index is not None else torch.cuda.current_device()
        if torch.cuda.device_count() > 1:
            try:
                retry_device = best_available_cuda_device(exclude_index=current_index)
                normalized_modules = {name: module.to("cpu") for name, module in modules.items()}
                return (
                    {name: module.to(retry_device) for name, module in normalized_modules.items()},
                    retry_device,
                    False,
                )
            except RuntimeError as retry_exc:
                if not _is_cuda_oom_error(retry_exc):
                    raise

        if not allow_cpu_fallback:
            raise RuntimeError(
                f"CUDA ran out of memory while moving models to {device}. "
                "The notebook kernel is likely still holding tensors from an earlier run. "
                "Free that GPU, pick a different one with DEVICE='cuda:N', or set DEVICE='auto'/'cpu' and try again. "
                f"Original error: {exc}"
            ) from exc

        moved = {name: module.to("cpu") for name, module in modules.items()}
        return moved, torch.device("cpu"), True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path_obj)


def denormalize_m11_to_01(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0


def tensor_image_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = denormalize_m11_to_01(tensor)
    array = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def save_triplet_batch(
    real_a: torch.Tensor,
    fake_b: torch.Tensor,
    real_b: torch.Tensor,
    output_path: str | Path,
    max_items: int = 4,
) -> None:
    items = min(max_items, real_a.size(0))
    strips: list[Image.Image] = []

    for idx in range(items):
        img_a = tensor_image_to_pil(real_a[idx])
        img_fake = tensor_image_to_pil(fake_b[idx])
        img_b = tensor_image_to_pil(real_b[idx])

        strip = Image.new("RGB", (img_a.width * 3, img_a.height))
        strip.paste(img_a, (0, 0))
        strip.paste(img_fake, (img_a.width, 0))
        strip.paste(img_b, (img_a.width * 2, 0))
        strips.append(strip)

    canvas = Image.new("RGB", (strips[0].width, strips[0].height * len(strips)))
    for idx, strip in enumerate(strips):
        canvas.paste(strip, (0, idx * strip.height))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def save_cyclegan_batch(
    real_a: torch.Tensor,
    fake_b: torch.Tensor,
    rec_a: torch.Tensor,
    real_b: torch.Tensor,
    fake_a: torch.Tensor,
    rec_b: torch.Tensor,
    output_path: str | Path,
    max_items: int = 4,
) -> None:
    items = min(max_items, real_a.size(0))
    rows: list[Image.Image] = []

    for idx in range(items):
        imgs = [
            tensor_image_to_pil(real_a[idx]),
            tensor_image_to_pil(fake_b[idx]),
            tensor_image_to_pil(rec_a[idx]),
            tensor_image_to_pil(real_b[idx]),
            tensor_image_to_pil(fake_a[idx]),
            tensor_image_to_pil(rec_b[idx]),
        ]
        width = imgs[0].width
        height = imgs[0].height
        row = Image.new("RGB", (width * len(imgs), height))
        for col, image in enumerate(imgs):
            row.paste(image, (col * width, 0))
        rows.append(row)

    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)))
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * row.height))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def compute_psnr_batch(fake: torch.Tensor, real: torch.Tensor) -> float:
    fake_01 = denormalize_m11_to_01(fake)
    real_01 = denormalize_m11_to_01(real)
    mse = torch.mean((fake_01 - real_01) ** 2, dim=(1, 2, 3)).clamp(min=1e-12)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.mean().item())


class ReplayBuffer:
    """CycleGAN image buffer."""

    def __init__(self, max_size: int = 50) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive.")
        self.max_size = max_size
        self.buffer: list[torch.Tensor] = []

    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        output: list[torch.Tensor] = []
        for element in data.detach():
            element = element.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(element.clone())
                output.append(element)
            elif random.random() < 0.5:
                idx = random.randrange(len(self.buffer))
                output.append(self.buffer[idx].clone())
                self.buffer[idx] = element.clone()
            else:
                output.append(element)
        return torch.cat(output, dim=0)


def linear_decay_multiplier(epoch: int, total_epochs: int, decay_start_epoch: int) -> float:
    if total_epochs <= decay_start_epoch:
        return 1.0
    if epoch < decay_start_epoch:
        return 1.0
    progress = (epoch - decay_start_epoch) / float(total_epochs - decay_start_epoch)
    return max(0.0, 1.0 - progress)

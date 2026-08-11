from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from adetailer.common import ensure_pil_image

# name -> (checkpoint file, hydra config inside the sam2 package, download url)
# Config names follow sam2.HF_MODEL_ID_TO_FILENAMES so build_sam2 can resolve them.
SAM_MODELS: dict[str, tuple[str, str, str]] = {
    "sam2_hiera_tiny.pt": (
        "sam2_hiera_tiny.pt",
        "configs/sam2/sam2_hiera_t.yaml",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    ),
    "sam2_hiera_small.pt": (
        "sam2_hiera_small.pt",
        "configs/sam2/sam2_hiera_s.yaml",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    ),
    "sam2_hiera_base_plus.pt": (
        "sam2_hiera_base_plus.pt",
        "configs/sam2/sam2_hiera_b+.yaml",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    ),
    "sam2_hiera_large.pt": (
        "sam2_hiera_large.pt",
        "configs/sam2/sam2_hiera_l.yaml",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    ),
    "sam2.1_hiera_tiny.pt": (
        "sam2.1_hiera_tiny.pt",
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    ),
    "sam2.1_hiera_small.pt": (
        "sam2.1_hiera_small.pt",
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    ),
    "sam2.1_hiera_base_plus.pt": (
        "sam2.1_hiera_base_plus.pt",
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    ),
    "sam2.1_hiera_large.pt": (
        "sam2.1_hiera_large.pt",
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    ),
}

_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()


def sam_available() -> bool:
    """True if the sam2 package is importable (installed by install.py)."""
    try:
        import sam2  # noqa: F401

        return True
    except ImportError:
        return False


def list_sam_models() -> list[str]:
    return list(SAM_MODELS.keys())


def _safe_mkdir(path: str | Path) -> None:
    path = Path(path)
    if not path.exists() and path.parent.exists():
        path.mkdir()


def _download(url: str, target: Path) -> None:
    import urllib.request

    tmp = target.with_suffix(target.suffix + ".part")
    _safe_mkdir(target.parent)
    expected: int | None = None
    with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as f:  # noqa: S310
        content_length = resp.headers.get("Content-Length")
        if content_length:
            expected = int(content_length)
        while chunk := resp.read(1024 * 256):
            f.write(chunk)
    if expected is not None and tmp.stat().st_size != expected:
        tmp.unlink(missing_ok=True)
        raise OSError(
            f"incomplete download for {target.name}: {tmp.stat().st_size} "
            f"bytes, expected {expected}"
        )
    tmp.rename(target)


def _build_sam_model(name: str, models_dir: str | Path, device: str) -> Any:
    import torch
    from sam2.build_sam import build_sam2

    checkpoint, config, url = SAM_MODELS[name]
    path = Path(models_dir) / checkpoint
    if not path.exists():
        _safe_mkdir(models_dir)
        print(f"[-] ADetailer: downloading SAM2 model {checkpoint} ...")
        _download(url, path)
        print(f"[-] ADetailer: SAM2 model {checkpoint} downloaded.")

    model = build_sam2(config_file=config, ckpt_path=str(path), device=device)
    model.eval()
    with torch.no_grad():
        # Warm-up on a tiny dummy input so the first real prediction is fast.
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        try:
            model.image_encoder(dummy)
        except Exception:
            pass
    return model


def _valid_boxes(
    bboxes: list[list[float]], width: int, height: int
) -> list[tuple[int, list[float]]]:
    """Return (original_index, clipped_box) for every usable box."""
    out: list[tuple[int, list[float]]] = []
    for i, bbox in enumerate(bboxes):
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        x1 = max(0.0, min(float(x1), width))
        y1 = max(0.0, min(float(y1), height))
        x2 = max(0.0, min(float(x2), width))
        y2 = max(0.0, min(float(y2), height))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((i, [x1, y1, x2, y2]))
    return out


def _align_masks(
    box_count: int,
    sam_masks: dict[int, Image.Image],
    fallback_masks: list[Image.Image],
    size: tuple[int, int],
) -> list[Image.Image]:
    """Emit exactly one mask per input bbox, falling back where SAM failed."""
    out: list[Image.Image] = []
    for i in range(box_count):
        if i in sam_masks:
            out.append(sam_masks[i])
        elif i < len(fallback_masks):
            out.append(fallback_masks[i])
        else:
            out.append(Image.new("L", size, 0))
    return out


def refine(
    name: str,
    image: Image.Image,
    bboxes: list[list[float]],
    fallback_masks: list[Image.Image],
    *,
    models_dir: str | Path,
    device: str,
    offload_device: str,
    keep_loaded: bool,
) -> list[Image.Image]:
    """
    Refine detection masks into precise SAM2 masks using each bbox as a box prompt.

    Returns the fallback masks unchanged when SAM is unavailable or no valid
    boxes are given. Empty/failed SAM masks fall back per-box to the detection
    mask, and every input bbox gets exactly one output mask, so the classic
    YOLO-only pipeline keeps working whenever SAM is not selected.
    """
    if not bboxes or not sam_available():
        return fallback_masks
    if name not in SAM_MODELS:
        msg = f"[-] ADetailer: Unknown SAM2 model {name!r}. Available: {list(SAM_MODELS)}"
        raise ValueError(msg)

    with _LOCK:
        model = _CACHE.get(name)
        if model is not None:
            if str(model.device) != str(device):
                model.to(device)  # wake up from a previous offload
        else:
            # Bound VRAM: keep only the active model resident; park the others
            # on CPU so switching models does not accumulate several in VRAM.
            for cached_name, cached in list(_CACHE.items()):
                if cached_name != name:
                    _offload(cached, "cpu")
            model = _build_sam_model(name, models_dir, device)
            _CACHE[name] = model

        try:
            masks = _predict(model, image, bboxes, fallback_masks)
        finally:
            if not keep_loaded:
                _offload(model, offload_device)
        return masks


def _predict(
    model: Any,
    image: Image.Image,
    bboxes: list[list[float]],
    fallback_masks: list[Image.Image],
) -> list[Image.Image]:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = ensure_pil_image(image, "RGB")
    width, height = image.size

    valid = _valid_boxes(bboxes, width, height)
    if not valid:
        return fallback_masks

    indices = [i for i, _ in valid]
    boxes = [box for _, box in valid]

    # set_image accepts numpy/PIL, moves the image to the model device itself,
    # and postprocesses masks back to the original image resolution.
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(np.array(image))

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(boxes, dtype=np.float32),
        multimask_output=False,
    )

    # predict returns (N, 1, H, W), or (1, H, W) for a single box.
    if masks.ndim == 4:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None, ...]
    if len(masks) != len(boxes):
        return fallback_masks

    sam_masks: dict[int, Image.Image] = {}
    for idx, mask in zip(indices, masks):
        binary = np.asarray(mask) > 0
        if binary.ndim == 3:
            binary = binary[0]
        if binary.any():
            sam_masks[idx] = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")

    return _align_masks(len(bboxes), sam_masks, fallback_masks, image.size)


def _offload(model: Any, offload_device: str) -> None:
    """Move a cached model to the offload device and free VRAM."""
    import torch

    device = getattr(model, "device", None)
    if device is None or str(device) == str(offload_device):
        return
    try:
        model.to(offload_device)
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

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
_LOCK = threading.Lock()  # serializes predict+offload on the shared cached model
_MAX_CACHED = 2  # bound CPU RAM: at most this many models kept around
_NAME_LOCKS: dict[str, threading.Lock] = {}
_NAME_LOCKS_GUARD = threading.Lock()

# architecture token -> hydra config inside the sam2 package; used to resolve
# the right config for custom checkpoints dropped into models/sam2
_SAM2_CONFIGS: dict[str, str] = {
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2_hiera_tiny": "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_small": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_base_plus": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_large": "configs/sam2/sam2_hiera_l.yaml",
}

_SAM2_DEFAULT_CONFIG = "configs/sam2/sam2_hiera_l.yaml"


def sam_available() -> bool:
    """True if the sam2 package is importable (installed by install.py)."""
    try:
        import sam2  # noqa: F401

        return True
    except ImportError:
        return False


def list_sam_models(models_dir: str | Path | None = None) -> list[str]:
    """Built-in model names plus custom .pt/.pth files found in models_dir."""
    names = list(SAM_MODELS.keys())
    if models_dir is not None:
        d = Path(models_dir)
        if d.is_dir():
            for p in sorted(d.glob("*.pt")) + sorted(d.glob("*.pth")):
                if p.name not in names:
                    names.append(p.name)
    return names


def _resolve_config(model_name: str) -> tuple[str, bool]:
    """Pick the sam2 config matching an architecture token in the filename.

    Returns (config, known). A custom checkpoint named after its base
    architecture (e.g. sam2_hiera_large_finetuned.pt) gets the matching
    config; anything else falls back to sam2_hiera_large with a warning.
    """
    for arch, cfg in _SAM2_CONFIGS.items():
        if arch in model_name:
            return cfg, True
    return _SAM2_DEFAULT_CONFIG, False


def _sanitize_name(name: str) -> str:
    """Reject anything that is not a plain filename (no path traversal)."""
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or ".." in name
        or "/" in name
        or "\\" in name
    ):
        msg = f"[-] ADetailer: invalid SAM2 model name {name!r}"
        raise ValueError(msg)
    return name


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
    size = tmp.stat().st_size
    if expected is not None and size != expected:
        tmp.unlink(missing_ok=True)
        raise OSError(
            f"incomplete download for {target.name}: {size} bytes, expected {expected}"
        )
    tmp.rename(target)


def _build_sam_model(name: str, models_dir: str | Path, device: str) -> Any:
    import torch
    from sam2.build_sam import build_sam2

    if name in SAM_MODELS:
        checkpoint, config, url = SAM_MODELS[name]
        path = Path(models_dir) / checkpoint
        if not path.exists():
            _safe_mkdir(models_dir)
            print(f"[-] ADetailer: downloading SAM2 model {checkpoint} ...")
            _download(url, path)
            print(f"[-] ADetailer: SAM2 model {checkpoint} downloaded.")
    else:
        # custom checkpoint dropped into models/sam2
        path = Path(models_dir) / name
        if not path.exists():
            msg = (
                f"[-] ADetailer: SAM2 model file not found: {path}. "
                "Put a .pt/.pth checkpoint in models/sam2, or pick a built-in model."
            )
            raise ValueError(msg)
        config, known = _resolve_config(name)
        if not known:
            print(
                f"[-] ADetailer: cannot guess the architecture of {name!r}; "
                f"using sam2_hiera_large. Name it like "
                "sam2_hiera_large_<yours>.pt or sam2.1_hiera_tiny_<yours>.pt "
                "to select the architecture automatically."
            )

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
    bboxes: list[list[float]],
    width: int,
    height: int,
    expansion: int = 0,
) -> list[tuple[int, list[float]]]:
    """Return (original_index, clipped_box) for every usable box.

    Boxes are clipped to the image and then expanded by `expansion` pixels
    (also clipped), giving SAM a bit more context around each detection.
    """
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
        if expansion > 0:
            x1 = max(0.0, x1 - expansion)
            y1 = max(0.0, y1 - expansion)
            x2 = min(float(width), x2 + expansion)
            y2 = min(float(height), y2 + expansion)
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


def _postprocess_mask(
    binary: np.ndarray, dilation: int, feather: int
) -> Image.Image:
    """Apply output threshold -> dilation/erosion -> optional feather."""
    if dilation:
        import cv2

        kernel = np.ones((2 * abs(dilation) + 1, 2 * abs(dilation) + 1), np.uint8)
        op = cv2.dilate if dilation > 0 else cv2.erode
        binary = op(binary.astype(np.uint8), kernel, iterations=1).astype(bool)

    img = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    if feather:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return img


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
    bbox_expansion: int = 0,
    mask_hint: bool = False,
    mask_hint_threshold: float = 0.5,
    mask_hint_use_negative: bool = False,
    threshold: float = 0.0,
    dilation: int = 0,
    feather: int = 0,
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
    name = _sanitize_name(name)

    # Look up without the lock (fast path); build under a per-name lock so a
    # slow download only blocks refinement of the same model, not everything.
    model = _CACHE.get(name)
    if model is None:
        with _name_lock(name):
            model = _CACHE.get(name)
            if model is None:
                # Park other cached models on CPU and enforce the cap while
                # holding the global lock: predict() also runs under it, so no
                # cached model's tensors can move mid-prediction. The download
                # itself stays outside _LOCK (only the per-name lock is held).
                with _LOCK:
                    for cached_name, cached in list(_CACHE.items()):
                        if cached_name != name:
                            _offload(cached, "cpu")
                    _evict_lru()
                model = _build_sam_model(name, models_dir, device)
                _CACHE[name] = model

    # Predict + offload under the global lock: webui jobs are sequential, so
    # this only guards against concurrent txt2img/img2img interleaving.
    with _LOCK:
        if str(model.device) != str(device):
            model.to(device)  # wake up from a previous offload
        # Refresh LRU recency so a model used right now is never the next
        # eviction victim (move-to-end on access = true LRU).
        if _CACHE.get(name) is model:
            _CACHE.pop(name, None)
            _CACHE[name] = model
        try:
            masks = _predict(
                model,
                image,
                bboxes,
                fallback_masks,
                bbox_expansion=bbox_expansion,
                mask_hint=mask_hint,
                mask_hint_threshold=mask_hint_threshold,
                mask_hint_use_negative=mask_hint_use_negative,
                threshold=threshold,
                dilation=dilation,
                feather=feather,
            )
        finally:
            if not keep_loaded:
                _offload(model, offload_device)
                _CACHE.pop(name, None)  # free the CPU copy too
        return masks


def _name_lock(name: str) -> threading.Lock:
    with _NAME_LOCKS_GUARD:
        lock = _NAME_LOCKS.get(name)
        if lock is None:
            lock = _NAME_LOCKS[name] = threading.Lock()
        return lock


def _evict_lru() -> None:
    """Drop the least-recently-used cached model to bound CPU RAM."""
    while len(_CACHE) > _MAX_CACHED:
        oldest = next(iter(_CACHE))
        _offload(_CACHE.pop(oldest), "cpu")


def _predict(
    model: Any,
    image: Image.Image,
    bboxes: list[list[float]],
    fallback_masks: list[Image.Image],
    *,
    bbox_expansion: int,
    mask_hint: bool,
    mask_hint_threshold: float,
    mask_hint_use_negative: bool,
    threshold: float,
    dilation: int,
    feather: int,
) -> list[Image.Image]:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = ensure_pil_image(image, "RGB")
    width, height = image.size

    valid = _valid_boxes(bboxes, width, height, expansion=bbox_expansion)
    if not valid:
        # Keep the one-mask-per-bbox contract even when every box is unusable.
        return _align_masks(len(bboxes), {}, fallback_masks, image.size)

    indices = [i for i, _ in valid]
    boxes = [box for _, box in valid]

    # set_image accepts numpy/PIL, moves the image to the model device itself,
    # and postprocesses masks back to the original image resolution.
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(np.array(image))
    # Mask hints are expected at the prompt encoder's mask input resolution
    # (typically 256x256), which it downsamples to image_embedding_size.
    mask_input_size = tuple(model.sam_prompt_encoder.mask_input_size)

    point_coords = None
    point_labels = None
    mask_input = None

    if mask_hint:
        # Binarize each detector mask as a hint: at original size for point
        # sampling, at the predictor's input size for mask_input.
        hints_orig: dict[int, np.ndarray] = {}
        hints_input: list[np.ndarray] = []
        for idx in indices:
            fallback = fallback_masks[idx] if idx < len(fallback_masks) else None
            if fallback is None:
                # No detector mask for this box: an all-zero dense mask prompt
                # is the neutral placeholder (an empty mask, not a "no hint"
                # signal) so the mask_input batch stays uniform. The box prompt
                # dominates the prediction for this box anyway.
                hints_input.append(
                    np.zeros((1, *mask_input_size), np.float32)
                )
                continue
            arr = np.array(ensure_pil_image(fallback, "L"))
            hint = arr > (mask_hint_threshold * 255)
            hints_orig[idx] = hint
            resized = np.array(
                Image.fromarray(hint.astype(np.uint8) * 255).resize(
                    mask_input_size, Image.BILINEAR
                )
            ) > 127
            hints_input.append(resized.astype(np.float32)[None, ...])

        mask_input = np.stack(hints_input, axis=0)  # (N, 1, H, W) at input size

        if mask_hint_use_negative:
            # SAM2 expects one point row per box (B, N, 2); build exactly two
            # points per box: a positive (hint centroid, or box center when no
            # hint) and a negative (background sample, or a harmless duplicate
            # positive when the box is fully covered by the hint).
            per_box_coords: list[list[list[float]]] = []
            per_box_labels: list[list[int]] = []
            for idx, box in zip(indices, boxes):
                hint = hints_orig.get(idx)
                if hint is None or not hint.any():
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    pos: tuple[float, float] = (cx, cy)
                else:
                    ys, xs = np.where(hint)
                    pos = (float(xs.mean()), float(ys.mean()))
                neg = (
                    _find_negative_point(hint, box, width, height)
                    if hint is not None and hint.any()
                    else None
                )
                box_coords: list[list[float]] = [list(pos)]
                box_labels: list[int] = [1]
                if neg is not None:
                    box_coords.append(neg)
                    box_labels.append(0)
                else:
                    box_coords.append(list(pos))  # no-op duplicate
                    box_labels.append(1)
                per_box_coords.append(box_coords)
                per_box_labels.append(box_labels)
            point_coords = np.array(per_box_coords, dtype=np.float32)
            point_labels = np.array(per_box_labels, dtype=np.int32)

    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=np.array(boxes, dtype=np.float32),
        mask_input=mask_input,
        multimask_output=False,
    )

    # predict returns (N, H, W) after the library's internal squeeze; keep a
    # defensive 4D branch in case a future version stops squeezing.
    if masks.ndim == 4:
        masks = masks[:, 0]
    if len(masks) != len(boxes):
        return _align_masks(len(bboxes), {}, fallback_masks, image.size)

    sam_masks: dict[int, Image.Image] = {}
    for idx, mask in zip(indices, masks):
        binary = np.asarray(mask) > threshold
        if binary.ndim == 3:
            binary = binary[0]
        if binary.any():
            sam_masks[idx] = _postprocess_mask(binary, dilation, feather)

    return _align_masks(len(bboxes), sam_masks, fallback_masks, image.size)


def _find_negative_point(
    hint: np.ndarray, box: list[float], width: int, height: int
) -> list[float] | None:
    """Pick a background point inside the box, far from the hint mask."""
    x1, y1, x2, y2 = (int(v) for v in box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    candidates = [
        (cx, cy),
        (x1 + (x2 - x1) // 4, y1 + (y2 - y1) // 4),
        (x2 - (x2 - x1) // 4, y1 + (y2 - y1) // 4),
        (x1 + (x2 - x1) // 4, y2 - (y2 - y1) // 4),
        (x2 - (x2 - x1) // 4, y2 - (y2 - y1) // 4),
    ]
    for px, py in candidates:
        if 0 <= py < height and 0 <= px < width and not hint[py, px]:
            return [float(px), float(py)]
    return None


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

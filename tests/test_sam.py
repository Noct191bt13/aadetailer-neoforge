from __future__ import annotations

from PIL import Image

from adetailer.sam import _align_masks, _valid_boxes, refine, sam_available


def _mask(size, value):
    return Image.new("L", size, value)


def test_refine_empty_bboxes_returns_fallback():
    img = Image.new("RGB", (64, 64))
    fallback = [_mask(img.size, 255)]
    result = refine(
        "sam2_hiera_tiny.pt",
        image=img,
        bboxes=[],
        fallback_masks=fallback,
        models_dir=".",
        device="cpu",
        offload_device="cpu",
        keep_loaded=True,
    )
    assert result is fallback


def test_valid_boxes_filters_and_clips():
    boxes = [
        [10, 10, 50, 50],  # ok
        [60, 10, 30, 50],  # inverted -> dropped
        [-10, -10, 100, 100],  # clipped to bounds
        [0, 0, 5, 5],  # ok
        [1, 2, 3],  # wrong length -> dropped
    ]
    valid = _valid_boxes(boxes, width=64, height=64)
    assert [i for i, _ in valid] == [0, 2, 3]
    assert valid[1][1] == [0.0, 0.0, 64.0, 64.0]


def test_valid_boxes_drops_degenerate_after_clipping():
    # fully outside the image collapses to zero area after clipping
    valid = _valid_boxes([[100, 100, 200, 200]], width=64, height=64)
    assert valid == []


def test_align_masks_one_per_input_box():
    size = (64, 64)
    fb = [_mask(size, 10), _mask(size, 20), _mask(size, 30)]
    sam = {0: _mask(size, 255)}
    out = _align_masks(3, sam, fb, size)
    assert len(out) == 3
    assert out[0].getpixel((0, 0)) == 255  # SAM result
    assert out[1].getpixel((0, 0)) == 20  # fallback
    assert out[2].getpixel((0, 0)) == 30  # fallback


def test_align_masks_handles_missing_fallback():
    size = (64, 64)
    out = _align_masks(2, {}, [_mask(size, 5)], size)
    assert len(out) == 2
    assert out[1].getpixel((0, 0)) == 0


def test_refine_unknown_model_raises():
    import pytest

    pytest.importorskip("sam2")
    img = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError):
        refine(
            "nope.pt",
            image=img,
            bboxes=[[0, 0, 10, 10]],
            fallback_masks=[_mask(img.size, 255)],
            models_dir=".",
            device="cpu",
            offload_device="cpu",
            keep_loaded=True,
        )


def test_sam_available_flag():
    assert isinstance(sam_available(), bool)

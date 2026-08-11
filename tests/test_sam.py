from __future__ import annotations

from PIL import Image

from adetailer.sam import (
    _align_masks,
    _find_negative_point,
    _postprocess_mask,
    _sanitize_name,
    _valid_boxes,
    refine,
    sam_available,
)


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
    with pytest.raises(ValueError, match="not found"):
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


def test_list_sam_models_discovers_custom(tmp_path):
    from adetailer.sam import list_sam_models

    (tmp_path / "my_finetune.pt").touch()
    (tmp_path / "sam2.1_hiera_large_finetuned.pt").touch()
    (tmp_path / "notes.txt").touch()

    names = list_sam_models(tmp_path)
    for builtin in (
        "sam2_hiera_tiny.pt",
        "sam2_hiera_small.pt",
        "sam2_hiera_base_plus.pt",
        "sam2_hiera_large.pt",
        "sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_small.pt",
        "sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_large.pt",
    ):
        assert builtin in names
    assert "my_finetune.pt" in names
    assert "sam2.1_hiera_large_finetuned.pt" in names
    assert "notes.txt" not in names


def test_sanitize_name_rejects_traversal():
    import pytest

    ok = "sam2_hiera_tiny.pt"
    assert _sanitize_name(ok) == ok
    for bad in [
        "../evil.pt",
        "a/b.pt",
        "..",
        "",
        "C:\\evil.pt",
        r"\\server\evil.pt",
    ]:
        with pytest.raises(ValueError):
            _sanitize_name(bad)


def test_valid_boxes_expansion():
    # [10,10,50,50] expanded by 8 px, clipped to image
    valid = _valid_boxes([[10, 10, 50, 50]], width=64, height=64, expansion=8)
    assert valid[0][1] == [2.0, 2.0, 58.0, 58.0]
    # expansion past the edge clamps
    valid = _valid_boxes([[60, 60, 62, 62]], width=64, height=64, expansion=10)
    assert valid[0][1] == [50.0, 50.0, 64.0, 64.0]
    # degenerate box stays dropped
    assert _valid_boxes([[20, 20, 10, 10]], width=64, height=64, expansion=5) == []


def test_postprocess_mask_dilation_and_feather():
    import numpy as np

    mask = np.zeros((32, 32), bool)
    mask[15:17, 15:17] = True

    img = _postprocess_mask(mask, dilation=0, feather=0)
    assert img.mode == "L" and img.getpixel((16, 16)) == 255

    dilated = _postprocess_mask(mask, dilation=2, feather=0)
    # 5x5 kernel grows the 2x2 blob: corner (13,13) becomes white
    assert dilated.getpixel((13, 13)) == 255

    eroded = _postprocess_mask(mask, dilation=-1, feather=0)
    # 3x3 erosion on a 2x2 blob removes it entirely
    assert eroded.getpixel((16, 16)) == 0

    blob = np.zeros((32, 32), bool)
    blob[12:20, 12:20] = True
    feathered = _postprocess_mask(blob, dilation=0, feather=3)
    assert feathered.mode == "L"
    assert feathered.getpixel((16, 16)) > 100  # core stays bright
    assert feathered.getpixel((5, 5)) == 0  # far background untouched
    px = feathered.getpixel((22, 16))  # 2px past the edge: soft transition
    assert 0 < px < 255


def test_find_negative_point_prefers_background():
    import numpy as np

    hint = np.zeros((100, 100), bool)
    hint[40:60, 40:60] = True  # box center covered, 1/4-offsets background
    neg = _find_negative_point(hint, [20.0, 20.0, 80.0, 80.0], 100, 100)
    assert neg is not None
    px, py = int(neg[0]), int(neg[1])
    assert not hint[py, px]  # must be background

    # fully-covered box -> no negative point
    hint2 = np.ones((100, 100), bool)
    assert _find_negative_point(hint2, [20.0, 20.0, 80.0, 80.0], 100, 100) is None


def test_resolve_config_by_filename():
    from adetailer.sam import _resolve_config

    cfg, known = _resolve_config("sam2.1_hiera_tiny_custom.pt")
    assert known and cfg == "configs/sam2.1/sam2.1_hiera_t.yaml"
    cfg, known = _resolve_config("sam2_hiera_base_plus_finetuned.pt")
    assert known and cfg == "configs/sam2/sam2_hiera_b+.yaml"
    # sam2.1 token must win over the bare sam2 token
    cfg, known = _resolve_config("sam2.1_hiera_large.pt")
    assert known and cfg == "configs/sam2.1/sam2.1_hiera_l.yaml"
    cfg, known = _resolve_config("my_finetune.pt")
    assert not known and cfg == "configs/sam2/sam2_hiera_l.yaml"

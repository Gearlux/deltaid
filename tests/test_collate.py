"""Tests for :func:`deltaid.segmentation.segmentation_collate_fn` (typed-bag collate)."""

import numpy as np
import pytest
import torch
from sampleflux import Image, Label, Mask, Sample

from deltaid.segmentation import segmentation_collate_fn
from deltaid.segmenter import _batch_input, _batch_target


def _make_sample(value: int = 0, h: int = 8, w: int = 8) -> Sample:
    """A per-sample Sample shaped like a typed segmentation source: Image input + Mask target + aux."""
    return Sample(
        {
            "image": Image(np.full((3, h, w), float(value), dtype="float32"), layout="CHW"),
            "mask": Mask(np.full((h, w), value, dtype="int64")),
            "pack_id": Label(f"pack{value}"),
            "iq_file": Label(f"f{value}.iq"),
        },
        roles={"mask": "target", "pack_id": "aux", "iq_file": "aux"},
    )


def test_collate_stacks_inputs_and_targets() -> None:
    batch = [_make_sample(0), _make_sample(1), _make_sample(2)]
    result = segmentation_collate_fn(batch)
    assert isinstance(result, Sample)
    x = _batch_input(result)
    y = _batch_target(result)
    assert x.shape == (3, 3, 8, 8)
    assert y.shape == (3, 8, 8)
    assert y.dtype == torch.int64
    assert int(y[0, 0, 0]) == 0
    assert int(y[2, 0, 0]) == 2


def test_collate_preserves_string_metadata_as_list() -> None:
    """Strings can't be torch.stack'd — they MUST end up in a per-sample metadata list."""
    batch = [_make_sample(0), _make_sample(1)]
    result = segmentation_collate_fn(batch)
    metas = result["metadata"]
    assert isinstance(metas, list)
    assert len(metas) == 2
    assert metas[0]["pack_id"] == "pack0"
    assert metas[1]["iq_file"] == "f1.iq"


def test_collate_promotes_non_int64_targets() -> None:
    s = Sample(
        {
            "image": Image(np.zeros((3, 4, 4), dtype="float32"), layout="CHW"),
            "mask": Mask(np.zeros((4, 4), dtype="int32")),
        },
        roles={"mask": "target"},
    )
    result = segmentation_collate_fn([s])
    assert _batch_target(result).dtype == torch.int64


def test_collate_handles_empty_metadata() -> None:
    s = Sample(
        {
            "image": Image(np.zeros((3, 4, 4), dtype="float32"), layout="CHW"),
            "mask": Mask(np.zeros((4, 4), dtype="int64")),
        },
        roles={"mask": "target"},
    )
    result = segmentation_collate_fn([s])
    assert result["metadata"] == [{}]


def test_collate_input_is_image_target_is_mask() -> None:
    from sampleflux import item_data, primary

    result = segmentation_collate_fn([_make_sample(0) for _ in range(2)])
    _, image_item = primary(result, "input")
    _, mask_item = primary(result, "target")
    assert isinstance(image_item, Image) and image_item.layout == "CHW"
    assert isinstance(mask_item, Mask)
    # Array items cannot hold a live tensor — their payloads are ndarrays recovered at the model boundary.
    assert isinstance(item_data(image_item), np.ndarray)
    assert isinstance(item_data(mask_item), np.ndarray)


def test_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        segmentation_collate_fn([])


def test_collate_rejects_non_tensor_input() -> None:
    bad = Sample(
        {"image": Label("not a tensor"), "mask": Mask(np.zeros((4, 4), dtype="int64"))},
        roles={"mask": "target"},
    )
    with pytest.raises(TypeError, match="input"):
        segmentation_collate_fn([bad])


def test_collate_rejects_non_tensor_target() -> None:
    bad = Sample(
        {"image": Image(np.zeros((3, 4, 4), dtype="float32"), layout="CHW"), "mask": Label("not a mask")},
        roles={"mask": "target"},
    )
    with pytest.raises(TypeError, match="target"):
        segmentation_collate_fn([bad])

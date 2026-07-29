"""Tests for the segmenter's use of the generic :func:`recordstream.collate_records`.

Deltaid no longer ships a bespoke segmentation collate: the segmentation record (an
:class:`~recordstream.Image` input under ``"image"`` + a fully-stackable :class:`~recordstream.Mask`
target under ``"target"``) is collated by ``recordstream.collate_records`` and read back at the
model boundary by the generic ``recordstream.batch_tensor`` / ``batch_metadata``.

The generic read-back behaviour is pinned in recordstream's own ``tests/test_batch.py`` — what
remains here is the SEGMENTATION-specific part: ``deltaid.segmenter._batch_target``'s int64
promotion for the pixel-class-id CrossEntropy contract, and this record shape round-tripping.
"""

import numpy as np
import pytest
import torch
from recordstream import Image, Label, Mask, Record, batch_metadata, batch_tensor, collate_records, item_data

from deltaid.segmenter import _batch_target


def _make_record(value: int = 0, h: int = 8, w: int = 8) -> Record:
    """A per-record dict shaped like a segmentation source: Image + Mask target + plain meta keys."""
    return {
        "image": Image(np.full((3, h, w), float(value), dtype="float32"), layout="CHW"),
        "target": Mask(np.full((h, w), value, dtype="int64")),
        "pack_id": f"pack{value}",
        "iq_file": f"f{value}.iq",
    }


def test_collate_stacks_inputs_and_targets() -> None:
    batch = collate_records([_make_record(0), _make_record(1), _make_record(2)])
    assert isinstance(batch, dict)
    x = batch_tensor(batch, "image")
    y = _batch_target(batch, "target")
    assert x.shape == (3, 3, 8, 8)
    assert y.shape == (3, 8, 8)
    assert y.dtype == torch.int64
    assert int(y[0, 0, 0]) == 0
    assert int(y[2, 0, 0]) == 2


def test_collate_preserves_string_metadata_as_list() -> None:
    """Strings can't be torch.stack'd — they MUST survive as a per-record metadata list."""
    batch = collate_records([_make_record(0), _make_record(1)])
    metas = batch_metadata(batch, exclude=("image", "target"))
    assert isinstance(metas, list)
    assert len(metas) == 2
    assert metas[0]["pack_id"] == "pack0"
    assert metas[1]["iq_file"] == "f1.iq"


def test_batch_target_promotes_non_int64_masks() -> None:
    """The one thing the generic reader deliberately does NOT do — a dtype opinion is per-task."""
    record: Record = {
        "image": Image(np.zeros((3, 4, 4), dtype="float32"), layout="CHW"),
        "target": Mask(np.zeros((4, 4), dtype="int32")),
    }
    assert batch_tensor(collate_records([record]), "target").dtype == torch.int32  # generic: as-is
    assert _batch_target(collate_records([record]), "target").dtype == torch.int64  # segmenter: promoted


def test_batch_target_stacks_defensive_label_list() -> None:
    """A Label target (which collate_records leaves as a per-record list) is stacked at the boundary."""
    batch = collate_records(
        [{"image": Image(np.zeros((3, 4, 4), dtype="float32"), layout="CHW"), "target": Label(v)} for v in (0, 1, 2)]
    )
    assert isinstance(batch["target"].value, list)  # collate_records did NOT stack it
    y = _batch_target(batch, "target")
    assert y.shape == (3,) and y.dtype == torch.int64


def test_collate_input_is_image_target_is_mask() -> None:
    batch = collate_records([_make_record(0) for _ in range(2)])
    image_item = batch["image"]
    mask_item = batch["target"]
    # collate_records gathers each item attr (e.g. Image.layout) as a per-record list.
    assert isinstance(image_item, Image) and image_item.layout == ["CHW", "CHW"]
    assert isinstance(mask_item, Mask)
    # Array items cannot hold a live tensor — their payloads are ndarrays recovered at the model boundary.
    assert isinstance(item_data(image_item), np.ndarray)
    assert isinstance(item_data(mask_item), np.ndarray)


def test_collate_records_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_records([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

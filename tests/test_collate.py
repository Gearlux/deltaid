"""Tests for the segmenter's use of the generic :func:`sampleflux.collate_records`.

Deltaid no longer ships a bespoke segmentation collate: the segmentation record (an
:class:`~sampleflux.Image` input under ``"image"`` + a fully-stackable :class:`~sampleflux.Mask`
target under ``"target"``) is collated by ``sampleflux.collate_records`` and the model boundary
recovers the batched tensors via ``deltaid.segmenter._batch_input`` / ``_batch_target`` (and
per-record metadata via ``_batch_metadata``).
"""

import numpy as np
import pytest
import torch
from sampleflux import Image, Label, Mask, Record, collate_records, item_data

from deltaid.segmenter import _batch_input, _batch_metadata, _batch_target


def _make_sample(value: int = 0, h: int = 8, w: int = 8) -> Record:
    """A per-record dict shaped like a segmentation source: Image + Mask target + plain meta keys."""
    return {
        "image": Image(np.full((3, h, w), float(value), dtype="float32"), layout="CHW"),
        "target": Mask(np.full((h, w), value, dtype="int64")),
        "pack_id": f"pack{value}",
        "iq_file": f"f{value}.iq",
    }


def test_collate_stacks_inputs_and_targets() -> None:
    batch = collate_records([_make_sample(0), _make_sample(1), _make_sample(2)])
    assert isinstance(batch, dict)
    x = _batch_input(batch, "image")
    y = _batch_target(batch, "target")
    assert x.shape == (3, 3, 8, 8)
    assert y.shape == (3, 8, 8)
    assert y.dtype == torch.int64
    assert int(y[0, 0, 0]) == 0
    assert int(y[2, 0, 0]) == 2


def test_collate_preserves_string_metadata_as_list() -> None:
    """Strings can't be torch.stack'd — they MUST survive as a per-record metadata list."""
    batch = collate_records([_make_sample(0), _make_sample(1)])
    metas = _batch_metadata(batch, "image", "target")
    assert isinstance(metas, list)
    assert len(metas) == 2
    assert metas[0]["pack_id"] == "pack0"
    assert metas[1]["iq_file"] == "f1.iq"


def test_batch_target_promotes_non_int64_masks() -> None:
    record: Record = {
        "image": Image(np.zeros((3, 4, 4), dtype="float32"), layout="CHW"),
        "target": Mask(np.zeros((4, 4), dtype="int32")),
    }
    assert _batch_target(collate_records([record]), "target").dtype == torch.int64


def test_batch_target_stacks_defensive_label_list() -> None:
    """A Label target (which collate_records leaves as a per-record list) is stacked at the boundary."""
    batch = collate_records(
        [{"image": Image(np.zeros((3, 4, 4), dtype="float32"), layout="CHW"), "target": Label(v)} for v in (0, 1, 2)]
    )
    assert isinstance(batch["target"].value, list)  # collate_records did NOT stack it
    y = _batch_target(batch, "target")
    assert y.shape == (3,) and y.dtype == torch.int64


def test_batch_metadata_none_without_extra_keys() -> None:
    record: Record = {
        "image": Image(np.zeros((3, 4, 4), dtype="float32"), layout="CHW"),
        "target": Mask(np.zeros((4, 4), dtype="int64")),
    }
    assert _batch_metadata(collate_records([record]), "image", "target") is None


def test_batch_metadata_reads_legacy_metadata_field() -> None:
    """A hand-built batch carrying a single list-valued ``metadata`` key is returned verbatim."""
    batch: Record = {
        "image": Image(np.zeros((2, 3, 4, 4), dtype="float32"), layout="CHW"),
        "target": Mask(np.zeros((2, 4, 4), dtype="int64")),
        "metadata": [{"idx": 0}, {"idx": 1}],
    }
    assert _batch_metadata(batch, "image", "target") == [{"idx": 0}, {"idx": 1}]


def test_collate_input_is_image_target_is_mask() -> None:
    batch = collate_records([_make_sample(0) for _ in range(2)])
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

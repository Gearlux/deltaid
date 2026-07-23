"""Tests for the segmenter's use of the generic :func:`sampleflux.typed_collate`.

Deltaid no longer ships a bespoke segmentation collate: the segmentation ``Sample`` (an
:class:`~sampleflux.Image` input + a fully-stackable :class:`~sampleflux.Mask` target) is collated by
``sampleflux.typed_collate`` and the model boundary recovers the batched tensors via
``deltaid.segmenter._batch_input`` / ``_batch_target`` (and per-sample metadata via ``_batch_metadata``).
"""

import numpy as np
import pytest
import torch
from sampleflux import Image, Label, Mask, Sample, item_data, primary, typed_collate

from deltaid.segmenter import _batch_input, _batch_metadata, _batch_target


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
    batch = typed_collate([_make_sample(0), _make_sample(1), _make_sample(2)])
    assert isinstance(batch, Sample)
    x = _batch_input(batch)
    y = _batch_target(batch)
    assert x.shape == (3, 3, 8, 8)
    assert y.shape == (3, 8, 8)
    assert y.dtype == torch.int64
    assert int(y[0, 0, 0]) == 0
    assert int(y[2, 0, 0]) == 2


def test_collate_preserves_string_metadata_as_list() -> None:
    """Strings can't be torch.stack'd — they MUST survive as a per-sample metadata list."""
    batch = typed_collate([_make_sample(0), _make_sample(1)])
    metas = _batch_metadata(batch)
    assert isinstance(metas, list)
    assert len(metas) == 2
    assert metas[0]["pack_id"] == "pack0"
    assert metas[1]["iq_file"] == "f1.iq"


def test_batch_target_promotes_non_int64_masks() -> None:
    s = Sample(
        {
            "image": Image(np.zeros((3, 4, 4), dtype="float32"), layout="CHW"),
            "mask": Mask(np.zeros((4, 4), dtype="int32")),
        },
        roles={"mask": "target"},
    )
    assert _batch_target(typed_collate([s])).dtype == torch.int64


def test_batch_target_stacks_defensive_label_list() -> None:
    """A Label target (which typed_collate leaves as a per-sample list) is stacked at the boundary."""
    batch = typed_collate(
        [
            Sample({"image": Image(np.zeros((3, 4, 4), dtype="float32"), layout="CHW"), "y": Label(v)}, {"y": "target"})
            for v in (0, 1, 2)
        ]
    )
    assert isinstance(primary(batch, "target")[1].value, list)  # typed_collate did NOT stack it
    y = _batch_target(batch)
    assert y.shape == (3,) and y.dtype == torch.int64


def test_batch_metadata_none_without_aux_fields() -> None:
    s = Sample(
        {
            "image": Image(np.zeros((3, 4, 4), dtype="float32"), layout="CHW"),
            "mask": Mask(np.zeros((4, 4), dtype="int64")),
        },
        roles={"mask": "target"},
    )
    assert _batch_metadata(typed_collate([s])) is None


def test_batch_metadata_reads_legacy_metadata_field() -> None:
    """A hand-built batch carrying a single list-valued ``metadata`` field is returned verbatim."""
    batch = Sample(
        {
            "image": Image(np.zeros((2, 3, 4, 4), dtype="float32"), layout="CHW"),
            "mask": Mask(np.zeros((2, 4, 4), dtype="int64")),
            "metadata": [{"idx": 0}, {"idx": 1}],
        },
        roles={"mask": "target", "metadata": "aux"},
    )
    assert _batch_metadata(batch) == [{"idx": 0}, {"idx": 1}]


def test_collate_input_is_image_target_is_mask() -> None:
    batch = typed_collate([_make_sample(0) for _ in range(2)])
    _, image_item = primary(batch, "input")
    _, mask_item = primary(batch, "target")
    # typed_collate gathers each item attr (e.g. Image.layout) as a per-sample list.
    assert isinstance(image_item, Image) and image_item.layout == ["CHW", "CHW"]
    assert isinstance(mask_item, Mask)
    # Array items cannot hold a live tensor — their payloads are ndarrays recovered at the model boundary.
    assert isinstance(item_data(image_item), np.ndarray)
    assert isinstance(item_data(mask_item), np.ndarray)


def test_typed_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        typed_collate([])

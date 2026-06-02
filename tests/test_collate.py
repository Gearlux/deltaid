"""Tests for :func:`deltaid.segmentation.segmentation_collate_fn`."""

import pytest
import torch
from dataflux.sample import Sample

from deltaid.segmentation import segmentation_collate_fn


def _make_sample(value: int = 0, h: int = 8, w: int = 8) -> Sample:
    return Sample(
        input=torch.full((3, h, w), float(value)),
        target=torch.full((h, w), value, dtype=torch.int64),
        metadata={"pack_id": f"pack{value}", "iq_file": f"f{value}.iq"},
    )


def test_collate_stacks_inputs_and_targets() -> None:
    batch = [_make_sample(0), _make_sample(1), _make_sample(2)]
    result = segmentation_collate_fn(batch)
    assert result.input.shape == (3, 3, 8, 8)
    assert result.target.shape == (3, 8, 8)
    assert result.target.dtype == torch.int64
    assert int(result.target[0, 0, 0]) == 0
    assert int(result.target[2, 0, 0]) == 2


def test_collate_preserves_string_metadata_as_list() -> None:
    """Strings can't be torch.stack'd — they MUST end up in a per_sample list."""
    batch = [_make_sample(0), _make_sample(1)]
    result = segmentation_collate_fn(batch)
    metas = result.meta["per_sample"]
    assert isinstance(metas, list)
    assert len(metas) == 2
    assert metas[0]["pack_id"] == "pack0"
    assert metas[1]["iq_file"] == "f1.iq"


def test_collate_promotes_non_int64_targets() -> None:
    s = Sample(
        input=torch.zeros(3, 4, 4),
        target=torch.zeros(4, 4, dtype=torch.int32),
        metadata={},
    )
    result = segmentation_collate_fn([s])
    assert result.target.dtype == torch.int64


def test_collate_handles_empty_metadata_dict() -> None:
    s = Sample(
        input=torch.zeros(3, 4, 4),
        target=torch.zeros(4, 4, dtype=torch.int64),
        metadata={},
    )
    result = segmentation_collate_fn([s])
    assert result.meta["per_sample"] == [{}]


def test_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        segmentation_collate_fn([])


def test_collate_rejects_non_tensor_input() -> None:
    bad = Sample(
        input="not a tensor",  # type: ignore[arg-type]
        target=torch.zeros(4, 4, dtype=torch.int64),
        metadata={},
    )
    with pytest.raises(TypeError, match="sample.input"):
        segmentation_collate_fn([bad])


def test_collate_rejects_non_tensor_target() -> None:
    bad = Sample(input=torch.zeros(3, 4, 4), target=[[0, 1], [1, 0]], metadata={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sample.target"):
        segmentation_collate_fn([bad])

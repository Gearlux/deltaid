"""Tests for :mod:`deltaid.segmentation` (loss + collate)."""

import math
from typing import List

import pytest
import torch
from dataflux.sample import Sample

from deltaid.segmentation import SegmentationLoss, segmentation_collate_fn


def _make_logits_and_target(n: int = 2, c: int = 3, h: int = 16, w: int = 16) -> tuple:
    logits = torch.randn(n, c, h, w, requires_grad=True)
    target = torch.randint(0, c, (n, h, w), dtype=torch.int64)
    return logits, target


# --- SegmentationLoss ------------------------------------------------------


def test_segmentation_loss_returns_finite_scalar_with_grad() -> None:
    loss_fn = SegmentationLoss()
    logits, target = _make_logits_and_target()
    loss = loss_fn(logits, target)
    assert loss.dim() == 0
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_segmentation_loss_accepts_int32_target_and_promotes_to_long() -> None:
    """Targets coming through some sources may be int32; we promote silently."""
    loss_fn = SegmentationLoss()
    logits = torch.randn(1, 2, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.int32)
    target[0, :4, :4] = 1
    loss = loss_fn(logits, target)
    assert math.isfinite(float(loss))


def test_class_weights_change_loss_value() -> None:
    """Up-weighting the FG class on an all-FG target yields a strictly larger sum-loss.

    Note: ``CrossEntropyLoss(weight=w, reduction='mean')`` divides by the sum
    of weights for the present targets, so weighting an all-one-class batch
    produces the *same* mean as the unweighted case (numerator and denominator
    both scale). Use ``reduction='sum'`` to make the weight effect visible.
    """
    logits = torch.zeros(1, 2, 4, 4)
    target = torch.ones(1, 4, 4, dtype=torch.int64)

    base = SegmentationLoss(reduction="sum")(logits, target)
    weighted = SegmentationLoss(class_weights=[1.0, 5.0], reduction="sum")(logits, target)
    assert float(weighted) == pytest.approx(5.0 * float(base))


def test_ignore_index_excludes_pixels_from_loss() -> None:
    """A target full of ignore_index produces zero (or NaN-tolerant) loss."""
    loss_fn = SegmentationLoss(ignore_index=-1)
    logits = torch.zeros(1, 2, 4, 4)
    target = torch.full((1, 4, 4), -1, dtype=torch.int64)
    loss = loss_fn(logits, target)
    # All pixels ignored → CE returns NaN under "mean" reduction (0/0). Verify
    # the call doesn't crash; production use should set reduction="sum" or
    # mix in non-ignore pixels.
    assert torch.isnan(loss) or float(loss) == 0.0


def test_class_weights_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="class_weights"):
        SegmentationLoss(class_weights=[])


def test_segmentation_loss_rejects_wrong_logit_shape() -> None:
    loss_fn = SegmentationLoss()
    target = torch.zeros(1, 8, 8, dtype=torch.int64)
    bad_logits = torch.randn(1, 2, 8)  # 3D, missing W
    with pytest.raises(ValueError, match=r"\[N,C,H,W\]"):
        loss_fn(bad_logits, target)


def test_segmentation_loss_rejects_wrong_target_shape() -> None:
    loss_fn = SegmentationLoss()
    logits = torch.randn(1, 2, 8, 8)
    bad_target = torch.zeros(1, 2, 8, 8, dtype=torch.int64)  # 4D
    with pytest.raises(ValueError, match=r"\[N,H,W\]"):
        loss_fn(logits, bad_target)


def test_class_weights_buffer_moves_with_module() -> None:
    """After .to('cpu') the weight buffer is still on cpu and usable."""
    loss_fn = SegmentationLoss(class_weights=[1.0, 2.0]).to("cpu")
    logits = torch.zeros(1, 2, 4, 4)
    target = torch.ones(1, 4, 4, dtype=torch.int64)
    # No exception — the underlying CrossEntropyLoss sees the buffer.
    loss_fn(logits, target)


# --- segmentation_collate_fn -----------------------------------------------


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
    metas = result.metadata["per_sample"]
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
    s = Sample(input=torch.zeros(3, 4, 4), target=torch.zeros(4, 4, dtype=torch.int64), metadata={})
    result = segmentation_collate_fn([s])
    assert result.metadata["per_sample"] == [{}]


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


# --- end-to-end integration: collate → loss --------------------------------


def test_collate_then_loss_runs_end_to_end() -> None:
    """Sanity check: collate produces shapes the loss accepts."""
    batch: List[Sample] = [_make_sample(0), _make_sample(1)]
    collated = segmentation_collate_fn(batch)
    # Pretend the model produced random logits of the right shape.
    n, _, h, w = collated.input.shape
    logits = torch.randn(n, 3, h, w)  # 3-class output
    loss = SegmentationLoss()(logits, collated.target)
    assert math.isfinite(float(loss))

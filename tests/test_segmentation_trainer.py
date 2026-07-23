"""Tests for LightningSegmenter."""

import numpy as np
import torch
import torch.nn as nn
from sampleflux import Image, Label, Mask, TypedSample
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from deltaid.segmentation import segmentation_collate_fn
from deltaid.segmenter import LightningSegmenter, _default_segmentation_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinySegmenter(nn.Module):
    """1x1 conv from 3 input channels to ``num_classes`` — trivial pixel classifier."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # type: ignore[no-any-return]


def _make_samples(n: int = 8, num_classes: int = 2, h: int = 8, w: int = 8) -> list[TypedSample]:
    return [
        TypedSample(
            {
                "image": Image(np.random.randn(3, h, w).astype("float32"), layout="CHW"),
                "mask": Mask(np.random.randint(0, num_classes, (h, w)).astype("int64")),
                "idx": Label(i),
            },
            roles={"mask": "target", "idx": "aux"},
        )
        for i in range(n)
    ]


def _make_batch(n: int = 4, num_classes: int = 2, h: int = 8, w: int = 8) -> TypedSample:
    # A batched TypedSample shaped like segmentation_collate_fn output: the primary input Image holds the
    # (pre-batched) [N, 3, H, W] tensor as an ndarray (an Image can't hold a live tensor), the target Mask
    # holds the [N, H, W] int64 mask, and per-sample metadata rides a "metadata" field.
    x = torch.randn(n, 3, h, w)
    y = torch.randint(0, num_classes, (n, h, w), dtype=torch.int64)
    return TypedSample(
        {"image": Image(x.numpy(), layout="CHW"), "mask": Mask(y.numpy()), "metadata": [{"idx": i} for i in range(n)]},
        roles={"mask": "target", "metadata": "aux"},
    )


# ---------------------------------------------------------------------------
# _default_segmentation_metrics
# ---------------------------------------------------------------------------


def test_default_metrics_is_metric_collection_with_miou_and_dice() -> None:
    m = _default_segmentation_metrics(num_classes=3, ignore_index=-100)
    assert isinstance(m, MetricCollection)
    assert "miou" in m
    assert "dice_macro" in m
    assert isinstance(m["miou"], MulticlassJaccardIndex)
    assert isinstance(m["dice_macro"], MulticlassF1Score)


def test_default_metrics_propagate_ignore_index() -> None:
    m = _default_segmentation_metrics(num_classes=3, ignore_index=255)
    assert m["miou"].ignore_index == 255  # type: ignore[attr-defined]
    assert m["dice_macro"].ignore_index == 255  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# LightningSegmenter defaults
# ---------------------------------------------------------------------------


def test_loss_fn_is_torch_cross_entropy() -> None:
    """Default loss is torch.nn.CrossEntropyLoss — no deltaid adapter class."""
    trainer = LightningSegmenter(
        model=_TinySegmenter(num_classes=2),
        train_set=_make_samples(8),
        num_classes=2,
    )
    assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)
    assert trainer.loss_fn.ignore_index == -100


def test_ignore_index_flows_into_default_loss_and_metrics() -> None:
    trainer = LightningSegmenter(
        model=_TinySegmenter(num_classes=3),
        train_set=_make_samples(8, num_classes=3),
        num_classes=3,
        ignore_index=255,
    )
    assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)
    assert trainer.loss_fn.ignore_index == 255
    assert trainer.train_metrics is not None
    assert trainer.train_metrics["train/miou"].ignore_index == 255  # type: ignore[attr-defined]
    assert trainer.val_metrics is not None
    assert trainer.val_metrics["val/dice_macro"].ignore_index == 255  # type: ignore[attr-defined]


def test_metrics_are_wired_with_correct_prefixes() -> None:
    trainer = LightningSegmenter(
        model=_TinySegmenter(num_classes=2),
        train_set=_make_samples(8),
        num_classes=2,
    )
    assert trainer.train_metrics is not None
    assert trainer.val_metrics is not None
    assert trainer.test_metrics is not None
    train_keys = [str(k) for k in trainer.train_metrics.keys()]
    val_keys = [str(k) for k in trainer.val_metrics.keys()]
    test_keys = [str(k) for k in trainer.test_metrics.keys()]
    assert all(k.startswith("train/") for k in train_keys), f"Expected train/ prefix, got: {train_keys}"
    assert all(k.startswith("val/") for k in val_keys), f"Expected val/ prefix, got: {val_keys}"
    assert all(k.startswith("test/") for k in test_keys), f"Expected test/ prefix, got: {test_keys}"
    assert "train/miou" in train_keys
    assert "train/dice_macro" in train_keys


def test_default_loader_uses_segmentation_collate_fn() -> None:
    from confluid.fluid import Class as ConfluidClass

    trainer = LightningSegmenter(
        model=_TinySegmenter(num_classes=2),
        train_set=_make_samples(8),
        num_classes=2,
    )
    assert isinstance(trainer.train_loader, ConfluidClass)
    assert trainer.train_loader.kwargs.get("collate_fn") is segmentation_collate_fn
    assert isinstance(trainer.val_loader, ConfluidClass)
    assert trainer.val_loader.kwargs.get("collate_fn") is segmentation_collate_fn


# ---------------------------------------------------------------------------
# Forward / step methods
# ---------------------------------------------------------------------------


def test_training_step_returns_scalar_loss() -> None:
    trainer = LightningSegmenter(
        model=_TinySegmenter(num_classes=2),
        train_set=_make_samples(8),
        num_classes=2,
    )
    batch = _make_batch(n=4, num_classes=2)
    loss = trainer.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_validation_step_returns_scalar_loss() -> None:
    trainer = LightningSegmenter(
        model=_TinySegmenter(num_classes=2),
        train_set=_make_samples(8),
        num_classes=2,
    )
    batch = _make_batch(n=4, num_classes=2)
    loss = trainer.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_metrics_update_on_training_step() -> None:
    trainer = LightningSegmenter(
        model=_TinySegmenter(num_classes=2),
        train_set=_make_samples(8),
        num_classes=2,
    )
    batch = _make_batch(n=4, num_classes=2)
    trainer.training_step(batch, 0)
    assert trainer.train_metrics is not None
    computed = trainer.train_metrics.compute()
    assert all(isinstance(v, torch.Tensor) for v in computed.values())


# ---------------------------------------------------------------------------
# Override hygiene — proves no deltaid adapter is required for custom loss/metrics
# ---------------------------------------------------------------------------


def test_user_can_override_loss_fn_with_any_upstream_class() -> None:
    """The trainer's `loss_fn` attribute is YAML-overridable (confluid post-construct).

    Simulate that by setting it directly after construction with an arbitrary
    upstream nn.Module — here a CE with explicit class_weights, the most common
    "custom loss" segmentation users want — and confirm it sticks.
    """
    trainer = LightningSegmenter(
        model=_TinySegmenter(num_classes=2),
        train_set=_make_samples(8),
        num_classes=2,
    )
    weighted_ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]), ignore_index=-100)
    trainer.loss_fn = weighted_ce
    batch = _make_batch(n=4, num_classes=2)
    # Confirm the step uses the overridden loss without raising.
    trainer.training_step(batch, 0)
    assert trainer.loss_fn is weighted_ce


def test_user_can_override_val_metrics_with_torchmetrics_class() -> None:
    trainer = LightningSegmenter(
        model=_TinySegmenter(num_classes=2),
        train_set=_make_samples(8),
        num_classes=2,
    )
    # Replace the default MetricCollection with a single bare torchmetrics metric.
    custom = MulticlassF1Score(num_classes=2, average="macro")
    trainer.val_metrics = MetricCollection({"val/f1_only": custom})
    batch = _make_batch(n=4, num_classes=2)
    trainer.validation_step(batch, 0)
    assert "val/f1_only" in trainer.val_metrics.keys()
    assert len(list(trainer.val_metrics.keys())) == 1

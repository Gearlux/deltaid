"""Tests for LightningSegmentationEvaluator."""

import json
from typing import Any, Dict, List

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score

from confluid import Class
from dataflux.core import Flux
from dataflux.sample import Sample
from deltaid.evaluator import LightningSegmentationEvaluator
from deltaid.segmentation import segmentation_collate_fn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinySegmenter(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # type: ignore[no-any-return]


def _make_flux(n: int = 8, num_classes: int = 2, h: int = 8, w: int = 8) -> Flux:
    samples = [
        Sample(
            input=torch.randn(3, h, w),
            target=torch.randint(0, num_classes, (h, w), dtype=torch.int64),
            metadata={"idx": i},
        )
        for i in range(n)
    ]
    return Flux(source=samples)


class _RecordingSink:
    def __init__(self) -> None:
        self.writes: List[Any] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
        self.writes.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_metrics_are_wired_on_construction() -> None:
    ev = LightningSegmentationEvaluator(
        model=_TinySegmenter(num_classes=2),
        dataset=_make_flux(8),
        num_classes=2,
    )
    assert ev.metrics is not None
    assert isinstance(ev.metrics, MetricCollection)


def test_metrics_have_test_prefix() -> None:
    ev = LightningSegmentationEvaluator(
        model=_TinySegmenter(num_classes=2),
        dataset=_make_flux(8),
        num_classes=2,
    )
    keys = list(ev.metrics.keys())  # type: ignore[union-attr]
    assert all(
        k.startswith("test/") for k in keys
    ), f"Expected test/ prefix, got: {keys}"
    assert "test/miou" in keys
    assert "test/dice_macro" in keys


def test_ignore_index_flows_into_default_metrics() -> None:
    ev = LightningSegmentationEvaluator(
        model=_TinySegmenter(num_classes=3),
        dataset=_make_flux(8, num_classes=3),
        num_classes=3,
        ignore_index=255,
    )
    assert ev.metrics is not None
    assert ev.metrics["test/miou"].ignore_index == 255  # type: ignore[attr-defined]
    assert ev.metrics["test/dice_macro"].ignore_index == 255  # type: ignore[attr-defined]


def test_default_loader_uses_segmentation_collate_fn() -> None:
    from confluid.fluid import Class as ConfluidClass

    ev = LightningSegmentationEvaluator(
        model=_TinySegmenter(num_classes=2),
        dataset=_make_flux(8),
        num_classes=2,
    )
    assert isinstance(ev.loader, ConfluidClass)
    assert ev.loader.kwargs.get("collate_fn") is segmentation_collate_fn


# ---------------------------------------------------------------------------
# test_step behaviour
# ---------------------------------------------------------------------------


def test_test_step_updates_metrics() -> None:
    ev = LightningSegmentationEvaluator(
        model=_TinySegmenter(num_classes=2),
        dataset=_make_flux(8),
        num_classes=2,
    )
    batch = Sample(
        input=torch.randn(4, 3, 8, 8),
        target=torch.randint(0, 2, (4, 8, 8), dtype=torch.int64),
        metadata={"per_sample": [{"idx": i} for i in range(4)]},
    )
    ev.test_step(batch, 0)
    assert ev.metrics is not None
    computed = ev.metrics.compute()
    assert all(isinstance(v, torch.Tensor) for v in computed.values())


# ---------------------------------------------------------------------------
# Override hygiene
# ---------------------------------------------------------------------------


def test_user_can_override_metrics_with_torchmetrics_class() -> None:
    """No deltaid adapter needed — assign any torchmetrics Metric directly."""
    ev = LightningSegmentationEvaluator(
        model=_TinySegmenter(num_classes=2),
        dataset=_make_flux(8),
        num_classes=2,
    )
    custom = MetricCollection(
        {"test/f1_only": MulticlassF1Score(num_classes=2, average="macro")}
    )
    ev.metrics = custom
    batch = Sample(
        input=torch.randn(4, 3, 8, 8),
        target=torch.randint(0, 2, (4, 8, 8), dtype=torch.int64),
        metadata={"per_sample": [{"idx": i} for i in range(4)]},
    )
    ev.test_step(batch, 0)
    assert "test/f1_only" in ev.metrics.keys()
    assert len(list(ev.metrics.keys())) == 1


# ---------------------------------------------------------------------------
# Full run with sidecar
# ---------------------------------------------------------------------------


def test_evaluator_run_writes_metrics_sidecar(tmp_path: Any) -> None:
    sidecar = tmp_path / "eval_metrics.json"
    ev = LightningSegmentationEvaluator(
        model=_TinySegmenter(num_classes=2),
        dataset=_make_flux(8),
        num_classes=2,
        loader=Class(
            DataLoader,
            batch_size=4,
            collate_fn=segmentation_collate_fn,
        ),
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
        experiment_name="seg_test_exp",
        run_name="seg_test_run",
        metrics_sidecar_path=str(sidecar),
    )
    ev.run()
    assert sidecar.exists(), "Sidecar must be written."
    payload = json.loads(sidecar.read_text())
    assert payload["experiment_name"] == "seg_test_exp"
    assert "metrics" in payload
    metric_values = payload["metrics"]
    assert any("miou" in k or "dice" in k for k in metric_values)
    for v in metric_values.values():
        assert isinstance(v, (int, float))


# ---------------------------------------------------------------------------
# Predictions-sink smoke
# ---------------------------------------------------------------------------


def test_evaluator_writes_to_predictions_sink(tmp_path: Any) -> None:
    sink = _RecordingSink()
    ev = LightningSegmentationEvaluator(
        model=_TinySegmenter(num_classes=2),
        dataset=_make_flux(8),
        num_classes=2,
        loader=Class(
            DataLoader,
            batch_size=4,
            collate_fn=segmentation_collate_fn,
        ),
        predictions_sink=sink,
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
    )
    ev.run()
    # 8 samples; the model emits per-batch [N, C, H, W] logits. The base
    # evaluator dispatches one .write() per per-sample entry it can find.
    # We assert the sink got at least one write and that .close() was called.
    assert sink.closed, ".close() must be invoked at end of run."
    assert len(sink.writes) >= 1

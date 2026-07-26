"""Tests for LightningSegmenter in the evaluation path (task='test')."""

import json
from typing import Any, Dict, List

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class
from sampleflux import Image, Mask, Record, collate_records
from sampleflux.core import Flux
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score

from deltaid.segmenter import LightningSegmenter


def _evaluator(**kw: Any) -> LightningSegmenter:
    """Build a test-path LightningSegmenter (maps ``dataset=`` → ``test_set``, sets task='test')."""
    dataset = kw.pop("dataset", None)
    return LightningSegmenter(test_set=dataset, task="test", **kw)


class _TinySegmenter(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # type: ignore[no-any-return]


def _make_flux(n: int = 8, num_classes: int = 2, h: int = 8, w: int = 8) -> Flux:
    samples = [
        {
            "image": Image(np.random.randn(3, h, w).astype("float32"), layout="CHW"),
            "target": Mask(np.random.randint(0, num_classes, (h, w)).astype("int64")),
            "idx": i,
        }
        for i in range(n)
    ]
    return Flux(source=samples)


def _make_batch(n: int = 4, num_classes: int = 2, h: int = 8, w: int = 8) -> Record:
    x = torch.randn(n, 3, h, w)
    y = torch.randint(0, num_classes, (n, h, w), dtype=torch.int64)
    return {
        "image": Image(x.numpy(), layout="CHW"),
        "target": Mask(y.numpy()),
        "metadata": [{"idx": i} for i in range(n)],
    }


class _RecordingSink:
    def __init__(self) -> None:
        self.writes: List[Any] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
        self.writes.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


def test_metrics_are_wired_on_construction() -> None:
    ev = _evaluator(model=_TinySegmenter(num_classes=2), dataset=_make_flux(8), num_classes=2)
    assert isinstance(ev.test_metrics, MetricCollection)


def test_metrics_have_test_prefix() -> None:
    ev = _evaluator(model=_TinySegmenter(num_classes=2), dataset=_make_flux(8), num_classes=2)
    keys = list(ev.test_metrics.keys())
    assert all(k.startswith("test/") for k in keys), f"Expected test/ prefix, got: {keys}"
    assert "test/miou" in keys
    assert "test/dice_macro" in keys


def test_ignore_index_flows_into_default_metrics() -> None:
    ev = _evaluator(
        model=_TinySegmenter(num_classes=3), dataset=_make_flux(8, num_classes=3), num_classes=3, ignore_index=255
    )
    assert ev.test_metrics["test/miou"].ignore_index == 255  # type: ignore[attr-defined]
    assert ev.test_metrics["test/dice_macro"].ignore_index == 255  # type: ignore[attr-defined]


def test_default_loader_uses_collate_records() -> None:
    from confluid.fluid import Class as ConfluidClass

    ev = _evaluator(model=_TinySegmenter(num_classes=2), dataset=_make_flux(8), num_classes=2)
    assert isinstance(ev.test_loader, ConfluidClass)
    assert ev.test_loader.kwargs.get("collate_fn") is collate_records


def test_test_step_updates_metrics() -> None:
    ev = _evaluator(model=_TinySegmenter(num_classes=2), dataset=_make_flux(8), num_classes=2)
    batch = _make_batch(n=4, num_classes=2)
    ev.test_step(batch, 0)
    computed = ev.test_metrics.compute()
    assert all(isinstance(v, torch.Tensor) for v in computed.values())


def test_user_can_override_metrics_with_torchmetrics_class() -> None:
    """No deltaid adapter needed — assign any torchmetrics Metric directly."""
    ev = _evaluator(model=_TinySegmenter(num_classes=2), dataset=_make_flux(8), num_classes=2)
    custom = MetricCollection({"test/f1_only": MulticlassF1Score(num_classes=2, average="macro")})
    ev.test_metrics = custom
    batch = _make_batch(n=4, num_classes=2)
    ev.test_step(batch, 0)
    assert "test/f1_only" in ev.test_metrics.keys()
    assert len(list(ev.test_metrics.keys())) == 1


def test_evaluator_run_writes_metrics_sidecar(tmp_path: Any) -> None:
    sidecar = tmp_path / "eval_metrics.json"
    ev = _evaluator(
        model=_TinySegmenter(num_classes=2),
        dataset=_make_flux(8),
        num_classes=2,
        experiment_name="seg_test_exp",
        run_name="seg_test_run",
    )
    ev.test_loader = Class(DataLoader, batch_size=4, collate_fn=collate_records)
    ev.lightning = Class(
        L.Trainer,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        default_root_dir=str(tmp_path),
    )
    ev.metrics_sidecar_path = str(sidecar)
    ev.run()
    assert sidecar.exists(), "Sidecar must be written."
    payload = json.loads(sidecar.read_text())
    assert payload["experiment_name"] == "seg_test_exp"
    metric_values = payload["metrics"]
    assert any("miou" in k or "dice" in k for k in metric_values)
    for v in metric_values.values():
        assert isinstance(v, (int, float))


def test_evaluator_writes_to_predictions_sink(tmp_path: Any) -> None:
    sink = _RecordingSink()
    ev = _evaluator(model=_TinySegmenter(num_classes=2), dataset=_make_flux(8), num_classes=2)
    ev.test_loader = Class(DataLoader, batch_size=4, collate_fn=collate_records)
    ev.predictions_sink = sink
    ev.lightning = Class(
        L.Trainer,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        default_root_dir=str(tmp_path),
    )
    ev.run()
    assert sink.closed, ".close() must be invoked at end of run."
    assert len(sink.writes) >= 1

"""LightningSegmenter: a standard PyTorch-Lightning semantic-segmentation module (train + eval).

A self-contained ``pytorch_lightning.LightningModule`` — its own ``training_step`` /
``validation_step`` / ``test_step`` / ``predict_step`` / ``configure_optimizers``, and a
``run()`` that builds the loaders + ``pl.Trainer`` and calls ``fit`` / ``test`` / ``predict``
per the ``task`` knob. It reads like a textbook Lightning example; marainer is used ONLY for
optional experiment tracking and the small ``ensure_record_dataset`` utility. Remove marainer and
this class still trains.

Segmentation specifics it owns: ``ignore_index``-aware ``CrossEntropyLoss`` + the mean-IoU / Dice
metric pair, wired from ``num_classes`` (explicit, or read off a concrete model head).

Minimal flat YAML::

    runnable: !class:deltaid.LightningSegmenter
      task: fit
      num_classes: 2
      ignore_index: -100
    model: !lazy:deltaid.deeplabv3plus.DeepLabV3PlusModel { num_classes: 2 }
    train_set: !ref:train_set
    val_set: !ref:val_set

    recordstream run train.yaml
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, get_hierarchy_from_instance, output
from loggair import get_logger
from marainer.lightning import (
    TrackingCallback,
    activate_for_lightning,
    attach_lightning_progress,
    coerce_metric_value,
    scalar_metrics,
)
from marainer.run_names import generate_run_name
from marainer.sinks import PredictionsSink
from marainer.torch import RecordSource, ensure_record_dataset
from marainer.tracking.base import Tracker
from pytorch_lightning.loggers import Logger
from recordstream import batch_metadata, batch_tensor, collate_records
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

logger = get_logger(__name__)

#: Which entry point :meth:`LightningSegmenter.run` dispatches to.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]


def _default_segmentation_metrics(num_classes: int, ignore_index: int) -> MetricCollection:
    """The deltaid default segmentation metric pair (mean IoU + macro Dice) from stock torchmetrics."""
    return MetricCollection(
        {
            "miou": MulticlassJaccardIndex(num_classes=num_classes, average="macro", ignore_index=ignore_index),
            "dice_macro": MulticlassF1Score(
                num_classes=num_classes, average="macro", ignore_index=ignore_index, multidim_average="global"
            ),
        }
    )


@configurable(task="segmentation", role="trainer")
class LightningSegmenter(TorchRunner, ProgressReporting, L.LightningModule):
    """A standard Lightning semantic segmenter — train, evaluate, test, or predict via ``task``.

    Args:
        model: Segmentation model emitting ``[N, num_classes, H, W]`` logits.
        train_set: Training dataset / Stream (for ``task="fit"``).
        test_set: Held-out eval dataset / Stream (for ``task="test"`` / ``"predict"``).
        num_classes: Number of output classes (incl. background); drives metric construction. Omit
            to read it off a concrete model head.
        ignore_index: Pixel value skipped by loss + metrics.
        task: Entry point :meth:`run` dispatches to.
        input_key: Record key carrying the model input tensor. Defaults to ``image``.
        target_key: Record key carrying the target mask. Defaults to ``target`` (the target ops' output default).
        max_epochs / batch_size / num_workers / logger: ergonomic flat-config knobs.
        experiment_name / run_name: tracking identity.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``optimizer``, ``train/val/test_loader``,
    ``lightning``, ``loss_fn``, ``predictions_sink``, ``metrics_sidecar_path``, ``trackers``,
    ``log_system_metrics``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nn.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        input_key: str = "image",
        target_key: str = "target",
        task: SegmenterTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model: Lazy[nn.Module] = model
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.task: SegmenterTask = task
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=_run_prefix(task))
        self.loss_fn: nn.Module = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.optimizer: Lazy[Optimizer] = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        loader_kw = dict(
            collate_fn=collate_records,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.lightning: Lazy[L.Trainer] = LazyClass(L.Trainer, max_epochs=max_epochs, logger=logger)
        self.train_metrics: Any = None
        self.val_metrics: Any = None
        self.test_metrics: Any = None
        # When num_classes is given, build the metric set eagerly (segmentation counts are usually
        # explicit); initialize() builds it from the model head when num_classes is omitted.
        if self.num_classes is not None:
            base = _default_segmentation_metrics(self.num_classes, self.ignore_index)
            self.train_metrics = base.clone(prefix="train/")
            self.val_metrics = base.clone(prefix="val/")
            self.test_metrics = base.clone(prefix="test/")
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self.trackers: Optional[List[Tracker]] = None
        self.log_system_metrics: bool = True
        self._trackers: List[Tracker] = []

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override of the marker
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table — a hand-written
        # {task: method} dict restates them and drifts (recordstream `docs/runnable.md`).
        run_entrypoint(self, self.task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, cast(nn.Module, self.model)(x))

    def initialize(self) -> None:
        """Flow the model, resolve ``num_classes`` (explicit or off the model head), and build metrics."""
        self.model = flow(self.model)
        if self.num_classes is None:
            nc = getattr(self.model, "num_classes", None)
            if not nc:
                raise RuntimeError("could not determine num_classes — set it on the segmenter or the model.")
            self.num_classes = int(nc)
        base = _default_segmentation_metrics(self.num_classes, self.ignore_index)
        if self.task == "fit":
            if self.train_metrics is None:
                self.train_metrics = base.clone(prefix="train/")
                self.val_metrics = base.clone(prefix="val/")
                self.test_metrics = base.clone(prefix="test/")
        elif self.task == "evaluate" and self.val_metrics is None:
            self.val_metrics = base.clone(prefix="val/")
        elif self.task in ("test", "predict") and self.test_metrics is None:
            self.test_metrics = base.clone(prefix="test/")

    # ---- standard Lightning step functions ---------------------------------- #
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch_tensor(batch, self.input_key), batch_tensor(batch, self.target_key, dtype=torch.int64)
        preds = self(x)
        loss: torch.Tensor = self.loss_fn(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.train_metrics, MetricCollection):
            self.train_metrics(preds, y)
            self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if isinstance(self.train_metrics, MetricCollection):
            self.train_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch_tensor(batch, self.input_key), batch_tensor(batch, self.target_key, dtype=torch.int64)
        preds = self(x)
        loss: torch.Tensor = self.loss_fn(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.val_metrics, MetricCollection):
            self.val_metrics.update(preds, y)
            self.log_dict(self.val_metrics, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        x = batch_tensor(batch, self.input_key)
        preds = self(x)
        if isinstance(self.test_metrics, MetricCollection):
            self.test_metrics.update(preds, batch_tensor(batch, self.target_key, dtype=torch.int64))
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True, batch_size=x.shape[0])
        if self.predictions_sink is not None:
            self._write_predictions(preds, batch)
        return preds

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        preds = self(batch_tensor(batch, self.input_key))
        if self.predictions_sink is not None:
            self._write_predictions(preds, batch)
        return preds

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"deltaid training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("LightningSegmenter.fit() requires a 'train_set'.")
        train_ds = ensure_record_dataset(flow(self.train_set))
        val_ds = ensure_record_dataset(flow(self.val_set)) if self.val_set is not None else None
        test_ds = ensure_record_dataset(flow(self.test_set)) if self.test_set is not None else None
        t_loader = flow(self.train_loader, dataset=train_ds)
        v_loader = flow(self.val_loader, dataset=val_ds) if val_ds is not None else None
        trainer = self._build_trainer()
        trainer.fit(model=self, train_dataloaders=t_loader, val_dataloaders=v_loader)
        logger.info("Training complete.")
        if test_ds is not None:
            trainer.test(model=self, dataloaders=flow(self.test_loader, dataset=test_ds))

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self._run_eval("validate", "val_set", "val_loader", "val_metrics")

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        self._run_eval("test", "test_set", "test_loader", "test_metrics")

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        logger.info(f"deltaid predicting: {self.experiment_name} ({self.run_name})")
        if self.test_set is None:
            raise ValueError("LightningSegmenter.predict() requires a 'test_set'.")
        dataset = ensure_record_dataset(flow(self.test_set))
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(self.test_loader, dataset=dataset)
        trainer = self._build_trainer()
        trainer.predict(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()

    def _run_eval(self, stage: str, dataset_slot: str, loader_slot: str, metrics_slot: str) -> None:
        logger.info(f"deltaid evaluating ({stage}): {self.experiment_name} ({self.run_name})")
        source = getattr(self, dataset_slot)
        if source is None:
            raise ValueError(f"LightningSegmenter.{stage}() requires a '{dataset_slot}'.")
        dataset = ensure_record_dataset(flow(source))
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(getattr(self, loader_slot), dataset=dataset)
        trainer = self._build_trainer()
        runner = trainer.validate if stage == "validate" else trainer.test
        runner(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()
        self._finalize_metrics(getattr(self, metrics_slot))

    def _build_trainer(self) -> L.Trainer:
        """Flow the ``pl.Trainer`` and attach the OPTIONAL marainer tracking (a few lines; removable)."""
        trainer = cast(L.Trainer, flow(self.lightning))
        if self.trackers:
            self._trackers = list(self.trackers)
            cb = TrackingCallback(self._trackers, params=self._config_params())
            trainer.callbacks.append(cb)  # type: ignore[attr-defined]
        attach_lightning_progress(trainer, self)
        if self.log_system_metrics:
            activate_for_lightning(trainer)
        return trainer

    def _config_params(self) -> Dict[str, Any]:
        """Flat ``{dotted-path: live value}`` of this runnable's configuration, for tracker params.

        Walks the FLOWED object graph (not the static signature), so it sees defaults the user
        never wrote into YAML plus post-construction keys. ``log_run_context`` keeps only the
        JSON-scalar values, so nested objects here are harmless. Best-effort by design: a
        tracking detail must never take down a training run.
        """
        try:
            return {path: value for path, (_, value, _) in get_hierarchy_from_instance(self).items()}
        except Exception:  # pragma: no cover - best effort
            return {}

    # ---- predictions + metrics ---------------------------------------------- #
    def _write_predictions(self, preds: Any, batch: Any) -> None:
        per_record_meta = (
            batch_metadata(batch, exclude=(self.input_key, self.target_key)) if isinstance(batch, dict) else None
        )
        items = (
            preds["predictions"]
            if isinstance(preds, dict) and "predictions" in preds
            else (preds if isinstance(preds, list) else [preds])
        )
        for i in range(len(items)):
            m = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            cast(PredictionsSink, self.predictions_sink).write(items[i], m)

    def _finalize_metrics(self, metrics: Any) -> None:
        if not isinstance(metrics, MetricCollection) or self.metrics_sidecar_path is None:
            return
        try:
            payload = coerce_metric_value(metrics.compute())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not compute final metrics for sidecar: {exc}")
            return
        wrapped = {"experiment_name": self.experiment_name, "run_name": self.run_name, "metrics": payload}
        out_path = Path(self.metrics_sidecar_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(wrapped, indent=2, sort_keys=True))
        logger.info(f"Wrote final metrics sidecar to {out_path}")

    # ---- @outputs ----------------------------------------------------------- #
    @property
    @output
    def trained_model(self) -> nn.Module:
        """The trained/live segmentation model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run(): the model is still a deferred Fluid.")
        return cast(nn.Module, self.model)

    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if self.task == "fit":
            try:
                return scalar_metrics(dict(self.trainer.callback_metrics))
            except RuntimeError as exc:
                raise RuntimeError("computed_metrics is only available after run().") from exc
        metrics = self.val_metrics if self.task == "evaluate" else self.test_metrics
        if not isinstance(metrics, MetricCollection):
            raise RuntimeError("computed_metrics is only available after run().")
        return scalar_metrics(metrics.compute())


def _run_prefix(task: str) -> str:
    return {"fit": "train", "predict": "predict"}.get(task, "evaluate")


__all__ = ["LightningSegmenter", "SegmenterTask", "_default_segmentation_metrics"]

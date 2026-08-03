"""LightningSegmenter: a standard PyTorch-Lightning semantic segmenter (train + eval in one class).

A self-contained ``pytorch_lightning.LightningModule`` — its own ``training_step`` /
``validation_step`` / ``test_step`` / ``predict_step`` / ``configure_optimizers``, and a ``run()``
that builds the loaders + ``pl.Trainer`` and calls ``fit`` / ``validate`` / ``test`` / ``predict``
per the ``task`` knob. It reads like a textbook Lightning example; marainer is used ONLY for
optional experiment tracking and a couple of small utilities. Remove marainer and this class still
trains — you lose the rich tracking, not the training.

Segmentation specifics it owns: deriving ``num_classes`` (from a concrete model head, an explicit
arg, or a walk of the mask column), an ``ignore_index``-aware ``CrossEntropyLoss``, and the mean-IoU
/ Dice metric pair sized from that count.

Minimal flat YAML (top-level keys wired by Confluid broadcasting)::

    runnable: !class:deltaid.LightningSegmenter
      task: fit
    model: !lazy:deltaid.SegmentationModelPytorch { arch: unet, encoder_name: resnet34 }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    val_set:   !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 8
    max_epochs: 20

    deltaid train train.yaml        # or: recordstream run train.yaml
"""

from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from loggair import get_logger
from marainer.artifacts import write_metrics_sidecar
from marainer.lightning import (
    TrackerLogger,
    TrackingCallback,
    activate_for_lightning,
    attach_lightning_progress,
    build_metric_collection,
    scalar_metrics,
)
from marainer.metric_slots import merge_metric_specs
from marainer.run_names import generate_run_name, run_prefix
from marainer.tracking.base import Tracker, as_tracker_list
from recordstream import (
    PredictionsSink,
    RecordSource,
    SegmentationOutput,
    batch_metadata,
    batch_tensor,
    collate_records,
    ensure_materialized,
    ensure_record_dataset,
    num_mask_classes,
    segmentation_output,
)
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

logger = get_logger(__name__)

#: Which entry point :meth:`LightningSegmenter.run` dispatches to.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]

#: What a metrics slot admits: a ``{name: template}`` mapping, a ready-made collection, or ``None``
#: for "not chosen". Named because FOUR slots share it (the shared set + one per split) and
#: `to_pydantic` reads the annotation for navigaitor's form-spec / StreamStudio widgets.
MetricsSpec = Optional[Union[Dict[str, Union[Metric, LazyClass[Metric]]], MetricCollection]]


def _per_record_masks(preds: Any) -> List[Any]:
    """Split a batched model output into ONE entry per record, for the per-record sink contract.

    A segmenter emits the whole batch at once (``mask`` ``[N, H, W]``, ``probs`` ``[N, C, H, W]``)
    while ``PredictionsSink.write(prediction, metadata)`` is per record — so a batched
    :class:`~recordstream.SegmentationOutput` is sliced row-wise here. Handing the batch over
    whole writes ONE entry for an N-record batch, with the sink reading row 0 as if it were the
    whole prediction.
    """
    if isinstance(preds, list):
        return preds
    if isinstance(preds, dict):
        lengths = {len(v) for v in preds.values() if hasattr(v, "__len__")}
        if len(lengths) == 1:
            n = lengths.pop()
            return [{k: v[i] for k, v in preds.items()} for i in range(n)]
    return [preds]


@configurable(task="segmentation", role="trainer", framework="torch")
class LightningSegmenter(TorchRunner, ProgressReporting, L.LightningModule):
    """A standard Lightning semantic segmenter — train, evaluate, test, or predict via ``task``.

    Args:
        model: Segmentation model emitting ``[N, num_classes, H, W]`` logits (wire ``!lazy:`` so
            :meth:`initialize` flows it with the resolved num_classes at run time).
        train_set: Training dataset / Stream (for ``task="fit"``).
        test_set: Held-out eval dataset / Stream (for ``task="test"`` / ``"predict"``; also tested
            after ``fit`` when set).
        num_classes: Output class count, background included; omit to resolve from the model or
            the mask column.
        ignore_index: Pixel value skipped by the loss AND the metrics — the standard way to
            exclude an "unlabelled" / "void" class. ``-100`` (torch's own default) means no pixel
            is excluded, since a class-id mask never contains it.
        task: Entry point :meth:`run` dispatches to — ``"fit"`` / ``"evaluate"`` / ``"test"`` / ``"predict"``.
        max_epochs / batch_size / num_workers: ergonomic flat-config knobs.
        trackers: marainer trackers — one or a list. They ARE this run's metric sinks: each is
            adapted to Lightning's logger interface, so there is no separate ``logger:`` slot of
            ``pytorch_lightning.loggers``.
        experiment_name / run_name: tracking identity.
        input_key: Record key carrying the model input image (default ``"image"``).
        target_key: Record key carrying the per-pixel mask (default ``"mask"`` — the key
            ``ConvertToMask`` writes, which is also albumentations' own vocabulary).
        metrics: Single-metric templates flowed per split (default mean IoU + macro Dice).
        train_metrics / val_metrics / test_metrics: That split's OWN templates, merged over
            ``metrics`` — how a metric is added to one split only (a per-class IoU on validation).
            Each slot holds the spec until :meth:`initialize` builds it, and the live prefixed
            ``MetricCollection`` after.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``lightning``, ``loss_fn``, ``predictions_sink``,
    ``metrics_sidecar_path``, ``log_system_metrics``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nn.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        task: SegmenterTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "mask",
        metrics: MetricsSpec = None,
        train_metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
    ) -> None:
        super().__init__()
        self.model: Lazy[nn.Module] = model
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: SegmenterTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.max_epochs = max_epochs
        # Deferred runtime slots — `Lazy[T]` DECLARES that the run methods flow this later, usually
        # with a kwarg that does not exist yet (`params=self.parameters()`, `dataset=`). For a
        # CONSTRUCTOR param the marker is machine-read (`confluid.lazy_param_names` reports it, and
        # liquifai's auto deep-flow SKIPS it instead of eagerly building it); for the BODY slots
        # below it is inert — deferral is enforced by the slot holding a `LazyClass` — but it still
        # earns its place by giving `to_pydantic` a real type where `Any` typed nothing at all.
        #
        # Sentinels: the loss and the metrics both need `num_classes`, which is only known once
        # `initialize()` has consulted the model or the data. `None` means "not chosen" — a live
        # default here could not be told apart from a user's explicit choice, so an explicitly
        # wired Dice loss would be silently replaced.
        # `Lazy[...]` on the loss because a `!class:` loss stays a deferred marker until it is
        # flowed, so this IS a slot this class flows.
        self.loss_fn: Optional[Lazy[nn.Module]] = None
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
        # No `logger=` here: the trackers ARE the metric sinks, adapted to Lightning's logger
        # interface in `_build_trainer` (a Tracker may still be a deferred Fluid at this point, so
        # the adapter is built at run time, not now).
        self.lightning: Lazy[L.Trainer] = LazyClass(L.Trainer, max_epochs=max_epochs)
        self.metrics: Any = metrics
        # Per-split slots, each dual-purpose the way `model` is: they hold that split's metrics
        # SPEC until `initialize()` builds it, and the live prefixed `MetricCollection` after.
        # `_metrics_built` is what tells the two states apart — `is None` cannot, once a slot may
        # legitimately arrive carrying a spec.
        self.train_metrics: Any = train_metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        self._metrics_built: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # Stored VERBATIM (lazy-init mandate: the constructor does no work). It may hold a single
        # tracker, a list, or deferred `Fluid` markers; `_live_trackers()` normalizes and flows it
        # at run time.
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.log_system_metrics: bool = True

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

    # ---- standard Lightning step functions ---------------------------------- #
    def _targets(self, batch: Any) -> torch.Tensor:
        """The per-pixel target tensor the loss expects — an ``[N, H, W]`` int64 class-id map.

        Both the reader and the dtype are recordstream's; what is segmentation's is naming
        ``int64``. It is not decoration: albumentations returns an int32 mask (measured), and
        ``CrossEntropyLoss`` rejects it with *"expected target dtype to be Long or Byte, but got
        Int"*. ``batch_tensor`` honours the contract the caller names rather than deciding one.
        """
        return batch_tensor(batch, self.target_key, self.device, dtype=torch.int64)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self._resolve_task_defaults()  # no-op once resolved; keeps a directly-called step working
        x = batch_tensor(batch, self.input_key, self.device)
        y = self._targets(batch)
        preds = self(x)
        loss: torch.Tensor = cast(nn.Module, self.loss_fn)(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.train_metrics, MetricCollection):
            # The per-BATCH values, SCALARS only — `log_dict` reduces every logged value through
            # Lightning's `convert_tensors_to_scalars`, which raises on a non-scalar metric.
            self.log_dict(scalar_metrics(self.train_metrics(preds, y)), on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        if isinstance(self.train_metrics, MetricCollection):
            self.train_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self._resolve_task_defaults()  # no-op once resolved; keeps a directly-called step working
        x = batch_tensor(batch, self.input_key, self.device)
        y = self._targets(batch)
        preds = self(x)
        loss: torch.Tensor = cast(nn.Module, self.loss_fn)(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.val_metrics, MetricCollection):
            self.val_metrics.update(preds, y)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        x = batch_tensor(batch, self.input_key, self.device)
        preds = self(x)
        if isinstance(self.test_metrics, MetricCollection):
            self.test_metrics.update(preds, self._targets(batch))
        if self.predictions_sink is not None:
            self._write_predictions(segmentation_output(preds), batch)
        return preds

    def predict_step(self, batch: Any, batch_idx: int) -> SegmentationOutput[torch.Tensor]:
        """Predict a batch as a :class:`~recordstream.SegmentationOutput`.

        Returning the CONTRACT rather than raw logits is what makes the output consumable: a
        predictions sink reads ``mask`` / ``probs`` by name and cannot softmax on the caller's
        behalf — only the model knows whether its head already did.
        """
        out = segmentation_output(self(batch_tensor(batch, self.input_key, self.device)))
        if self.predictions_sink is not None:
            self._write_predictions(out, batch)
        return out

    # ---- epoch-end metric reporting ----------------------------------------- #
    # Reset happens at epoch START, not end, so the LAST epoch's state survives the run —
    # `_write_metrics_sidecar` and the `computed_metrics` @output both `compute()` after
    # `trainer.validate()` / `.test()` has returned.
    def on_validation_epoch_start(self) -> None:
        if isinstance(self.val_metrics, MetricCollection):
            self.val_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self._report_epoch_metrics(self.val_metrics)

    def on_test_epoch_start(self) -> None:
        if isinstance(self.test_metrics, MetricCollection):
            self.test_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self._report_epoch_metrics(self.test_metrics)

    def _report_epoch_metrics(self, collection: Any) -> None:
        """Compute one split's metrics once at epoch end and log the scalars.

        Computing here rather than handing the whole `MetricCollection` to `log_dict` is what
        lets a non-scalar metric be wired at all: `log_dict` reduces every value through
        Lightning's `convert_tensors_to_scalars`, which raises on anything that is not one
        number — and a per-class IoU (``average=None`` -> a ``[C]`` vector) is exactly the metric
        a segmentation run wants on validation. `scalar_metrics` keeps the scalars; the rest is
        dropped here rather than crashing the epoch (see `TASKS.md` for reporting it properly).
        """
        if not isinstance(collection, MetricCollection):
            return
        self.log_dict(scalar_metrics(collection.compute()))

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"deltaid training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("LightningSegmenter.fit() requires a 'train_set'.")
        train_ds = ensure_record_dataset(self.train_set)
        val_ds = ensure_record_dataset(self.val_set) if self.val_set is not None else None
        test_ds = ensure_record_dataset(self.test_set) if self.test_set is not None else None
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
        dataset = ensure_record_dataset(self.test_set)
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
        dataset = ensure_record_dataset(source)
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(getattr(self, loader_slot), dataset=dataset)
        trainer = self._build_trainer()
        runner = trainer.validate if stage == "validate" else trainer.test
        runner(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()
        self._write_metrics_sidecar(getattr(self, metrics_slot))

    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot.

        Written back rather than returned fresh each call: ``flow()`` on a LIVE object returns
        that same object (idempotent), but a deferred ``Fluid`` would be REBUILT on every call —
        two MLflow runs for one training run.
        """
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    def _build_trainer(self) -> L.Trainer:
        """Flow the ``pl.Trainer`` and attach the OPTIONAL marainer tracking (a few lines; removable).

        The trackers ARE the Trainer's loggers: each is flowed (it may still be a deferred
        ``Fluid``) and adapted through ``TrackerLogger``, so Lightning's per-step metrics land in
        the same backends as the provenance and the manifest. There is no separate ``logger:``
        slot — one wiring, one place a run's output goes.
        """
        trackers = self._live_trackers()
        adapters = [TrackerLogger(t) for t in trackers]
        # A runtime kwarg wins over the stored one, so an explicitly wired `lightning:` block keeps
        # every knob it set and only gains the loggers.
        trainer = cast(L.Trainer, flow(self.lightning, logger=adapters or False))
        if trackers:
            # `runnable=self` rather than a params dict: marainer derives the hyperparameters from
            # the flowed object graph AND dumps the resolved config, so no two backends can answer
            # "what were the hyperparameters?" differently.
            cb = TrackingCallback(trackers, runnable=self)
            trainer.callbacks.append(cb)  # type: ignore[attr-defined]
        attach_lightning_progress(trainer, self)
        if self.log_system_metrics:
            activate_for_lightning(trainer)
        return trainer

    # ---- segmentation specifics --------------------------------------------- #
    def initialize(self) -> None:
        """Resolve ``num_classes``, then flow the model + the loss + the metrics with it."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()

    def _initialize_fit(self) -> None:
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)

        nc = self._num_classes_without_data()
        if nc is None:
            if self.train_set is None:
                raise RuntimeError(
                    "could not determine num_classes — set it on the segmenter or the model, or wire a train_set."
                )
            # The last resort, and the expensive one: decode the mask column to find the largest
            # class id. Announced because it is a full pass over the targets — for segmentation
            # the count is usually a property of the LABEL SCHEME, so setting it explicitly is
            # both cheaper and more honest about a class that appears in no training image.
            logger.info(f"Deriving num_classes from the {self.target_key!r} column (a full pass over the masks)...")
            nc = num_mask_classes(self.train_set, self.target_key)
            logger.info(f"Derived num_classes={nc} from the mask column.")
        self.num_classes = int(nc)

        self.model = flow(self.model, num_classes=self.num_classes)
        self._resolve_task_defaults()
        if not self._metrics_built:
            self.train_metrics = self._build_split_metrics(self.train_metrics, "train/")
            self.val_metrics = self._build_split_metrics(self.val_metrics, "val/")
            self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
            self._metrics_built = True

    def _initialize_eval(self) -> None:
        # The trained model's head is authoritative for eval — flow it first, read num_classes off
        # it. NOT the eval dataset, which may be a subset whose masks never show the last class and
        # would mis-size the metrics against the head.
        self.model = flow(self.model)
        nc = getattr(self.model, "num_classes", None) or self.num_classes
        if not nc:
            raise RuntimeError("could not determine num_classes from the model (it must expose `num_classes`).")
        self.num_classes = int(nc)
        self._resolve_task_defaults()
        eval_slot = "val_set" if self.task == "evaluate" else "test_set"
        setattr(self, eval_slot, self._prepare(getattr(self, eval_slot)))
        metrics_slot = "val_metrics" if self.task == "evaluate" else "test_metrics"
        prefix = "val/" if self.task == "evaluate" else "test/"
        if not self._metrics_built:
            setattr(self, metrics_slot, self._build_split_metrics(getattr(self, metrics_slot), prefix))
            self._metrics_built = True

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process.

        ``ensure_record_dataset`` does the first; ``ensure_materialized`` does the second, and it
        matters as soon as ``num_workers`` is non-zero. A ``DataLoader`` worker may be a FORKED
        child, and if that child is the first to touch a lazy source, the download runs there —
        reaching ``_scproxy`` / CoreFoundation on macOS, which is not fork-safe. The result is a
        SIGSEGV with no Python traceback, surfacing only as ``DataLoader worker exited
        unexpectedly``. Reading one record here, in the parent, leaves the child nothing to build.
        """
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _num_classes_without_data(self) -> Optional[int]:
        """The class count from the model or the config — everything short of walking the masks."""
        if not isinstance(self.model, Fluid):
            model_nc = getattr(self.model, "num_classes", None)
            if model_nc:
                if self.num_classes is not None and int(self.num_classes) != int(model_nc):
                    logger.warning(
                        f"Segmenter num_classes={self.num_classes} != model num_classes={model_nc}; using model's."
                    )
                return int(model_nc)
        if self.num_classes is not None:
            return int(self.num_classes)
        if isinstance(self.model, Fluid):
            pinned = self.model.kwargs.get("num_classes")
            if pinned:
                return int(pinned)
        return None

    def _resolve_task_defaults(self) -> None:
        """Fill the ``loss_fn`` / ``metrics`` sentinels once ``num_classes`` is known.

        Both defaults need the class count, which `initialize()` derives — so they cannot be
        constructor defaults. A value the user DID wire is left untouched (that is what the `None`
        sentinel buys: "not chosen" is distinguishable from "chosen to be the default"), and this
        is idempotent so a directly-invoked step function works without `initialize()`.
        """
        if self.loss_fn is None:
            # Per-pixel softmax over the channel dim — the standard segmentation loss, with the
            # void class excluded from BOTH the loss and the metrics by the same knob.
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            logger.debug(f"loss_fn defaulted to CrossEntropyLoss(ignore_index={self.ignore_index}).")
        if self.metrics is None and self.num_classes:
            # The segmentation pair: mean IoU (the field's primary score) and macro Dice — the
            # same statistic as macro F1, which is what torchmetrics calls it.
            self.metrics = {
                "miou": LazyClass[Metric](MulticlassJaccardIndex, average="macro", ignore_index=self.ignore_index),
                "dice_macro": LazyClass[Metric](
                    MulticlassF1Score, average="macro", ignore_index=self.ignore_index, multidim_average="global"
                ),
            }

    def _build_split_metrics(self, spec: Any, prefix: str) -> MetricCollection:
        """One split's live, prefixed collection — its own spec merged over the shared one.

        The combination rule is :func:`marainer.metric_slots.merge_metric_specs`, shared with every
        other backend and with the classification task, so a config moving between them cannot
        change shape; what is torch's alone is the rest of this line — sizing the templates with
        ``num_classes`` and prefixing the collection.
        """
        assert self.num_classes is not None
        return build_metric_collection(merge_metric_specs(self.metrics, spec), num_classes=self.num_classes).clone(
            prefix=prefix
        )

    # ---- predictions + metrics ---------------------------------------------- #
    def _write_predictions(self, preds: Any, batch: Any) -> None:
        """Write ONE prediction per record to the wired sink, paired with that record's metadata."""
        per_record_meta = (
            batch_metadata(batch, exclude=(self.input_key, self.target_key)) if isinstance(batch, dict) else None
        )
        items = _per_record_masks(preds)
        for i, item in enumerate(items):
            m = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            cast(PredictionsSink, self.predictions_sink).write(item, m)

    def _write_metrics_sidecar(self, metrics: Any) -> None:
        """Hand the run's final scalars to marainer, which owns where run artifacts land."""
        if not isinstance(metrics, MetricCollection):
            return
        try:
            payload = scalar_metrics(metrics.compute())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not compute final metrics for the sidecar: {exc}")
            return
        write_metrics_sidecar(self.metrics_sidecar_path, payload, self.experiment_name, self.run_name)

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


__all__ = ["LightningSegmenter", "MetricsSpec", "SegmenterTask"]

"""``TorchSegmenter`` — the same segmentation run as a PLAIN PyTorch training loop.

The second backend, and the one with nothing underneath it. The Lightning backend hands the epoch
to a ``Trainer``; this one writes the loop out — ``zero_grad`` / ``backward`` / ``step``, an
explicit device, an explicit ``model.train()`` / ``model.eval()``, an explicit ``torch.no_grad()``
on the eval pass.

**Why it earns its place in a project whose point is that the config is the program.** It is the
control experiment. The claim every other backend makes is "swap the runnable, keep the config";
this one removes the training *framework* entirely and the config still does not change — same
dataset, same ``preprocess`` chain, same ``SegmentationModelPytorch``, same ``torch.nn`` and smp
losses, same ``torchmetrics``, same artifacts. Its ``!scope:framework=torch`` block in the example
config is the smallest in the file, which is the point. If a knob only worked because Lightning
implemented it, this backend is where that shows up.

It is also the smallest correct answer to "how do I train a segmenter on plain torch?", which is a
real question.

**What it shares with the Lightning backend, and what it does not.** The model, the losses and
metrics (:mod:`deltaid.torch_segmentation`, ``framework="torch"``), the batch shape
(``[N, 3, H, W]`` + an ``[N, H, W]`` int64 mask via ``collate_records``) and the run artifacts are
the SAME objects. What differs is the loop and what a loop implies: there is no ``Trainer``, so
there is no ``lightning:`` slot and no framework loggers (tracking goes through ``trackers:``,
which every backend has), no callback protocol, and no distributed strategy. Multi-GPU is
deliberately out of scope — that is precisely what Lightning is for, and pretending otherwise here
would be a worse ``Trainer``.

The `num_classes` resolution and the metric slots are DUPLICATED from the Lightning backend rather
than shared, deliberately and temporarily: an engine-neutral base is deferred until the detection
project has been through the same treatment, so it can be designed against three tasks instead of
one. See ``TASKS.md``.

Minimal flat YAML — the Lightning block with one key changed::

    runnable: !class:deltaid.TorchSegmenter
    model: !lazy:deltaid.SegmentationModelPytorch { arch: unet }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    num_classes: 3
    max_epochs: 5

Run it with ``deltaid train cfg.yaml --framework torch``.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from loggair import get_logger
from marainer.artifacts import write_metrics_sidecar
from marainer.metric_slots import merge_metric_specs
from marainer.run_names import generate_run_name, run_prefix
from marainer.torch import Device, build_metric_collection, resolve_device, scalar_metrics
from marainer.tracking.base import Tracker, as_tracker_list
from marainer.tracking.provenance import capture_log_file
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

#: Which entry point :meth:`TorchSegmenter.run` dispatches to — the same vocabulary every other
#: backend uses, because the verb is a property of the TASK, not of the engine.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]

#: What a metrics slot admits. Declared here rather than imported from the Lightning backend for
#: the reason this whole module exists — that import would pull ``pytorch_lightning`` into a
#: deliberately framework-free one (``test_it_needs_no_training_framework`` scans for exactly it).
MetricsSpec = Optional[Union[Dict[str, Union[Metric, LazyClass[Metric]]], MetricCollection]]

#: Re-exported so ``TorchSegmenter``'s ``device`` knob can be read from this module the way every
#: other name in its signature can. It is marainer's: a ``Trainer`` picks an accelerator for you, a
#: hand-written loop has to pick one itself, and every such loop picks identically — so the choice
#: is a torch-platform helper, not this backend's private function.
__all__ = ["Device", "MetricsSpec", "SegmenterTask", "TorchSegmenter"]


@configurable(task="segmentation", role="trainer", framework="torch")
class TorchSegmenter(TorchRunner, ProgressReporting):
    """Train, evaluate or predict a semantic segmenter with a hand-written PyTorch loop.

    Args:
        model: Segmentation model emitting ``[N, num_classes, H, W]`` logits — wire ``!lazy:`` so
            :meth:`initialize` can inject the resolved ``num_classes`` at run time.
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``; also tested after a fit).
        num_classes: Class count, background included. Omitted, it is derived from the model or
            the mask column.
        ignore_index: Pixel value skipped by the loss AND the metrics — the standard way to
            exclude an "unlabelled" / "void" class.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        num_workers: DataLoader worker processes. Persistent workers follow from this, exactly as
            in the Lightning backend — there is no separate knob.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the per-pixel mask.
        metrics: torchmetrics templates, flowed per split. ``None`` picks mean IoU + macro Dice.
        train_metrics / val_metrics / test_metrics: That split's OWN templates, merged over
            ``metrics`` — how a metric is added to one split only (a per-class IoU on validation).
        device: Where to run — see :data:`Device`.
        checkpoint_dir: Directory for ``model.pt`` (a plain ``state_dict``). ``None`` writes no
            weights, which is what a smoke run wants.
        grad_clip: Clip gradients to this global norm before each ``step()``. ``None`` disables.
        trackers: marainer trackers — one or a list.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``loss_fn``, ``predictions_sink``, ``metrics_sidecar_path``.
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
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "mask",
        metrics: MetricsSpec = None,
        train_metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        device: Device = "auto",
        checkpoint_dir: Optional[str] = None,
        grad_clip: Optional[float] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
    ) -> None:
        super().__init__()
        # `nn.Module`, not `Lazy[nn.Module]`: the DEFERRED-ness is declared once, on the ctor
        # param above, which is what `lazy_param_names` / `to_pydantic` read. Restating it here
        # would make the attribute a union no method could call through.
        self.model: nn.Module = cast(nn.Module, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: SegmenterTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.device_choice: Device = device
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        # Deferred runtime slots, built as `LazyClass` so a config can override any DataLoader
        # knob (`pin_memory`, `drop_last`, a sampler) without this class enumerating them — the
        # same construction the Lightning backend uses.
        loader_kw: Dict[str, Any] = dict(
            collate_fn=collate_records,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.optimizer: Lazy[Optimizer] = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        # Sentinels: both defaults need `num_classes`, which is only known once `initialize()` has
        # resolved it. A live default here could not be told apart from a user's explicit choice.
        self.loss_fn: Optional[Lazy[nn.Module]] = None
        self.metrics: Any = metrics
        # Dual-purpose the way `model` is: that split's metrics SPEC until `initialize()` builds
        # it, the live prefixed collection after. `_metrics_built` tells the two states apart —
        # `is None` cannot, once a slot may legitimately arrive carrying a spec.
        self.train_metrics: Any = train_metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        self._metrics_built: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # Stored VERBATIM (lazy-init mandate: the constructor does no work).
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}
        self._device: torch.device = torch.device("cpu")

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table (recordstream docs/runnable.md).
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve ``num_classes``, build the model, and place everything on a device."""
        self._device = resolve_device(self.device_choice)
        logger.info(f"deltaid torch backend on {self._device}.")
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        if not isinstance(self.model, nn.Module):
            raise TypeError(
                f"TorchSegmenter.model must be a torch.nn.Module, got {type(self.model).__name__}. "
                "Wire `!lazy:deltaid.SegmentationModelPytorch` (or any nn.Module) into the model slot."
            )
        # The model is BUILT by the `flow()` in the initialize methods, whether it arrived
        # deferred or live — confluid fires `solidify()` on its idempotency pass-through too, so a
        # config wiring `!class:deltaid.SegmentationModelPytorch()` no longer reaches the optimizer
        # with an unbuilt backbone and NO parameters.
        self.model.to(self._device)
        # The loss moves too, and it is not decoration: a weighted `CrossEntropyLoss` holds its
        # class weights in a BUFFER, and a CPU buffer against MPS/CUDA logits raises mid-batch.
        loss = flow(self.loss_fn)
        if isinstance(loss, nn.Module):
            self.loss_fn = loss.to(self._device)

    def _initialize_fit(self) -> None:
        """Resolve ``num_classes``, then build the model and the metrics with it.

        Deliberately the same sequence as the Lightning backend's method of this name — none of it
        is engine-specific, and it is duplicated rather than shared until the base can be designed
        against detection too (see the module docstring).
        """
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)

        nc = self._num_classes_without_data()
        if nc is None:
            if self.train_set is None:
                raise RuntimeError(
                    "could not determine num_classes — set it on the segmenter or the model, or wire a train_set."
                )
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
        """Model-driven setup: the trained head's width is authoritative, not the eval split's.

        An eval dataset may be a subset whose masks never show every class, so deriving the count
        from it would mis-size the metrics against the head that produced the logits.
        """
        self.model = flow(self.model)
        nc = getattr(self.model, "num_classes", None) or self.num_classes
        if not nc:
            raise RuntimeError("could not determine num_classes from the model (it must expose `num_classes`).")
        self.num_classes = int(nc)
        self.test_set = self._prepare(self.test_set)
        self.val_set = self._prepare(self.val_set)
        self._resolve_task_defaults()
        if not self._metrics_built:
            self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
            self.val_metrics = self._build_split_metrics(self.val_metrics, "val/")
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

        Identical choices to the Lightning backend, and that is the point: a config moved between
        the two must not silently change its loss. Idempotent, so an explicitly wired value is
        never replaced.
        """
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            logger.debug(f"loss_fn defaulted to CrossEntropyLoss(ignore_index={self.ignore_index}).")
        if self.metrics is None and self.num_classes:
            self.metrics = {
                "miou": LazyClass[Metric](MulticlassJaccardIndex, average="macro", ignore_index=self.ignore_index),
                "dice_macro": LazyClass[Metric](
                    MulticlassF1Score, average="macro", ignore_index=self.ignore_index, multidim_average="global"
                ),
            }

    def _build_split_metrics(self, spec: Any, prefix: str) -> MetricCollection:
        """One split's live, prefixed collection — its own spec merged over the shared one.

        The combination rule is :func:`~marainer.metric_slots.merge_metric_specs`, shared with
        every other backend so a config moving between them cannot change shape.
        """
        assert self.num_classes is not None
        return build_metric_collection(merge_metric_specs(self.metrics, spec), num_classes=self.num_classes).clone(
            prefix=prefix
        )

    # ---- the batch boundary -------------------------------------------------- #
    def _targets(self, batch: Any) -> torch.Tensor:
        """The per-pixel target tensor the loss expects — an ``[N, H, W]`` int64 class-id map.

        Both the reader and the dtype are recordstream's; what is segmentation's is naming
        ``int64``, because albumentations returns an int32 mask and ``CrossEntropyLoss`` rejects
        it with *"expected target dtype to be Long or Byte, but got Int"*.
        """
        return batch_tensor(batch, self.target_key, self._device, dtype=torch.int64)

    def _loaders(self) -> Dict[str, Optional[DataLoader[Any]]]:
        """The deferred loader slots, flowed with the datasets this run resolved."""
        return {
            "train": flow(self.train_loader, dataset=self.train_set) if self.train_set is not None else None,
            "val": flow(self.val_loader, dataset=self.val_set) if self.val_set is not None else None,
            "test": flow(self.test_loader, dataset=self.test_set) if self.test_set is not None else None,
        }

    # ---- the loop ------------------------------------------------------------ #
    def _train_epoch(self, loader: DataLoader[Any], optimizer: Optimizer) -> float:
        """One training pass, returning the mean loss. The whole reason this backend exists."""
        model, loss_fn = self.model, cast(nn.Module, flow(self.loss_fn))
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            x = batch_tensor(batch, self.input_key, self._device)
            y = self._targets(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
            if self.train_metrics is not None:
                self.train_metrics.update(logits.detach(), y)
        return total / batches if batches else float("nan")

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader[Any], metrics: Optional[MetricCollection]) -> float:
        """One evaluation pass, returning the mean loss. ``no_grad`` because nothing here trains."""
        model, loss_fn = self.model, cast(nn.Module, flow(self.loss_fn))
        model.eval()
        total, batches = 0.0, 0
        for batch in loader:
            x = batch_tensor(batch, self.input_key, self._device)
            y = self._targets(batch)
            logits = model(x)
            total += float(loss_fn(logits, y))
            batches += 1
            if metrics is not None:
                metrics.update(logits, y)
        return total / batches if batches else float("nan")

    def _epoch_metrics(self, collection: Optional[MetricCollection]) -> Dict[str, float]:
        """Compute and RESET a split's metrics — the bookkeeping a ``Trainer`` hides.

        Only the SCALARS are returned: `Tracker.log_metrics` is a `{name: float}` channel, and
        `scalar_metrics` drops the rest. For segmentation the useful non-scalar is a per-class
        VECTOR (`MulticlassJaccardIndex(average=None)`), which currently vanishes here — see
        `TASKS.md`. That is the same gap the Lightning backend has, deliberately: the two
        backends must report identically.
        """
        if collection is None:
            return {}
        values = scalar_metrics(collection.compute())
        collection.reset()
        return values

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        """Epochs, in the open: forward, backward, step, validate, log, checkpoint."""
        logger.info(f"deltaid torch training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("TorchSegmenter.fit() requires a 'train_set'.")
        loaders = self._loaders()
        train_loader = cast(DataLoader[Any], loaders["train"])
        optimizer = cast(Optimizer, flow(self.optimizer, params=self.model.parameters()))
        trackers = self._live_trackers()
        manifest = self._log_run_context(trackers)
        self._move_metrics()

        for epoch in range(self.max_epochs):
            logs: Dict[str, float] = {"train/loss": self._train_epoch(train_loader, optimizer)}
            logs.update(self._epoch_metrics(self.train_metrics))
            if loaders["val"] is not None:
                logs["val/loss"] = self._eval_epoch(cast(DataLoader[Any], loaders["val"]), self.val_metrics)
                logs.update(self._epoch_metrics(self.val_metrics))
            for tracker in trackers:
                tracker.log_metrics(logs, step=epoch)
            for key, value in logs.items():
                self._history.setdefault(key, []).append(value)
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
            self._computed_metrics = dict(logs)

        logger.info("Training complete.")
        self._save_checkpoint()
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        if loaders["test"] is not None:
            self._computed_metrics.update(self._score(cast(DataLoader[Any], loaders["test"])))
        # Uploads the log file, stamps the manifest's terminal status AND finalizes every tracker.
        # Not optional once a manifest exists: left unfinished it stays `running`, and marainer's
        # atexit hook then calls a successful run `interrupted`.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        """Score the VALIDATION split — the `evaluate` / `test` split is the only difference."""
        if self.val_set is None:
            raise ValueError("TorchSegmenter.evaluate() requires a 'val_set'.")
        self._move_metrics()
        loader = cast(DataLoader[Any], self._loaders()["val"])
        self._computed_metrics = self._score(loader, split="val", metrics=self.val_metrics)
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("TorchSegmenter.test() requires a 'test_set'.")
        self._move_metrics()
        self._computed_metrics = self._score(cast(DataLoader[Any], self._loaders()["test"]))
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def _score(
        self, loader: DataLoader[Any], split: str = "test", metrics: Optional[MetricCollection] = None
    ) -> Dict[str, float]:
        """Evaluate over ``loader`` and report — shared by ``test``, ``evaluate`` and the post-fit pass."""
        trackers = self._live_trackers()
        collection = metrics if metrics is not None else self.test_metrics
        results = {f"{split}/loss": self._eval_epoch(loader, collection)}
        results.update(self._epoch_metrics(collection))
        for tracker in trackers:
            tracker.log_metrics(results)
        logger.info(f"Evaluation complete: {results}")
        return results

    @entrypoint("predict", role="predictor", primary=True)
    @torch.no_grad()
    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each batch with its metadata."""
        logger.info(f"deltaid torch predicting: {self.experiment_name} ({self.run_name})")
        if self.test_set is None:
            raise ValueError("TorchSegmenter.predict() requires a 'test_set'.")
        self.model.eval()
        sink = flow(self.predictions_sink)
        loader = cast(DataLoader[Any], self._loaders()["test"])
        for batch in loader:
            out = segmentation_output(self.model(batch_tensor(batch, self.input_key, self._device)))
            if sink is not None:
                self._write_predictions(out, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, out: SegmentationOutput[torch.Tensor], batch: Any, sink: PredictionsSink) -> None:
        """One prediction per record, sliced from the batched output the model emits.

        The model returns ``mask`` ``[N, H, W]`` and ``probs`` ``[N, C, H, W]`` while the sink
        contract is per-record, so handing the batch over whole would write ONE entry for N
        records with the sink reading row 0 as if it were the whole prediction.
        """
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        for i in range(len(out["mask"])):
            item: SegmentationOutput[torch.Tensor] = {
                "logits": out["logits"][i],
                "probs": out["probs"][i],
                "mask": out["mask"][i],
            }
            meta = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            sink.write(item, meta)

    # ---- artifacts + tracking ----------------------------------------------- #
    def _save_checkpoint(self) -> None:
        """A plain ``state_dict`` — no framework format, because there is no framework."""
        if not self.checkpoint_dir:
            return
        directory = Path(self.checkpoint_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.pt"
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved weights to {path}")

    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot.

        Written back rather than returned fresh each call: ``flow()`` on a LIVE object returns
        that same object (idempotent), but a deferred ``Fluid`` would be REBUILT on every call —
        two MLflow runs for one training run.
        """
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    def _log_run_context(self, trackers: List[Tracker]) -> Any:
        """Provenance through marainer's plain FUNCTIONS — hookup style (1).

        There is no callback protocol to hang a ``TrackingCallback`` on, which is exactly the case
        marainer's function-level API exists for. ``runnable=self`` rather than a hand-written
        params dict: marainer derives the hyperparameters from the flowed object graph and dumps
        the resolved config, so this backend and the Lightning one cannot answer "what were the
        hyperparameters?" differently. Returns the run manifest so ``fit()`` can finish it.
        """
        if not trackers:
            return None
        from marainer.tracking.provenance import log_run_context

        return log_run_context(trackers, {"device": str(self._device)}, runnable=self)

    def _move_metrics(self) -> None:
        """Metric state lives on the same device as the logits it consumes."""
        for collection in (self.train_metrics, self.val_metrics, self.test_metrics):
            if collection is not None:
                collection.to(self._device)

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> nn.Module:
        """The live trained model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run() — the model is still deferred.")
        return self.model

    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if not self._computed_metrics:
            raise RuntimeError("computed_metrics is only available after run().")
        return dict(self._computed_metrics)

    @property
    @output
    def history(self) -> Dict[str, List[float]]:
        """Per-epoch metric history — what a ``Trainer``'s logger would have accumulated."""
        return {key: list(values) for key, values in self._history.items()}

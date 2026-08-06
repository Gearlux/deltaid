"""``MlxSegmenter`` — the same segmentation run, on Apple's MLX.

The EIGHTH and last backend, and the one whose batch path does genuinely LESS work than the
others rather than the same work differently. Everything upstream of the model is recordstream's
unchanged — the record stream, the fork guard, ``RecordSequence`` batching, the predictions sink
and contract, ``write_metrics_sidecar`` and the ``@entrypoint`` dispatch. What is written here is
:meth:`MlxSegmenter._to_arrays` (the batch shape), the step and the epoch loop. The METRICS are
not written here: the named ones are :mod:`matrainer.metric_functions`' reductions over a confusion
matrix this loop accumulates.

**Three MLX facts shape the loop**, and each is a real difference rather than a spelling one:

* **Unified memory.** CPU and GPU share it on Apple silicon, so there is NO device transfer and
  no ``.to(device)`` anywhere below — this is the only backend for which that is true.
* **Arrays are LAZY.** ``mx.eval`` is what forces computation, and omitting it does NOT raise: the
  graph grows until memory runs out, so the run just gets slower and then dies. ``fit()``
  therefore evaluates once per step over the parameters, the optimizer state AND the loss —
  deliberately and visibly.
* **Train/eval is module STATE, not a per-call argument.** ``self.model.train(bool)`` is what
  ``nn.BatchNorm`` reads, so the mode is set once per phase rather than passed per call. A
  ``training=False`` default on ``__call__`` would silently flip the network to eval on every
  forward — including the one inside the training step, where ``nn.value_and_grad`` calls
  ``model(x)`` with no flag to pass, and batch-norm would then never train.

**Channels-LAST** (``[N, H, W, C]``), which is MLX's native layout, so a MLX run's ``preprocess``
chain is the Keras / TensorFlow / JAX one — it stops before the CHW ``ToTensor``.

**Losses, metrics and optimizers are NAMES, not ``!class:`` tags.** ``mlx.nn.losses`` ships plain
functions, so there is no class for a tag to resolve or for confluid's registry to tag — the
fastai and JAX situation. Note the asymmetry with JAX though: **optax ships FACTORIES
(``optax.adam(lr)``) while ``mlx.optimizers`` ships CLASSES**, so ``_as_optimizer``
INSTANTIATES here where the JAX one CALLS.

**Apple silicon ONLY** — the extra's marker keeps ``pip install deltaid[mlx]`` succeeding (and
empty) elsewhere, so this module is importable only where MLX is.

Minimal flat YAML::

    runnable: !class:deltaid.MlxSegmenter
    model: !lazy:deltaid.MlxUNet
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess_channels_last }
    num_classes: 3
"""

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_functions import accumulate_confusion, iou_per_class, mean_iou, overall_accuracy, zero_confusion
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import merge_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, as_tracker_list
from matrainer.tracking.provenance import capture_log_file, log_run_context
from matrainer.tracking.render import per_class_vectors, render_vector_as_figure, render_vector_as_table
from recordstream import (
    PredictionsSink,
    RecordSource,
    SegmentationOutput,
    batch_metadata,
    batch_values,
    ensure_materialized,
    ensure_record_dataset,
    num_mask_classes,
)
from recordstream.keras import RecordSequence
from recordstream.runnable import ProgressReporting, entrypoint, run_entrypoint

logger = get_logger(__name__)

#: Which entry point :meth:`MlxSegmenter.run` dispatches to.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]

#: What the ``loss`` / ``metrics`` / ``optimizer`` slots accept: a NAME, or the callable itself.
#: Spelled out rather than left as ``Any`` so ``to_pydantic`` has a real type to build from.
MlxLoss = Union[str, Callable[..., Any]]
MlxMetric = Union[str, Callable[..., Any]]
MlxOptimizer = Union[str, Callable[..., Any]]

#: What the ``callbacks`` slot holds. MLX ships no callback protocol — this backend owns its loop
#: — so a callback is a plain callable invoked once per epoch with the epoch index and its logs.
EpochCallback = Callable[[int, Dict[str, float]], None]

__all__ = ["EpochCallback", "MlxLoss", "MlxMetric", "MlxOptimizer", "MlxSegmenter", "SegmenterTask"]


def _valid_mask(labels: Any, ignore_index: int) -> Any:
    """The pixels that count — everything except the void class."""
    return labels != ignore_index


#: Metric NAMES this backend resolves — each derived from the split's CONFUSION MATRIX rather than
#: computed per batch and averaged, so a number here is the number ``torchmetrics`` reports for the
#: same predictions and a cross-engine table compares models rather than aggregations.
#:
#: The arithmetic is ``matrainer.metric_functions``' and not written out here for the
#: ``metric_slots`` reason one layer up: three engines in this project need it identically. What
#: was written here before — a Python loop over the classes in ``mx`` ops — was correct FOR ONE
#: BATCH and wrong once averaged over an epoch, which is the failure that module exists to name.
_METRICS: Dict[str, Callable[[Any], Any]] = {
    "mean_iou": mean_iou,
    "pixel_accuracy": overall_accuracy,
    "iou_per_class": iou_per_class,
}

#: What a run measures when ``metrics`` is left unset.
_DEFAULT_METRICS = ("mean_iou", "pixel_accuracy")


@configurable(task="segmentation", role="trainer", framework="mlx")
class MlxSegmenter(ProgressReporting):
    """Train, evaluate or predict a semantic segmenter with MLX.

    Args:
        model: The segmentation model — wire ``!lazy:deltaid.MlxUNet`` so ``initialize()`` can
            inject the resolved ``num_classes``.
        train_set: Training source (``task="fit"``).
        val_set: Validation source, scored at the end of each epoch when wired.
        test_set: Evaluation / prediction source.
        num_classes: Class count, background included. Omitted, it is derived from the model or
            the mask column.
        ignore_index: Pixel value excluded from the loss AND the metrics. ``-100`` (torch's own
            default, kept for parity) excludes nothing, since a class-id mask never contains it.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring.
        callbacks: Callables invoked as ``callback(epoch, logs)`` after each epoch.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the per-pixel mask.
        learning_rate: LR for the optimizer.
        optimizer: An ``mlx.optimizers`` NAME (``"Adam"`` / ``"AdamW"`` / ``"SGD"``), a class, or
            an instance.
        loss: An ``mlx.nn.losses`` NAME or a callable over ``(logits, labels)``. ``None`` picks
            ``cross_entropy``, which handles the ``[N, H, W, C]`` / ``[N, H, W]`` pair natively
            and agrees with ``torch.nn.CrossEntropyLoss`` exactly (measured).
        metrics: Metric names (``"mean_iou"`` / ``"pixel_accuracy"`` / ``"iou_per_class"``) or
            callables. ``None`` picks the first two.
        train_metrics / val_metrics / test_metrics: That split's OWN metrics, merged over
            ``metrics``. All three are served: this loop is ours, so each split gets its own set —
            and the training one costs no second forward pass (see :meth:`_loss_and_grad`).
        class_names: Labels for the per-class metric report; indices otherwise.
        plot_metrics: ALSO send each per-class metric to the trackers as a matplotlib bar chart.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. This loop already writes the canonical
            ``<split>/<name>``, so the default translates nothing; wiring ``target:`` is what
            makes it record in another engine's spelling.

    Post-construction slots (YAML / broadcasting): ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nn.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        val_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        task: SegmenterTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy[EpochCallback]]] = None,
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "mask",
        learning_rate: float = 1e-3,
        optimizer: MlxOptimizer = "Adam",
        loss: Optional[MlxLoss] = None,
        metrics: Optional[Sequence[MlxMetric]] = None,
        train_metrics: Optional[Sequence[MlxMetric]] = None,
        val_metrics: Optional[Sequence[MlxMetric]] = None,
        test_metrics: Optional[Sequence[MlxMetric]] = None,
        class_names: Optional[List[str]] = None,
        plot_metrics: bool = False,
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__()
        # `nn.Module`, not `Lazy[nn.Module]`: the deferred-ness is declared once, on the ctor
        # param, which is what `lazy_param_names` / `to_pydantic` read. `nn.value_and_grad` and
        # `optimizer.update` both require a real `nn.Module`, so that IS the slot's contract once
        # `initialize()` has flowed it.
        self.model: nn.Module = cast(nn.Module, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = val_set
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
        self.learning_rate = learning_rate
        self.optimizer: MlxOptimizer = optimizer
        # Sentinels: `None` means "not chosen", never "no loss".
        self.loss: Optional[MlxLoss] = loss
        self.metrics: Optional[Sequence[MlxMetric]] = metrics
        # Per-split ADDITIONS, merged over `metrics` by the shared rule. ALL THREE are served:
        # this loop belongs to the project, so each split gets its own set. The claim this slot's
        # absence used to rest on ("the training pass computes the LOSS and nothing else") was true
        # of the loop as written, not of MLX — the step now returns its logits beside the loss, so
        # a training metric costs ONE reduction per batch.
        self.train_metrics: Optional[Sequence[MlxMetric]] = train_metrics
        self.val_metrics: Optional[Sequence[MlxMetric]] = val_metrics
        self.test_metrics: Optional[Sequence[MlxMetric]] = test_metrics
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics
        # Stored VERBATIM (lazy-init): entries may be deferred markers, flowed where used.
        self.callbacks: Optional[List[Lazy[EpochCallback]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        # Stored VERBATIM for the same reason `trackers` is: wired from YAML it arrives as a
        # deferred marker, and `as_metric_names()` builds it where the metrics are reported.
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table (recordstream docs/runnable.md).
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve ``num_classes``, then build the model at that width."""
        logger.info(f"MLX device: {mx.default_device()}")
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        self.model = flow(self.model, num_classes=self.num_classes)

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
            logger.info(f"Deriving num_classes from the {self.target_key!r} column (a full pass over the masks)...")
            nc = num_mask_classes(self.train_set, self.target_key)
            logger.info(f"Derived num_classes={nc} from the mask column.")
        self.num_classes = int(nc)

    def _initialize_eval(self) -> None:
        """Model-driven setup: the trained head's width is authoritative, not the eval split's."""
        nc = self._num_classes_without_data()
        if not nc:
            raise RuntimeError("could not determine num_classes — set it, or wire a model that exposes it.")
        self.num_classes = int(nc)
        self.test_set = self._prepare(self.test_set)
        self.val_set = self._prepare(self.val_set)

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process (the fork guard)."""
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

    # ---- the batch boundary -------------------------------------------------- #
    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        """recordstream's batching adapter, wearing THIS task's batch shape."""
        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=partial(self._to_arrays, with_targets=with_targets),
        )

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy ``(x, y)`` the step consumes.

        **Channels-LAST**, MLX's native layout, so a MLX config's chain stops after the joint
        resize. Targets are **int32**, not the int64 the torch backends name: MLX indexes with
        int32 and its ``cross_entropy`` expects it.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int32")

    # ---- the loop -------------------------------------------------------------- #
    def _loss_fn(self) -> Callable[..., Any]:
        """The resolved loss as a callable over ``(logits, labels)`` returning a scalar.

        ``ignore_index`` is applied HERE rather than passed to MLX, because ``nn.losses`` takes no
        such argument: the per-pixel loss is masked and the mean is taken over the VALID pixels
        only. Dividing by the full pixel count instead would shrink the loss in proportion to how
        much of the image is void — a silent, image-dependent learning-rate change.
        """
        loss = _as_loss(self.loss)
        ignore_index = self.ignore_index

        def compute(logits: Any, labels: Any) -> Any:
            valid = _valid_mask(labels, ignore_index)
            # The ignored label is not a valid class id, so it would index out of range inside the
            # loss; clamped to 0 first and then masked out of the mean.
            safe = mx.where(valid, labels, mx.zeros_like(labels))
            per_pixel = loss(logits, safe)
            return mx.sum(mx.where(valid, per_pixel, 0.0)) / mx.maximum(mx.sum(valid), 1)

        return compute

    def _loss_and_grad(self) -> Any:
        """The step: a loss over ``(model, x, y)``, differentiated by ``nn.value_and_grad``.

        It returns ``(loss, logits)`` and MLX differentiates the FIRST element of a returned tuple
        (verified), so the logits ride out beside the loss for free — which is what lets the
        training metrics be scored from the very tensor the gradient was taken through. Re-running
        the model to get them would cost a second forward pass AND measure a different network,
        since the module is in TRAIN mode here (dropout, batch norm).
        """
        loss_fn = self._loss_fn()

        def compute(model: Any, x: Any, y: Any) -> Any:
            logits = model(x)
            return loss_fn(logits, y), logits

        return nn.value_and_grad(self.model, compute)

    def _split_metrics(self, split: str) -> Sequence[MlxMetric]:
        """The metric list for one split — its own slot merged over the shared ``metrics``.

        The combination rule is the workspace-wide one
        (:func:`~matrainer.metric_slots.merge_metric_specs`), so a config moving between backends
        keeps its shape.
        """
        spec = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics"))
        return spec if spec is not None else list(_DEFAULT_METRICS)

    def _partition_metrics(self, split: str) -> Any:
        """One split's metrics, split into the CONFUSION-based ones and the per-batch ones.

        Partitioned by NAME, before resolution, which is why only a NAMED metric can be
        confusion-based: a user-supplied callable is scored per batch, since nothing here can know
        what it would want accumulated.
        """
        confusion: Dict[str, Callable[[Any], Any]] = {}
        batchwise: List[MlxMetric] = []
        for metric in self._split_metrics(split):
            name = flow(metric)
            if isinstance(name, str) and name in _METRICS:
                confusion[name] = _METRICS[name]
            else:
                batchwise.append(metric)
        return confusion, batchwise

    def _split_confusion(self, confusion: Dict[str, Callable[[Any], Any]]) -> Optional[np.ndarray]:
        """A zeroed accumulator when this split wants one, else ``None``."""
        if not confusion:
            return None
        if not self.num_classes:
            raise RuntimeError("MlxSegmenter: a confusion-matrix metric needs num_classes.")
        return zero_confusion(self.num_classes)

    def _metric_values(self, logits: Any, labels: Any, metrics: Sequence[MlxMetric]) -> Dict[str, Any]:
        """One batch's values for the metrics that are NOT confusion-based — user callables.

        Averaged over batches by the caller, which is the best a loop can do for a function whose
        aggregation it cannot know. Returned as numpy so they can be summed.
        """
        out: Dict[str, Any] = {}
        for metric in metrics:
            fn = _as_metric(metric)
            value = fn(logits, labels, self.ignore_index)
            mx.eval(value)  # lazy arrays: nothing has been computed until here
            out[getattr(fn, "__name__", str(metric)).lstrip("_")] = np.asarray(value)
        return out

    def _evaluate(self, sequence: RecordSequence, split: str = "val") -> Dict[str, Any]:
        """One pass in EVAL mode, returning ``{"loss": …, **metrics}``.

        The LOSS is averaged over batches, which is what a loss is. A NAMED metric is not: it
        accumulates into this split's confusion matrix and is reduced ONCE at the end, so the
        number matches what torchmetrics reports for the same predictions. Only a user-supplied
        CALLABLE is still batch-averaged.
        """
        loss_fn = self._loss_fn()
        confusion, batchwise = self._partition_metrics(split)
        matrix = self._split_confusion(confusion)
        total, batches = 0.0, 0
        sums: Dict[str, Any] = {}
        # MLX carries train/eval as module STATE, so the mode is set ONCE here rather than passed
        # per call — which is what lets this slot hold any `nn.Module`, not just one whose
        # `__call__` happens to accept a `training=` keyword. `fit()` sets it back to True at the
        # top of each epoch, so a mid-fit validation pass cannot leak eval mode.
        self.model.train(False)
        for index in range(len(sequence)):
            x, y = sequence[index]
            xa, ya = mx.array(x), mx.array(y)
            logits = self.model(xa)
            loss = loss_fn(logits, ya)
            mx.eval(loss, logits)
            total += float(loss)
            batches += 1
            if matrix is not None:
                accumulate_confusion(matrix, np.asarray(logits), np.asarray(ya), self.ignore_index)
            for key, value in self._metric_values(logits, ya, batchwise).items():
                sums[key] = value if key not in sums else sums[key] + value
        if not batches:
            return {"loss": float("nan")}
        values: Dict[str, Any] = {"loss": total / batches}
        values.update({k: v / batches for k, v in sums.items()})
        if matrix is not None:
            values.update({name: fn(matrix) for name, fn in confusion.items()})
        return values

    def _report(self, logs: Dict[str, Any], split: str = "train", step: Optional[int] = None) -> Dict[str, float]:
        """Split one epoch's values into the SCALARS a tracker takes and the vectors it does not.

        The names are canonical (``<split>/<name>``) from here rather than translated later: this
        loop is ours, so there is no engine convention to preserve — and the same prefix has to
        reach the per-class TEXT channel below, where a `val_iou_per_class` beside another
        backend's `val/iou_per_class` would be two names for one thing.

        The same split every other backend makes: ``log_metrics`` is a ``{name: float}`` channel,
        so a ``[C]`` per-class IoU goes out through ``log_text`` (and ``log_figure`` when
        ``plot_metrics`` asks) rather than being dropped. Which values are vectors is matrainer's
        rule, by SHAPE.
        """
        named = {f"{split}/{k}": v for k, v in logs.items()}
        scalars = {k: float(v) for k, v in named.items() if np.asarray(v).ndim == 0}
        trackers = self._live_trackers()
        for payload in per_class_vectors(named, self.class_names or None):
            table = render_vector_as_table(payload["name"], payload["values"], payload["class_names"])
            logger.info(table)
            figure = (
                render_vector_as_figure(payload["name"], payload["values"], payload["class_names"])
                if self.plot_metrics and trackers
                else None
            )
            for tracker in trackers:
                tracker.log_text(payload["name"], table, step=step)
                if figure is not None:
                    tracker.log_figure(payload["name"], figure, step=step)
        return scalars

    # ---- the four entry points ----------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"deltaid mlx training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("MlxSegmenter.fit() requires a 'train_set'.")
        # Provenance through matrainer's plain FUNCTIONS — `runnable=self` so the hyperparameters
        # and the resolved config come from matrainer's ONE derivation.
        trackers = self._live_trackers()
        names = as_metric_names(self.metric_names)
        manifest = log_run_context(trackers, runnable=self)
        network = self.model
        optimizer = _as_optimizer(self.optimizer, self.learning_rate)
        step = self._loss_and_grad()
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None

        # Resolved ONCE, outside the epoch loop: the partition depends only on the config, while
        # the accumulator has to be zeroed per EPOCH rather than per batch.
        train_confusion, train_batchwise = self._partition_metrics("train")

        scalars: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            network.train(True)
            total, batches = 0.0, 0
            sums: Dict[str, Any] = {}
            matrix = self._split_confusion(train_confusion)
            for index in range(len(train_seq)):
                x, y = train_seq[index]
                ya = mx.array(y)
                (loss, logits), grads = step(network, mx.array(x), ya)
                optimizer.update(network, grads)
                # MLX is LAZY: without this the graph grows every step until memory runs out, and
                # nothing warns — the run just gets slower and then dies. One eval per step, over
                # the parameters AND the optimizer state, is what forces the work. The logits join
                # it because the metrics below read them.
                mx.eval(network.parameters(), optimizer.state, loss, logits)
                total += float(loss)
                batches += 1
                # Scored from the logits the gradient was taken through, so these numbers describe
                # the network AS IT TRAINED — the module is in train mode here, so a gap against
                # the validation column is expected rather than a defect.
                if matrix is not None:
                    accumulate_confusion(matrix, np.asarray(logits), np.asarray(ya), self.ignore_index)
                for key, value in self._metric_values(logits, ya, train_batchwise).items():
                    sums[key] = value if key not in sums else sums[key] + value
            scalars = {"train/loss": total / batches if batches else float("nan")}
            if batches:
                # The SAME reduction the eval pass takes (`_evaluate`) — accumulated once per
                # epoch, not averaged per batch — so the two columns of one row are the same
                # computation on different data.
                values: Dict[str, Any] = {k: v / batches for k, v in sums.items()}
                if matrix is not None:
                    values.update({name: fn(matrix) for name, fn in train_confusion.items()})
                scalars.update(self._report(values, split="train", step=epoch))
            if val_seq is not None:
                scalars.update(self._report(self._evaluate(val_seq), split="val", step=epoch))
            # A `fit()` would have done this; a hand-written loop must. Forgetting it is
            # INVISIBLE — training silently runs on one fixed row order forever.
            train_seq.on_epoch_end()
            for callback in self.callbacks or []:
                flow(callback)(epoch, scalars)
            for tracker in trackers:
                tracker.log_metrics(names(scalars), step=epoch)
            for key, value in scalars.items():
                self._history.setdefault(key, []).append(value)
            logger.info(
                f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in scalars.items())
            )

        self._computed_metrics = names(scalars)
        logger.info("Training complete.")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        # Uploads the log file AND stamps the manifest's terminal status. Not optional once a
        # manifest exists: left unfinished it stays `running`, and matrainer's atexit hook then
        # calls a successful run `interrupted`.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("MlxSegmenter.test() requires a 'test_set'.")
        values = self._evaluate(self._sequence(self.test_set), split="test")
        self._computed_metrics = as_metric_names(self.metric_names)(self._report(values, split="test"))
        for tracker in self._live_trackers():
            tracker.log_metrics(self._computed_metrics)
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each batch with its metadata.

        Walks ``sequence.batches()`` — the COLLATED record batches — rather than the arrays,
        because the sink needs the metadata that lives beside the image and ``_to_arrays`` has
        already dropped it. One walk, so output and metadata cannot desync.
        """
        if self.test_set is None:
            raise ValueError("MlxSegmenter.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        sequence = self._sequence(self.test_set, with_targets=False)
        self.model.train(False)
        for index, batch in enumerate(sequence.batches()):
            x = sequence[index][0]
            logits = self.model(mx.array(x))
            mx.eval(logits)
            if sink is not None:
                self._write_predictions(np.asarray(logits), batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions ---------------------------------------------------------- #
    def _write_predictions(self, logits: np.ndarray, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.SegmentationOutput` per record, paired with its metadata.

        The contract is recordstream's and framework-neutral — generic in its array type, so the
        SAME sink that consumes the torch backends' output consumes this numpy one with no
        adapter. The channel axis is LAST here; the emitted ``mask`` is ``[H, W]`` either way.
        """
        probs = _softmax(logits)
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        for i in range(len(logits)):
            out: SegmentationOutput[np.ndarray] = {
                "logits": logits[i],
                "probs": probs[i],
                "mask": np.argmax(logits[i], axis=-1).astype("int64"),
            }
            meta = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            sink.write(out, meta)

    # ---- tracking -------------------------------------------------------------- #
    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot."""
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    # ---- @outputs -------------------------------------------------------------- #
    @property
    @output
    def trained_model(self) -> Any:
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
        """Per-epoch scalar history — what a `Trainer`'s logger would have accumulated."""
        return {key: list(values) for key, values in self._history.items()}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the CHANNEL axis, which is LAST here.

    On numpy rather than MLX because the arrays are already back on the host by the time the sink
    sees them — re-entering MLX for it would build a graph per batch for no gain.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))


def _as_loss(value: Optional[MlxLoss]) -> Callable[..., Any]:
    """An MLX loss from a callable or its NAME in ``mlx.nn.losses``.

    ``None`` picks ``cross_entropy``, which takes ``[..., C]`` logits against ``[...]`` integer
    labels — so the ``[N, H, W, C]`` / ``[N, H, W]`` pair needs no reshape — and agrees with
    ``torch.nn.CrossEntropyLoss`` exactly on the same arrays (measured).

    ``reduction`` is left at its default ``"none"``: the per-pixel values are what
    :meth:`MlxSegmenter._loss_fn` masks before averaging, and a loss that reduced for us would
    have already folded the ignored pixels in.

    The string form is not sugar: MLX's losses are plain FUNCTIONS, so there is no class for a
    ``!class:`` tag to name or for confluid's registry to tag.
    """
    if value is None:
        return cast(Callable[..., Any], nn.losses.cross_entropy)
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    loss = getattr(nn.losses, str(built), None)
    if loss is None:
        raise ValueError(
            f"Unknown MLX loss {built!r}. Names come from `mlx.nn.losses` — e.g. cross_entropy, "
            "binary_cross_entropy, hinge_loss."
        )
    return cast(Callable[..., Any], loss)


def _as_metric(value: MlxMetric) -> Callable[..., Any]:
    """A metric CALLABLE — the per-batch kind, over ``(logits, labels, ignore_index)``.

    A NAME in :data:`_METRICS` never reaches here: ``_partition_metrics`` routes it to the
    confusion accumulator, whose functions take a MATRIX. What is left is resolving a user-supplied
    callable and refusing an unknown name, naming the ones that exist.
    """
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    metric = _METRICS.get(str(built))
    if metric is None:
        raise ValueError(f"Unknown metric {built!r}. Known names: {', '.join(sorted(_METRICS))}.")
    return metric


def _as_optimizer(value: MlxOptimizer, learning_rate: float) -> Any:
    """An ``mlx.optimizers`` optimizer from an instance, a class, or its NAME.

    Unlike optax (FACTORIES) MLX ships CLASSES, so a name is resolved and INSTANTIATED with the
    learning rate — keeping the shared ``learning_rate`` knob the one place an LR is written.
    """
    built = flow(value)
    if isinstance(built, optim.Optimizer):
        return built
    if isinstance(built, type) and issubclass(built, optim.Optimizer):
        return built(learning_rate=learning_rate)  # type: ignore[call-arg]  # each concrete one takes it
    factory = getattr(optim, str(built), None)
    if factory is None:
        raise ValueError(f"Unknown MLX optimizer {built!r}. Names come from `mlx.optimizers` — e.g. Adam, AdamW, SGD.")
    return factory(learning_rate=learning_rate)

"""``MlxSegmenter`` — the same segmentation run, on Apple's MLX.

The EIGHTH and last backend, and the one whose batch path does genuinely LESS work than the
others rather than the same work differently. Everything upstream of the model is recordstream's
unchanged — the record stream, the fork guard, ``RecordSequence`` batching, the predictions sink
and contract, ``write_metrics_sidecar`` and the ``@entrypoint`` dispatch. What is written here is
:meth:`MlxSegmenter._to_arrays` (the batch shape), the step, the epoch loop, and three per-pixel
metrics.

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
from marainer.artifacts import write_metrics_sidecar
from marainer.metric_slots import merge_metric_specs
from marainer.run_names import generate_run_name, run_prefix
from marainer.tracking.base import Tracker, as_tracker_list
from marainer.tracking.provenance import capture_log_file, log_run_context
from marainer.tracking.render import per_class_vectors, render_vector_as_figure, render_vector_as_table
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


def _pixel_accuracy(logits: Any, labels: Any, ignore_index: int) -> Any:
    """Fraction of NON-IGNORED pixels whose argmax matches the label. Channel axis LAST."""
    valid = _valid_mask(labels, ignore_index)
    correct = (mx.argmax(logits, axis=-1) == labels) & valid
    total = mx.sum(valid)
    return mx.where(total > 0, mx.sum(correct) / mx.maximum(total, 1), mx.array(0.0))


def _mean_iou(logits: Any, labels: Any, ignore_index: int) -> Any:
    """Mean IoU over the classes PRESENT in this batch.

    Classes absent from both prediction and label are skipped rather than scored 0 — scoring them
    would drag the mean down with classes the batch never had a chance to get right, which is what
    ``MulticlassJaccardIndex`` does too.
    """
    per_class = _iou_per_class(logits, labels, ignore_index)
    present = ~mx.isnan(per_class)
    total = mx.sum(present)
    return mx.where(total > 0, mx.sum(mx.where(present, per_class, 0.0)) / mx.maximum(total, 1), mx.array(0.0))


def _iou_per_class(logits: Any, labels: Any, ignore_index: int) -> Any:
    """One IoU per class — the ``[C]`` VECTOR a segmentation report actually wants.

    An absent class scores ``nan`` rather than 0, so the per-class table says "not present in this
    batch" instead of "the model got this wrong".

    A Python loop over the classes rather than a vectorized reduction, because MLX has no ``vmap``
    over a traced closure the way JAX does and the class count is small — and nothing here is
    inside a compiled step, so the loop costs one graph node per class and no retrace.
    """
    valid = _valid_mask(labels, ignore_index)
    predicted = mx.argmax(logits, axis=-1)
    values = []
    for class_id in range(int(logits.shape[-1])):
        pred_c = (predicted == class_id) & valid
        true_c = (labels == class_id) & valid
        union = mx.sum(pred_c | true_c)
        values.append(mx.where(union > 0, mx.sum(pred_c & true_c) / mx.maximum(union, 1), mx.array(float("nan"))))
    return mx.stack(values)


#: Metric NAMES this backend resolves. Plain functions, so there is nothing for confluid to tag.
#: Each takes ``(logits, labels, ignore_index)``.
_METRICS: Dict[str, Callable[..., Any]] = {
    "mean_iou": _mean_iou,
    "pixel_accuracy": _pixel_accuracy,
    "iou_per_class": _iou_per_class,
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
        trackers: marainer trackers — one or a list. This run's ONLY output wiring.
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
        val_metrics / test_metrics: That split's OWN metrics, merged over ``metrics``. There is no
            ``train_metrics``: the training pass computes the loss and nothing else.
        class_names: Labels for the per-class metric report; indices otherwise.
        plot_metrics: ALSO send each per-class metric to the trackers as a matplotlib bar chart.

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
        val_metrics: Optional[Sequence[MlxMetric]] = None,
        test_metrics: Optional[Sequence[MlxMetric]] = None,
        class_names: Optional[List[str]] = None,
        plot_metrics: bool = False,
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
        # No `train_metrics`, and that is a property of this loop rather than an omission: the
        # training pass computes the LOSS and nothing else.
        self.val_metrics: Optional[Sequence[MlxMetric]] = val_metrics
        self.test_metrics: Optional[Sequence[MlxMetric]] = test_metrics
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics
        # Stored VERBATIM (lazy-init): entries may be deferred markers, flowed where used.
        self.callbacks: Optional[List[Lazy[EpochCallback]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
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
        """The step: a loss over ``(model, x, y)``, differentiated by ``nn.value_and_grad``."""
        loss_fn = self._loss_fn()

        def compute(model: Any, x: Any, y: Any) -> Any:
            return loss_fn(model(x), y)

        return nn.value_and_grad(self.model, compute)

    def _split_metrics(self, split: str) -> Sequence[MlxMetric]:
        """The metric list for one eval split — its own slot merged over the shared ``metrics``.

        The combination rule is the workspace-wide one
        (:func:`~marainer.metric_slots.merge_metric_specs`), so a config moving between backends
        keeps its shape.
        """
        spec = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics"))
        return spec if spec is not None else list(_DEFAULT_METRICS)

    def _metric_values(self, logits: Any, labels: Any, metrics: Sequence[MlxMetric]) -> Dict[str, Any]:
        """One batch's metric values, SCALARS AND VECTORS alike (as numpy, so they can be summed)."""
        out: Dict[str, Any] = {}
        for metric in metrics:
            fn = _as_metric(metric)
            value = fn(logits, labels, self.ignore_index)
            mx.eval(value)  # lazy arrays: nothing has been computed until here
            out[getattr(fn, "__name__", str(metric)).lstrip("_")] = np.asarray(value)
        return out

    def _evaluate(self, sequence: RecordSequence, split: str = "val") -> Dict[str, Any]:
        """One pass in EVAL mode, returning ``{"loss": …, **metrics}``."""
        loss_fn = self._loss_fn()
        metrics = self._split_metrics(split)
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
            for key, value in self._metric_values(logits, ya, metrics).items():
                sums[key] = value if key not in sums else sums[key] + value
        if not batches:
            return {"loss": float("nan")}
        return {"loss": total / batches, **{k: v / batches for k, v in sums.items()}}

    def _report(self, logs: Dict[str, Any], prefix: str = "", step: Optional[int] = None) -> Dict[str, float]:
        """Split one epoch's values into the SCALARS a tracker takes and the vectors it does not.

        The same split every other backend makes: ``log_metrics`` is a ``{name: float}`` channel,
        so a ``[C]`` per-class IoU goes out through ``log_text`` (and ``log_figure`` when
        ``plot_metrics`` asks) rather than being dropped. Which values are vectors is marainer's
        rule, by SHAPE.
        """
        named = {f"{prefix}{k}": v for k, v in logs.items()}
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
        # Provenance through marainer's plain FUNCTIONS — `runnable=self` so the hyperparameters
        # and the resolved config come from marainer's ONE derivation.
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        network = self.model
        optimizer = _as_optimizer(self.optimizer, self.learning_rate)
        step = self._loss_and_grad()
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None

        scalars: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            network.train(True)
            total, batches = 0.0, 0
            for index in range(len(train_seq)):
                x, y = train_seq[index]
                loss, grads = step(network, mx.array(x), mx.array(y))
                optimizer.update(network, grads)
                # MLX is LAZY: without this the graph grows every step until memory runs out, and
                # nothing warns — the run just gets slower and then dies. One eval per step, over
                # the parameters AND the optimizer state, is what forces the work.
                mx.eval(network.parameters(), optimizer.state, loss)
                total += float(loss)
                batches += 1
            scalars = {"loss": total / batches if batches else float("nan")}
            if val_seq is not None:
                scalars.update(self._report(self._evaluate(val_seq), prefix="val_", step=epoch))
            # A `fit()` would have done this; a hand-written loop must. Forgetting it is
            # INVISIBLE — training silently runs on one fixed row order forever.
            train_seq.on_epoch_end()
            for callback in self.callbacks or []:
                flow(callback)(epoch, scalars)
            for tracker in trackers:
                tracker.log_metrics(scalars, step=epoch)
            for key, value in scalars.items():
                self._history.setdefault(key, []).append(value)
            logger.info(
                f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in scalars.items())
            )

        self._computed_metrics = scalars
        logger.info("Training complete.")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        # Uploads the log file AND stamps the manifest's terminal status. Not optional once a
        # manifest exists: left unfinished it stays `running`, and marainer's atexit hook then
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
        self._computed_metrics = self._report(values, prefix="test_")
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
    """A metric from a callable or its NAME (``"mean_iou"`` / ``"pixel_accuracy"`` / …)."""
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

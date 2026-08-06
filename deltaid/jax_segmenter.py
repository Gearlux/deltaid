"""``JaxSegmenter`` — the same segmentation run, on a jitted ``optax`` loop.

The seventh backend, and the one that shows what "owns the loop" means when the framework hands
you nothing. Lightning, Keras, ``Trainer`` and ``Learner`` each bring an epoch loop; JAX brings
function transformations and expects you to write the loop yourself. So this is the closest
sibling of :mod:`deltaid.tensorflow_segmenter` — both replace ``fit()`` with a traced step
function — and the differences between them are exactly the differences between the two
programming models:

* **The step is a PURE FUNCTION of (model, batch).** ``nnx.value_and_grad`` differentiates it and
  ``nnx.Optimizer`` applies the update in place, which is flax's object-oriented face over what is
  underneath a functional ``(params, grads) -> params``.
* **It is jitted with ``nnx.jit``, ONCE**, and reused across epochs. A fresh ``jit`` per epoch
  would retrace the whole graph every time — the same trap the TensorFlow backend documents for
  ``tf.function``.
* **Randomness is explicit.** The model carries its own ``nnx.Rngs`` rather than reading a global
  seed, which is what makes ``seed:`` on ``FlaxUNet`` a real knob.

Everything else is shared and reused, not rewritten: the record stream, the fork guard, the
predictions sink and contract, ``write_metrics_sidecar``, and the ``@entrypoint`` dispatch. What
is written here is :meth:`JaxSegmenter._to_arrays` (the batch shape), the step functions and the
epoch loop. The METRICS are not written here either: the named ones are
:mod:`matrainer.metric_functions`' reductions over a confusion matrix this loop accumulates.

**Channels-LAST**, like the Keras and TensorFlow backends: the image arrives ``[N, H, W, C]``, so
a JAX run's ``preprocess`` chain stops after the joint albumentations resize rather than adding
the CHW ``ToTensor`` the torch chain ends with. The MASK is ``[N, H, W]`` either way.

**Losses and metrics are NAMES, not ``!class:`` tags, and that is a JAX fact.** ``optax`` losses
(``softmax_cross_entropy_with_integer_labels`` and friends) are plain *functions*, so there is no
class for a tag to resolve or for confluid's registry to tag — exactly the situation the fastai
backend's ``fastai.metrics`` names are in. Hence no ``jax_segmentation.py`` and
``FRAMEWORKS["jax"].registrations is None``.

Minimal flat YAML::

    runnable: !class:deltaid.JaxSegmenter
    model: !lazy:deltaid.FlaxUNet
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess_channels_last }
    num_classes: 3
"""

from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from flax import nnx
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

#: Which entry point :meth:`JaxSegmenter.run` dispatches to.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]

#: What the ``loss`` / ``metrics`` / ``optimizer`` slots accept: a NAME resolved against ``optax``
#: (or the built-in metric table), or the callable itself. Spelled out rather than left as ``Any``
#: so ``to_pydantic`` — form specs, MCP schemas, editor widgets — has a real type to build from.
JaxLoss = Union[str, Callable[..., Any]]
JaxMetric = Union[str, Callable[..., Any]]
JaxOptimizer = Union[str, Callable[..., Any]]

#: What the ``callbacks`` slot holds. There is no JAX callback CLASS to name — this backend owns
#: its loop, so a callback is a plain callable invoked once per epoch with the epoch index and
#: that epoch's metric logs. Spelling the signature out is what makes the slot checkable.
EpochCallback = Callable[[int, Dict[str, float]], None]

__all__ = ["EpochCallback", "JaxLoss", "JaxMetric", "JaxOptimizer", "JaxSegmenter", "SegmenterTask"]


def _valid_mask(labels: Any, ignore_index: int) -> Any:
    """The pixels that count — everything except the void class.

    The LOSS's helper. The metrics need no equivalent since they moved onto the confusion matrix:
    ``accumulate_confusion`` takes ``ignore_index`` and drops those positions before counting.
    """
    return labels != ignore_index


#: Metric NAMES this backend resolves — each derived from the split's CONFUSION MATRIX rather than
#: computed per batch and averaged, so a number here is the number ``torchmetrics`` reports for the
#: same predictions and the sweep's columns compare models rather than aggregations.
#:
#: The arithmetic is ``matrainer.metric_functions``' and not written out here for the
#: ``metric_slots`` reason one layer up: three engines in this project need it identically, and the
#: alternative is the same twenty lines copied into each. What was written here before — a jitted
#: ``jax.vmap`` per-class reduction — was correct FOR ONE BATCH and wrong once averaged over an
#: epoch, which is the failure that module exists to name.
#:
#: Plain functions, so there is nothing for confluid to tag; see the module docstring.
_METRICS: Dict[str, Callable[[Any], Any]] = {
    "mean_iou": mean_iou,
    "pixel_accuracy": overall_accuracy,
    "iou_per_class": iou_per_class,
}

#: What a run measures when ``metrics`` is left unset.
_DEFAULT_METRICS = ("mean_iou", "pixel_accuracy")


@configurable(task="segmentation", role="trainer", framework="jax")
class JaxSegmenter(ProgressReporting):
    """Train, evaluate or predict a semantic segmenter with a jitted optax loop.

    Args:
        model: The segmentation model — wire ``!lazy:deltaid.FlaxUNet`` so ``initialize()`` can
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
        callbacks: Callables invoked as ``callback(epoch, logs)`` after each epoch — JAX ships no
            callback protocol, so this is the plainest thing that works.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the per-pixel mask.
        learning_rate: Peak LR for the optimizer.
        optimizer: An ``optax`` optimizer NAME (``"adam"`` / ``"adamw"`` / ``"sgd"``) or a callable
            returning a ``GradientTransformation``.
        loss: An ``optax`` loss NAME or a callable over ``(logits, labels)``. ``None`` picks
            ``softmax_cross_entropy_with_integer_labels``, which agrees with
            ``torch.nn.CrossEntropyLoss`` to seven decimals (measured).
        metrics: Metric names (``"mean_iou"`` / ``"pixel_accuracy"`` / ``"iou_per_class"``) or
            callables. ``None`` picks the first two.
        train_metrics / val_metrics / test_metrics: That split's OWN metrics, merged over
            ``metrics``. All three are served: this loop is ours, so each split gets its own set —
            and the training one costs no second forward pass (see :meth:`_make_steps`).
        class_names: Labels for the per-class metric report; indices otherwise.
        plot_metrics: ALSO send each per-class metric to the trackers as a matplotlib bar chart.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. This loop already writes the canonical
            ``<split>/<name>``, so the default translates nothing; wiring ``target:`` is what
            makes it record in another engine's spelling.
        jit: Trace the step functions with ``nnx.jit``. On by default — that IS the JAX story.
            Turn it off to read a Python traceback from inside a step.

    Post-construction slots (YAML / broadcasting): ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nnx.Module]],
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
        optimizer: JaxOptimizer = "adam",
        loss: Optional[JaxLoss] = None,
        metrics: Optional[Sequence[JaxMetric]] = None,
        train_metrics: Optional[Sequence[JaxMetric]] = None,
        val_metrics: Optional[Sequence[JaxMetric]] = None,
        test_metrics: Optional[Sequence[JaxMetric]] = None,
        class_names: Optional[List[str]] = None,
        plot_metrics: bool = False,
        metric_names: Optional[Lazy[MetricNames]] = None,
        jit: bool = True,
    ) -> None:
        super().__init__()
        # `nnx.Module`, not `Lazy[nnx.Module]`: the deferred-ness is declared once, on the ctor
        # param, which is what `lazy_param_names` / `to_pydantic` read. `nnx.Optimizer(model, …)`
        # and `nnx.value_and_grad` both require a real `nnx.Module`, so that IS the slot's
        # contract once `initialize()` has flowed it.
        self.model: nnx.Module = cast(nnx.Module, model)
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
        self.optimizer: JaxOptimizer = optimizer
        # Sentinels: `None` means "not chosen", never "no loss". A live default could not be told
        # apart from a user's explicit choice, so an explicitly wired focal loss would be
        # silently replaced.
        self.loss: Optional[JaxLoss] = loss
        self.metrics: Optional[Sequence[JaxMetric]] = metrics
        # Per-split ADDITIONS, merged over `metrics` by the shared rule. ALL THREE are served:
        # this loop belongs to the project, so each split gets its own set. The claim this slot's
        # absence used to rest on ("the training pass computes the LOSS and nothing else") was true
        # of the loop as written, not of JAX — the step now returns its logits as an aux output, so
        # a training metric costs ONE reduction per batch.
        self.train_metrics: Optional[Sequence[JaxMetric]] = train_metrics
        self.val_metrics: Optional[Sequence[JaxMetric]] = val_metrics
        self.test_metrics: Optional[Sequence[JaxMetric]] = test_metrics
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics
        self.jit = jit
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
        logger.info(f"JAX devices: {jax.devices()}")
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
        """recordstream's batching adapter, wearing THIS task's batch shape.

        The same ``PyDataset`` the Keras backend uses, and reused rather than reimplemented for
        exactly the reason the spec exists: batching a record stream is not a JAX problem.
        """
        from functools import partial

        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=partial(self._to_arrays, with_targets=with_targets),
        )

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy ``(x, y)`` the step functions consume.

        **Channels-LAST**, so a JAX config's preprocessing chain stops after the joint resize
        rather than adding the CHW ``ToTensor`` the torch chain ends with. ``int64`` is named
        explicitly for the same reason every other backend names it — albumentations returns
        int32, and optax's integer-label loss indexes with the label.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int64")

    # ---- the traced steps ---------------------------------------------------- #
    def _loss_fn(self) -> Callable[..., Any]:
        """The resolved loss as a callable over ``(logits, labels)`` returning a scalar.

        ``ignore_index`` is applied HERE rather than passed to optax, because optax's losses take
        no such argument: the per-pixel loss is masked and the mean is taken over the VALID pixels
        only. Dividing by the full pixel count instead would shrink the loss in proportion to how
        much of the image is void — a silent, image-dependent learning-rate change.
        """
        loss = _as_loss(self.loss)
        ignore_index = self.ignore_index

        def compute(logits: Any, labels: Any) -> Any:
            valid = _valid_mask(labels, ignore_index)
            # The ignored label is not a valid class id, so it would index out of range inside the
            # loss; clamped to 0 first and then masked out of the mean.
            safe = jnp.where(valid, labels, 0)
            per_pixel = loss(logits, safe)
            return jnp.sum(jnp.where(valid, per_pixel, 0.0)) / jnp.maximum(jnp.sum(valid), 1)

        return compute

    def _make_steps(self) -> Any:
        """Build the train and eval steps once, and jit them once.

        Built once and reused across epochs on purpose: a fresh ``nnx.jit`` each epoch retraces the
        whole graph every time, which is slower than the eager loop it replaced — the same trap
        the TensorFlow backend documents for ``tf.function``.
        """
        loss_fn = self._loss_fn()

        def train_step(model: Any, optimizer: Any, x: Any, y: Any) -> Any:
            def forward(m: Any) -> Any:
                # `has_aux`: the logits come back BESIDE the loss, so the training metrics are
                # scored from the very tensor the gradient was taken through. Re-running the model
                # to get them would cost a second forward pass AND measure a different network,
                # since `training=True` leaves dropout and batch norm in train mode — a train/val
                # gap that is expected would then read as a bug.
                logits = m(x, training=True)
                return loss_fn(logits, y), logits

            (loss, logits), grads = nnx.value_and_grad(forward, has_aux=True)(model)
            optimizer.update(model, grads)
            return loss, logits

        def eval_step(model: Any, x: Any, y: Any) -> Any:
            logits = model(x, training=False)
            return loss_fn(logits, y), logits

        if self.jit:
            return nnx.jit(train_step), nnx.jit(eval_step)
        return train_step, eval_step

    # ---- metrics -------------------------------------------------------------- #
    def _split_metrics(self, split: str) -> Sequence[JaxMetric]:
        """The metric list for one eval split — its own slot merged over the shared ``metrics``.

        The combination rule is the workspace-wide one
        (:func:`~matrainer.metric_slots.merge_metric_specs`), so a config moving between backends
        keeps its shape. All THREE splits have a slot, because this loop is ours.
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
        batchwise: List[JaxMetric] = []
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
            raise RuntimeError("JaxSegmenter: a confusion-matrix metric needs num_classes.")
        return zero_confusion(self.num_classes)

    def _metric_values(self, logits: Any, labels: Any, metrics: Sequence[JaxMetric]) -> Dict[str, Any]:
        """One batch's values for the metrics that are NOT confusion-based — user callables.

        Returns the raw values rather than floats, because such a callable may return a ``[C]``
        array and coercing it here would be the drop this project already decided against. These
        are averaged over batches by the caller, which is the best a loop can do for a function
        whose aggregation it cannot know.
        """
        out: Dict[str, Any] = {}
        for metric in metrics:
            fn = _as_metric(metric)
            name = getattr(fn, "__name__", str(metric)).lstrip("_")
            out[name] = fn(logits, labels, self.ignore_index)
        return out

    def _run_epoch(self, sequence: RecordSequence, step: Any, model: Any, split: str = "val") -> Dict[str, Any]:
        """One pass over ``sequence`` in EVAL mode, returning ``{"loss": …, **metrics}``.

        The LOSS is averaged over batches, which is what a loss is. The METRICS are not: a named
        one accumulates into this split's confusion matrix and is reduced ONCE at the end, so the
        number matches what torchmetrics reports for the same predictions (see
        ``matrainer.metric_functions``). Only a user-supplied CALLABLE is still batch-averaged,
        which is why the vector arithmetic below stays element-wise.
        """
        confusion, batchwise = self._partition_metrics(split)
        matrix = self._split_confusion(confusion)
        total, batches = 0.0, 0
        sums: Dict[str, Any] = {}
        for index in range(len(sequence)):
            x, y = sequence[index]
            labels = jnp.asarray(y)
            loss, logits = step(model, jnp.asarray(x), labels)
            total += float(loss)
            batches += 1
            if matrix is not None:
                accumulate_confusion(matrix, np.asarray(logits), np.asarray(labels), self.ignore_index)
            for key, value in self._metric_values(logits, labels, batchwise).items():
                sums[key] = value if key not in sums else sums[key] + value
        if not batches:
            return {"loss": float("nan")}
        values: Dict[str, Any] = {"loss": total / batches}
        values.update({k: np.asarray(v) / batches for k, v in sums.items()})
        if matrix is not None:
            values.update({name: fn(matrix) for name, fn in confusion.items()})
        return values

    def _report(self, logs: Dict[str, Any], split: str = "train", step: Optional[int] = None) -> Dict[str, float]:
        """Split one epoch's values into the SCALARS a tracker takes and the vectors it does not.

        The names are canonical (``<split>/<name>``) from here rather than translated later: this
        loop is ours, so there is no engine convention to preserve — and the same prefix has to
        reach the per-class TEXT channel below, where a `val_iou_per_class` beside another
        backend's `val/iou_per_class` would be two names for one thing.

        The same split every other backend makes, and for the same reason: ``log_metrics`` is a
        ``{name: float}`` channel, so a ``[C]`` per-class IoU has to go out through ``log_text``
        (and ``log_figure`` when ``plot_metrics`` asks) or be dropped. Which values are vectors is
        matrainer's rule, by SHAPE — see :func:`~matrainer.tracking.render.per_class_vectors`.
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
        logger.info(f"deltaid jax training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("JaxSegmenter.fit() requires a 'train_set'.")
        # Provenance through matrainer's plain FUNCTIONS — `runnable=self` so the hyperparameters
        # and the resolved config come from matrainer's ONE derivation rather than a dict this
        # backend transcribes for itself.
        trackers = self._live_trackers()
        names = as_metric_names(self.metric_names)
        manifest = log_run_context(trackers, runnable=self)
        model = self.model
        optimizer = nnx.Optimizer(model, _as_optimizer(self.optimizer, self.learning_rate), wrt=nnx.Param)
        train_step, eval_step = self._make_steps()
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None

        # Resolved ONCE, outside the epoch loop: the partition depends only on the config, while
        # the accumulator has to be zeroed per EPOCH rather than per batch.
        train_confusion, train_batchwise = self._partition_metrics("train")

        scalars: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            total, batches = 0.0, 0
            sums: Dict[str, Any] = {}
            matrix = self._split_confusion(train_confusion)
            for index in range(len(train_seq)):
                x, y = train_seq[index]
                labels = jnp.asarray(y)
                loss, logits = train_step(model, optimizer, jnp.asarray(x), labels)
                total += float(loss)
                batches += 1
                # Scored from the logits the gradient was taken through, so these numbers describe
                # the network AS IT TRAINED — dropout and batch norm in train mode. A gap against
                # the validation column is therefore expected rather than a defect.
                if matrix is not None:
                    accumulate_confusion(matrix, np.asarray(logits), np.asarray(labels), self.ignore_index)
                for key, value in self._metric_values(logits, labels, train_batchwise).items():
                    sums[key] = value if key not in sums else sums[key] + value
            scalars = {"train/loss": total / batches if batches else float("nan")}
            if batches:
                # The SAME reduction the eval pass takes (`_run_epoch`) — accumulated once per
                # epoch, not averaged per batch — so the two columns of one row are the same
                # computation on different data.
                values: Dict[str, Any] = {k: np.asarray(v) / batches for k, v in sums.items()}
                if matrix is not None:
                    values.update({name: fn(matrix) for name, fn in train_confusion.items()})
                scalars.update(self._report(values, split="train", step=epoch))
            if val_seq is not None:
                scalars.update(self._report(self._run_epoch(val_seq, eval_step, model), split="val", step=epoch))
            # A `fit()` would have done this for us; a hand-written loop has to. Forgetting it is
            # INVISIBLE — training just runs on one fixed row order forever.
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
            raise ValueError("JaxSegmenter.test() requires a 'test_set'.")
        _, eval_step = self._make_steps()
        values = self._run_epoch(self._sequence(self.test_set), eval_step, self.model, split="test")
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
            raise ValueError("JaxSegmenter.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            x = sequence[index][0]
            logits = np.asarray(self.model(jnp.asarray(x), training=False))
            if sink is not None:
                self._write_predictions(logits, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions ---------------------------------------------------------- #
    def _write_predictions(self, logits: np.ndarray, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.SegmentationOutput` per record, paired with its metadata.

        The contract is recordstream's and framework-neutral — generic in its array type, so the
        SAME sink that consumes the torch backends' output consumes this numpy one with no
        adapter. The channel axis is LAST here, so the softmax and argmax run over ``-1``; the
        emitted ``mask`` is ``[H, W]`` either way.
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
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot.

        Written back rather than returned fresh each call: ``flow()`` on a LIVE object returns that
        same object (idempotent), but a deferred ``Fluid`` would be REBUILT on every call — two
        MLflow runs for one training run.
        """
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

    Written out rather than taken from ``jax.nn`` because this runs on numpy, after the arrays are
    already back on the host — re-entering JAX for it would trace a graph per batch for no gain.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))


def _as_loss(value: Optional[JaxLoss]) -> Callable[..., Any]:
    """An optax loss from a callable or its NAME in ``optax.losses``.

    ``None`` picks ``softmax_cross_entropy_with_integer_labels``, which is the per-pixel twin of
    ``torch.nn.CrossEntropyLoss`` — it takes ``[..., C]`` logits against ``[...]`` integer labels,
    so the ``[N, H, W, C]`` / ``[N, H, W]`` pair needs no reshape, and it agrees with torch to
    seven decimals on the same arrays (measured).

    The string form is not sugar: optax's losses are plain FUNCTIONS, so there is no class for a
    ``!class:`` tag to name or for confluid's registry to tag.
    """
    if value is None:
        return cast(Callable[..., Any], optax.losses.softmax_cross_entropy_with_integer_labels)
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    loss = getattr(optax.losses, str(built), None) or getattr(optax, str(built), None)
    if loss is None:
        raise ValueError(
            f"Unknown optax loss {built!r}. Names come from `optax.losses` — e.g. "
            "softmax_cross_entropy_with_integer_labels, sigmoid_binary_cross_entropy."
        )
    return cast(Callable[..., Any], loss)


def _as_metric(value: JaxMetric) -> Callable[..., Any]:
    """A metric CALLABLE — the per-batch kind, over ``(logits, labels, ignore_index)``.

    A NAME in :data:`_METRICS` never reaches here: ``_partition_metrics`` routes it to the
    confusion accumulator, whose functions take a MATRIX and have a different signature entirely.
    What is left for this to do is resolve a user-supplied callable and refuse an unknown name,
    naming the ones that exist — which is why the lookup stays.
    """
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    metric = _METRICS.get(str(built))
    if metric is None:
        raise ValueError(f"Unknown metric {built!r}. Known names: {', '.join(sorted(_METRICS))}.")
    return metric


def _as_optimizer(value: JaxOptimizer, learning_rate: float) -> Any:
    """An ``optax.GradientTransformation`` from a callable or its NAME in ``optax``.

    A NAME is resolved and CALLED with the learning rate, because optax optimizers are FACTORIES
    (``optax.adam(1e-3)``) rather than classes — so the ``learning_rate`` knob stays the single
    place the LR is written, shared with every other backend's config.
    """
    built = flow(value)
    if callable(built) and not isinstance(built, str):
        return built(learning_rate) if not hasattr(built, "update") else built
    factory = getattr(optax, str(built), None)
    if factory is None:
        raise ValueError(f"Unknown optax optimizer {built!r}. Names come from `optax` — e.g. adam, adamw, sgd.")
    return factory(learning_rate)

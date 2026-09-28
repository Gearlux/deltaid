"""``JaxSegmenter`` — the same segmentation run, on a jitted ``optax`` loop.

Since P6 (2026-08-08) this is ``(SegmentationBase, matrainer.jax.JaxLoopRunnable)``: the loop
machinery — the jit-ONCE step builders (``nnx.value_and_grad`` with the logits as an aux
output), the epoch driver with the per-epoch reshuffle, the plain-function tracking hookup,
the per-epoch callback fan-out, the record-batch predict walk — lives in
:class:`matrainer.jax.loop.JaxLoopRunnable`; the ``num_classes`` resolution (model head →
explicit → lazy kwarg → the announced mask walk) in
:class:`~deltaid.segmentation_base.SegmentationBase`. What stays HERE is segmentation's jax
vocabulary:

* **``ignore_index`` is applied in :meth:`_loss_fn`**, because optax's losses take no such
  argument: the per-pixel loss is masked and the mean taken over the VALID pixels only, with
  the ignored label CLAMPED to 0 first (it is not a valid class id and would index out of
  range). Dividing by the full pixel count instead would shrink the loss in proportion to how
  much of the image is void — a silent, image-dependent learning-rate change.
* **The named metrics are reductions over the split's CONFUSION MATRIX**
  (:mod:`matrainer.metric_functions` — accumulated once per pass, never batch-averaged), with
  the per-class VECTOR routed to the trackers' text channel through the fleet's renderers.
* **Channels-LAST** — the mask reads int64 ``[N, H, W]``, and only the image line of a
  config's preprocess chain differs from torch's.

Losses and metrics are NAMES, not ``!class:`` tags, and that is a JAX fact — hence no
``jax_segmentation.py`` and ``FRAMEWORKS["jax"].registrations is None``.
"""

from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import jax.numpy as jnp
import numpy as np
from confluid import Mandatory, Partial, configurable, flow
from flax import nnx
from loggair import get_logger
from matrainer.jax.loop import EpochCallback, JaxLoopRunnable, JaxLoss, JaxMetric, JaxOptimizer, as_optax_loss
from matrainer.metric_functions import accumulate_confusion, iou_per_class, mean_iou, overall_accuracy, zero_confusion
from matrainer.metric_names import MetricNames
from matrainer.metric_slots import merge_metric_specs
from matrainer.tracking.base import Tracker
from matrainer.tracking.render import per_class_vectors, render_vector_as_figure, render_vector_as_table
from recordstream import RecordSource, SegmentationOutput, batch_values

from deltaid.segmentation_base import SegmentationBase

logger = get_logger(__name__)

#: Which entry point :meth:`JaxSegmenter.run` dispatches to.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]

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
#: Plain functions, so there is nothing for confluid to tag; see the module docstring.
_METRICS: Dict[str, Callable[[Any], Any]] = {
    "mean_iou": mean_iou,
    "pixel_accuracy": overall_accuracy,
    "iou_per_class": iou_per_class,
}

#: What a run measures when ``metrics`` is left unset.
_DEFAULT_METRICS = ("mean_iou", "pixel_accuracy")


@configurable(task="segmentation", role="trainer", framework="jax")
class JaxSegmenter(SegmentationBase, JaxLoopRunnable):
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
            and the training one costs no second forward pass (the step's aux output).
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
        model: Mandatory[Partial[nnx.Module]],
        train_set: Optional[Partial[RecordSource]] = None,
        val_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        task: SegmenterTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        callbacks: Optional[List[Partial[EpochCallback]]] = None,
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
        metric_names: Optional[Partial[MetricNames]] = None,
        jit: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            jit=jit,
            loss=loss,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metric_names=metric_names,
        )
        self.val_set = val_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics

    # ---- setup: the eval path injects the width BEFORE the flow ------------------------------ #
    def _initialize_eval(self) -> None:
        """Model-driven where a live head exists; a DEFERRED spec model takes the segmenter's count.

        The task base's eval path flows the model FIRST and reads the head — right for a
        loaded checkpoint, fatal for a ``!lazy:`` U-Net whose build needs the width (its
        ``solidify`` raises without it, and flowing plainly IS that build). So this engine's
        eval resolves the count first (a live model's head still wins inside the ladder) and
        injects it into the flow, exactly as the fit path does.
        """
        nc = self._num_classes_without_data()
        if not nc:
            raise RuntimeError("could not determine num_classes — set it, or wire a model that exposes it.")
        self.num_classes = int(nc)
        self.model = flow(self.model, num_classes=self.num_classes)
        self._resolve_task_defaults()
        eval_slot = "val_set" if self.task == "evaluate" else "test_set"
        setattr(self, eval_slot, self._prepare(getattr(self, eval_slot)))
        self._build_eval_metrics()

    # ---- the task hooks ---------------------------------------------------------------------- #
    def _default_jax_loss(self) -> Optional[JaxLoss]:
        """The per-pixel twin of ``torch.nn.CrossEntropyLoss`` — it takes ``[..., C]`` logits
        against ``[...]`` integer labels, so the ``[N, H, W, C]`` / ``[N, H, W]`` pair needs no
        reshape, and it agrees with torch to seven decimals on the same arrays (measured)."""
        return "softmax_cross_entropy_with_integer_labels"

    def _default_jax_metrics(self) -> Optional[List[Any]]:
        return list(_DEFAULT_METRICS)

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """float32 image, int64 ``[N, H, W]`` mask — int64 named explicitly for the same reason
        the torch backends name it (albumentations returns int32). Channels-LAST, so a JAX
        config's preprocess chain stops after the joint albumentations resize."""
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int64")

    def _loss_fn(self) -> Callable[..., Any]:
        """The resolved loss with the ``ignore_index`` mask — see the module docstring.

        The sentinel is asked through the task hook when still unresolved, so a
        directly-driven instance works without ``initialize()``.
        """
        chosen = self.loss if self.loss is not None else self._default_jax_loss()
        assert chosen is not None, "the task hook always answers for this task"
        loss = as_optax_loss(chosen)
        ignore_index = self.ignore_index

        def compute(logits: Any, labels: Any) -> Any:
            valid = _valid_mask(labels, ignore_index)
            # The ignored label is not a valid class id, so it would index out of range inside
            # the loss; clamped to 0 first and then masked out of the mean.
            safe = jnp.where(valid, labels, 0)
            per_pixel = loss(logits, safe)
            return jnp.sum(jnp.where(valid, per_pixel, 0.0)) / jnp.maximum(jnp.sum(valid), 1)

        return compute

    # ---- the metric state: batchwise sums + ONE confusion matrix per split ------------------- #
    def _partition_metrics(self, split: str) -> Any:
        """One split's metrics, split into the CONFUSION-based ones and the per-batch callables.

        By NAME, before resolution, which is why only a NAMED metric can be confusion-based:
        a user-supplied callable keeps the ``(logits, labels, ignore_index)`` protocol and its
        per-batch mean, since nothing here can know what it would want accumulated.
        """
        confusion: Dict[str, Callable[[Any], Any]] = {}
        batchwise: List[JaxMetric] = []
        spec = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics"))
        for metric in spec if spec is not None else list(_DEFAULT_METRICS):
            name = flow(metric)
            if isinstance(name, str) and name in _METRICS:
                confusion[name] = _METRICS[name]
            else:
                batchwise.append(metric)
        return confusion, batchwise

    def _new_split_state(self, split: str) -> Any:
        confusion, batchwise = self._partition_metrics(split)
        matrix = None
        if confusion:
            if not self.num_classes:
                raise RuntimeError("JaxSegmenter: a confusion-matrix metric needs num_classes.")
            matrix = zero_confusion(self.num_classes)
        return {"confusion": confusion, "batchwise": batchwise, "matrix": matrix, "sums": {}, "batches": 0}

    def _score_batch(self, state: Any, outputs: Any, arrays: Tuple[Any, ...], split: str) -> None:
        labels = arrays[1]
        state["batches"] += 1
        if state["matrix"] is not None:
            accumulate_confusion(state["matrix"], np.asarray(outputs), np.asarray(labels), self.ignore_index)
        for metric in state["batchwise"]:
            fn = _as_metric(metric)
            name = getattr(fn, "__name__", str(metric)).lstrip("_")
            value = fn(outputs, labels, self.ignore_index)
            state["sums"][name] = value if name not in state["sums"] else state["sums"][name] + value

    def _split_scores(self, state: Any, split: str, step: Optional[int] = None) -> Dict[str, float]:
        """Reduce once per pass, then route: scalars back to the driver, VECTORS to the text
        channel (worst-class-first table + optional bar chart) under the canonical name."""
        if not state["batches"]:
            return {}
        values: Dict[str, Any] = {k: np.asarray(v) / state["batches"] for k, v in state["sums"].items()}
        if state["matrix"] is not None:
            values.update({name: fn(state["matrix"]) for name, fn in state["confusion"].items()})
        return self._report_vectors(values, split, step)

    def _report_vectors(self, values: Dict[str, Any], split: str, step: Optional[int]) -> Dict[str, float]:
        """The fleet rule: ``log_metrics`` is a ``{name: float}`` channel, so a ``[C]``
        per-class IoU goes out through ``log_text`` (and ``log_figure`` when asked) or is
        dropped. Which values are vectors is matrainer's rule, by SHAPE."""
        named = {f"{split}/{k}": v for k, v in values.items()}
        scalars = {k.split("/", 1)[1]: float(v) for k, v in named.items() if np.asarray(v).ndim == 0}
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

    # ---- predictions -------------------------------------------------------------------------- #
    def _prediction_output(self, preds: Any) -> Any:
        """The batched :class:`~recordstream.SegmentationOutput` — softmax/argmax over the LAST
        axis (channels-last), the mask int64; numpy, so the SAME sink that consumes the torch
        backends' output consumes it with no adapter. The machinery slices per record."""
        logits = np.asarray(preds)
        out: SegmentationOutput[np.ndarray] = {
            "logits": logits,
            "probs": _softmax(logits),
            "mask": np.argmax(logits, axis=-1).astype("int64"),
        }
        return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the CHANNEL axis, which is LAST here.

    Written out rather than taken from ``jax.nn`` because this runs on numpy, after the arrays
    are already back on the host — re-entering JAX for it would trace a graph per batch for no
    gain.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))


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

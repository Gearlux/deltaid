"""``TensorFlowSegmenter`` — the same segmentation run on a native TensorFlow loop.

The fair question this backend answers: why, when ``KERAS_BACKEND=tensorflow deltaid train
--framework keras`` already runs the identical config? Because ``tf.data``, ``tf.GradientTape``
under ``@tf.function`` (which is what makes ``jit_compile``/XLA a knob at all) and the
SavedModel export are TensorFlow's and not Keras's. It is a SIBLING of
:class:`~deltaid.keras_segmenter.KerasSegmenter`, not a subclass — it narrows a precondition
its would-be parent does not have (the engine guard) — and still declares ``framework="keras"``:
the axis names the API a class is WIRED TO.

Since P5 (2026-08-07) this is ``(SegmentationKerasHooks, matrainer.tensorflow.
TensorFlowRunnable)``: the loop machinery — the ``tf.data`` pipeline, the traced steps, the
hand-driven ``CallbackList``, the per-split metric objects (MERGED per split and CLONED per
instance — the narrowing the compile/fit sibling cannot do), the engine guard, the SavedModel
``_export`` — lives in :class:`matrainer.tensorflow.loop.TensorFlowRunnable`; the task half in
:class:`deltaid.segmentation_base.SegmentationBase`; and the task's keras vocabulary (the
``from_logits=True`` default loss, the ``num_classes``-sized defaults, the int64 mask read,
the channels-last prediction contract) in
:class:`deltaid.segmentation_keras.SegmentationKerasHooks` — shared verbatim with the
compile/fit sibling, which is the point. What stays here is the FULL flat constructor.

One normalization worth knowing: the export signature is now DISCOVERED (the model's declared
``input_shape``, else a measured batch) where this module used to hard-code ``channels = 3``;
and ``evaluate()`` scores the VAL split (the fleet harmonization).
"""

from typing import Iterable, List, Literal, Optional, Union

from confluid import Mandatory, Partial, configurable
from loggair import get_logger
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker
from recordstream import RecordSource

from deltaid._tensorflow import keras
from deltaid.segmentation_keras import SegmentationKerasHooks

# AFTER the `_tensorflow` shim import: the shim setdefaults KERAS_BACKEND before anything reads
# it (the matrainer module does the same itself — belt-and-braces, and it keeps this module's
# import story identical to every other deltaid TF-touching module).
from matrainer.tensorflow.loop import TensorFlowNetwork, TensorFlowRunnable  # isort: skip

logger = get_logger(__name__)

#: Which entry point :meth:`TensorFlowSegmenter.run` dispatches to.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]

#: A Keras loss / metric / optimizer: a live object, a string alias, or a deferred marker.
KerasLoss = Union[str, keras.losses.Loss]
KerasMetric = Union[str, keras.metrics.Metric]
KerasOptimizer = Union[str, keras.optimizers.Optimizer]

__all__ = [
    "KerasLoss",
    "KerasMetric",
    "KerasOptimizer",
    "SegmenterTask",
    "TensorFlowNetwork",
    "TensorFlowSegmenter",
]


@configurable(task="segmentation", role="trainer", framework="keras")
class TensorFlowSegmenter(SegmentationKerasHooks, TensorFlowRunnable):
    """Train, evaluate or predict a semantic segmenter with a native TensorFlow loop.

    Args:
        model: A ``keras.Model`` or raw ``tf.Module`` emitting ``[N, H, W, C]`` logits —
            channels LAST. Wire ``!lazy:deltaid.TensorFlowUNet`` (or ``KerasUNet`` — ``tf.keras``
            IS Keras 3).
        train_set: Training source (``task="fit"``).
        test_set: Evaluation / prediction source.
        num_classes: Class count, background included. Omitted, it is derived from the model or
            the mask column.
        ignore_index: Pixel value excluded from the default metrics (Keras's ``ignore_class``).
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        jit_compile: Compile the traced steps with XLA.
        prefetch: ``dataset.prefetch(AUTOTUNE)`` between the sequence and the loop.
        export_dir: Where ``fit()`` writes the SavedModel. ``None`` skips the export.
        trackers: matrainer trackers — one or a list.
        callbacks: Keras callbacks, driven by hand through a ``CallbackList``.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the per-pixel mask.
        optimizer / loss: Keras objects, or their string aliases.
        metrics: Shared metrics; ``train_metrics`` / ``val_metrics`` / ``test_metrics`` are that
            split's OWN, merged over ``metrics``. Unlike the Keras backend this loop CAN narrow
            per split — each split gets its own metric OBJECTS, because they are stateful.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Partial[TensorFlowNetwork]],
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        task: SegmenterTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        jit_compile: bool = False,
        prefetch: bool = True,
        export_dir: Optional[str] = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        callbacks: Optional[List[Partial[keras.callbacks.Callback]]] = None,
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "mask",
        optimizer: Partial[KerasOptimizer] = "adam",
        loss: Optional[Partial[KerasLoss]] = None,
        metrics: Optional[List[Partial[KerasMetric]]] = None,
        train_metrics: Optional[List[Partial[KerasMetric]]] = None,
        val_metrics: Optional[List[Partial[KerasMetric]]] = None,
        test_metrics: Optional[List[Partial[KerasMetric]]] = None,
        metric_names: Optional[Partial[MetricNames]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            jit_compile=jit_compile,
            prefetch=prefetch,
            export_dir=export_dir,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)

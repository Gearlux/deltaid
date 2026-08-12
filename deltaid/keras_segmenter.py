"""``KerasSegmenter`` — the same segmentation run, on Keras 3 instead of Lightning.

Since P5 (2026-08-07) this is ``(SegmentationKerasHooks, matrainer.keras.KerasRunnable)``: the
compile/fit machinery — build + compile, the ``History`` read-back, the entry points, the
``RecordSequence`` batch boundary, the tracking callback — lives in
:class:`matrainer.keras.trainer.KerasRunnable`; the task half (``num_classes`` resolution, the
fit/eval initialization split, the announced mask walk) in
:class:`deltaid.segmentation_base.SegmentationBase`; and the task's KERAS VOCABULARY (the
``from_logits=True`` default loss, the ``num_classes``-sized MeanIoU + pixel-accuracy
defaults, the int64 mask read, the channels-last prediction contract, the MeanIoU-on-torch
warning) in :class:`deltaid.segmentation_keras.SegmentationKerasHooks` — written once for this
concrete AND its native-TensorFlow sibling. What stays here is the FULL flat constructor (the
confluid introspection surface).

**TWO real differences from the torch backends, both about the ENGINE and both living in the
config where a reader can see them:**

1. **Keras is channels-LAST.** The image arrives ``[N, H, W, C]``, so the ``preprocess`` chain
   for a Keras run simply STOPS after the joint albumentations resize rather than adding the
   CHW ``ToTensor`` the torch chain ends with. The MASK is ``[N, H, W]`` either way.
2. **`from_logits=True` is mandatory on the loss** — see the hooks module for the measurement.

``metrics`` / ``train_metrics`` / ``val_metrics`` / ``test_metrics`` are UNIONED (the family's
``_compiled_metrics``): ``keras.Model.compile`` takes ONE list — see the machinery base.

Keras 3 picks its engine from ``KERAS_BACKEND``; this module never sets it, so
``deltaid train --framework keras`` runs on whichever the operator chose.

Minimal flat YAML::

    runnable: !class:deltaid.KerasSegmenter
    model: !lazy:deltaid.KerasUNet
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess_keras }
    num_classes: 3
"""

from typing import Iterable, List, Literal, Optional, Union

from confluid import Mandatory, Partial, configurable
from loggair import get_logger
from matrainer.keras.runnable import KerasLoss, KerasMetric, KerasOptimizer
from matrainer.keras.trainer import KerasRunnable
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker
from recordstream import RecordSource
from recordstream.keras import keras

from deltaid.segmentation_keras import SegmentationKerasHooks

logger = get_logger(__name__)

#: Which entry point :meth:`KerasSegmenter.run` dispatches to — the same vocabulary every other
#: backend uses, because the verb is a property of the TASK, not of the engine.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]

__all__ = ["KerasLoss", "KerasMetric", "KerasOptimizer", "KerasSegmenter", "SegmenterTask"]


@configurable(task="segmentation", role="trainer", framework="keras")
class KerasSegmenter(SegmentationKerasHooks, KerasRunnable):
    """Train, evaluate or predict a semantic segmenter with Keras 3.

    Args:
        model: The segmentation model — wire ``!lazy:`` so ``initialize()`` can inject the
            resolved ``num_classes``.
        train_set: Training source (``task="fit"``).
        test_set: Evaluation / prediction source.
        num_classes: Class count, background included. Omitted, it is derived from the model or
            the mask column.
        ignore_index: Pixel value excluded from the default metrics (Keras's ``ignore_class``).
            The default ``-100`` matches the torch backends and excludes nothing, since a
            class-id mask never contains it.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring.
        callbacks: ``keras.callbacks.Callback`` instances. A file-writing one gets its parent
            directory created, because Keras opens the path without doing so.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the per-pixel mask.
        optimizer: Keras optimizer — a string alias (``"adam"``) or an instance.
        loss: Keras loss. ``None`` picks ``SparseCategoricalCrossentropy(from_logits=True)``.
        metrics / train_metrics / val_metrics / test_metrics: Keras metrics. This engine
            computes their UNION — see the machinery base.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. Keras writes ``loss`` / ``val_loss``;
            the default translates that to the canonical ``train/loss`` / ``val/loss`` on the way
            to the trackers and the sidecar. What Keras hands its OWN callbacks is untouched —
            a wired ``EarlyStopping(monitor="val_loss")`` must keep finding that name.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Partial[keras.Model]],
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        task: SegmenterTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
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

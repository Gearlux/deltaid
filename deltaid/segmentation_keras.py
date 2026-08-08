"""``SegmentationKerasHooks`` — the segmentation task's KERAS-family hook implementations.

The keras twin of :mod:`deltaid.segmentation_torch`: the task hooks written once for BOTH
Keras-family machinery bases — the compile/fit member (:class:`matrainer.keras.KerasRunnable`)
and the native-TensorFlow loop (:class:`matrainer.tensorflow.TensorFlowRunnable`) — so the two
concretes cannot drift apart on what per-pixel classification means in the keras vocabulary. A
concrete backend composes ``(SegmentationKerasHooks, <Machinery>Runnable)``.

What the task says, in this vocabulary:

* **The default loss is ``SparseCategoricalCrossentropy(from_logits=True)``** — and
  ``from_logits`` is not optional: the head is a bare 1x1 conv emitting raw logits, and
  Keras's default expects probabilities — handed logits it normalizes by the row sum and CLIPS
  rather than raising (measured: 1.4546 correct vs 4.6293 wrong on one batch — finite, and
  simply wrong).
* **The default metrics need ``num_classes``** — ``MeanIoU(num_classes=…, sparse_y_pred=False,
  ignore_class=…)`` (``sparse_y_pred=False`` because the predictions are raw logits, not
  argmaxed ids — a shape error otherwise, measured) plus pixel accuracy. The hook answers
  ``None`` until the task's ``initialize`` has resolved the count; sentinel resolution is
  idempotent and asks again.
* **The mask reads int64** — albumentations returns int32 (measured), and the batch shape is
  ``[N, H, W]`` in every engine (a class-id map has no channel axis), so only the image line
  of a config's preprocess chain differs from torch's.
* **A prediction is the :class:`~recordstream.SegmentationOutput` contract** — softmax and
  argmax over the LAST axis (Keras is channels-last) where the torch backends use dim 1; the
  emitted ``mask`` is ``[H, W]`` either way.

Plus one warning that belongs to the compile/fit member: :meth:`_check_compiled_metrics` says
so when the IoU family meets the TORCH engine, because the run will look HUNG rather than slow
(``keras.ops.scatter`` is a Python loop there — ~160 s per 8x224x224 batch against 0.009 s on
TensorFlow, measured; per-pixel, the scatter's index count IS the pixel count). The
native-TensorFlow sibling never needs it — it pins the engine.
"""

from typing import Any, List, Optional, Sequence, cast

import numpy as np
from confluid import Lazy
from loggair import get_logger
from matrainer.keras.runnable import KerasLoss, KerasMetric
from recordstream import SegmentationOutput, batch_values
from recordstream.keras import keras, keras_backend

from deltaid.segmentation_base import SegmentationBase

logger = get_logger(__name__)

__all__ = ["SegmentationKerasHooks"]


class SegmentationKerasHooks(SegmentationBase):
    """The keras-typed segmentation hooks, shared by the Keras and TensorFlow concretes."""

    def _default_keras_loss(self) -> Optional[Lazy[KerasLoss]]:
        return keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def _default_keras_metrics(self) -> Optional[List[Lazy[KerasMetric]]]:
        if not self.num_classes:
            return None  # asked again once initialize() has resolved the count
        # `ignore_class` is Keras's `ignore_index`. `None` when nothing is excluded, because
        # Keras treats the sentinel as a real class id rather than as "no exclusion".
        ignore = self.ignore_index if self.ignore_index >= 0 else None
        return [
            keras.metrics.MeanIoU(
                num_classes=self.num_classes, sparse_y_pred=False, ignore_class=ignore, name="mean_iou"
            ),
            keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy"),
        ]

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """float32 image, int64 ``[N, H, W]`` mask — int64 named explicitly for the same reason
        the torch backends name it (albumentations returns int32)."""
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int64")

    def _check_compiled_metrics(self, metrics: Sequence[Any]) -> None:
        """Say so when the IoU family meets the torch engine, because the run will look HUNG.

        Nothing to fix in this project — the metric is upstream's and correct, and the engine
        is the operator's choice — so this warns instead of substituting a different metric: a
        default that silently changed with the engine would break the one claim the project
        makes, that the same config gives the same run.
        """
        if keras_backend() != "torch":
            return
        slow = [m for m in metrics if isinstance(m, (keras.metrics.MeanIoU, keras.metrics.IoU))]
        if not slow:
            return
        logger.warning(
            f"{', '.join(type(m).__name__ for m in slow)} on the TORCH engine will make this run "
            f"appear to hang: keras.ops.scatter is a Python loop there, ~160 s per 8x224x224 batch "
            f"against 0.009 s on TensorFlow (measured). Re-run with KERAS_BACKEND=tensorflow (or "
            f"jax), or drop the IoU metric and keep pixel accuracy."
        )

    def _prediction_output(self, preds: Any) -> Any:
        """The batched :class:`~recordstream.SegmentationOutput` — softmax/argmax over the LAST
        axis (channels-last), the mask int64. Framework-neutral, so the same sink that consumes
        the torch backends' output consumes this numpy one; the machinery slices per record."""
        logits = np.asarray(preds)
        out: SegmentationOutput[np.ndarray] = {
            "logits": logits,
            "probs": _softmax(logits),
            "mask": np.argmax(logits, axis=-1).astype("int64"),
        }
        return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the CHANNEL axis, which is LAST in Keras.

    Written out rather than taken from the engine because this runs on numpy, after
    ``model.predict`` has already handed the arrays back — reaching for a backend op here would
    re-enter whichever engine Keras is bound to for no gain.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))

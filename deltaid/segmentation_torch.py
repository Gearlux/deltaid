"""``SegmentationTorchHooks`` — the torch-family implementations of the task hooks.

The segmentation answers to the trainer API's typed hooks
(:class:`matrainer.torch.runnable.TorchTrainerRunnable`): the int64 mask read, the
``ignore_index``-aware loss + metric defaults, the ``num_classes`` sizing keyword, the
prediction contract, and the per-class vector reporting. ONE module serves every torch-family
backend; the torch-free task logic lives in :mod:`deltaid.segmentation_base`, and a concrete
backend composes ``(SegmentationTorchHooks, <Framework>Runnable)``.

This module imports torch, so a keras/jax/mlx backend must never import it — the same rule as
:mod:`deltaid.torch_segmentation` (the registered loss/metric set, a DIFFERENT module: that
one registers upstream classes, this one implements hooks).
"""

from typing import Any, Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn
from confluid import PartialClass
from loggair import get_logger
from matrainer.torch.metrics import MetricsSpec
from matrainer.torch.runnable import TorchTrainerRunnable
from matrainer.tracking.base import Tracker
from matrainer.tracking.render import report_vectors
from recordstream import batch_tensor, segmentation_output
from recordstream.items import Record
from recordstream.outputs import SegmentationOutput
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from deltaid.segmentation_base import SegmentationBase

logger = get_logger(__name__)

__all__ = ["SegmentationTorchHooks"]


class SegmentationTorchHooks(SegmentationBase, TorchTrainerRunnable):
    """The torch half of the segmentation task, mixed into every torch-family backend."""

    def _targets(self, batch: Record) -> Tensor:
        """The per-pixel target tensor the loss expects — an ``[N, H, W]`` int64 class-id map.

        Both the reader and the dtype are recordstream's; what is segmentation's is naming
        ``int64``. It is not decoration: albumentations returns an int32 mask (measured), and
        ``CrossEntropyLoss`` rejects it with *"expected target dtype to be Long or Byte, but
        got Int"*. ``batch_tensor`` honours the contract the caller names rather than
        deciding one.
        """
        return batch_tensor(batch, self.target_key, self.device, dtype=torch.int64)

    def _default_loss(self) -> nn.Module:
        # Per-pixel softmax over the channel dim — the standard segmentation loss, with the
        # void class excluded from BOTH the loss and the metrics by the same knob.
        return nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def _default_metrics(self) -> MetricsSpec:
        if not self.num_classes:
            return None  # not knowable yet — the idempotent resolution asks again later
        # The segmentation pair: mean IoU (the field's primary score) and macro Dice — the
        # same statistic as macro F1, which is what torchmetrics calls it.
        return {
            # `mean_iou`, not `miou`: the JAX / MLX / HuggingFace backends name their metrics
            # from their own function registry and cannot say `miou`, so a different word here
            # would put one quantity in two columns of the same cross-backend comparison.
            "mean_iou": PartialClass[Metric](MulticlassJaccardIndex, average="macro", ignore_index=self.ignore_index),
            "dice_macro": PartialClass[Metric](
                MulticlassF1Score, average="macro", ignore_index=self.ignore_index, multidim_average="global"
            ),
        }

    def _metric_flow_dims(self) -> Dict[str, Union[int, float]]:
        assert self.num_classes is not None
        return {"num_classes": self.num_classes}

    def _prediction_output(self, preds: Tensor) -> SegmentationOutput[Tensor]:
        return segmentation_output(preds)

    def _report_nonscalars(self, values: Mapping[str, Any], step: Optional[int], trackers: List[Tracker]) -> None:
        """Per-class VECTORS (a ``MulticlassJaccardIndex(average=None)``) to the log + trackers.

        Selection is matrainer's shape rule (a flat numeric sequence of length >= 2); the
        rendering is the worst-first table + sorted bar chart — a different rendering from the
        classification task's heatmap ON PURPOSE (one axis of meaning, a shared scale, and
        what a reader wants is a RANKING).
        """
        report_vectors(values, self.class_names or None, trackers, step=step, plot=self.plot_metrics)

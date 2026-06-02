"""LightningSegmentationTrainer: specialised semantic-segmentation trainer.

Subclass of :class:`~marainer.trainer.LightningTrainer` that bakes in
segmentation-specific defaults so a minimal YAML config only needs to specify
the dataset, model backbone, and number of classes — no adapter wiring for
loss, collate, or metrics required.

Baked-in defaults (all overridable via YAML ``@configurable`` attributes):

* ``loss_fn`` — :class:`torch.nn.CrossEntropyLoss` with ``ignore_index``
  threaded from the trainer's ``ignore_index`` knob. No deltaid-owned loss
  adapter exists; override directly with any upstream-library loss class
  (e.g. ``segmentation_models_pytorch.losses.DiceLoss``,
  ``kornia.losses.FocalLoss``).
* ``train/val/test_loader`` ``collate_fn`` —
  :func:`~deltaid.segmentation.segmentation_collate_fn`.
* ``optimizer`` — ``AdamW(lr=1e-4, weight_decay=0.05)``.
* Metrics — :class:`~torchmetrics.classification.MulticlassJaccardIndex`
  (mean IoU) + :class:`~torchmetrics.classification.MulticlassF1Score`
  (Dice-equivalent, macro) per split. ``ignore_index`` propagates so
  don't-care pixels are excluded from both loss and metric accumulation.

Minimal YAML example::

    trainer: !class:deltaid.trainer.LightningSegmentationTrainer
      num_classes: 2
      ignore_index: -100
      model: !class:deltaid.deeplabv3plus.DeepLabV3PlusModel
        num_classes: !ref:num_classes
      train_set: !ref:train_set
      val_set: !ref:val_set

Override with an upstream-library loss (no deltaid wrapper needed)::

    trainer: !class:deltaid.trainer.LightningSegmentationTrainer
      num_classes: 2
      model: !ref:model
      train_set: !ref:train_set
      val_set: !ref:val_set
      loss_fn: !class:torch.nn.CrossEntropyLoss
        weight: [1.0, 5.0]
        ignore_index: -100
      val_metrics: !class:torchmetrics.classification.MulticlassJaccardIndex
        num_classes: !ref:num_classes
        average: macro
        ignore_index: -100
"""

from typing import Any, Optional

import torch
import torch.nn as nn
from confluid import LazyClass, configurable
from marainer.trainer import LightningTrainer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from deltaid.segmentation import segmentation_collate_fn


def _default_segmentation_metrics(num_classes: int, ignore_index: int) -> MetricCollection:
    """Build the deltaid default segmentation metric pair from torchmetrics.

    No deltaid-owned wrapper: returns a stock :class:`MetricCollection` of
    upstream torchmetrics classes. ``ignore_index`` flows through to both
    metrics so don't-care pixels are excluded from accumulation.
    """
    return MetricCollection(
        {
            "miou": MulticlassJaccardIndex(
                num_classes=num_classes,
                average="macro",
                ignore_index=ignore_index,
            ),
            "dice_macro": MulticlassF1Score(
                num_classes=num_classes,
                average="macro",
                ignore_index=ignore_index,
                multidim_average="global",
            ),
        }
    )


@configurable(task="segmentation", role="trainer")
class LightningSegmentationTrainer(LightningTrainer):
    """Specialised semantic-segmentation trainer with baked-in defaults.

    Mirrors :class:`~marainer.trainer.LightningTrainer`'s minimal-ctor discipline:
    only ``model`` / ``train_set`` (+ identity ``num_classes`` / ``ignore_index`` /
    ``experiment_name`` / ``run_name``); the deferred ``optimizer`` /
    ``train/val/test_loader`` (``segmentation_collate_fn``) / ``lightning`` slots
    and the ``loss_fn`` + metric set are body attributes configured
    post-construction (YAML / broadcasting; a deferred slot uses ``!lazy:``).

    Args:
        model: Segmentation model that emits ``[N, num_classes, H, W]`` logits.
        train_set: Training dataset or Flux pipeline.
        num_classes: Number of output classes including background; drives
            metric construction. Provide it explicitly to build the metric set —
            dataset-derived ``num_classes`` was removed with ``marainer.dimensions``
            (TASKS follow-up: restore via an ``initialize()`` hook like sonair's).
        ignore_index: Pixel value in ``target`` that contributes zero loss
            and is skipped during metric accumulation. Propagates to both the
            default ``loss_fn`` and every default metric. When the user
            overrides ``loss_fn`` / ``*_metrics`` via YAML, those upstream
            classes are responsible for their own ``ignore_index``; this knob
            becomes a no-op for the overridden path.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
    """

    def __init__(
        self,
        model: Any,
        train_set: Any,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        experiment_name: str = "deltaid_segmentation",
        run_name: str = "train",
    ) -> None:
        super().__init__(model=model, train_set=train_set, experiment_name=experiment_name, run_name=run_name)
        # Store config ONLY — no functional work. Deferred slots are LazyClass;
        # the base run() flows them. Every slot is overridable post-construction.
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.optimizer = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        self.train_loader = LazyClass(DataLoader, collate_fn=segmentation_collate_fn)
        self.val_loader = LazyClass(DataLoader, collate_fn=segmentation_collate_fn)
        self.test_loader = LazyClass(DataLoader, collate_fn=segmentation_collate_fn)
        if self.num_classes is not None:
            base = _default_segmentation_metrics(self.num_classes, self.ignore_index)
            self.train_metrics = base.clone(prefix="train/")
            self.val_metrics = base.clone(prefix="val/")
            self.test_metrics = base.clone(prefix="test/")
        # When num_classes is omitted, metrics stay None (the base default). Set
        # num_classes explicitly to enable segmentation metrics — dataset-derived
        # num_classes was removed with marainer.dimensions (TASKS follow-up: restore
        # via an initialize() hook mirroring sonair's classification trainer).

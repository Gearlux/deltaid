"""The **torch** segmentation losses and metrics — registered, never wrapped.

One module per ENGINE, because a loss is engine-bound: everything below is ``torch.nn`` /
``torchmetrics`` / ``segmentation_models_pytorch.losses`` and carries ``framework="torch"``, so a
picker asked for "a segmentation loss" for a non-torch trainer is not offered one. Each engine's
set is its own ``confluid.configurables`` entry point, so on an install without torch this module
simply fails to import and a discovery scan skips it — which is the correct answer, not a degraded
one.

**Nothing here is a deltaid class.** These are off-the-shelf upstream classes tagged for discovery,
which is the workspace model: tag and register what exists; navigaitor scans and generates. A
segmentation loss that *reimplements* Dice or Focal does not belong here — smp already ships both.

Loss: the default is :class:`torch.nn.CrossEntropyLoss` (raw ``[N, C, H, W]`` logits + an
``[N, H, W]`` int64 class-id mask), built by the segmenter with its ``ignore_index``. The rest of
the family is registered so a config can name one:

* ``torch.nn`` — ``CrossEntropyLoss`` (with ``ignore_index`` / ``label_smoothing`` / ``weight``),
  ``BCEWithLogitsLoss`` (binary / one-hot targets), ``NLLLoss`` (for a log-softmax head).
* ``smp.losses`` — the region-overlap family that a class-imbalanced segmentation run actually
  wants: ``DiceLoss`` / ``JaccardLoss`` (optimize the metric directly), ``FocalLoss`` (down-weight
  easy background pixels), ``TverskyLoss`` (trade precision against recall), ``LovaszLoss`` (a
  smooth surrogate for IoU), plus ``SoftCrossEntropyLoss`` / ``SoftBCEWithLogitsLoss`` (label
  smoothing) and ``MCCLoss``. **They take a ``mode``** (``"binary"`` / ``"multiclass"`` /
  ``"multilabel"``) as their first argument — a segmentation run over a class-id mask wants
  ``mode: multiclass``, and getting it wrong is loud rather than silent.

Metrics: the ``torchmetrics.classification`` multiclass family applies per-pixel unchanged — a
segmentation prediction is a classification of every pixel — so ``MulticlassJaccardIndex`` (mean
IoU, the field's primary score) and ``MulticlassF1Score`` (Dice) are the segmenter's defaults, with
the accuracy / precision / recall / confusion family registered alongside. Two knobs matter here
and nowhere else: ``ignore_index`` (exclude the void class from the score as well as the loss) and
``average`` (``"macro"`` for one number, ``None`` for the PER-CLASS vector a segmentation report
usually wants — wire that one into ``val_metrics:`` alone).

Typical wiring (YAML)::

    loss_fn: !class:DiceLoss@framework=torch
      mode: multiclass

    val_metrics:
      iou_per_class: !class:MulticlassJaccardIndex { average: null }
"""

import segmentation_models_pytorch as smp
from confluid import register
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

# smp's losses are classes (unlike optax's / fastai's functions), so `register` reaches them
# directly — no shim. Bound to module-level names so `scan_module`'s `__all__`-gated pass picks
# them up: it filters by `__module__`, which for every class here is the upstream package.
DiceLoss = smp.losses.DiceLoss
JaccardLoss = smp.losses.JaccardLoss
FocalLoss = smp.losses.FocalLoss
TverskyLoss = smp.losses.TverskyLoss
LovaszLoss = smp.losses.LovaszLoss
SoftCrossEntropyLoss = smp.losses.SoftCrossEntropyLoss
SoftBCEWithLogitsLoss = smp.losses.SoftBCEWithLogitsLoss
MCCLoss = smp.losses.MCCLoss

#: Off-the-shelf losses a per-pixel run can wire. `torch.nn` first (the default and its two
#: siblings), then smp's region-overlap family.
_SEGMENTATION_LOSSES = (
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    NLLLoss,
    DiceLoss,
    JaccardLoss,
    FocalLoss,
    TverskyLoss,
    LovaszLoss,
    SoftCrossEntropyLoss,
    SoftBCEWithLogitsLoss,
    MCCLoss,
)

#: The torchmetrics multiclass family, which scores a per-pixel prediction unchanged.
_SEGMENTATION_METRICS = (
    MulticlassJaccardIndex,
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix,
)

for _loss in _SEGMENTATION_LOSSES:
    register(_loss, task="segmentation", role="loss", framework="torch")

for _metric in _SEGMENTATION_METRICS:
    register(_metric, task="segmentation", role="metric", framework="torch")


# Every registered class MUST be named here: `confluid.discovery.scan_module` filters members by
# `__module__` (which is `torch` / `torchmetrics` / `segmentation_models_pytorch` for these), so
# the `__all__`-gated pass is what carries them into StreamStudio's palette.
# `tests/test_categories.py` pins this list against the registrations above.
__all__ = [
    # losses — role="loss"
    "CrossEntropyLoss",
    "BCEWithLogitsLoss",
    "NLLLoss",
    "DiceLoss",
    "JaccardLoss",
    "FocalLoss",
    "TverskyLoss",
    "LovaszLoss",
    "SoftCrossEntropyLoss",
    "SoftBCEWithLogitsLoss",
    "MCCLoss",
    # metrics — role="metric"
    "MulticlassJaccardIndex",
    "MulticlassF1Score",
    "MulticlassAccuracy",
    "MulticlassPrecision",
    "MulticlassRecall",
    "MulticlassConfusionMatrix",
]

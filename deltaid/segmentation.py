"""Generic semantic-segmentation adapters that plug into Marainer's
:class:`~marainer.trainer.LightningTrainer`.

Segmentation is a much simpler fit for the base Trainer than detection: a
segmentation model takes ``[N, 3, H, W]`` images and emits ``[N, C, H, W]``
logits, and the loss / metric pair both consume ``(logits, target_mask)``
directly. No bundled batch type, no step-output dict, no per-mode forward
divergence — the Trainer's standard ``preds = self(batch.input)`` /
``loss = self.loss_fn(preds, batch.target)`` flow works unchanged.

This module provides:

* :class:`SegmentationLoss` — a :class:`torch.nn.CrossEntropyLoss` wrapper
  exposed for Confluid configuration (so per-class weights / ignore_index
  can be set from YAML without writing a Python class).
* :func:`segmentation_collate_fn` — a DataLoader collate that stacks
  per-sample ``input`` and ``target`` tensors into batched tensors and keeps
  per-sample metadata as a list (the default collate would crash on
  string-valued metadata such as ``pack_id`` or ``iq_file``).

Most torchmetrics segmentation metrics
(:class:`torchmetrics.classification.MulticlassJaccardIndex`,
:class:`~torchmetrics.classification.MulticlassF1Score`, ...) accept logits
directly as ``preds`` and emit a scalar by default — no adapter is needed,
they can be wired into ``val_metrics`` straight from YAML.

Typical wiring (YAML)::

    loss_fn: !class:deltaid.segmentation.SegmentationLoss
      class_weights: [1.0, 5.0]   # optional — bias toward foreground class

    val_metrics: !class:torchmetrics.classification.MulticlassJaccardIndex
      num_classes: 2

    train_loader: !class:torch.utils.data.DataLoader
      collate_fn: !ref:deltaid.segmentation.segmentation_collate_fn
"""

from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn
from confluid import configurable
from dataflux.sample import Sample
from logflow import get_logger
from torch import Tensor

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loss adapter
# ---------------------------------------------------------------------------


@configurable
class SegmentationLoss(nn.Module):
    """:class:`torch.nn.CrossEntropyLoss` wrapped for Confluid configuration.

    The base Trainer calls ``loss_fn(preds, target)`` where ``preds`` are
    raw logits ``[N, C, H, W]`` and ``target`` is an int64 class-id mask
    ``[N, H, W]`` (this is exactly what
    :class:`waivefront.segmentation_ops.SignalRegionsToSegmentationTargetOp`
    produces). PyTorch's CrossEntropyLoss handles the spatial dims natively
    when ``target`` is rank-3 — it averages the per-pixel cross-entropy.

    Args:
        class_weights: Optional per-class loss multipliers, length
            ``num_classes``. Useful when the foreground class covers a tiny
            fraction of pixels (typical for spectrogram annotations) — bumping
            the foreground weight prevents the model from collapsing to
            "predict background everywhere."
        ignore_index: Pixel value in ``target`` that contributes zero loss.
            Defaults to ``-100`` (the PyTorch default), so any pixel set to
            ``-100`` in the target mask is excluded from training. Useful for
            "don't-care" regions.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``. Defaults to ``"mean"``
            — averages over the batch and spatial dims, returning a single
            scalar that Lightning's logger can consume directly.
    """

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        weight: Optional[Tensor] = None
        if class_weights is not None:
            if len(class_weights) == 0:
                raise ValueError("class_weights must contain at least one entry.")
            weight = torch.tensor(class_weights, dtype=torch.float32)
        self.class_weights = list(class_weights) if class_weights is not None else None
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        # Register as a buffer (when present) so .to(device) moves it
        # automatically. When absent, no buffer is registered — CrossEntropyLoss
        # then defaults to unweighted, which is exactly what we want.
        if weight is not None:
            self.register_buffer("_weight_buf", weight, persistent=False)
        self._ce = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.dim() != 4:
            raise ValueError(f"SegmentationLoss expects logits [N,C,H,W], got shape {tuple(logits.shape)}.")
        if target.dim() != 3:
            raise ValueError(
                f"SegmentationLoss expects target [N,H,W] int64 (class ids), got shape {tuple(target.shape)}."
            )
        if target.dtype != torch.int64:
            target = target.long()
        # Re-bind weight from buffer in case the module was moved to another
        # device — CrossEntropyLoss captures the weight reference at
        # construction time and won't pick up the buffer's moved state.
        self._ce.weight = getattr(self, "_weight_buf", None)
        return cast(Tensor, self._ce(logits, target))


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def segmentation_collate_fn(batch: List[Sample]) -> Sample:
    """Collate per-sample segmentation samples into a batched Trainer-ready Sample.

    Each input ``Sample`` is expected to carry:

    * ``sample.input`` — ``[3, H, W]`` float tensor (e.g. produced by
      :class:`waivefront.segmentation_ops.SignalRegionsToSegmentationTargetOp`).
    * ``sample.target`` — ``[H, W]`` int64 class-id mask.

    The returned :class:`~dataflux.sample.Sample` has:

    * ``.input`` — ``[N, 3, H, W]`` stacked image tensor.
    * ``.target`` — ``[N, H, W]`` stacked int64 mask.
    * ``.metadata`` — ``{"per_sample": [<per-sample metadata dicts>]}``,
      preserved as a list so string-valued fields (``pack_id``, ``iq_file``)
      survive collation. The default DataLoader collate would crash on those.

    All per-sample tensors must already share the same spatial dims; mixed
    sizes must be resized upstream (e.g. via Waivefront's
    :class:`~waivefront.visualizers.RenderSpectrogramImageOp`).
    """
    if not batch:
        raise ValueError("segmentation_collate_fn received an empty batch.")
    images: List[Tensor] = []
    targets: List[Tensor] = []
    metas: List[Dict[str, Any]] = []
    for s in batch:
        if not isinstance(s.input, Tensor):
            raise TypeError(
                f"segmentation_collate_fn expects sample.input to be a torch.Tensor; " f"got {type(s.input).__name__}."
            )
        if not isinstance(s.target, Tensor):
            raise TypeError(
                f"segmentation_collate_fn expects sample.target to be a torch.Tensor; "
                f"got {type(s.target).__name__}."
            )
        images.append(s.input)
        targets.append(s.target.long() if s.target.dtype != torch.int64 else s.target)
        metas.append(s.metadata)

    return Sample(
        input=torch.stack(images),
        target=torch.stack(targets),
        metadata={"per_sample": metas},
    )


__all__ = [
    "SegmentationLoss",
    "segmentation_collate_fn",
]

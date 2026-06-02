"""Semantic-segmentation collate helper used by deltaid's trainer/evaluator.

Segmentation training plugs into the **unchanged**
:class:`~marainer.trainer.LightningTrainer` step flow: a model takes
``[N, 3, H, W]`` images and emits ``[N, C, H, W]`` logits, and the loss /
metric pair both consume ``(logits, target_mask)`` directly. No bundled
batch type, no step-output dict, no per-mode forward divergence.

The loss and metrics themselves are off-the-shelf upstream classes —
:class:`torch.nn.CrossEntropyLoss`, :class:`torchmetrics.classification.MulticlassJaccardIndex`,
:class:`torchmetrics.classification.MulticlassF1Score`, etc. — wired in by
:class:`~deltaid.trainer.LightningSegmentationTrainer` as defaults. There is
deliberately no deltaid-owned loss or metric adapter class: anyone who
wants to swap in :class:`segmentation_models_pytorch.losses.DiceLoss`,
:class:`kornia.losses.FocalLoss`, or any other upstream class can do so
directly from YAML without writing a deltaid wrapper.

The one helper that *does* need to live in deltaid is the DataLoader
collate function, because torch's default collate crashes on the
string-valued per-sample metadata (``pack_id``, ``iq_file``, …) that the
waivefront op chain emits — see :func:`segmentation_collate_fn` below.
"""

from typing import Any, Dict, List

import torch
from dataflux.sample import Sample
from logflow import get_logger
from torch import Tensor

logger = get_logger(__name__)


def segmentation_collate_fn(batch: List[Sample]) -> Sample:
    """Collate per-sample segmentation samples into a batched Trainer-ready Sample.

    Each input ``Sample`` is expected to carry:

    * ``sample.input`` — ``[3, H, W]`` float tensor (e.g. produced by
      :class:`waivefront.targets.RegionsToSegmentationMaskOp`).
    * ``sample.target`` — ``[H, W]`` int64 class-id mask.

    The returned :class:`~dataflux.sample.Sample` has:

    * ``.input`` — ``[N, 3, H, W]`` stacked image tensor.
    * ``.target`` — ``[N, H, W]`` stacked int64 mask.
    * ``.metadata`` — ``{"per_sample": [<per-sample metadata dicts>]}``,
      preserved as a list so string-valued fields (``pack_id``, ``iq_file``)
      survive collation. The default DataLoader collate would crash on those.

    All per-sample tensors must already share the same spatial dims; mixed
    sizes must be resized upstream (e.g. via Waivefront's
    :class:`~dataflux.ops.image.ConvertToImageOp`).
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
        metas.append(s.meta)  # per-sample dict (the batch nests the list under metadata["per_sample"])

    return Sample(
        input=torch.stack(images),
        target=torch.stack(targets),
        metadata={"per_sample": metas},
    )


__all__ = ["segmentation_collate_fn"]

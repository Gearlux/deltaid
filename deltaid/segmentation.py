"""Semantic-segmentation collate helper used by deltaid's segmenter.

Segmentation plugs into the segmenter's own step flow: a model takes
``[N, 3, H, W]`` images and emits ``[N, C, H, W]`` logits, and the loss /
metric pair both consume ``(logits, target_mask)`` directly. No step-output
dict, no per-mode forward divergence — the segmenter recovers the input
tensor + target mask from the batched :class:`~sampleflux.Sample` (via
``_batch_input`` / ``_batch_target`` in :mod:`deltaid.segmenter`).

The loss and metrics themselves are off-the-shelf upstream classes —
:class:`torch.nn.CrossEntropyLoss`, :class:`torchmetrics.classification.MulticlassJaccardIndex`,
:class:`torchmetrics.classification.MulticlassF1Score`, etc. — wired in by
:class:`~deltaid.segmenter.LightningSegmenter` as defaults. There is
deliberately no deltaid-owned loss or metric adapter class: anyone who
wants to swap in :class:`segmentation_models_pytorch.losses.DiceLoss`,
:class:`kornia.losses.FocalLoss`, or any other upstream class can do so
directly from YAML without writing a deltaid wrapper.

The one helper that *does* need to live in deltaid is the DataLoader
collate function: the typed sources yield a :class:`~sampleflux.Sample`
(an :class:`~sampleflux.Image` input + a :class:`~sampleflux.Mask` target),
and the collate stacks them + preserves the per-sample metadata (``pack_id``,
``iq_file``, …) that torch's default collate crashes on — see
:func:`segmentation_collate_fn` below.
"""

from typing import Any, Dict, List

import numpy as np
import torch
from loggair import get_logger
from sampleflux import Image, Label, Mask, Sample, item_data, primary
from torch import Tensor

logger = get_logger(__name__)


def _as_tensor(value: Any, what: str) -> Tensor:
    """Coerce an item payload (a torch tensor or numpy array) to a torch tensor.

    A typed source carries the image as an :class:`~sampleflux.Image` and the mask as a
    :class:`~sampleflux.Mask` — both ``np.ndarray`` subclasses (an array item cannot hold a live
    tensor, ``NDArrayItem.__new__`` runs ``np.asarray``), so ``item_data`` returns an ndarray that
    is converted back with a cheap, memory-sharing ``torch.as_tensor``. A field carrying a raw
    tensor verbatim passes through unchanged. Any other payload is a loud collate error.
    """
    if isinstance(value, Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)
    raise TypeError(
        f"segmentation_collate_fn expects the {what} item payload to be a torch.Tensor or np.ndarray; "
        f"got {type(value).__name__}."
    )


def _sample_metadata(sample: Sample) -> Dict[str, Any]:
    """Gather a :class:`~sampleflux.Sample`'s NON-primary fields into a plain metadata dict.

    The typed bag has no free-form metadata dict — auxiliary features (``pack_id`` / ``iq_file`` …)
    ride as their own fields. This collects every field that is not the primary input or the primary
    target into ``{name: value}`` (a :class:`~sampleflux.Label`'s ``.value``, else the item payload)
    so string-valued fields survive collation and predictions sinks can correlate outputs.
    """
    input_key, _ = primary(sample, "input")
    try:
        target_key: Any = primary(sample, "target")[0]
    except KeyError:
        target_key = None
    return {
        key: (item.value if isinstance(item, Label) else item_data(item))
        for key, item in sample.items()
        if key not in (input_key, target_key)
    }


def segmentation_collate_fn(batch: List[Sample]) -> Sample:
    """Collate per-sample segmentation :class:`~sampleflux.Sample`\\ s into ONE batched bag.

    The default DataLoader collate the deltaid segmenter wires in. Per sample it reads the primary
    ``input`` item's payload (``item_data(primary(s, "input"))``) as a ``[3, H, W]`` image tensor and
    the primary ``target`` item's payload as a ``[H, W]`` class-id mask (forced to ``int64``), then
    stacks into ONE batched :class:`~sampleflux.Sample`:

    * field ``"image"`` — an :class:`~sampleflux.Image` (``layout="CHW"``) over the stacked
      ``[N, 3, H, W]`` payload. An :class:`~sampleflux.Image` cannot hold a live ``torch.Tensor``
      (``NDArrayItem.__new__`` runs ``np.asarray``), so the payload lands as an ndarray and the model
      boundary (:func:`deltaid.segmenter._batch_input`) converts it back with a cheap ``torch.as_tensor``;
    * field ``"mask"`` (role ``target``) — a :class:`~sampleflux.Mask` over the stacked ``[N, H, W]``
      int64 mask payload;
    * field ``"metadata"`` (role ``aux``) — a ``list`` of the per-sample metadata dicts
      (:func:`_sample_metadata`, the non-primary fields) so string-valued fields survive.

    All per-sample tensors must already share the same spatial dims (``torch.stack`` requires it);
    mixed sizes must be resized upstream (e.g. via Waivefront's ``ConvertToImageOp``).
    """
    if not batch:
        raise ValueError("segmentation_collate_fn received an empty batch.")
    images: List[Tensor] = []
    targets: List[Tensor] = []
    metas: List[Dict[str, Any]] = []
    for s in batch:
        _, img_item = primary(s, "input")
        image = _as_tensor(item_data(img_item), "input")
        _, mask_item = primary(s, "target")
        mask = _as_tensor(item_data(mask_item), "target")
        images.append(image)
        targets.append(mask.long() if mask.dtype != torch.int64 else mask)
        metas.append(_sample_metadata(s))

    return Sample(
        {
            "image": Image(torch.stack(images), layout="CHW"),
            "mask": Mask(torch.stack(targets)),
            "metadata": metas,
        },
        roles={"mask": "target", "metadata": "aux"},
    )


__all__ = ["segmentation_collate_fn"]

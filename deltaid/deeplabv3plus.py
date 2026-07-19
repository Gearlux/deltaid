"""DeepLabV3+ semantic-segmentation wrapper for the Marainer Trainer flow.

Wraps :class:`segmentation_models_pytorch.DeepLabV3Plus` (smp) into a plain
:class:`torch.nn.Module` whose ``forward(images) -> logits`` signature plugs
straight into the **unchanged** base :class:`marainer.trainer.LightningTrainer`. Marainer
already calls ``preds = self(batch.input)`` and
``loss = self.loss_fn(preds, batch.target)`` — for segmentation that's all the
glue we need. No bundled batch type, no step-output dict, no per-mode forward
divergence.

The wrapper exists for three reasons:

1. **Confluid configurability**: ``@configurable`` exposes every smp constructor
   knob (encoder choice, ImageNet weights, decoder channels, ...) for YAML
   wiring while keeping a sensible default that matches the most common
   spectrogram-segmentation case (resnet34 + ImageNet).
2. **Checkpoint loading**: ``checkpoint_path`` strips Lightning's ``"model."``
   prefix via :func:`marainer.checkpoint.load_lightning_checkpoint`, so a
   marainer-trained model resumes / evaluates cleanly without forcing the
   caller to know about Lightning's wrapping.
3. **Diagnostic logging**: same per-batch warning pattern as
   :class:`raidar.faster_rcnn.FasterRCNNModel` — surface zero-positive batches
   and prediction-collapse early so users don't waste epochs on a silently
   broken pipeline.

Typical wiring (YAML)::

    model: !class:deltaid.deeplabv3plus.DeepLabV3PlusModel
      num_classes: 2
      encoder_name: resnet34
      encoder_weights: imagenet
    loss_fn: !class:deltaid.segmentation.SegmentationLoss
    val_metrics: !class:torchmetrics.classification.MulticlassJaccardIndex
      num_classes: 2
"""

from typing import Optional, cast

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from confluid import configurable
from loggair import get_logger
from marainer.checkpoint import load_lightning_checkpoint
from torch import Tensor

logger = get_logger(__name__)


@configurable(task="segmentation", role="model")
class DeepLabV3PlusModel(nn.Module):
    """DeepLabV3+ segmentation model wrapped for the Marainer Trainer flow.

    ``forward(images) -> logits`` only — segmentation models compute loss
    outside the model (unlike torchvision detectors), so the Trainer's
    standard ``loss_fn(preds, target)`` call works without a batch wrapper.

    Args:
        num_classes: Output channel count, including background as class 0.
            For binary foreground/background segmentation, pass ``2``.
        encoder_name: smp encoder identifier — ``"resnet34"``, ``"resnet50"``,
            ``"efficientnet-b0"``, ``"mit_b0"`` (SegFormer), ``"mobilenet_v2"``,
            etc. See https://smp.readthedocs.io/en/latest/encoders.html for
            the full list. Defaults to ``"resnet34"`` — fast and strong on
            small datasets.
        encoder_weights: Pretrained weights tag for the encoder. ``"imagenet"``
            (default) downloads ImageNet-1k weights on first use; ``None``
            initializes the encoder from scratch (useful for offline CI and
            when the input domain is far from natural images).
        in_channels: Input channel count. Defaults to 3 (RGB images, including
            the rendered spectrogram path used by the waivefront RFUAV configs).
        decoder_channels: Width of the decoder convolutions. smp's default
            (256) is appropriate for most cases.
        decoder_atrous_rates: Tuple of dilation rates for the ASPP module.
            smp's default ``(12, 24, 36)`` matches the original DeepLabV3+
            paper and works well at 512px input.
        activation: Final activation applied to the logits. Defaults to
            ``None`` so the model emits raw logits — the standard
            :class:`torch.nn.CrossEntropyLoss` path expects logits and applies
            softmax internally for stable gradients.
        checkpoint_path: Optional path to a Lightning ``.ckpt`` (or plain
            torch state-dict) saved by a previous training run. Loaded after
            the encoder weights so checkpoint values win. ``"model."`` prefix
            from Lightning wrapping is stripped automatically.
        checkpoint_strict: Forwarded to ``load_state_dict(strict=...)``.
            Defaults to False so e.g. a re-trained classifier head with a
            different ``num_classes`` doesn't reject the whole load.
    """

    def __init__(
        self,
        num_classes: int = 2,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        activation: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
        self.num_classes = int(num_classes)
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = int(in_channels)
        # Stored as ``self.segmenter`` (not ``self.model``) so a state_dict
        # saved through marainer's Lightning wrapper — which adds a leading
        # ``model.`` prefix — round-trips cleanly. ``load_lightning_checkpoint``
        # strips that single ``model.`` prefix, leaving keys like
        # ``segmenter.encoder.conv1.weight`` that match this wrapper's layout.
        # If we used ``self.model``, the stripping would also chew through our
        # own attribute name, producing keys (``encoder.conv1.weight``) that
        # don't match anything.
        self.segmenter = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self.in_channels,
            classes=self.num_classes,
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
            activation=activation,
        )
        if checkpoint_path:
            load_lightning_checkpoint(self, checkpoint_path, strict=checkpoint_strict)

    def forward(self, images: Tensor) -> Tensor:
        """Run the segmentation model and emit per-pixel class logits.

        Args:
            images: ``[N, in_channels, H, W]`` float tensor. ``H`` and ``W``
                must each be a multiple of 32 (DeepLabV3+ stride constraint).

        Returns:
            ``[N, num_classes, H, W]`` raw logits.
        """
        if images.dim() != 4:
            raise ValueError(f"DeepLabV3PlusModel expects 4D [N,C,H,W] images, got shape {tuple(images.shape)}.")
        n, c, h, w = images.shape
        if c != self.in_channels:
            raise ValueError(
                f"DeepLabV3PlusModel was built for in_channels={self.in_channels}, " f"got images with {c} channels."
            )
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(
                f"DeepLabV3+ requires H and W to each be multiples of 32 (encoder stride), "
                f"got H={h}, W={w}. Resize upstream."
            )

        logits = cast(Tensor, self.segmenter(images))

        # Per-batch diagnostic: surface argmax-class collapse early. If the
        # model only ever predicts background, training is silently broken —
        # warn so the user can check class balance / loss weights.
        with torch.no_grad():
            pred_classes = logits.argmax(dim=1)
            unique_pred = torch.unique(pred_classes).tolist()
            if len(unique_pred) == 1 and unique_pred[0] == 0 and self.num_classes > 1:
                logger.warning(
                    f"[diag][deeplabv3plus] batch of {n} image(s): argmax predicts ONLY "
                    f"background (class 0) across all pixels — classifier head may be "
                    f"undertrained or class imbalance may need addressing via class_weights."
                )
            else:
                logger.trace(
                    f"[diag][deeplabv3plus] batch={n} pred_classes={sorted(unique_pred)} "
                    f"logits_range=[{float(logits.min()):.3f},{float(logits.max()):.3f}]"
                )

        return logits


__all__ = ["DeepLabV3PlusModel"]

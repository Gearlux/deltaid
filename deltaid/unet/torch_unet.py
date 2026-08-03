"""``TorchUNet`` — the shared U-Net spec in ``torch.nn``.

The torch entry in the cross-framework comparison :mod:`deltaid.unet.spec` describes, and the
opposite kind of model from the zoo wrapper it sits beside:
:class:`deltaid.smp_model.SegmentationModelPytorch` reaches twelve architectures over ~800
pretrained encoders, this one wraps nothing at all. For real work prefer the zoo wrapper — a small
net trained from scratch loses to a pretrained encoder every time — but smp is torch-only, so a
comparison built on it could never include Keras, Flax or MLX.

It serves ALL THREE torch backends. ``LightningSegmenter``, ``TorchSegmenter`` and
``FastaiSegmenter`` differ in the training loop, not in what a model is, so each takes any
``torch.nn.Module`` and this is one — the same reason all three already share
``SegmentationModelPytorch``.

**This is the channels-FIRST implementation, and the ``input_shape`` knob still reads
``(H, W, C)``.** The other implementations will consume NHWC because keras, tf, flax and mlx all
default to it; torch is NCHW. Keeping the constructor knob channels-last means one config line —
``input_shape: [224, 224, 3]`` — drives every implementation, and only ``C`` affects the parameter
count anyway. What changes is the TENSOR: ``forward`` takes ``[N, C, H, W]``, which is exactly what
the torch preprocessing chain already produces, since it ends in a ``ToTensor`` transposing HWC to
CHW.

It IS a ``torch.nn.Module``, the same decision as ``SegmentationModelPytorch`` being one: a
segmenter drives whatever is wired into its ``model`` slot without asking what it got.
"""

from typing import List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from confluid import configurable
from loggair import get_logger
from torch import Tensor, nn

from deltaid.unet.spec import BOTTLENECK_WIDTH, ENCODER_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE, size_divisor

logger = get_logger(__name__)

__all__ = ["TorchUNet"]


def _conv_block(in_channels: int, width: int, kernel_size: int) -> nn.Sequential:
    """``[conv -> BN -> ReLU] x2`` — the spec's repeated unit, in torch's vocabulary.

    Written out rather than borrowed from a library: the point of the spec is that every
    implementation is the same network, which only holds if each one says so in its own
    framework's primitives.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, width, kernel_size, padding="same"),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
        nn.Conv2d(width, width, kernel_size, padding="same"),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
    )


class _UNet(nn.Module):
    """The spec's layer graph. Private: the ``@configurable`` wrapper is the public surface.

    A ``Sequential`` cannot express this — the decoder consumes a skip from the matching encoder
    level, so ``forward`` has to hold the intermediate activations. That is the one structural
    difference from the classification convnet, and it is what a U-Net IS.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        widths: Tuple[int, ...],
        bottleneck_width: int,
        kernel_size: int,
        pool_size: int,
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.divisor = size_divisor(widths, pool_size)

        self.encoders = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.encoders.append(_conv_block(channels, width, kernel_size))
            channels = width

        self.bottleneck = _conv_block(channels, bottleneck_width, kernel_size)
        channels = bottleneck_width

        # The decoder's input at each level is `channels + skip` because of the concat — which is
        # what makes a U-Net's decoder wider than its encoder, and is the whole mechanism.
        self.decoders = nn.ModuleList()
        for width in reversed(widths):
            self.decoders.append(_conv_block(channels + width, width, kernel_size))
            channels = width

        # 1x1: one logit per class, per pixel. Raw — no activation, like every head here.
        self.head = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, images: Tensor) -> Tensor:
        height, width = images.shape[-2:]
        if height % self.divisor or width % self.divisor:
            # Checked here rather than left to fail: the concat below would otherwise raise a
            # shape error naming two intermediate tensors, which says nothing about the input.
            raise ValueError(
                f"TorchUNet needs H and W divisible by {self.divisor} (one halving per encoder "
                f"level), got H={height}, W={width}. Resize upstream."
            )
        skips: List[Tensor] = []
        x = images
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, self.pool_size)

        x = self.bottleneck(x)

        for decoder in self.decoders:
            # Nearest-neighbour, PARAMETER-FREE — a learned ConvTranspose2d's parameter count and
            # output size depend on padding conventions that differ between frameworks, which
            # would make "the same network" untrue by construction (see the spec module).
            x = F.interpolate(x, scale_factor=self.pool_size, mode="nearest")
            x = torch.cat([x, skips.pop()], dim=1)
            x = decoder(x)

        return cast(Tensor, self.head(x))


@configurable(task="segmentation", role="model", lazy=True, framework="torch")
class TorchUNet(nn.Module):
    """The shared U-Net spec (:mod:`deltaid.unet.spec`), implemented in ``torch.nn``.

    Args:
        num_classes: Width of the per-pixel head. Sentinel ``0`` is invalid but permitted so the
            class is zero-arg constructible; a segmenter flows the resolved count in at run time,
            so it is not written in YAML.
        input_shape: ``(H, W, C)`` — the SAME channels-last knob every implementation takes, so
            one config line drives any of them. The tensor ``forward`` consumes is
            ``[N, C, H, W]``; only ``C`` is read here.
        widths: Channels per encoder level. Changing it breaks comparability with the other
            implementations — see :mod:`deltaid.unet.spec`.
        bottleneck_width: Channels at the bottom of the U.
        seed: Seed for the parameter initializers, so a config reproduces its weights.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = ENCODER_WIDTHS,
        bottleneck_width: int = BOTTLENECK_WIDTH,
        seed: int = 0,
    ) -> None:
        # `nn.Module.__init__` first — a torch module refuses submodule assignment before it.
        super().__init__()
        # Lazy: store only. No parameter allocation — the workspace zero-arg mandate.
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.bottleneck_width = bottleneck_width
        self.seed = seed
        # Public submodule name, NOT `_net`, for the reason the zoo wrapper uses `segmenter`: a
        # state_dict saved through a Lightning wrapper round-trips as `net.*`, and a leading
        # underscore reads as "private" for something checkpoints address by name.
        self.net: Optional[nn.Module] = None

    def solidify(self) -> nn.Module:
        """Build (once) and return the live network — confluid's ``flow()``-time hook.

        Required configuration is validated HERE rather than in ``__init__``, which is what keeps
        zero-arg construction working. ``flow()`` calls this, so ``self.parameters()`` is
        populated before an optimizer reads it.
        """
        if self.net is None:
            if not self.num_classes:
                raise ValueError(
                    "TorchUNet.num_classes is not set — a segmenter flows the resolved count in "
                    "at run time, or set it explicitly."
                )
            torch.manual_seed(self.seed)
            self.net = _UNet(
                num_classes=self.num_classes,
                in_channels=int(self.input_shape[-1]),
                widths=self.widths,
                bottleneck_width=self.bottleneck_width,
                kernel_size=KERNEL_SIZE,
                pool_size=POOL_SIZE,
            )
            logger.info(f"Built the shared U-Net spec in torch.nn ({self.num_classes} classes, widths={self.widths}).")
        return cast(nn.Module, self.net)

    @property
    def backbone(self) -> nn.Module:
        """The live network, built on first access — the ergonomic twin of the zoo wrapper's.

        ``flow()`` never touches property getters (confluid's lazy-init rule 3), which is why
        :meth:`solidify` exists as a method as well; this is what makes a directly constructed,
        never-flowed instance usable.
        """
        return self.solidify()

    def forward(self, images: Tensor) -> Tensor:
        """Per-pixel class logits ``[N, num_classes, H, W]`` from an ``[N, C, H, W]`` batch."""
        return cast(Tensor, self.backbone(images))

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`deltaid.unet.spec` predicts.

        BatchNorm's running mean/variance are torch BUFFERS rather than parameters, so they are
        excluded by construction here — unlike an MLX implementation, which would have to filter
        them out by name because MLX keeps them in the same tree as the weights.
        """
        return int(sum(p.numel() for p in self.backbone.parameters() if p.requires_grad))

    def __repr__(self) -> str:
        """Short and honest before the build — the default would print an empty module."""
        if self.net is None:
            return f"TorchUNet(num_classes={self.num_classes}, widths={self.widths}, unbuilt)"
        return cast(str, super().__repr__())

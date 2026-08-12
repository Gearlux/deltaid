"""``MlxUNet`` — the shared U-Net spec in ``mlx.nn``, the MLX backend's model.

Like the Flax one this wraps no zoo, and for a comparable reason: the only real MLX vision zoo,
``mlx-image``, hard-pins ``huggingface_hub==0.24.0``, ``albumentations==1.4.1`` and
``Pillow==10.4.0``, and it ships classification backbones rather than segmentation heads anyway.
That hub downgrade is precisely the recurring skew ``aisland doctor`` checks for — it would break
``import pytorch_lightning`` for every other backend — so a shared venv cannot have it. The MLX
backend implements :mod:`deltaid.unet.spec` natively instead.

MLX is **Apple-silicon only**, which is a property of the framework rather than a choice here: the
extra already carries ``sys_platform == 'darwin' and platform_machine == 'arm64'``, and this module
is only importable where that holds.

Three MLX facts shape the code, and all three are the classification twin's too:

* **Channels-last is native.** ``mlx.nn.Conv2d`` takes ``NHWC`` directly, so this consumes the
  same preprocessing chain as the Keras, TensorFlow and JAX runs with no transpose.
* **Arrays are lazy.** Nothing computes until ``mx.eval`` (or a value is read). That matters in
  the training loop rather than here — see :mod:`deltaid.mlx_segmenter` — but it is why this
  module never inspects a result to "check" it.
* **``parameters()`` skips underscore-prefixed attributes.** ``nn.Module`` is a dict, and its
  ``valid_parameter_filter`` ends in ``… and not key.startswith("_")`` (measured). So the lazily
  built network is held in the PUBLIC attribute ``net`` — ``self._model``, the name its Keras /
  TensorFlow / Flax twins use, would make the whole network invisible to ``parameters()`` and
  therefore to the optimizer, and the loop would run forever applying gradients to nothing.

One fact is this task's alone: **MLX has no max-pool over NHWC in every version**, so the
downsample is a reshape-and-max — which is exactly what pooling is, and is portable across
versions. The upsample is its inverse, a repeat along H and W, sized from the SKIP so the concat
agrees by construction.
"""

from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from confluid import configurable
from loggair import get_logger

from deltaid.unet.spec import BOTTLENECK_WIDTH, ENCODER_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE, size_divisor

logger = get_logger(__name__)

__all__ = ["MlxUNet"]


def _pool(x: Any, size: int = POOL_SIZE) -> Any:
    """Max-pool ``[N, H, W, C]`` by ``size``, written as a reshape.

    MLX ships no NHWC max-pool in every version, and folding each window into its own axis and
    taking the max over it IS pooling — portable, and no slower.
    """
    batch, height, width, channels = x.shape
    x = x.reshape(batch, height // size, size, width // size, size, channels)
    return x.max(axis=(2, 4))


def _upsample_to(x: Any, size: Tuple[int, int]) -> Any:
    """Nearest-neighbour upsample of ``[N, H, W, C]`` to ``size`` — parameter-free, per the spec.

    A repeat along H and W rather than a learned ``ConvTranspose``: a transposed convolution's
    parameter count and output size depend on padding conventions that differ between frameworks,
    which would make "the same network" untrue by construction.

    Sized from the SKIP the caller is about to concatenate, so the two agree by construction
    rather than by both happening to double.
    """
    batch, height, width, channels = x.shape
    factor_h, factor_w = int(size[0]) // height, int(size[1]) // width
    x = mx.repeat(x, factor_h, axis=1)
    return mx.repeat(x, factor_w, axis=2)


class _ConvBlock(nn.Module):
    """``[conv -> BN -> ReLU] x2`` — the spec's repeated unit, in ``mlx.nn``.

    NOTE it does NOT pool: in a U-Net the skip is taken from the block's output *before* the
    downsample, so the pooling belongs to the caller.
    """

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2)
        self.norm1 = nn.BatchNorm(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2)
        self.norm2 = nn.BatchNorm(width)

    def __call__(self, x: Any) -> Any:
        x = nn.relu(self.norm1(self.conv1(x)))
        return nn.relu(self.norm2(self.conv2(x)))


class _UNet(nn.Module):
    """The spec's layer graph: encoder + skips → bottleneck → decoder → 1x1 head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        widths: Tuple[int, ...],
        bottleneck_width: int,
    ) -> None:
        super().__init__()
        encoders = []
        channels = in_channels
        for width in widths:
            encoders.append(_ConvBlock(channels, width))
            channels = width
        self.encoders = encoders

        self.bottleneck = _ConvBlock(channels, bottleneck_width)
        channels = bottleneck_width

        # The decoder's input at each level is `channels + skip` because of the concat — what
        # makes a U-Net's decoder wider than its encoder, and the whole mechanism.
        decoders = []
        for width in reversed(widths):
            decoders.append(_ConvBlock(channels + width, width))
            channels = width
        self.decoders = decoders

        # 1x1: one logit per class, per pixel. Raw — no activation.
        self.head = nn.Conv2d(channels, num_classes, kernel_size=1)

    def __call__(self, x: Any) -> Any:
        skips: List[Any] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = _pool(x)

        x = self.bottleneck(x)

        for decoder in self.decoders:
            skip = skips.pop()
            x = _upsample_to(x, (skip.shape[1], skip.shape[2]))
            x = mx.concatenate([x, skip], axis=-1)
            x = decoder(x)

        return self.head(x)


@configurable(task="segmentation", role="model", lazy=True, framework="mlx")
class MlxUNet(nn.Module):
    """The shared U-Net spec (:mod:`deltaid.unet.spec`), implemented in ``mlx.nn``.

    **It IS an ``nn.Module``**, the same decision as ``KerasUNet`` being a ``keras.Model`` and
    ``TorchUNet`` an ``nn.Module``: the segmenter drives whatever is wired into its ``model`` slot
    without asking what it got. Concretely that is what lets ``MlxSegmenter`` call
    ``nn.value_and_grad(self.model, …)`` and ``optimizer.update(self.model, grads)`` on the wrapper
    itself, so the slot can hold a bare ``mlx.nn.Module`` a user brought.

    Args:
        num_classes: Width of the per-pixel head. Sentinel ``0`` is invalid but permitted so the
            class is zero-arg constructible; a segmenter flows the resolved count in at run time.
        input_shape: ``(H, W, C)`` — channels LAST, which is MLX's native layout.
        widths: Channels per encoder level. Changing it breaks comparability with the other
            implementations — see :mod:`deltaid.unet.spec`.
        bottleneck_width: Channels at the bottom of the U.
        seed: Seed for parameter init, so a config reproduces its weights.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = ENCODER_WIDTHS,
        bottleneck_width: int = BOTTLENECK_WIDTH,
        seed: int = 0,
    ) -> None:
        # `nn.Module.__init__` first — it initialises the dict every attribute lands in.
        super().__init__()
        # Partial: store only. These are ints/tuples, which MLX's parameter filter admits only for
        # dict/list/array, so none of them is mistaken for a weight.
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.bottleneck_width = bottleneck_width
        self.seed = seed
        # PUBLIC `net`, not `_model` — see the module docstring: an underscore-prefixed attribute
        # is filtered out of `parameters()`, which would hide the whole network.
        self.net: Optional[nn.Module] = None

    def solidify(self) -> Any:
        """Build (once) and return the live ``nn.Module`` — confluid's ``flow()``-time hook."""
        if self.net is not None:
            return self.net
        if not self.num_classes:
            raise ValueError(
                "MlxUNet.num_classes is not set — a segmenter flows the resolved count in at "
                "run time, or set it explicitly."
            )
        mx.random.seed(self.seed)
        self.net = _UNet(
            num_classes=self.num_classes,
            in_channels=int(self.input_shape[-1]),
            widths=self.widths,
            bottleneck_width=self.bottleneck_width,
        )
        logger.info(
            f"Built the shared U-Net spec in mlx.nn ({self.num_classes} classes, "
            f"widths={self.widths}, device={mx.default_device()})."
        )
        return self.net

    @property
    def model(self) -> Any:
        """The live inner ``nn.Module``, built on first access.

        The ergonomic accessor for a directly-constructed instance that was never flowed. A caller
        that just wants to train this does NOT need it: the wrapper is itself an ``nn.Module``, so
        ``parameters()`` / ``update()`` / ``train()`` reach through.
        """
        return self.solidify()

    def __call__(self, x: Any, training: Optional[bool] = None) -> Any:
        """Per-pixel class logits from an ``[N, H, W, C]`` batch — channels LAST.

        ``training`` defaults to ``None`` meaning *leave the module's mode alone*, which is not the
        ``False`` its Keras and Flax twins default to — and the difference is MLX's, not a style
        choice. MLX carries train/eval as module STATE (``self.train(bool)``, read by
        ``nn.BatchNorm``), so a per-call flag defaulting to ``False`` would silently flip the
        network to eval on every forward — including the one inside the training step, where
        ``nn.value_and_grad`` calls ``model(x)`` with no flag to pass, and batch-norm would then
        never train.
        """
        net = self.model  # builds on first access
        divisor = self.size_divisor
        height, width = int(x.shape[1]), int(x.shape[2])
        if height % divisor or width % divisor:
            raise ValueError(
                f"MlxUNet: H and W must each be divisible by {divisor} "
                f"(2 ** {len(self.widths)} encoder levels), got {height}x{width}."
            )
        if training is not None:
            self.train(training)
        return net(x)

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`deltaid.unet.spec` predicts.

        Counted off the leaves of the parameter tree. MLX keeps batch-norm's running mean and
        variance in that SAME tree, so they are excluded by NAME — the other implementations hold
        them as separate non-trainable state and the spec counts neither. Without the filter the
        match breaks, which is the one wrinkle this framework adds.
        """
        from mlx.utils import tree_flatten

        self.solidify()
        leaves: Any = tree_flatten(self.parameters())
        return int(
            sum(
                value.size
                for path, value in leaves
                if not path.endswith("running_mean") and not path.endswith("running_var")
            )
        )

    @property
    def size_divisor(self) -> int:
        """The multiple ``H`` and ``W`` must each be, for the skips to line up on the way back."""
        return size_divisor(self.widths, POOL_SIZE)

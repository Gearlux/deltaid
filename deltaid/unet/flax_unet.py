"""``FlaxUNet`` — the shared U-Net spec in ``flax.nnx``, the JAX backend's model.

The JAX entry in the comparison :mod:`deltaid.unet.spec` describes, and it wraps no zoo — which
here is not a design choice but the only option. **There is no Flax segmentation zoo to wrap:**
``transformers`` dropped Flax entirely in v5 (there is no ``FlaxAutoModelForSemanticSegmentation``)
and ``segmentation_models_pytorch`` is torch-only by name. So the JAX backend implements the spec
natively, which is exactly why the spec exists — see that module for why one architecture written
five times is worth more than five different zoo models.

It IS an ``nnx.Module``, the same decision as ``KerasUNet`` being a ``keras.Model`` and
``TorchUNet`` an ``nn.Module``: the segmenter drives whatever is wired into its ``model`` slot
without asking what it got, and ``nnx.state`` / ``nnx.Optimizer`` reach the parameters straight
through the wrapper.

**Channels-LAST** (``[N, H, W, C]``), like every implementation except the torch one, so it
consumes the same preprocessing chain as the Keras and TensorFlow runs — the chain that stops
before ``ToTensor``.

Two Flax facts shape the code and both were measured rather than read off the docs:

* **An attribute first assigned ``None`` in ``__init__`` becomes STATIC in nnx's pytree**, and
  assigning a Module to it later raises *"Cannot assign data value … to static attribute"*.
  ``nnx.data`` is the fix flax itself names, and it is what keeps zero-arg construction (the
  workspace lazy-init mandate) compatible with building at ``flow()`` time.
* **``nnx.max_pool`` and ``jax.image.resize`` are the pool/upsample pair**, and the upsample is
  sized from the SKIP rather than computed as ``2 *`` the current size — the same rule the
  TensorFlow implementation follows, so an odd intermediate size fails the explicit divisor check
  up front instead of at the concat.
"""

from typing import Any, List, Optional, Tuple

from confluid import configurable
from flax import nnx
from loggair import get_logger

from deltaid.unet.spec import BOTTLENECK_WIDTH, ENCODER_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE, size_divisor

logger = get_logger(__name__)

__all__ = ["FlaxUNet"]


class _ConvBlock(nnx.Module):
    """``[conv -> BN -> ReLU] x2`` — the spec's repeated unit, in ``flax.nnx``.

    NOTE it does NOT pool. In a U-Net the skip is taken from the block's output *before* the
    downsample, so the pooling belongs to the caller — unlike the classification convnet, whose
    block owns its pool because nothing reads the pre-pool activation.
    """

    def __init__(self, in_channels: int, width: int, rngs: Any) -> None:
        self.conv1 = nnx.Conv(in_channels, width, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), rngs=rngs)
        self.norm1 = nnx.BatchNorm(width, rngs=rngs)
        self.conv2 = nnx.Conv(width, width, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), rngs=rngs)
        self.norm2 = nnx.BatchNorm(width, rngs=rngs)

    def __call__(self, x: Any, training: bool = False) -> Any:
        # `use_running_average` is the INVERSE of "training": in eval we consume the running
        # statistics rather than the batch's, exactly as keras and torch do.
        x = nnx.relu(self.norm1(self.conv1(x), use_running_average=not training))
        return nnx.relu(self.norm2(self.conv2(x), use_running_average=not training))


class _UNet(nnx.Module):
    """The spec's layer graph: encoder + skips → bottleneck → decoder → 1x1 head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        widths: Tuple[int, ...],
        bottleneck_width: int,
        seed: int,
    ) -> None:
        rngs = nnx.Rngs(seed)
        self.encoders: Any = nnx.List([])
        channels = in_channels
        for width in widths:
            self.encoders.append(_ConvBlock(channels, width, rngs))
            channels = width

        self.bottleneck = _ConvBlock(channels, bottleneck_width, rngs)
        channels = bottleneck_width

        # The decoder's input at each level is `channels + skip` because of the concat — what
        # makes a U-Net's decoder wider than its encoder, and the whole mechanism.
        self.decoders: Any = nnx.List([])
        for width in reversed(widths):
            self.decoders.append(_ConvBlock(channels + width, width, rngs))
            channels = width

        # 1x1: one logit per class, per pixel. Raw — no activation.
        self.head = nnx.Conv(channels, num_classes, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: Any, training: bool = False) -> Any:
        import jax.numpy as jnp

        skips: List[Any] = []
        for encoder in self.encoders:
            x = encoder(x, training=training)
            skips.append(x)
            x = nnx.max_pool(x, window_shape=(POOL_SIZE, POOL_SIZE), strides=(POOL_SIZE, POOL_SIZE))

        x = self.bottleneck(x, training=training)

        for decoder in self.decoders:
            skip = skips.pop()
            # Sized from the SKIP, not computed as `2 * current`: nearest-neighbour resize to the
            # exact shape the concat needs is what makes the two agree by construction.
            x = _upsample_to(x, skip.shape[1:3])
            x = jnp.concatenate([x, skip], axis=-1)
            x = decoder(x, training=training)

        return self.head(x)


def _upsample_to(x: Any, size: Tuple[int, int]) -> Any:
    """Nearest-neighbour resize of ``[N, H, W, C]`` to ``size`` — parameter-free, per the spec.

    ``jax.image.resize`` rather than a learned ``ConvTranspose``: a transposed convolution's
    parameter count and output size depend on padding conventions that differ between frameworks,
    which would make "the same network" untrue by construction.
    """
    import jax

    return jax.image.resize(x, (x.shape[0], int(size[0]), int(size[1]), x.shape[3]), method="nearest")


@configurable(task="segmentation", role="model", lazy=True, framework="jax")
class FlaxUNet(nnx.Module):
    """The shared U-Net spec (:mod:`deltaid.unet.spec`), implemented in ``flax.nnx``.

    Args:
        num_classes: Width of the per-pixel head. Sentinel ``0`` is invalid but permitted so the
            class is zero-arg constructible; a segmenter flows the resolved count in at run time.
        input_shape: ``(H, W, C)`` — channels LAST, which is also this framework's tensor layout.
            Only ``C`` reaches the parameter count.
        widths: Channels per encoder level. Changing it breaks comparability with the other
            implementations — see :mod:`deltaid.unet.spec`.
        bottleneck_width: Channels at the bottom of the U.
        seed: PRNG seed for parameter init. JAX makes randomness explicit, so this is a real knob
            rather than a global ``set_seed`` — the same config gives the same weights.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = ENCODER_WIDTHS,
        bottleneck_width: int = BOTTLENECK_WIDTH,
        seed: int = 0,
    ) -> None:
        # Partial: store only. No parameter allocation — the workspace zero-arg mandate.
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.bottleneck_width = bottleneck_width
        self.seed = seed
        self._model: Optional[Any] = None

    def solidify(self) -> Any:
        """Build (once) and return the live inner ``nnx.Module`` — confluid's ``flow()``-time hook."""
        if self._model is not None:
            return self._model
        if not self.num_classes:
            raise ValueError(
                "FlaxUNet.num_classes is not set — a segmenter flows the resolved count in at "
                "run time, or set it explicitly."
            )
        # `nnx.data` is required, not decorative — see the module docstring.
        self._model = nnx.data(
            _UNet(
                num_classes=self.num_classes,
                in_channels=int(self.input_shape[-1]),
                widths=self.widths,
                bottleneck_width=self.bottleneck_width,
                seed=self.seed,
            )
        )
        logger.info(f"Built the shared U-Net spec in flax.nnx ({self.num_classes} classes, widths={self.widths}).")
        return self._model

    @property
    def model(self) -> Any:
        """The live inner ``nnx.Module``, built on first access.

        The ergonomic accessor for a directly-constructed instance that was never flowed —
        ``flow()`` does not touch property getters, which is why :meth:`solidify` exists too.
        """
        return self.solidify()

    def __call__(self, x: Any, training: bool = False) -> Any:
        """Per-pixel class logits from an ``[N, H, W, C]`` batch — channels LAST."""
        divisor = self.size_divisor
        height, width = int(x.shape[1]), int(x.shape[2])
        if height % divisor or width % divisor:
            # Checked up front: the concat would otherwise raise naming two intermediate arrays,
            # which says nothing about the input the caller gave.
            raise ValueError(
                f"FlaxUNet: H and W must each be divisible by {divisor} "
                f"(2 ** {len(self.widths)} encoder levels), got {height}x{width}."
            )
        return self.model(x, training=training)

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`deltaid.unet.spec` predicts.

        Counted over ``nnx.Param`` only, which is what excludes BatchNorm's running mean and
        variance: nnx files those as ``nnx.BatchStat``, a different variable type in the same
        state tree. The spec counts neither, and each implementation excludes them the way its
        own framework requires.
        """
        import jax

        self.solidify()
        return int(sum(int(v.size) for v in jax.tree_util.tree_leaves(nnx.state(self, nnx.Param))))

    @property
    def size_divisor(self) -> int:
        """The multiple ``H`` and ``W`` must each be, for the skips to line up on the way back."""
        return size_divisor(self.widths, POOL_SIZE)

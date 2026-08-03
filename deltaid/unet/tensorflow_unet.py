"""``TensorFlowUNet`` — the shared U-Net spec in raw ``tf.nn`` ops.

The TensorFlow entry in the comparison :mod:`deltaid.unet.spec` describes, and it wraps nothing
at all. It is what makes that comparison include a real TensorFlow implementation rather than
Keras-on-TF standing in for one — :class:`deltaid.unet.keras_unet.KerasUNet` is already the Keras
entry, and it runs on whichever engine Keras is bound to.

Tagged ``framework="keras"`` rather than ``"tensorflow"``, which looks wrong and is deliberate:
the framework axis names the API a class is **wired to**, and ``TensorFlowSegmenter`` — the only
trainer that can consume this — declares ``"keras"`` itself. A ``"tensorflow"`` tag would be truer
to what the code imports and would hide the class from the one slot picker that should offer it.

**It is a ``tf.Module``, not a ``keras.Model``.** A ``tf.Module`` gives ``trainable_variables``
and ``__call__`` — everything a traced loop needs — and nothing else, which is the honest shape
for something that wraps no framework machinery.

BatchNorm is written out rather than borrowed, because borrowing it would mean
``keras.layers.BatchNormalization`` and then this would be a Keras model wearing a TensorFlow
name. Its moving statistics are ``trainable=False`` variables, so they are STATE rather than
parameters — which is what lets the parameter count match the other implementations, since the
frameworks disagree about whether to count them.
"""

from typing import Any, List, Optional, Tuple

from confluid import configurable
from loggair import get_logger

# `deltaid._tensorflow` pins KERAS_BACKEND=tensorflow BEFORE importing keras; importing `tf`
# through it keeps that ordering in one place.
from deltaid._tensorflow import tf
from deltaid.unet.spec import BOTTLENECK_WIDTH, ENCODER_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE, size_divisor

logger = get_logger(__name__)

__all__ = ["TensorFlowUNet"]


def _glorot_uniform(shape: Tuple[int, ...], seed: int) -> Any:
    """Glorot-uniform weights from ``tf.random``, written out rather than taken from an initializer.

    **`tf.initializers` IS Keras**, not TensorFlow: since TF 2.16 `tf.keras` is Keras 3, and
    `tf.initializers.GlorotUniform` resolves to `keras.initializers.GlorotUniform`. Called in a
    process where Keras is bound to the TORCH backend — which is exactly this workspace, since
    `recordstream.keras` defaults `KERAS_BACKEND` to the first engine installed and torch wins —
    it returns a **torch tensor**, and `tf.Variable` then dies converting an MPS tensor to numpy
    (measured: *"can't convert mps:0 device type tensor to numpy"*). So a module claiming to be
    raw `tf.nn` cannot reach for `tf.initializers` at all, and the four lines below are what
    "raw TensorFlow" actually costs.

    ``stateless_uniform`` rather than ``tf.random.uniform``: the seed is the whole argument for a
    reproducible model, and the stateful op reads a GLOBAL seed that another library can move.

    The limit is Glorot's: ``sqrt(6 / (fan_in + fan_out))``, with the fans computed exactly as
    ``keras.src.initializers.random_initializers.compute_fans`` does — so the weights this
    produces are drawn from the same distribution the other implementations' initializers use,
    and "the same network" stays true of more than the parameter COUNT.
    """
    receptive_field = 1
    for dim in shape[:-2]:
        receptive_field *= int(dim)
    if len(shape) == 2:  # a Dense kernel `(in, out)` — no receptive field at all
        fan_in, fan_out = int(shape[0]), int(shape[1])
    else:  # a conv kernel `(..., in, out)`
        fan_in, fan_out = int(shape[-2]) * receptive_field, int(shape[-1]) * receptive_field
    limit = (6.0 / float(fan_in + fan_out)) ** 0.5
    return tf.random.stateless_uniform(shape, seed=(seed, 0), minval=-limit, maxval=limit, dtype=tf.float32)


class _ConvBlock(tf.Module):
    """``[conv -> BN -> ReLU] x2`` — the spec's repeated unit, in raw ``tf.nn`` ops.

    NOTE it does NOT pool. In a U-Net the skip connection is taken from the block's output
    *before* the downsample, so the pooling belongs to the caller — unlike the classification
    convnet, whose block owns its pool because nothing reads the pre-pool activation.
    """

    def __init__(self, in_channels: int, width: int, seed: int, momentum: float = 0.99) -> None:
        super().__init__(name=f"block_{in_channels}_{width}")
        self.momentum = momentum
        self.kernels: List[Any] = []
        self.biases: List[Any] = []
        self.gammas: List[Any] = []
        self.betas: List[Any] = []
        self.moving_means: List[Any] = []
        self.moving_vars: List[Any] = []
        channels = in_channels
        for index in range(2):
            kernel = _glorot_uniform((KERNEL_SIZE, KERNEL_SIZE, channels, width), seed=seed + index)
            self.kernels.append(tf.Variable(kernel, name=f"kernel_{index}"))
            self.biases.append(tf.Variable(tf.zeros((width,)), name=f"bias_{index}"))
            self.gammas.append(tf.Variable(tf.ones((width,)), name=f"gamma_{index}"))
            self.betas.append(tf.Variable(tf.zeros((width,)), name=f"beta_{index}"))
            self.moving_means.append(tf.Variable(tf.zeros((width,)), trainable=False, name=f"moving_mean_{index}"))
            self.moving_vars.append(tf.Variable(tf.ones((width,)), trainable=False, name=f"moving_var_{index}"))
            channels = width

    def __call__(self, x: Any, training: bool = False) -> Any:
        for index in range(2):
            x = tf.nn.conv2d(x, self.kernels[index], strides=1, padding="SAME") + self.biases[index]
            if training:
                # Batch statistics while training, folded into the moving estimates the eval
                # path uses — the update Keras/torch do for you inside their layer.
                mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
                self.moving_means[index].assign(self.moving_means[index] * self.momentum + mean * (1.0 - self.momentum))
                self.moving_vars[index].assign(
                    self.moving_vars[index] * self.momentum + variance * (1.0 - self.momentum)
                )
            else:
                mean, variance = self.moving_means[index], self.moving_vars[index]
            x = tf.nn.batch_normalization(
                x, mean, variance, self.betas[index], self.gammas[index], variance_epsilon=1e-3
            )
            x = tf.nn.relu(x)
        return x


class _UNet(tf.Module):
    """The spec's layer graph: encoder + skips → bottleneck → decoder → 1x1 head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        widths: Tuple[int, ...],
        bottleneck_width: int,
        seed: int,
    ) -> None:
        super().__init__(name="deltaid_unet")
        self.encoders: List[_ConvBlock] = []
        channels = in_channels
        for index, width in enumerate(widths):
            self.encoders.append(_ConvBlock(channels, width, seed=seed + index * 2))
            channels = width

        self.bottleneck = _ConvBlock(channels, bottleneck_width, seed=seed + len(widths) * 2)
        channels = bottleneck_width

        # The decoder's input at each level is `channels + skip` because of the concat — what
        # makes a U-Net's decoder wider than its encoder, and the whole mechanism.
        self.decoders: List[_ConvBlock] = []
        for index, width in enumerate(reversed(widths)):
            self.decoders.append(_ConvBlock(channels + width, width, seed=seed + (len(widths) + index + 1) * 2))
            channels = width

        head_kernel = _glorot_uniform((1, 1, channels, num_classes), seed=seed + 999)
        self.head_kernel = tf.Variable(head_kernel, name="head_kernel")
        self.head_bias = tf.Variable(tf.zeros((num_classes,)), name="head_bias")

    def __call__(self, x: Any, training: bool = False) -> Any:
        skips: List[Any] = []
        for encoder in self.encoders:
            x = encoder(x, training=training)
            skips.append(x)
            x = tf.nn.max_pool2d(x, ksize=POOL_SIZE, strides=POOL_SIZE, padding="VALID")

        x = self.bottleneck(x, training=training)

        for decoder in self.decoders:
            skip = skips.pop()
            # NEAREST resize — parameter-free, per the spec. Sized from the SKIP rather than by
            # a fixed factor, so an odd intermediate size cannot silently misalign the concat.
            x = tf.image.resize(x, tf.shape(skip)[1:3], method="nearest")
            x = tf.concat([x, skip], axis=-1)
            x = decoder(x, training=training)

        # 1x1: one logit per class, per pixel. Raw — no activation.
        return tf.nn.conv2d(x, self.head_kernel, strides=1, padding="SAME") + self.head_bias


@configurable(task="segmentation", role="model", lazy=True, framework="keras")
class TensorFlowUNet(tf.Module):
    """The shared U-Net spec (:mod:`deltaid.unet.spec`), implemented in raw ``tf.nn`` ops.

    Args:
        num_classes: Width of the per-pixel head. Sentinel ``0`` is invalid but permitted so the
            class is zero-arg constructible; a segmenter flows the resolved count in at run time.
        input_shape: ``(H, W, C)`` — channels LAST, which is also this engine's tensor layout.
        widths: Channels per encoder level. Changing it breaks comparability with the other
            implementations — see :mod:`deltaid.unet.spec`.
        bottleneck_width: Channels at the bottom of the U.
        seed: Seed for the variable initializers, so a config reproduces its weights.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = ENCODER_WIDTHS,
        bottleneck_width: int = BOTTLENECK_WIDTH,
        seed: int = 0,
    ) -> None:
        # `tf.Module.__init__` first — it sets up the name scope the variables register in.
        super().__init__(name="deltaid_tensorflow_unet")
        # Lazy: store only. No variables allocated until `solidify()`.
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.bottleneck_width = bottleneck_width
        self.seed = seed
        self._model: Optional[_UNet] = None

    def solidify(self) -> Any:
        """Build (once) and return the live network — confluid's ``flow()``-time hook."""
        if self._model is None:
            if not self.num_classes:
                raise ValueError(
                    "TensorFlowUNet.num_classes is not set — a segmenter flows the resolved "
                    "count in at run time, or set it explicitly."
                )
            tf.random.set_seed(self.seed)
            self._model = _UNet(
                num_classes=self.num_classes,
                in_channels=int(self.input_shape[-1]),
                widths=self.widths,
                bottleneck_width=self.bottleneck_width,
                seed=self.seed,
            )
            logger.info(f"Built the shared U-Net spec in tf.nn ({self.num_classes} classes, widths={self.widths}).")
        return self._model

    @property
    def model(self) -> Any:
        """The live network, built on first access."""
        return self.solidify()

    def __call__(self, x: Any, training: bool = False) -> Any:
        """Per-pixel class logits from an ``[N, H, W, C]`` batch — channels LAST."""
        return self.model(x, training=training)

    @property
    def trainable_variables(self) -> Any:
        """The variables the optimizer updates.

        Building on ACCESS, so a segmenter can call ``optimizer.build(model.trainable_variables)``
        before the first traced step — Keras allocates its slot variables on the first
        ``apply_gradients``, and creating variables during a retrace is an error.
        """
        return self.model.trainable_variables

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`deltaid.unet.spec` predicts.

        The moving statistics are ``trainable=False``, so they are excluded by construction —
        the same exclusion the spec makes.
        """
        return int(sum(int(tf.size(v)) for v in self.trainable_variables))

    @property
    def size_divisor(self) -> int:
        """The multiple ``H`` and ``W`` must each be, for the skips to line up on the way back."""
        return size_divisor(self.widths, POOL_SIZE)

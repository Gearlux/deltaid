"""``KerasUNet`` — the shared U-Net spec in ``keras.layers``.

The Keras entry in the comparison :mod:`deltaid.unet.spec` describes, and the opposite kind of
model from the zoo wrapper: :class:`deltaid.smp_model.SegmentationModelPytorch` reaches twelve
architectures over ~800 pretrained encoders, this one wraps nothing at all. It exists because
**smp is torch-only** — there is no Keras equivalent to reach for, so without this the Keras
backend would have no segmentation model and the cross-framework comparison would stop at torch.

Being written in ``keras.layers`` it runs on whichever engine Keras is bound to, so BOTH the
``keras`` and ``tensorflow`` backends wire it. That is also why it is not called
``TensorFlowUNet``: it is not TensorFlow-specific. The genuinely TF-native implementation, in raw
``tf.nn`` ops, is its sibling :mod:`deltaid.unet.tensorflow_unet`.

**Channels-LAST**, like every Keras model: ``call`` takes ``[N, H, W, C]`` and returns
``[N, H, W, num_classes]``. The ``input_shape`` knob is ``(H, W, C)`` in every implementation —
here it happens to match the tensor, where torch's does not.

It IS a ``keras.Model``, the same decision as `SegmentationModelPytorch` being an ``nn.Module``:
a segmenter drives whatever is in its ``model`` slot without asking what it got.
"""

from typing import Any, Optional, Tuple

from confluid import configurable
from loggair import get_logger

from deltaid._keras import keras, keras_backend
from deltaid.unet.spec import BOTTLENECK_WIDTH, ENCODER_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE, size_divisor

logger = get_logger(__name__)

__all__ = ["KerasUNet"]


def _conv_block(x: Any, width: int) -> Any:
    """``[conv -> BN -> ReLU] x2`` — the spec's repeated unit, in Keras's vocabulary."""
    for _ in range(2):
        x = keras.layers.Conv2D(width, KERNEL_SIZE, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
    return x


@configurable(task="segmentation", role="model", lazy=True, framework="keras")
class KerasUNet(keras.Model):
    """The shared U-Net spec (:mod:`deltaid.unet.spec`), implemented in ``keras.layers``.

    Args:
        num_classes: Width of the per-pixel head. Sentinel ``0`` is invalid but permitted so the
            class is zero-arg constructible; a segmenter flows the resolved count in at run time.
        input_shape: ``(H, W, C)`` — channels LAST, which is also this engine's tensor layout.
        widths: Channels per encoder level. Changing it breaks comparability with the other
            implementations — see :mod:`deltaid.unet.spec`.
        bottleneck_width: Channels at the bottom of the U.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = ENCODER_WIDTHS,
        bottleneck_width: int = BOTTLENECK_WIDTH,
    ) -> None:
        # Lazy: store only, and `super().__init__()` FIRST — a `keras.Layer` refuses attribute
        # assignment before it.
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.bottleneck_width = bottleneck_width
        self._model: Optional[Any] = None

    def solidify(self) -> Any:
        """Build (once) and return the live ``keras.Model`` — confluid's ``flow()``-time hook."""
        if self._model is not None:
            return self._model
        if not self.num_classes:
            raise ValueError(
                "KerasUNet.num_classes is not set — a segmenter flows the resolved count in at "
                "run time, or set it explicitly."
            )
        # `(None, None, C)`, NOT the configured `(H, W, C)`. A U-Net is fully convolutional, so
        # the torch implementation accepts any size the encoder can halve cleanly — and Keras's
        # Functional API PINS the spatial dims unless they are None, so building at the
        # configured size makes the model reject every other one (measured: a 224-built model
        # raises *"expected shape=(None, 224, 224, 3), found shape=(2, 64, 64, 3)"*). Only `C`
        # is fixed, which is also the only part of `input_shape` the parameter count reads.
        inputs = keras.Input(shape=(None, None, int(self.input_shape[-1])))
        x = inputs
        skips = []
        for width in self.widths:
            x = _conv_block(x, width)
            skips.append(x)
            x = keras.layers.MaxPooling2D(POOL_SIZE)(x)

        x = _conv_block(x, self.bottleneck_width)

        for width in reversed(self.widths):
            # `UpSampling2D` defaults to NEAREST and has no parameters — the spec's choice, so
            # the implementations stay comparable (a learned transpose conv's parameter count
            # depends on padding conventions that differ between frameworks).
            x = keras.layers.UpSampling2D(POOL_SIZE)(x)
            x = keras.layers.Concatenate()([x, skips.pop()])
            x = _conv_block(x, width)

        # 1x1: one logit per class, per pixel. Raw — no activation, so the same
        # `from_logits=True` losses apply as everywhere else in this project.
        outputs = keras.layers.Conv2D(self.num_classes, 1, name="predictions")(x)
        self._model = keras.Model(inputs, outputs, name="deltaid_unet")
        logger.info(
            f"Built the shared U-Net spec in keras ({keras_backend()} backend, "
            f"{self.num_classes} classes, widths={self.widths})."
        )
        return self._model

    @property
    def model(self) -> Any:
        """The live ``keras.Model``, built on first access."""
        return self.solidify()

    def build(self, input_shape: Any = None) -> None:
        """Keras's build hook — the ONE place a subclassed model may create state.

        A network created on the first ``call()`` dies with *"You cannot add new elements of
        state … to a layer that is already built"*, and ``build()`` runs before the first call,
        which is what keeps a never-flowed instance usable.
        """
        self.solidify()
        super().build(input_shape)

    def call(self, x: Any, training: bool = False) -> Any:
        """Per-pixel class logits from an ``[N, H, W, C]`` batch — channels LAST."""
        return self.model(x, training=training)

    @property
    def output_shape(self) -> Any:
        """Functional-only on a subclassed model, so it is forwarded."""
        return self.model.output_shape

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`deltaid.unet.spec` predicts.

        BatchNorm's moving mean/variance are Keras NON-trainable weights, so reading
        ``trainable_weights`` excludes them by construction — the same exclusion the spec makes.
        """
        return int(sum(int(w.size if hasattr(w, "size") else w.numpy().size) for w in self.model.trainable_weights))

    @property
    def size_divisor(self) -> int:
        """The multiple ``H`` and ``W`` must each be, for the skips to line up on the way back."""
        return size_divisor(self.widths, POOL_SIZE)

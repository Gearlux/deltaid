"""The ONE U-Net architecture, written once so several frameworks can implement it.

Every other model in deltaid wraps somebody's zoo — ``segmentation_models_pytorch``, and later a
Hub checkpoint. This is the opposite: a small network specified here and implemented natively per
framework. It exists because **the zoos do not line up**. smp is torch-only: there is no smp for
Keras, Flax or MLX, so a cross-framework comparison built on zoos would compare four different
networks and tell you nothing about the frameworks. Implementing ONE architecture several times
makes the comparison mean something.

For real work prefer the zoo wrapper — a small net trained from scratch loses to a pretrained
encoder every time. This is the control, not the recommendation.

**The architecture** — a textbook U-Net, deliberately small::

    input (H, W, C)
      encoder:  for w in widths:            # (16, 32, 64)
                    x = conv_block(x, w)    # [conv3x3(w) -> BN -> ReLU] x2
                    skips.append(x)
                    x = maxpool(2)
      bottleneck: x = conv_block(x, bottleneck_width)          # 128
      decoder:  for w in reversed(widths):
                    x = upsample(2)         # NEAREST — parameter-free, see below
                    x = concat([x, skips.pop()])               # the U-Net skip
                    x = conv_block(x, w)
      head:     conv1x1(num_classes)        # raw per-pixel logits, NO activation

The head emits raw logits like every other model in this project, so the same
``CrossEntropyLoss`` (which applies its own log-softmax) and the same logit-space metrics apply.

**Three choices that exist to keep the implementations comparable, not because they are best:**

* **Upsampling is parameter-free** (nearest-neighbour), not a learned ``ConvTranspose2d``. A
  transposed convolution's parameter count and output size depend on padding conventions that
  differ between frameworks — so it would make "the same network" untrue by construction.
* **The spatial size must halve cleanly** ``len(widths)`` times, so ``H`` and ``W`` must each be
  divisible by ``2 ** len(widths)``. With three levels that is 8; 224 works.
* **Layout is per-implementation, and only the tensor differs.** Keras, TensorFlow, Flax and MLX
  default to NHWC; torch is NCHW. The ``input_shape`` KNOB stays ``(H, W, C)`` in every
  implementation so one config line drives any of them, and only ``C`` reaches the parameter
  count.

**The constants below are the single source of truth**, imported by each implementation for its
constructor defaults, so "same architecture" is true by construction for the knobs rather than by
several copies agreeing. The LAYER ORDER is prose — different APIs cannot share code — and is held
together instead by :func:`trainable_parameter_count`, which DERIVES the count from the
architecture. Each implementation is checked against that CLAIM rather than against whichever one
was written first.
"""

from typing import Tuple

__all__ = [
    "BOTTLENECK_WIDTH",
    "ENCODER_WIDTHS",
    "INPUT_SHAPE",
    "KERNEL_SIZE",
    "POOL_SIZE",
    "size_divisor",
    "trainable_parameter_count",
]

#: Channels per encoder level, doubling. Three levels — small enough to train in seconds on CPU,
#: deep enough that the skip connections and the up/down path actually matter to the comparison.
ENCODER_WIDTHS: Tuple[int, ...] = (16, 32, 64)

#: Channels at the bottom of the U, below the last encoder level.
BOTTLENECK_WIDTH: int = 128

#: Square conv kernel, ``padding="same"`` so only the pooling changes the spatial size.
KERNEL_SIZE: int = 3

#: Square max-pool window and stride — and the upsampling factor that undoes it.
POOL_SIZE: int = 2

#: Default input, matching the other model wrappers' ``(H, W, C)`` — channels LAST.
INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)


def size_divisor(widths: Tuple[int, ...] = ENCODER_WIDTHS, pool: int = POOL_SIZE) -> int:
    """The multiple ``H`` and ``W`` must each be, for the skips to line up on the way back.

    Every encoder level halves the spatial size and every decoder level doubles it; a level whose
    input was odd comes back one pixel short and the concat fails. Stated as a function rather
    than a constant because it follows from ``widths``, and an implementation that exposes
    ``widths`` as a knob must be able to recompute it.
    """
    return int(pool ** len(widths))


def _conv_block(in_channels: int, width: int, kernel_size: int) -> int:
    """Parameters in one ``[conv -> BN -> ReLU] x2`` block.

    Per conv: ``k*k*cin*cout`` weights + ``cout`` biases. Per BatchNorm: ``2*cout`` (scale and
    shift). Running mean/variance are STATE, not parameters, and the frameworks disagree about
    whether to include them — so they are excluded here, and each implementation is responsible
    for excluding them the way its own framework requires.
    """
    k = kernel_size * kernel_size
    first = k * in_channels * width + width + 2 * width
    second = k * width * width + width + 2 * width
    return first + second


def trainable_parameter_count(
    num_classes: int,
    widths: Tuple[int, ...] = ENCODER_WIDTHS,
    bottleneck_width: int = BOTTLENECK_WIDTH,
    in_channels: int = 3,
    kernel_size: int = KERNEL_SIZE,
) -> int:
    """The parameter count the spec implies — what every implementation must report.

    Derived from the architecture rather than measured from one framework, so it is a CLAIM the
    implementations are each checked against instead of one of them being the reference.

    With the defaults and 3 classes this is 488,739.
    """
    total = 0
    channels = in_channels
    for width in widths:
        total += _conv_block(channels, width, kernel_size)
        channels = width

    total += _conv_block(channels, bottleneck_width, kernel_size)
    channels = bottleneck_width

    # The decoder mirrors the encoder. Each level upsamples (no parameters), CONCATENATES the
    # skip from the matching encoder level, then runs one conv block — so its input width is
    # `channels + skip`, which is what makes a U-Net's decoder wider than its encoder.
    for width in reversed(widths):
        total += _conv_block(channels + width, width, kernel_size)
        channels = width

    # The head is 1x1: one logit per class, per pixel.
    return total + channels * num_classes + num_classes

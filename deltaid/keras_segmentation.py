"""The **keras** segmentation losses and metrics — registered, never wrapped.

One module per ENGINE, because a loss is engine-bound: a ``torch.nn`` loss cannot be handed to
``KerasSegmenter`` and a ``keras.losses`` one cannot be handed to ``LightningSegmenter``. The torch
twin is :mod:`deltaid.torch_segmentation`; each is its own ``confluid.configurables`` entry point,
so on an install without Keras this module simply fails to import and a discovery scan skips it —
which is the correct answer, not a degraded one.

Keras is reached as ``keras.losses.X`` / ``keras.metrics.X`` ATTRIBUTES via
``from deltaid._keras import keras`` — NEVER ``from keras.losses import X``, which isort sorts
above the first-party import and would beat ``recordstream.keras``'s ``KERAS_BACKEND`` default
into place.

**The target contract decides the set, and it is the same split the torch side spells as
`Multiclass*` vs `Multilabel*`.** A segmentation target here is an ``[N, H, W]`` int64 class-id
map against ``[N, H, W, C]`` raw logits (channels LAST — this is Keras), so the offered set is the
SPARSE family plus the metrics that can be told to argmax for themselves:

* **`SparseCategoricalCrossentropy`** — the default, and ``from_logits=True`` is MANDATORY. The
  head is a bare 1x1 conv with no activation, and Keras's string/default losses expect
  PROBABILITIES: handed logits they normalize by the row sum instead of a softmax and then CLIP
  rather than raise, so the loss stays finite and merely WRONG. Measured on one batch:
  ``from_logits=True`` gives 1.4546, matching ``torch.nn.CrossEntropyLoss`` exactly, while
  ``from_logits=False`` gives 4.6293.
* **`MeanIoU` / `IoU` / `OneHotMeanIoU`** — the segmentation scores. ``MeanIoU`` needs
  **``sparse_y_pred=False``** to accept raw logits (it then argmaxes internally); with the default
  ``sparse_y_pred=True`` it raises a shape error (measured: *"The size of tensor a (128) must
  match tensor b"*). Its ``ignore_class`` is the equivalent of torch's ``ignore_index``.
* **`SparseCategoricalAccuracy`** — per-pixel accuracy; reads the ``[N, H, W]`` int ids directly.

The ``*Crossentropy`` NAMESAKES are registered as losses only. Keras publishes several names under
both ``keras.losses`` and ``keras.metrics``; a config picks one with ``!class:X@role=loss`` or the
dotted path, and a bare ambiguous name raises naming both.

Typical wiring (YAML)::

    loss: !class:SparseCategoricalCrossentropy@framework=keras
      from_logits: true

    metrics:
      - !class:keras.metrics.MeanIoU {num_classes: 3, sparse_y_pred: false}
"""

from confluid import register

from deltaid._keras import keras

# Bound to module-level names so `scan_module`'s `__all__`-gated pass picks them up: it filters
# by `__module__`, which for every class here is `keras.src.*`.
SparseCategoricalCrossentropy = keras.losses.SparseCategoricalCrossentropy
CategoricalCrossentropy = keras.losses.CategoricalCrossentropy
BinaryCrossentropy = keras.losses.BinaryCrossentropy
CategoricalFocalCrossentropy = keras.losses.CategoricalFocalCrossentropy
BinaryFocalCrossentropy = keras.losses.BinaryFocalCrossentropy

MeanIoU = keras.metrics.MeanIoU
IoU = keras.metrics.IoU
OneHotMeanIoU = keras.metrics.OneHotMeanIoU
SparseCategoricalAccuracy = keras.metrics.SparseCategoricalAccuracy

#: Off-the-shelf losses a per-pixel Keras run can wire. The sparse one is the default; the rest
#: serve a one-hot or binary target shape a config may deliberately produce.
_SEGMENTATION_LOSSES = (
    SparseCategoricalCrossentropy,
    CategoricalCrossentropy,
    BinaryCrossentropy,
    CategoricalFocalCrossentropy,
    BinaryFocalCrossentropy,
)

#: The metrics that CAN consume this project's raw-logits head. A metric that cannot is not
#: offered — not registering removes it from PICKERS only; a dotted path still resolves from YAML.
_SEGMENTATION_METRICS = (
    MeanIoU,
    IoU,
    OneHotMeanIoU,
    SparseCategoricalAccuracy,
)

for _loss in _SEGMENTATION_LOSSES:
    register(_loss, task="segmentation", role="loss", framework="keras")

for _metric in _SEGMENTATION_METRICS:
    register(_metric, task="segmentation", role="metric", framework="keras")


# Every registered class MUST be named here: `confluid.discovery.scan_module` filters members by
# `__module__` (`keras.src.*` for these), so the `__all__`-gated pass is what carries them into
# the StreamStudio palette. `tests/test_categories.py` pins this list against the registrations.
__all__ = [
    # losses — role="loss"
    "SparseCategoricalCrossentropy",
    "CategoricalCrossentropy",
    "BinaryCrossentropy",
    "CategoricalFocalCrossentropy",
    "BinaryFocalCrossentropy",
    # metrics — role="metric"
    "MeanIoU",
    "IoU",
    "OneHotMeanIoU",
    "SparseCategoricalAccuracy",
]

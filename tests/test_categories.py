# mypy: disable-error-code="attr-defined"
"""Discovery tags for deltaid's ``@configurable`` classes — task / role / framework.

These drive navigaitor's ``list_classes(task=, role=, framework=)`` scan, its generated task
configs, and the visual-editor form-spec picker. A dropped tag empties the corresponding picker
SILENTLY — the class still resolves from a dotted path in YAML, so nothing else notices — which is
why they are pinned here rather than left to be discovered by a user with a short menu.
"""

import importlib.util
from typing import Any, Tuple

from confluid.registry import get_registry

import deltaid.keras_segmentation as keras_segmentation
import deltaid.torch_segmentation as torch_segmentation
from deltaid.huggingface_model import HuggingFaceSegmentationModel
from deltaid.huggingface_segmenter import HuggingFaceSegmenter
from deltaid.keras_segmenter import KerasSegmenter
from deltaid.lightning_segmenter import LightningSegmenter
from deltaid.smp_model import SegmentationModelPytorch

#: `deltaid.tensorflow_segmenter` imports tensorflow at import time, and `tensorflow` is a
#: SELECTABLE extra this workspace may not have installed. The tags it carries are still worth
#: pinning, so the two tests below reach for it only where the extra is present — every OTHER
#: backend's tags stay checked on every machine, which is the coverage that matters here.
_HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


def test_model_tags() -> None:
    assert SegmentationModelPytorch.__confluid_task__ == "segmentation"
    assert SegmentationModelPytorch.__confluid_role__ == "model"
    assert SegmentationModelPytorch.__confluid_framework__ == "torch"
    #: The derived legacy category, which the older pickers still read.
    assert SegmentationModelPytorch.__confluid_category__ == "segmentation_model"


def test_segmenter_tags() -> None:
    """One class, `role="trainer"` — its evaluate/test/predict capabilities are the SAME class.

    They are declared by the `@entrypoint`-annotated methods, not by a second class and not by a
    `role="evaluator"` tag, so there is intentionally no separate evaluator here.
    """
    assert LightningSegmenter.__confluid_task__ == "segmentation"
    assert LightningSegmenter.__confluid_role__ == "trainer"
    assert LightningSegmenter.__confluid_framework__ == "torch"


def test_tags_enumerable_via_registry() -> None:
    registry = get_registry()
    assert "SegmentationModelPytorch" in registry.list_classes(task="segmentation", role="model")
    assert "LightningSegmenter" in registry.list_classes(task="segmentation", role="trainer")
    assert "SegmentationModelPytorch" in registry.list_classes(category="segmentation_model")


def test_the_framework_axis_separates_engines() -> None:
    """`framework` says what a class can be WIRED TO, so a picker cannot offer a mismatch."""
    registry = get_registry()
    assert {"SegmentationModelPytorch", "LightningSegmenter"} <= registry.list_classes(
        task="segmentation", framework="torch"
    )


def test_losses_and_metrics_are_registered_with_their_roles() -> None:
    registry = get_registry()
    losses = registry.list_classes(task="segmentation", role="loss", framework="torch")
    metrics = registry.list_classes(task="segmentation", role="metric", framework="torch")
    # The default loss, and the region-overlap family a class-imbalanced run actually wants.
    assert {"CrossEntropyLoss", "DiceLoss", "JaccardLoss", "FocalLoss", "TverskyLoss"} <= losses
    # The segmenter's own default pair, plus the wider multiclass family.
    assert {"MulticlassJaccardIndex", "MulticlassF1Score", "MulticlassConfusionMatrix"} <= metrics


def test_all_exports_every_registered_loss_and_metric() -> None:
    """`__all__` is LOAD-BEARING here, not decoration.

    ``confluid.discovery.scan_module`` filters members by ``__module__`` — which for these is
    ``torch`` / ``torchmetrics`` / ``segmentation_models_pytorch``, never deltaid — so the
    ``__all__``-gated pass is the ONLY thing that carries them into the StreamStudio palette. A
    class registered but not exported is registered nowhere a GUI can see.
    """
    registered = {cls.__name__ for cls in torch_segmentation._SEGMENTATION_LOSSES}
    registered |= {cls.__name__ for cls in torch_segmentation._SEGMENTATION_METRICS}
    assert registered == set(torch_segmentation.__all__), (
        "the registrations and __all__ have drifted: "
        f"registered-not-exported={sorted(registered - set(torch_segmentation.__all__))}, "
        f"exported-not-registered={sorted(set(torch_segmentation.__all__) - registered)}"
    )


def test_the_loss_is_a_plain_upstream_class_not_a_deltaid_wrapper() -> None:
    """The workspace model: tag and register what exists; never reimplement Dice or Focal."""
    for cls in torch_segmentation._SEGMENTATION_LOSSES:
        assert not cls.__module__.startswith("deltaid"), f"{cls.__name__} is a deltaid-owned loss"


# --------------------------------------------------------------------------- #
# The Keras engine — its own module, its own tag, its own entry point
# --------------------------------------------------------------------------- #
def test_the_keras_losses_and_metrics_are_registered_under_their_own_framework() -> None:
    """A loss is ENGINE-BOUND: a `torch.nn` one cannot be handed to `KerasSegmenter`, so the two
    sets are tagged apart and a picker filtered by `framework=` cannot offer a mismatch."""
    registry = get_registry()
    losses = registry.list_classes(task="segmentation", role="loss", framework="keras")
    metrics = registry.list_classes(task="segmentation", role="metric", framework="keras")
    assert {"SparseCategoricalCrossentropy", "CategoricalFocalCrossentropy"} <= losses
    assert {"MeanIoU", "IoU", "SparseCategoricalAccuracy"} <= metrics


def test_the_two_engines_registrations_do_not_overlap() -> None:
    """The reason `framework=` exists on the axis at all: `CrossEntropyLoss` and
    `SparseCategoricalCrossentropy` mean the same thing and are NOT interchangeable objects."""
    registry = get_registry()
    torch_losses = registry.list_classes(task="segmentation", role="loss", framework="torch")
    keras_losses = registry.list_classes(task="segmentation", role="loss", framework="keras")
    assert torch_losses & keras_losses == set()


def test_the_keras_module_exports_every_class_it_registers() -> None:
    """Same `__all__` rule as the torch module, for the same reason: `scan_module` filters by
    `__module__` — `keras.src.*` here — so the `__all__`-gated pass is the ONLY thing carrying
    these into the StreamStudio palette."""
    registered = {cls.__name__ for cls in keras_segmentation._SEGMENTATION_LOSSES}
    registered |= {cls.__name__ for cls in keras_segmentation._SEGMENTATION_METRICS}
    assert registered == set(keras_segmentation.__all__), (
        "the registrations and __all__ have drifted: "
        f"registered-not-exported={sorted(registered - set(keras_segmentation.__all__))}, "
        f"exported-not-registered={sorted(set(keras_segmentation.__all__) - registered)}"
    )


def test_the_keras_loss_is_a_plain_upstream_class_too() -> None:
    for cls in keras_segmentation._SEGMENTATION_LOSSES:
        assert not cls.__module__.startswith("deltaid"), f"{cls.__name__} is a deltaid-owned loss"


# --------------------------------------------------------------------------- #
# Every backend carries the same three tags
# --------------------------------------------------------------------------- #
def test_every_segmenter_is_a_segmentation_trainer() -> None:
    """A dropped `role` silently empties navigaitor's `role="trainer"` scan — which is what
    builds `TASK_REGISTRY` and every generated task config."""
    backends: Tuple[Any, ...] = (LightningSegmenter, KerasSegmenter, HuggingFaceSegmenter)
    if _HAS_TENSORFLOW:
        from deltaid.tensorflow_segmenter import TensorFlowSegmenter

        backends += (TensorFlowSegmenter,)
    for cls in backends:
        assert cls.__confluid_task__ == "segmentation", cls.__name__
        assert cls.__confluid_role__ == "trainer", cls.__name__


def test_the_framework_tag_names_the_api_a_class_is_wired_to() -> None:
    """`TensorFlowSegmenter` declares `keras`, NOT `tensorflow`, and that is deliberate: the axis
    names what its SLOTS admit, and every one of them holds a `keras.losses` / `keras.metrics` /
    `keras.Model`. Its TensorFlow-ness is the loop, which no picker wires."""
    assert KerasSegmenter.__confluid_framework__ == "keras"
    assert HuggingFaceSegmenter.__confluid_framework__ == "huggingface"
    if _HAS_TENSORFLOW:
        from deltaid.tensorflow_segmenter import TensorFlowSegmenter

        assert TensorFlowSegmenter.__confluid_framework__ == "keras"


def test_the_hub_wrapper_is_a_segmentation_model() -> None:
    registry = get_registry()
    assert HuggingFaceSegmentationModel.__confluid_role__ == "model"
    assert HuggingFaceSegmentationModel.__confluid_framework__ == "huggingface"
    assert "HuggingFaceSegmentationModel" in registry.list_classes(task="segmentation", role="model")

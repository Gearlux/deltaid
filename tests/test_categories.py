# mypy: disable-error-code="attr-defined"
"""Discovery-category coverage for deltaid ``@configurable`` classes.

The ``category=`` tag drives navigaitor's ``list_configurable_classes(category=...)``
and the visual-editor form-spec picker (``get_node_form_spec``). Pinned here so a
dropped tag fails CI instead of silently emptying the segmentation-model picker.
"""

from confluid.registry import get_registry

from deltaid.deeplabv3plus import DeepLabV3PlusModel


def test_segmentation_model_tagged() -> None:
    assert DeepLabV3PlusModel.__confluid_category__ == "segmentation_model"


def test_category_enumerable_via_registry() -> None:
    assert "DeepLabV3PlusModel" in get_registry().list_classes(category="segmentation_model")

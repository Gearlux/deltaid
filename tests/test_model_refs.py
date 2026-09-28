"""``SegmentationModelPytorch`` names the weight artifacts it loads (``matrainer.uri``)."""

from pathlib import Path

from matrainer.uri import model_refs

from deltaid import SegmentationModelPytorch


def test_the_encoder_weight_selection_is_the_reference() -> None:
    """smp routes each encoder's weights through a different upstream registry, so the honest
    identifier is the triple smp itself was asked for, not a repository URL that would be wrong
    for most of its ~800 encoders."""
    refs = model_refs(SegmentationModelPytorch(arch="unet", encoder_name="resnet34"))
    assert [r.uri for r in refs] == ["smp://unet/resnet34?weights=imagenet"]
    assert refs[0].url is None  # no single repository to point at; none is invented
    assert refs[0].role == "pretrained"


def test_the_architecture_and_encoder_both_appear() -> None:
    refs = model_refs(SegmentationModelPytorch(arch="deeplabv3plus", encoder_name="mit_b0"))
    assert refs[0].uri == "smp://deeplabv3plus/mit_b0?weights=imagenet"


def test_a_checkpoint_is_a_second_entry_with_its_own_role(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch=3.ckpt"
    refs = model_refs(SegmentationModelPytorch(checkpoint_path=str(checkpoint)))
    assert [r.role for r in refs] == ["pretrained", "checkpoint"]
    assert refs[1].uri == checkpoint.resolve().as_uri()


def test_an_encoder_trained_from_scratch_reports_nothing() -> None:
    assert model_refs(SegmentationModelPytorch(encoder_weights=None)) == []


def test_asking_does_not_build_the_model() -> None:
    """Provenance must be readable from a wrapper the segmenter has not solidified."""
    model = SegmentationModelPytorch(arch="unet", encoder_name="resnet34")
    assert model_refs(model)
    # The lazy cache is `segmenter`, deliberately NOT `model` (see the class docstring on how the
    # Lightning `model.` prefix strip would chew through a slot of that name).
    assert model.segmenter is None

"""``FourierNeuralOperator`` — a segmentation model whose weights live on frequencies, not pixels."""

from pathlib import Path
from typing import Any

import pytest
import torch
from confluid import PartialClass, flow, to_pydantic
from confluid.registry import get_registry
from matrainer.uri import model_refs

from deltaid import FourierNeuralOperator
from deltaid.torch_segmenter import TorchSegmenter

from .test_torch_segmenter import _records


def _tiny(**kwargs: Any) -> FourierNeuralOperator:
    params: dict[str, Any] = {"num_classes": 3, "width": 8, "modes": (4, 4), "layers": 2, "projection_width": 16}
    params.update(kwargs)
    return FourierNeuralOperator(**params)


class TestTheOutput:
    def test_it_emits_per_pixel_logits_at_the_input_resolution(self) -> None:
        assert _tiny()(torch.zeros(2, 3, 64, 48)).shape == (2, 3, 64, 48)

    def test_any_size_runs_with_the_same_weights(self) -> None:
        """No stride rule: a size no encoder could halve, a picture smaller than the modes, and a
        large one all run through one set of weights."""
        model = _tiny()
        for height, width in [(37, 53), (6, 6), (128, 96)]:
            assert model(torch.rand(1, 3, height, width)).shape == (1, 3, height, width)

    def test_the_plain_operator_runs_at_full_resolution(self) -> None:
        assert _tiny(patch_size=1)(torch.rand(1, 3, 20, 30)).shape == (1, 3, 20, 30)

    def test_the_answer_follows_the_picture_when_it_is_resampled(self) -> None:
        """What resolution independence means: a smooth picture resampled to twice the rows gives
        the same logits, resampled the same way, up to the interpolation's error."""
        model = _tiny(patch_size=1).eval()
        rows = torch.linspace(0, 1, 32).view(1, 1, 32, 1)
        columns = torch.linspace(0, 1, 32).view(1, 1, 1, 32)
        picture = torch.sin(2 * torch.pi * rows) * torch.cos(2 * torch.pi * columns)
        picture = picture.expand(1, 3, 32, 32).contiguous()
        doubled = torch.nn.functional.interpolate(picture, size=(64, 32), mode="bilinear", align_corners=True)
        with torch.no_grad():
            small = model(picture)
            large = model(doubled)
        resampled = torch.nn.functional.interpolate(small, size=(64, 32), mode="bilinear", align_corners=True)
        assert torch.allclose(large, resampled, atol=0.05 * small.abs().max().item())

    def test_a_non_4d_input_is_rejected_with_a_clear_message(self) -> None:
        with pytest.raises(ValueError, match="4D"):
            _tiny()(torch.zeros(3, 32, 32))

    def test_a_channel_mismatch_is_rejected_with_a_clear_message(self) -> None:
        with pytest.raises(ValueError, match="in_channels=3"):
            _tiny()(torch.zeros(1, 1, 32, 32))


class TestTheBuild:
    def test_zero_arg_construction_builds_nothing(self) -> None:
        model = FourierNeuralOperator()
        assert model.net is None
        assert list(model.parameters()) == []

    def test_num_classes_is_validated_at_build_time(self) -> None:
        with pytest.raises(ValueError, match="num_classes is not set"):
            FourierNeuralOperator().solidify()

    def test_an_even_local_kernel_is_refused_when_built(self) -> None:
        model = _tiny(local_kernel=2)
        with pytest.raises(ValueError, match="must be odd"):
            model.solidify()

    def test_a_wider_local_kernel_builds_that_convolution(self) -> None:
        model = _tiny(local_kernel=3)
        assert model.backbone.local[0].kernel_size == (3, 3)

    def test_without_coordinates_the_lift_takes_only_the_picture(self) -> None:
        assert _tiny(coordinates=False).backbone.lift.in_channels == 3
        assert _tiny().backbone.lift.in_channels == 5

    def test_solidify_is_idempotent_and_shares_the_backbone_cache(self) -> None:
        model = _tiny()
        assert model.solidify() is model.backbone is model.net

    def test_the_seed_makes_the_weights_reproducible(self) -> None:
        a, b, c = _tiny(seed=7), _tiny(seed=7), _tiny(seed=8)
        weights = [next(m.backbone.parameters()) for m in (a, b, c)]
        assert torch.equal(weights[0], weights[1])
        assert not torch.equal(weights[0], weights[2])

    def test_the_repr_is_honest_before_the_build(self) -> None:
        assert "unbuilt" in repr(FourierNeuralOperator(num_classes=3))
        model = _tiny()
        model.solidify()
        assert "unbuilt" not in repr(model)

    def test_the_derived_num_classes_reaches_the_head(self) -> None:
        """What `!lazy:` wiring buys: the segmenter injects the resolved count at flow time."""
        model = flow(PartialClass(FourierNeuralOperator, width=8, modes=(4, 4), layers=1), num_classes=5)
        assert model(torch.zeros(1, 3, 16, 16)).shape[1] == 5


class TestCheckpointsAndProvenance:
    def test_a_lightning_checkpoint_loads_back(self, tmp_path: Path) -> None:
        """Lightning prefixes `model.`; the strip leaves `net.*` keys that match this layout."""
        trained = _tiny(seed=1)
        trained.solidify()
        checkpoint = tmp_path / "last.ckpt"
        torch.save({"state_dict": {f"model.{k}": v for k, v in trained.state_dict().items()}}, checkpoint)
        loaded = _tiny(seed=2, checkpoint_path=str(checkpoint))
        picture = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            assert torch.equal(loaded(picture), trained(picture))

    def test_only_a_checkpoint_is_an_external_weight(self, tmp_path: Path) -> None:
        assert model_refs(_tiny()) == []
        checkpoint = tmp_path / "last.ckpt"
        refs = model_refs(_tiny(checkpoint_path=str(checkpoint)))
        assert [r.role for r in refs] == ["checkpoint"]
        assert refs[0].uri == checkpoint.resolve().as_uri()


class TestDeployment:
    def test_torchscript_reproduces_it(self) -> None:
        model = _tiny().eval()
        picture = torch.rand(1, 3, 40, 24)
        scripted = torch.jit.script(model.backbone)
        with torch.no_grad():
            assert torch.allclose(scripted(picture), model(picture), atol=1e-6)

    def test_torch_export_reproduces_it(self) -> None:
        model = _tiny().eval()
        picture = torch.rand(1, 3, 32, 32)
        exported = torch.export.export(model.backbone, (picture,))
        with torch.no_grad():
            assert torch.allclose(exported.module()(picture), model(picture), atol=1e-6)


class TestDiscoveryAndTraining:
    def test_it_is_listed_as_a_torch_segmentation_model(self) -> None:
        assert "FourierNeuralOperator" in get_registry().list_classes(
            task="segmentation", role="model", framework="torch"
        )

    def test_the_range_marks_reach_the_schema(self) -> None:
        """What a form-spec or a StreamStudio widget reads its bounds from."""
        schema = to_pydantic(FourierNeuralOperator).model_json_schema()["properties"]
        assert schema["width"]["minimum"] == 1
        assert schema["local_kernel"]["maximum"] == 15

    def test_a_torch_segmenter_trains_it(self) -> None:
        """Wired as any torch model is: the loss descends on data a model can fit."""
        segmenter = TorchSegmenter(
            model=PartialClass(FourierNeuralOperator, width=8, modes=(4, 4), layers=2, projection_width=16),
            train_set=_records(8, learnable=True),
            num_classes=3,
            max_epochs=10,
            batch_size=4,
            device="cpu",
        )
        segmenter.run()
        losses = segmenter.history["train/loss"]
        assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"

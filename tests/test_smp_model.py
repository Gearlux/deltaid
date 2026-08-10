"""``SegmentationModelPytorch`` — one wrapper over every architecture smp ships.

The class replaced a `DeepLabV3PlusModel` that named one smp class directly. What is pinned here
is the difference that made that worth doing (``arch`` is a closed, upstream-checked set), plus the
lazy-build contract every ``@configurable`` model in the workspace obeys.
"""

from typing import get_args

import pytest
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from deltaid.smp_model import SEGMENTATION_ARCHITECTURES, SegmentationArchitecture, SegmentationModelPytorch


def _tiny(**kwargs: object) -> SegmentationModelPytorch:
    """A model small enough to build in a test — no pretrained download."""
    params: dict = {"encoder_name": "resnet18", "encoder_weights": None, "num_classes": 3}
    params.update(kwargs)
    return SegmentationModelPytorch(**params)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The architecture set — a CLOSED Literal, checked against smp itself
# --------------------------------------------------------------------------- #
class TestTheArchitectureSet:
    def test_the_literal_matches_what_smp_ships(self) -> None:
        """Both directions.

        A `Literal` cannot be built at runtime and still be a type, so it is written out — which
        means an smp release adding an architecture would leave it silently unofferable in every
        picker, and one REMOVING an architecture would leave a choice that fails at build time.
        This is the check that makes the hand-written list safe.
        """
        assert set(SEGMENTATION_ARCHITECTURES) == set(smp.MODEL_ARCHITECTURES_MAPPING)

    def test_the_runtime_tuple_is_derived_from_the_literal(self) -> None:
        """ONE source of truth — never a second hand-written list beside the type."""
        assert SEGMENTATION_ARCHITECTURES == get_args(SegmentationArchitecture)

    def test_it_covers_more_than_the_one_architecture_it_replaced(self) -> None:
        """The point of the class: `arch` is the whole difference between twelve models."""
        assert {"unet", "deeplabv3plus", "fpn", "segformer"} <= set(SEGMENTATION_ARCHITECTURES)
        assert len(SEGMENTATION_ARCHITECTURES) >= 12

    @pytest.mark.parametrize("arch", ["unet", "fpn", "deeplabv3plus"])
    def test_each_architecture_builds_and_emits_per_pixel_logits(self, arch: str) -> None:
        model = _tiny(arch=arch)
        out = model(torch.zeros(2, 3, 64, 64))
        assert out.shape == (2, 3, 64, 64)  # [N, num_classes, H, W]

    def test_an_unknown_architecture_is_rejected_at_CONSTRUCTION(self) -> None:
        """What the closed Literal buys beyond the picker: confluid validates it with pydantic.

        A typo'd arch fails at the call site, naming every valid value — earlier and more
        informatively than smp's own `KeyError` deep inside `create_model`.
        """
        with pytest.raises(Exception, match="unet"):  # pydantic ValidationError, listing the set
            _tiny(arch="not-an-arch")

    def test_an_architecture_set_AFTER_construction_is_caught_at_build(self) -> None:
        """The post-construction paradigm means the constructor is not the only way in.

        `configure()` and a plain setattr both bypass the pydantic check, so `solidify()` keeps
        its own guard — otherwise the failure would be smp's KeyError several frames down.
        """
        model = _tiny()
        model.arch = "not-an-arch"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="arch must be one of"):
            model.solidify()


# --------------------------------------------------------------------------- #
# Lazy init / zero-arg construction (the workspace mandate)
# --------------------------------------------------------------------------- #
class TestLazyConstruction:
    def test_zero_arg_construction_works_and_builds_nothing(self) -> None:
        model = SegmentationModelPytorch()
        assert model.segmenter is None, "the constructor must do NO functional work"
        assert model.num_classes == 0, "0 is the unset sentinel, validated at build time"

    def test_num_classes_is_validated_when_built_not_in_the_constructor(self) -> None:
        model = SegmentationModelPytorch(encoder_weights=None)  # no num_classes
        with pytest.raises(ValueError, match="num_classes must be >= 1"):
            model.solidify()

    def test_solidify_is_idempotent(self) -> None:
        model = _tiny()
        assert model.solidify() is model.solidify()

    def test_the_backbone_property_builds_a_never_flowed_model(self) -> None:
        """`flow()` never touches property getters, so `solidify` is the hook it calls; the
        property is the ergonomic accessor for a directly-constructed model. ONE cache."""
        model = _tiny()
        assert model.backbone is model.segmenter

    def test_forward_builds_on_first_use(self) -> None:
        model = _tiny()
        model(torch.zeros(1, 3, 64, 64))
        assert model.segmenter is not None

    def test_the_derived_num_classes_reaches_the_head(self) -> None:
        """What `!lazy:` wiring buys: the segmenter injects the resolved count at flow time."""
        from confluid import LazyClass, flow

        model = flow(
            LazyClass(SegmentationModelPytorch, encoder_name="resnet18", encoder_weights=None),
            num_classes=5,
        )
        assert model.num_classes == 5
        assert model(torch.zeros(1, 3, 64, 64)).shape[1] == 5


# --------------------------------------------------------------------------- #
# The wrapper's own contract
# --------------------------------------------------------------------------- #
class TestTheWrapperContract:
    def test_it_is_a_torch_module(self) -> None:
        """Every model wrapper IS its framework's own model type, so no caller reaches through it."""
        assert isinstance(_tiny(), nn.Module)

    def test_the_cache_is_a_public_submodule_named_segmenter(self) -> None:
        """A state_dict saved through Lightning gains a `model.` prefix that
        `load_lightning_checkpoint` strips — leaving `segmenter.*` keys that match this layout.
        Naming the cache `self.model` would make the strip chew through this attribute too."""
        model = _tiny()
        model.solidify()
        assert any(key.startswith("segmenter.") for key in model.state_dict())

    def test_smp_kwargs_pass_through_verbatim(self) -> None:
        """The pass-through mandate: this wrapper must not shadow smp's names or rewrite defaults."""
        model = _tiny(arch="unet", decoder_channels=(128, 64, 32, 16, 8))
        assert model.model_kwargs == {"decoder_channels": (128, 64, 32, 16, 8)}
        model.solidify()  # smp accepts it — a shadowed name would raise here

    def test_a_non_4d_input_is_rejected_with_a_clear_message(self) -> None:
        with pytest.raises(ValueError, match="4D"):
            _tiny()(torch.zeros(3, 64, 64))

    def test_a_channel_mismatch_is_rejected_with_a_clear_message(self) -> None:
        with pytest.raises(ValueError, match="in_channels=3"):
            _tiny()(torch.zeros(1, 1, 64, 64))

    def test_the_stride_rule_is_smps_to_enforce_not_ours(self) -> None:
        """smp checks it against the ENCODER's real `output_stride` and names the padded size."""
        with pytest.raises(RuntimeError, match="divisible by"):
            _tiny(arch="deeplabv3plus")(torch.zeros(2, 3, 60, 60))

    def test_a_hardcoded_divisibility_check_here_would_reject_valid_input(self) -> None:
        """The measurement that says the rule must NOT be re-implemented in this wrapper.

        The predecessor hardcoded ``H % 32 != 0 -> raise``. But ``requires_divisible_input_shape``
        is per-ARCHITECTURE: smp 0.5's ``Unet`` sets it False and pads internally, so 60x60 is
        perfectly valid input — which that check would have refused. DeepLabV3+ (the one
        architecture the predecessor supported) sets it True, which is why the bug was invisible
        there and would have surfaced the moment ``arch`` became a knob.
        """
        assert smp.create_model("unet", encoder_name="resnet18", encoder_weights=None, classes=3) is not None
        out = _tiny(arch="unet")(torch.zeros(2, 3, 60, 60))  # not a multiple of 32 — and fine
        assert out.shape == (2, 3, 60, 60)


def test_the_seed_actually_reproduces_the_weights() -> None:
    """The knob is only worth having if it BINDS — assert on the built weights, not the API.

    The decoder is always a fresh random init, so the zoo sweep rows moved between identical
    runs (2026-08-09) until the wrapper seeded its build — the restoration project's 2026-08-06
    wrapper-seeding idiom, applied here.
    """
    import torch

    def first_param(seed: int) -> torch.Tensor:
        model = _tiny(seed=seed)
        model.solidify()
        return next(model.parameters()).detach().clone()

    a, b, other = first_param(0), first_param(0), first_param(1)
    assert torch.equal(a, b), "the same seed must rebuild the same weights"
    assert not torch.equal(a, other), "a different seed must not — else it is seeding nothing"

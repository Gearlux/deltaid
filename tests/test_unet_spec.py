"""The shared U-Net spec, and the implementation(s) checked against its CLAIM.

The spec DERIVES a trainable-parameter count from the architecture rather than measuring one
implementation, so each implementation is checked against the claim instead of one of them being
the reference. That is what makes "these are the same network" a checkable statement once a second
framework arrives — and it is the only thing holding the implementations together, since different
APIs cannot share the layer code.
"""

from pathlib import Path
from typing import Any, Tuple

import pytest
import torch

from deltaid.unet.spec import (
    BOTTLENECK_WIDTH,
    ENCODER_WIDTHS,
    INPUT_SHAPE,
    KERNEL_SIZE,
    POOL_SIZE,
    size_divisor,
    trainable_parameter_count,
)
from deltaid.unet.torch_unet import TorchUNet


def _tiny(**kwargs: object) -> TorchUNet:
    params: dict = {"num_classes": 3}
    params.update(kwargs)
    return TorchUNet(**params)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The spec itself
# --------------------------------------------------------------------------- #
class TestTheSpec:
    def test_the_constants_are_the_single_source_of_truth(self) -> None:
        """Each implementation imports these for its constructor defaults, so "same
        architecture" is true by construction for the knobs rather than by copies agreeing."""
        assert ENCODER_WIDTHS == (16, 32, 64)
        assert BOTTLENECK_WIDTH == 128
        assert KERNEL_SIZE == 3
        assert POOL_SIZE == 2
        assert INPUT_SHAPE == (224, 224, 3)  # (H, W, C) — channels LAST in every implementation

    def test_the_count_is_derived_not_measured(self) -> None:
        """The documented number, so a change to the derivation is visible in a diff."""
        assert trainable_parameter_count(3) == 488_739

    def test_only_the_head_depends_on_the_class_count(self) -> None:
        """A 1x1 head: one weight per input channel per class, plus a bias each."""
        per_class = trainable_parameter_count(4) - trainable_parameter_count(3)
        assert per_class == ENCODER_WIDTHS[0] + 1

    def test_the_size_divisor_follows_from_the_widths(self) -> None:
        """Every encoder level halves the spatial size and every decoder level doubles it, so a
        level whose input was odd comes back one pixel short and the concat fails."""
        assert size_divisor() == 2 ** len(ENCODER_WIDTHS) == 8
        assert size_divisor((8, 16)) == 4


# --------------------------------------------------------------------------- #
# The torch implementation, against the claim
# --------------------------------------------------------------------------- #
class TestTorchUNetMatchesTheSpec:
    @pytest.mark.parametrize("num_classes", [2, 3, 10])
    def test_the_parameter_count_matches_the_derivation(self, num_classes: int) -> None:
        assert _tiny(num_classes=num_classes).parameter_count == trainable_parameter_count(num_classes)

    @pytest.mark.parametrize("widths", [(8, 16), (16, 32, 64), (4, 8, 16, 32)])
    def test_it_matches_under_a_changed_architecture_too(self, widths: Tuple[int, ...]) -> None:
        """The derivation tracks the knobs, so it is not a constant that happens to agree once."""
        model = _tiny(widths=widths)
        assert model.parameter_count == trainable_parameter_count(3, widths=widths)

    def test_batchnorm_running_stats_are_excluded(self) -> None:
        """They are STATE, not parameters, and frameworks disagree about whether to count them —
        so the spec excludes them and each implementation must too. In torch they are BUFFERS,
        so `requires_grad` filtering does it; an MLX implementation would need a name filter."""
        model = _tiny()
        model.solidify()
        buffers = sum(b.numel() for b in model.backbone.buffers())
        assert buffers > 0, "the network has BatchNorm, so this test is not vacuous"
        assert model.parameter_count == trainable_parameter_count(3)


# --------------------------------------------------------------------------- #
# The implementation's own behaviour
# --------------------------------------------------------------------------- #
class TestTorchUNet:
    def test_it_emits_per_pixel_logits_at_the_input_resolution(self) -> None:
        """A segmentation head: one logit per class PER PIXEL, and the U restores the size."""
        out = _tiny()(torch.zeros(2, 3, 64, 64))
        assert out.shape == (2, 3, 64, 64)

    def test_a_size_the_encoder_cannot_halve_is_refused_clearly(self) -> None:
        """Checked up front: the concat would otherwise raise naming two intermediate tensors,
        which says nothing about the input the caller gave."""
        with pytest.raises(ValueError, match="divisible by 8"):
            _tiny()(torch.zeros(1, 3, 60, 60))

    def test_the_input_shape_knob_is_channels_LAST_even_though_the_tensor_is_not(self) -> None:
        """One config line — `input_shape: [224, 224, 3]` — drives every implementation; only C
        is read here. Reading it as NCHW would build the first conv for 224 input channels:
        still runnable, and silently a different network."""
        model = _tiny(input_shape=(224, 224, 1))
        model.solidify()
        first_conv = next(m for m in model.backbone.modules() if isinstance(m, torch.nn.Conv2d))
        assert first_conv.in_channels == 1

    def test_it_is_a_torch_module(self) -> None:
        """Every model wrapper IS its framework's own model type, so no loop reaches through it."""
        assert isinstance(_tiny(), torch.nn.Module)

    def test_the_seed_makes_the_weights_reproducible(self) -> None:
        a, b = _tiny(seed=7), _tiny(seed=7)
        assert torch.equal(next(a.backbone.parameters()), next(b.backbone.parameters()))

    # ---- lazy / zero-arg construction (the workspace mandate) --------------- #
    def test_zero_arg_construction_builds_nothing(self) -> None:
        model = TorchUNet()
        assert model.net is None, "the constructor must allocate no parameters"

    def test_num_classes_is_validated_at_build_time_not_in_the_constructor(self) -> None:
        with pytest.raises(ValueError, match="num_classes is not set"):
            TorchUNet().solidify()

    def test_solidify_is_idempotent_and_shares_the_backbone_cache(self) -> None:
        model = _tiny()
        assert model.solidify() is model.backbone is model.net

    def test_the_repr_is_honest_before_the_build(self) -> None:
        """The default would print an empty module and read as a broken model."""
        assert "unbuilt" in repr(TorchUNet(num_classes=3))
        model = _tiny()
        model.solidify()
        assert "unbuilt" not in repr(model)

    def test_the_derived_num_classes_reaches_the_head(self) -> None:
        """What `!lazy:` wiring buys: the segmenter injects the resolved count at flow time."""
        from confluid import LazyClass, flow

        model = flow(LazyClass(TorchUNet), num_classes=5)
        assert model(torch.zeros(1, 3, 32, 32)).shape[1] == 5


# --------------------------------------------------------------------------- #
# The other implementations, against the SAME claim
# --------------------------------------------------------------------------- #
# Each is checked against `trainable_parameter_count`, never against `TorchUNet` — the spec is the
# reference, so no implementation is privileged and a drift in ANY of them fails here rather than
# turning whichever was written first into the de-facto answer.
class TestKerasUNetMatchesTheSpec:
    @pytest.mark.parametrize("num_classes", [2, 3, 10])
    def test_the_parameter_count_matches_the_derivation(self, num_classes: int) -> None:
        from deltaid.unet.keras_unet import KerasUNet

        assert KerasUNet(num_classes=num_classes).parameter_count == trainable_parameter_count(num_classes)

    @pytest.mark.parametrize("widths", [(8, 16), (16, 32, 64), (4, 8, 16, 32)])
    def test_it_matches_under_a_changed_architecture_too(self, widths: Tuple[int, ...]) -> None:
        from deltaid.unet.keras_unet import KerasUNet

        assert KerasUNet(num_classes=3, widths=widths).parameter_count == trainable_parameter_count(3, widths=widths)

    def test_batchnorm_moving_statistics_are_excluded(self) -> None:
        """In Keras they are non-trainable WEIGHTS rather than a separate buffer collection, so
        `trainable_weights` is what makes the exclusion, where torch got it from `requires_grad`.

        Sizes come from `w.shape`, never `w.numpy()`: on the torch backend a Keras weight is an
        MPS tensor and `numpy()` raises on it (measured).
        """
        import math

        from deltaid.unet.keras_unet import KerasUNet

        model = KerasUNet(num_classes=3)
        model.solidify()
        non_trainable = sum(math.prod(w.shape) for w in model.model.non_trainable_weights)
        assert non_trainable > 0, "the network has BatchNorm, so this test is not vacuous"
        assert model.parameter_count == trainable_parameter_count(3)

    def test_it_accepts_a_spatial_size_other_than_the_configured_one(self) -> None:
        """The `input_shape` knob names the CHANNEL count and nothing else here.

        A Keras Functional model PINS whatever spatial dims its `Input` declares, so building it
        from the configured `(H, W, C)` would refuse every other size — including the crop a
        prediction run happens to hand it. The build therefore declares `(None, None, C)`.
        """
        import numpy as np

        from deltaid.unet.keras_unet import KerasUNet

        model = KerasUNet(num_classes=3, input_shape=(224, 224, 3))
        out = model(np.zeros((2, 64, 64, 3), dtype="float32"))
        assert tuple(out.shape) == (2, 64, 64, 3)


class TestTensorFlowUNetMatchesTheSpec:
    @pytest.mark.parametrize("num_classes", [2, 3, 10])
    def test_the_parameter_count_matches_the_derivation(self, num_classes: int) -> None:
        from deltaid.unet.tensorflow_unet import TensorFlowUNet

        assert TensorFlowUNet(num_classes=num_classes).parameter_count == trainable_parameter_count(num_classes)

    @pytest.mark.parametrize("widths", [(8, 16), (16, 32, 64), (4, 8, 16, 32)])
    def test_it_matches_under_a_changed_architecture_too(self, widths: Tuple[int, ...]) -> None:
        from deltaid.unet.tensorflow_unet import TensorFlowUNet

        model = TensorFlowUNet(num_classes=3, widths=widths)
        assert model.parameter_count == trainable_parameter_count(3, widths=widths)

    def test_batchnorm_moving_statistics_are_excluded(self) -> None:
        """Raw `tf.nn` here, so the moving mean/variance are ordinary `tf.Variable`s that happen
        to be `trainable=False` — the only thing separating them from the parameters."""
        from deltaid.unet.tensorflow_unet import TensorFlowUNet

        model = TensorFlowUNet(num_classes=3)
        model.solidify()
        non_trainable = sum(int(v.shape.num_elements()) for v in model.variables if not v.trainable)
        assert non_trainable > 0, "the network has BatchNorm, so this test is not vacuous"
        assert model.parameter_count == trainable_parameter_count(3)

    def test_it_emits_per_pixel_logits_at_the_input_resolution(self) -> None:
        import numpy as np

        from deltaid.unet.tensorflow_unet import TensorFlowUNet

        out = TensorFlowUNet(num_classes=3)(np.zeros((2, 64, 64, 3), dtype="float32"))
        assert tuple(out.shape) == (2, 64, 64, 3)


class TestFlaxUNetMatchesTheSpec:
    @pytest.mark.parametrize("num_classes", [2, 3, 10])
    def test_the_parameter_count_matches_the_derivation(self, num_classes: int) -> None:
        from deltaid.unet.flax_unet import FlaxUNet

        assert FlaxUNet(num_classes=num_classes).parameter_count == trainable_parameter_count(num_classes)

    @pytest.mark.parametrize("widths", [(8, 16), (16, 32, 64), (4, 8, 16, 32)])
    def test_it_matches_under_a_changed_architecture_too(self, widths: Tuple[int, ...]) -> None:
        from deltaid.unet.flax_unet import FlaxUNet

        assert FlaxUNet(num_classes=3, widths=widths).parameter_count == trainable_parameter_count(3, widths=widths)

    def test_batchnorm_statistics_are_excluded(self) -> None:
        """nnx files them as `nnx.BatchStat`, a DIFFERENT variable type in the same state tree, so
        counting `nnx.Param` only is what makes the exclusion — where torch got it from
        `requires_grad` and MLX needs a name filter."""
        import jax
        from flax import nnx

        from deltaid.unet.flax_unet import FlaxUNet

        model = FlaxUNet(num_classes=3)
        model.solidify()
        stats = sum(int(v.size) for v in jax.tree_util.tree_leaves(nnx.state(model, nnx.BatchStat)))
        assert stats > 0, "the network has BatchNorm, so this test is not vacuous"
        assert model.parameter_count == trainable_parameter_count(3)

    def test_it_emits_per_pixel_logits_at_the_input_resolution(self) -> None:
        import numpy as np

        from deltaid.unet.flax_unet import FlaxUNet

        out = FlaxUNet(num_classes=3)(np.zeros((2, 64, 64, 3), dtype="float32"))
        assert tuple(out.shape) == (2, 64, 64, 3)

    def test_a_size_the_encoder_cannot_halve_is_refused_clearly(self) -> None:
        import numpy as np

        from deltaid.unet.flax_unet import FlaxUNet

        with pytest.raises(ValueError, match="divisible by 8"):
            FlaxUNet(num_classes=3)(np.zeros((1, 60, 60, 3), dtype="float32"))


class TestMlxUNetMatchesTheSpec:
    @pytest.mark.parametrize("num_classes", [2, 3, 10])
    def test_the_parameter_count_matches_the_derivation(self, num_classes: int) -> None:
        from deltaid.unet.mlx_unet import MlxUNet

        assert MlxUNet(num_classes=num_classes).parameter_count == trainable_parameter_count(num_classes)

    @pytest.mark.parametrize("widths", [(8, 16), (16, 32, 64), (4, 8, 16, 32)])
    def test_it_matches_under_a_changed_architecture_too(self, widths: Tuple[int, ...]) -> None:
        from deltaid.unet.mlx_unet import MlxUNet

        assert MlxUNet(num_classes=3, widths=widths).parameter_count == trainable_parameter_count(3, widths=widths)

    def test_batchnorm_statistics_are_excluded_BY_NAME(self) -> None:
        """The one wrinkle MLX adds: it keeps the running mean and variance in the SAME parameter
        tree as the weights, where the other four hold them as separate non-trainable state. So
        the exclusion is a name filter, and without it the five-way match breaks."""
        from mlx.utils import tree_flatten

        from deltaid.unet.mlx_unet import MlxUNet

        model = MlxUNet(num_classes=3)
        model.solidify()
        # `tree_flatten` is typed as returning a flat list, so the (path, value) pairs are
        # indexed rather than unpacked — mypy rejects the star-unpack on its declared type.
        leaves: Any = tree_flatten(model.parameters())
        stats = [entry[0] for entry in leaves if str(entry[0]).endswith(("running_mean", "running_var"))]
        assert stats, "the running statistics are in the parameter tree, so this test is not vacuous"
        assert model.parameter_count == trainable_parameter_count(3)

    def test_it_emits_per_pixel_logits_at_the_input_resolution(self) -> None:
        import mlx.core as mx
        import numpy as np

        from deltaid.unet.mlx_unet import MlxUNet

        out = MlxUNet(num_classes=3)(mx.array(np.zeros((2, 64, 64, 3), dtype="float32")))
        mx.eval(out)
        assert tuple(out.shape) == (2, 64, 64, 3)

    def test_a_size_the_encoder_cannot_halve_is_refused_clearly(self) -> None:
        import mlx.core as mx
        import numpy as np

        from deltaid.unet.mlx_unet import MlxUNet

        with pytest.raises(ValueError, match="divisible by 8"):
            MlxUNet(num_classes=3)(mx.array(np.zeros((1, 60, 60, 3), dtype="float32")))


def test_all_five_implementations_agree_with_the_spec_and_each_other() -> None:
    """The claim the package exists to make, stated once.

    Not "they equal TorchUNet" — they equal the DERIVATION, which is what makes a cross-framework
    benchmark compare frameworks rather than five different networks.
    """
    from deltaid.unet.flax_unet import FlaxUNet
    from deltaid.unet.keras_unet import KerasUNet
    from deltaid.unet.mlx_unet import MlxUNet
    from deltaid.unet.tensorflow_unet import TensorFlowUNet

    counts = {
        "torch": _tiny().parameter_count,
        "keras": KerasUNet(num_classes=3).parameter_count,
        "tensorflow": TensorFlowUNet(num_classes=3).parameter_count,
        "flax": FlaxUNet(num_classes=3).parameter_count,
        "mlx": MlxUNet(num_classes=3).parameter_count,
    }
    assert set(counts.values()) == {trainable_parameter_count(3)}, counts


def test_every_implementation_is_an_instance_of_its_own_frameworks_model_type() -> None:
    """No loop reaches THROUGH a wrapper: `model.fit(...)` must work on the object a config
    wired, so each implementation IS what its engine calls a model."""
    import mlx.nn
    import tensorflow as tf
    from flax import nnx

    from deltaid._keras import keras
    from deltaid.unet.flax_unet import FlaxUNet
    from deltaid.unet.keras_unet import KerasUNet
    from deltaid.unet.mlx_unet import MlxUNet
    from deltaid.unet.tensorflow_unet import TensorFlowUNet

    assert isinstance(_tiny(), torch.nn.Module)
    assert isinstance(KerasUNet(num_classes=3), keras.Model)
    assert isinstance(TensorFlowUNet(num_classes=3), tf.Module)
    assert isinstance(FlaxUNet(num_classes=3), nnx.Module)
    assert isinstance(MlxUNet(num_classes=3), mlx.nn.Module)


# --------------------------------------------------------------------------- #
# Why the package exists at all
# --------------------------------------------------------------------------- #
def test_the_unet_package_imports_nothing() -> None:
    """LOAD-BEARING: `deltaid/__init__.py` resolves every backend name lazily so a machine
    without torch never imports it. An `__init__` here re-exporting the implementations would
    drag in torch the moment anything under the package — including the dependency-free
    `spec.py` — was touched.
    """
    import ast
    import pathlib

    import deltaid.unet

    source = pathlib.Path(str(deltaid.unet.__file__)).read_text()
    imports = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == [], f"deltaid/unet/__init__.py must import nothing, found: {len(imports)}"


def test_no_tensorflow_module_reaches_back_into_keras_through_a_tf_alias() -> None:
    """`tf.initializers` / `tf.losses` / `tf.metrics` / `tf.keras` are ALL Keras since TF 2.16.

    `tf.initializers.GlorotUniform` *is* `keras.initializers.GlorotUniform` — the same object, not
    a wrapper — so in a process where Keras is bound to the TORCH engine (this workspace's
    default, since `recordstream.keras` picks the first engine installed) it returns a **torch
    tensor**, and `tf.Variable` then dies converting an MPS tensor to numpy.

    Scanned as SOURCE across the whole package rather than asserted about one module, because the
    mistake is a natural one to make in the next raw-TensorFlow module too — and it is invisible
    in a single-file run: `TensorFlowUNet` passed alone and failed the moment a Keras test had run
    first, which is exactly how this was found.
    """
    import re

    root = Path(__file__).resolve().parents[1] / "deltaid"
    aliases = ("tf.initializers", "tf.losses", "tf.metrics", "tf.optimizers", "tf.keras.")
    offenders = {}
    for path in sorted(root.rglob("*.py")):
        code = "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("#"))
        code = re.sub(r'"""(?:.|\n)*?"""', "", code)  # the aliases are NAMED in the explanations
        hits = [alias for alias in aliases if alias in code]
        if hits:
            offenders[path.relative_to(root).as_posix()] = hits
    assert not offenders, (
        f"these reach Keras through a `tf.` alias: {offenders}. Since TF 2.16 `tf.keras` IS "
        "Keras 3, so on a torch-bound engine they return torch tensors. Write the primitive out "
        "(see `_glorot_uniform`) or reach for `keras` explicitly and mean it."
    )


def test_the_tensorflow_unet_builds_after_keras_has_claimed_the_torch_engine() -> None:
    """The regression itself, reproduced in ONE process — the only place it is visible.

    `KerasUNet` first (which binds keras to whichever engine this session picked), then
    `TensorFlowUNet`. With `tf.initializers` in the build this raises *"can't convert mps:0 device
    type tensor to numpy"* out of `tf.Variable`; with `_glorot_uniform` it does not.
    """
    from deltaid.unet.keras_unet import KerasUNet
    from deltaid.unet.tensorflow_unet import TensorFlowUNet

    KerasUNet(num_classes=3).solidify()
    assert TensorFlowUNet(num_classes=3).parameter_count == trainable_parameter_count(3)


def test_the_spec_module_imports_only_the_standard_library() -> None:
    """So the CLAIM can be checked without any framework installed — which is what lets a
    framework-free CI job verify that the implementations agree."""
    import ast
    import pathlib
    import sys

    import deltaid.unet.spec

    source = pathlib.Path(str(deltaid.unet.spec.__file__)).read_text()
    modules = {
        (node.module or "").split(".")[0] for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)
    }
    assert modules <= set(sys.stdlib_module_names), f"non-stdlib imports: {modules - set(sys.stdlib_module_names)}"

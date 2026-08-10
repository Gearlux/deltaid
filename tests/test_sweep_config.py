"""The sweep overlay resolves to the right runnable + model for every cell of its matrix.

`examples/sweep/all_backends.yaml` is an overlay: it `include:`s the example and overrides only
what a sweep needs. Three things about that composition are easy to get wrong and silent when
wrong, and all three are checked here rather than discovered eleven training runs later.

**A scope block REPLACES, it does not merge.** `include:` deep-merges two *mappings*, and a
`!scope:` block is a `ScopeBlock` object rather than a dict — so an overlay block with a key the
included file already uses discards that block wholesale. A `tensorflow:` block holding only
`export_dir` would drop the runnable, the model and the channels-LAST preprocess chain, and
`--framework tensorflow` would train a `LightningSegmenter` over CHW tensors under the right name
with nothing raising.

**A deep-merged key keeps its ORIGINAL POSITION, so a new top-level key wins over it.** The
overlay's `metrics:` is new here and lands at the end of the merged document, while `fastai:` is a
key the example already has and stays where the example put it — early. As a sibling key inside
that block, `metrics:` was silently overwritten by the top-level one (measured: fastai resolved to
a torchmetrics mapping, which a `Learner` cannot consume). An explicit CONSTRUCTOR kwarg is immune.

**A metric set must reach only the engines that can consume it.** The mechanism here is confluid's
broadcast gate rather than a nulled key: a MAPPING does not reach a slot annotated
`Optional[List[...]]`, which is how the five non-torch backends declare theirs.

Config-level on purpose, like `test_cli_frameworks.py`: a machine need not have all eight
frameworks installed, so flowing the classes would reduce the check to whichever extras happen to
be present — exactly the machine where a copy-pasted block lands.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import confluid
import pytest

SWEEP = Path(__file__).resolve().parent.parent / "examples" / "sweep" / "all_backends.yaml"

#: (backend, variant) -> (runnable class, model class). The matrix the driver script walks.
#: A variant an engine structurally cannot have is simply absent: there is no non-torch
#: segmentation ZOO to wrap (smp is torch-only, which is why the shared spec exists), and a
#: `transformers.Trainer` drives models that compute their own loss, so it takes no U-Net.
MATRIX: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("lightning", "zoo"): ("LightningSegmenter", "SegmentationModelPytorch"),
    ("lightning", "unet"): ("LightningSegmenter", "TorchUNet"),
    ("torch", "zoo"): ("TorchSegmenter", "SegmentationModelPytorch"),
    ("torch", "unet"): ("TorchSegmenter", "TorchUNet"),
    ("fastai", "zoo"): ("FastaiSegmenter", "SegmentationModelPytorch"),
    ("fastai", "unet"): ("FastaiSegmenter", "TorchUNet"),
    ("keras", "unet"): ("KerasSegmenter", "KerasUNet"),
    ("tensorflow", "unet"): ("TensorFlowSegmenter", "TensorFlowUNet"),
    ("huggingface", "zoo"): ("HuggingFaceSegmenter", "HuggingFaceSegmentationModel"),
    ("jax", "unet"): ("JaxSegmenter", "FlaxUNet"),
    ("mlx", "unet"): ("MlxSegmenter", "MlxUNet"),
}

TRACKERS = {
    "mlflow": "MlflowTracker",
    "tensorboard": "TensorBoardTracker",
    "csv": "CsvTracker",
    "aim": "AimTracker",
    "clearml": "ClearMLTracker",
    "wandb": "WandbTracker",
}

#: The engines whose metrics are NAMES or their own engine's objects, so the shared torchmetrics
#: mapping must not reach them.
NON_TORCH = ("keras", "tensorflow", "fastai", "jax", "mlx", "huggingface")


def resolve(framework: str, variant: str = "zoo", tracker: str = "mlflow") -> Dict[str, Any]:
    scopes = [f"framework={framework}", f"tracker={tracker}"]
    if variant == "unet":
        scopes.append("model=unet")
    document = confluid.load(confluid.load_config(SWEEP), flow=False, scopes=scopes)
    assert isinstance(document, dict)
    return document


def target_name(value: Any) -> Optional[str]:
    """The class name behind a `!class:` / `!lazy:` marker, without importing it."""
    target = getattr(value, "target", None)
    if target is None:
        return None
    return str(target).rsplit(".", 1)[-1]


def constructor_annotation(module: str, parameter: str) -> str:
    """The source text of one `__init__` parameter's annotation, read STATICALLY.

    `ast` rather than `typing.get_type_hints`: resolving a hint imports the module, and a module
    whose framework is not installed here is exactly the one whose annotation nobody would check.
    """
    source = (Path(__file__).resolve().parent.parent / "deltaid" / module).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for argument in [*node.args.args, *node.args.kwonlyargs]:
                if argument.arg == parameter and argument.annotation is not None:
                    return ast.unparse(argument.annotation)
    raise AssertionError(f"{module} declares no `{parameter}` constructor parameter")


def test_the_sweep_spec_declares_exactly_this_matrix() -> None:
    """The declarative spec (`matrainer sweep`'s input) and this table must not drift — a
    cell in one and not the other is either a run nobody checks or a check nothing runs."""
    from matrainer.sweep import load_spec

    spec = load_spec(SWEEP.parent / "sweep.yaml")
    default = spec.models[0].name if spec.models else None
    declared = {
        (row.name, model) for row in spec.frameworks for model in (row.models or ([default] if default else []))
    }
    assert declared == set(MATRIX)


@pytest.mark.parametrize("cell", sorted(MATRIX), ids=lambda c: f"{c[0]}-{c[1]}")
def test_every_cell_resolves_to_its_own_runnable_and_model(cell: Tuple[str, str]) -> None:
    """The regression pin for the replace-not-merge trap described in the module docstring."""
    framework, variant = cell
    expected_runnable, expected_model = MATRIX[cell]
    document = resolve(framework, variant)

    assert target_name(document["runnable"]) == expected_runnable
    assert target_name(document.get("model")) == expected_model


def test_the_matrix_covers_every_backend_the_cli_advertises() -> None:
    """A ninth backend must not slip into the CLI without a row in the sweep.

    The sweep is how the claim "one config, every backend" is EXECUTED, so a backend the CLI
    offers and the sweep never runs is a claim nothing checks.
    """
    from deltaid.cli import FRAMEWORKS

    assert {backend for backend, _ in MATRIX} == set(FRAMEWORKS)


def test_the_torch_backed_backends_share_one_implementation_of_the_spec() -> None:
    """Three backends, one `TorchUNet` — the example's top-level block, inherited unchanged.

    If one of them ever resolves to something else, a per-backend block has been added where the
    shared one was doing the job, which is the duplication `unet/spec.py` exists to avoid.
    """
    names = {target_name(resolve(fw, "unet")["model"]) for fw in ("lightning", "torch", "fastai")}
    assert names == {"TorchUNet"}


def test_each_non_torch_engine_gets_its_own_implementation_of_the_spec() -> None:
    """The point of the sweep: the SAME architecture, in each framework's own module type.

    Nothing in the overlay arranges this — each framework block wires its implementation
    unconditionally and is spliced AFTER the example's top-level `unet:` block, so the block that
    gives the torch backends `TorchUNet` is simply overridden. That is a property of document
    ORDER, which is exactly the kind of thing an edit can reverse silently, so it is pinned here:
    without it, `--framework mlx --model unet` would train a torch model under the MLX name.
    """
    assert {fw: target_name(resolve(fw, "unet")["model"]) for fw in ("keras", "tensorflow", "jax", "mlx")} == {
        "keras": "KerasUNet",
        "tensorflow": "TensorFlowUNet",
        "jax": "FlaxUNet",
        "mlx": "MlxUNet",
    }


@pytest.mark.parametrize("tracker", sorted(TRACKERS))
def test_each_tracker_scope_selects_exactly_that_backend(tracker: str) -> None:
    trackers = resolve("lightning", tracker=tracker)["trackers"]

    assert [target_name(t) for t in trackers] == [TRACKERS[tracker]]


@pytest.mark.parametrize("tracker", sorted(TRACKERS))
def test_the_tracker_takes_its_names_from_the_run(tracker: str) -> None:
    """`!ref:` rather than a literal, so the ONE pair the driver sets reaches the tracker too.

    A literal here would drift from whatever the run is actually called — the one thing a sweep
    must not do, since the tracker is where the runs are compared.
    """
    wired = resolve("lightning", tracker=tracker)["trackers"][0]

    for key in ("experiment_name", "run_name"):
        assert type(wired.kwargs[key]).__name__ == "Reference", f"{tracker}: {key} is not a !ref:"


def test_a_metric_set_never_reaches_an_engine_that_cannot_consume_it() -> None:
    """The shared `metrics:` is a torchmetrics TEMPLATE MAPPING, which only torch can consume.

    Unlike a per-engine `metrics: null`, what keeps it away from the other five is confluid's
    BROADCAST GATE: it classifies by the receiving slot's annotation, and every non-torch segmenter
    declares `metrics: Optional[List[...]]`, which a mapping does not satisfy. So the check is on
    the ANNOTATION rather than on the document key — the key is present for every framework and
    that is fine.

    Read with `ast` rather than by importing: a machine need not carry all eight frameworks, and an
    import-based check would silently pass on the one that does not have the backend it would test.
    """
    document = resolve("lightning")
    assert isinstance(document["metrics"], dict), "the shared set must be a torchmetrics template MAPPING"
    assert set(document["metrics"]) == {"mean_iou", "pixel_accuracy"}

    for framework in NON_TORCH:
        declaration = constructor_annotation(f"{framework}_segmenter.py", "metrics")
        assert declaration.startswith("Optional[List[") or declaration.startswith("Optional[Sequence["), (
            f"{framework} declares `metrics: {declaration}` — a mapping-admitting annotation would "
            f"let the sweep's torchmetrics templates broadcast into an engine that cannot compile them"
        )


def test_every_row_reports_mean_iou() -> None:
    """The one quantity all eleven runs share, and the reason the sweep is a comparison.

    The NAME is what a table lines up on, so this checks what each block ASKS FOR rather than the
    metric object. Seven engines can produce the word `mean_iou` directly; fastai cannot — its
    `fastai.metrics` publishes the quantity as `JaccardCoeffMulti`, whose recorder column is
    `jaccard_coeff_multi` — so the overlay's `metric_names.renames` carries it into the same column.
    """
    assert set(resolve("lightning")["metrics"]) >= {"mean_iou"}

    # keras / tensorflow / jax / mlx / huggingface all DEFAULT to mean IoU, so an unset slot is the
    # right answer for them; huggingface names it anyway because it also pins its output directory.
    assert "mean_iou" in resolve("huggingface")["runnable"].kwargs["metrics"]

    fastai_metrics = resolve("fastai")["runnable"].kwargs["metrics"]
    assert "JaccardCoeffMulti" in fastai_metrics, "fastai's mean IoU must be a NAME it can resolve"
    renames = resolve("fastai")["metric_names"].kwargs["renames"]
    assert renames["jaccard_coeff_multi"] == "mean_iou", "fastai's column must land under the shared name"


def test_the_naming_knob_reaches_every_backend() -> None:
    """One top-level `metric_names:` key, eight differently-sourced backends.

    It broadcasts rather than being restated per block, which is the whole design of the slot —
    `source` is per-CALL (the backend knows which engine wrote the names) while `target` and
    `renames` are per-CONFIG.
    """
    for framework in ("lightning", *NON_TORCH):
        wired = resolve(framework).get("metric_names")
        assert target_name(wired) == "MetricNames", f"{framework} does not receive the naming knob"


def test_every_row_reports_pixel_accuracy_too() -> None:
    """The second full column, and the one that took a deltaid-owned metric to fill.

    `fastai.metrics` has no per-pixel accuracy reachable by NAME — its `accuracy` argmaxes over
    `axis=-1`, which on an `[N, C, H, W]` prediction is the WIDTH — so `pixel_accuracy` here is
    deltaid's own CONFIGURED ALIAS (`fastai_segmenter.PixelAccuracy`), resolved by `_as_metric`.
    Everyone else reports it from the shared torchmetrics template (lightning / torch) or from its
    own engine defaults (keras / tensorflow / jax / mlx / huggingface).
    """
    assert "pixel_accuracy" in resolve("lightning")["metrics"]
    assert "pixel_accuracy" in resolve("huggingface")["runnable"].kwargs["metrics"]
    assert "pixel_accuracy" in resolve("fastai")["runnable"].kwargs["metrics"]


def test_the_sweep_writes_everything_under_one_tree() -> None:
    """Eleven runs must not scatter exports and Trainer checkpoints across the project root.

    The tree is `runs/` — the ONE gitignored output root every project carries (workspace
    `AGENTS.md`, "Run Output Lands In A Gitignored `runs/`"). `sweep_logs/` keeps only the
    driver's per-cell `.log` files; everything a RUN produces belongs here.
    """
    assert resolve("tensorflow", "unet")["runnable"].kwargs["export_dir"].startswith("./runs/")
    assert resolve("huggingface")["runnable"].kwargs["output_dir"].startswith("./runs/")

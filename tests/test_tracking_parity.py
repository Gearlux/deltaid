"""Every backend tracks a run the SAME way — checked across the fleet, by reading the source.

A backend that wires `trackers:` and then uses them for nothing completes, reports green, and
records a thin run: no git state, no config, no hyperparameters. That is invisible from a passing
suite and a successful training run, which is why it is checked here rather than left to each
backend's own file.

The scans are STATIC (`ast` / text) on purpose, for the same reason the classification project's
twin gives: no venv carries every backend's framework, so an import-based check silently reduces
to whichever extras happen to be installed — exactly the machine where a copy-pasted omission
lands.
"""

import ast
from pathlib import Path
from typing import Dict, Set

import pytest

SOURCE_DIR = Path(__file__).resolve().parent.parent / "deltaid"

#: Which backend hands its trackers to which matrainer entry point. The two are not
#: interchangeable: a tracking CALLBACK owns the run lifecycle (it captures on fit start and
#: finishes on end), while the plain functions leave that to the caller — so a backend using the
#: functions MUST also finish the manifest itself.
CALLBACK_BACKENDS: Set[str] = {"lightning_segmenter", "keras_segmenter", "tensorflow_segmenter"}
FUNCTION_BACKENDS: Set[str] = {
    "torch_segmenter",
    "fastai_segmenter",
    "huggingface_segmenter",
    "jax_segmenter",
    "mlx_segmenter",
}

#: The callback spellings, one per framework family. Kept as a SET rather than one name because
#: the adapters are genuinely different objects — `matrainer.lightning.TrackingCallback` is a
#: `pl.Callback`, `matrainer.keras.keras_tracking_callback` builds a `keras.callbacks.Callback` —
#: and a backend must reach exactly one of them.
TRACKING_CALLBACKS = ("TrackingCallback", "keras_tracking_callback")


def built_on_shared_base(source: str) -> bool:
    """The 2026-08-06 consolidation: a backend on a shared machinery base inherits its
    tracking wiring — `runnable=self` + `TrackingCallback` in `_build_trainer` (lightning) /
    `_log_run_context` and the per-epoch `tracker.log_metrics` in `fit` (the owned loop), the
    once-flowed trackers, the naming knob's application, the fork warm in the task base's
    `initialize` — all pinned in matrainer's own suite. What such a module must not do is
    SHADOW that wiring."""
    markers = (
        "LightningRunnable",
        "TorchLoopRunnable",
        "FastaiRunnable",
        "KerasRunnable",
        "TensorFlowRunnable",
        "HuggingFaceRunnable",
        "JaxLoopRunnable",
        "MlxLoopRunnable",
    )
    return any(marker in source for marker in markers)


def assert_does_not_shadow_the_shared_wiring(stem: str, source: str) -> None:
    shared = (
        # the Lightning machinery base's wiring
        "_build_trainer",
        "_run_eval",
        "_report_epoch_metrics",
        # the owned-loop machinery base's wiring
        "fit",
        "_score",
        "_epoch_metrics",
        "_log_run_context",
        # the keras-family machinery bases' wiring (the tracking callback + the loop)
        "_callbacks",
        "_run_epoch",
        # both
        "_live_trackers",
        "initialize",
    )
    for name in shared:
        # Anchored with the paren: the entries are METHOD names, and a hook that merely
        # STARTS with one (`_score_batch` vs the loop's `_score`) is not a shadow.
        assert f"def {name}(" not in source, (
            f"{stem} is built on the shared base but shadows {name} — the inherited wiring "
            "no longer applies; extend via the hooks instead."
        )


def _backends() -> Dict[str, str]:
    """``{module stem: source}`` for every backend on disk."""
    return {path.stem: path.read_text() for path in sorted(SOURCE_DIR.glob("*_segmenter.py"))}


def test_the_checked_list_is_every_backend_on_disk() -> None:
    """The check that stops a fourth backend quietly skipping everything below."""
    on_disk = set(_backends())
    declared = CALLBACK_BACKENDS | FUNCTION_BACKENDS
    assert on_disk == declared, f"backends on disk but unchecked: {sorted(on_disk - declared)}"


@pytest.mark.parametrize("stem", sorted(CALLBACK_BACKENDS | FUNCTION_BACKENDS))
def test_every_backend_hands_matrainer_the_RUNNABLE(stem: str) -> None:
    """`runnable=self`, never a hand-built params dict.

    matrainer owns the ONE hyperparameter derivation (it walks the flowed object graph and dumps
    the resolved config), so two backends transcribing their own would give two answers to "what
    were the hyperparameters?" for the same config.
    """
    source = _backends()[stem]
    if built_on_shared_base(source):
        assert_does_not_shadow_the_shared_wiring(stem, source)
        return
    assert "runnable=self" in source


@pytest.mark.parametrize("stem", sorted(CALLBACK_BACKENDS | FUNCTION_BACKENDS))
def test_no_backend_keeps_a_private_hyperparameter_derivation(stem: str) -> None:
    source = _backends()[stem]
    for banned in ("_config_params", "_tracked_params"):
        assert banned not in source, f"{stem} re-derives its own hyperparameters ({banned})"


@pytest.mark.parametrize("stem", sorted(FUNCTION_BACKENDS))
def test_a_function_driven_backend_finishes_its_manifest(stem: str) -> None:
    """`capture_log_file(trackers, manifest=…, status="success")` at the end of `fit()`.

    Not optional once a manifest exists: left unfinished it stays `running`, and matrainer's
    atexit hook then labels a SUCCESSFUL run `interrupted`.
    """
    source = _backends()[stem]
    if built_on_shared_base(source):
        # Both calls live in the shared loop base's `fit()` (pinned in matrainer's suite).
        assert_does_not_shadow_the_shared_wiring(stem, source)
        return
    assert "capture_log_file" in source
    assert "log_run_context" in source


@pytest.mark.parametrize("stem", sorted(CALLBACK_BACKENDS))
def test_a_callback_driven_backend_lets_the_callback_own_the_lifecycle(stem: str) -> None:
    """A tracking callback captures on start and finishes on end, so the backend must NOT also
    call `capture_log_file` — that would finish the manifest twice."""
    source = _backends()[stem]
    if built_on_shared_base(source):
        assert "capture_log_file" not in source
        return  # the shared base wires TrackingCallback, whose end hook owns the lifecycle
    assert any(name in source for name in TRACKING_CALLBACKS), f"{stem} reaches no tracking callback"
    assert "capture_log_file" not in source


@pytest.mark.parametrize("stem", sorted(CALLBACK_BACKENDS | FUNCTION_BACKENDS))
def test_no_backend_finalizes_a_tracker_itself(stem: str) -> None:
    """Ending the run belongs to `capture_log_file` (or, for Lightning, to the adapter Lightning
    finalizes) — a backend calling `Tracker.finalize` is another place for the tracker's status
    to disagree with the manifest's."""
    assert ".finalize(" not in _backends()[stem]


@pytest.mark.parametrize("stem", sorted(CALLBACK_BACKENDS | FUNCTION_BACKENDS))
def test_every_backend_flows_its_trackers_once_and_writes_them_back(stem: str) -> None:
    """`flow()` on a live object is idempotent, but a deferred `Fluid` is REBUILT on every call —
    which for an MLflow tracker means two runs recorded for one training run."""
    source = _backends()[stem]
    if built_on_shared_base(source):
        assert_does_not_shadow_the_shared_wiring(stem, source)
        return
    assert "_live_trackers" in source
    assert "self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]" in source


@pytest.mark.parametrize("stem", sorted(CALLBACK_BACKENDS | FUNCTION_BACKENDS))
def test_trackers_is_a_constructor_param_on_every_backend(stem: str) -> None:
    """confluid gates LIST broadcast on the receiving slot's ANNOTATION, so as a
    post-construction attribute a `trackers: [...]` YAML key silently does NOTHING — and every
    unit test still passes, because tests assign `.trackers` in Python."""
    tree = ast.parse(_backends()[stem])
    init = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    params = {a.arg for a in (*init.args.posonlyargs, *init.args.args, *init.args.kwonlyargs)}
    assert "trackers" in params, f"{stem} does not declare `trackers` as a constructor param"


@pytest.mark.parametrize("stem", sorted(CALLBACK_BACKENDS | FUNCTION_BACKENDS))
def test_every_backend_takes_the_metric_naming_knob(stem: str) -> None:
    """One `metric_names:` key must reach ALL EIGHT, or a comparison's columns stop lining up.

    Every tracker records the key it is handed, and the engines disagree on what to hand it:
    `val_loss` under Keras, `valid_loss` under fastai, `eval_loss` under a `transformers.Trainer`,
    `val/loss` here and in the torch loop. A backend left out of this records its curves in
    columns of its own — which looks like a sparse table rather than a bug.
    """
    tree = ast.parse(_backends()[stem])
    init = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    annotations = {a.arg: ast.unparse(a.annotation) for a in init.args.args if a.annotation is not None}

    assert "metric_names" in annotations, f"{stem} cannot be told how to name what it records"
    assert "Lazy[MetricNames]" in annotations["metric_names"], (
        f"{stem}: the slot must be `Optional[Lazy[MetricNames]]` — wired from YAML it arrives as "
        f"a deferred marker (got {annotations['metric_names']})"
    )


@pytest.mark.parametrize("stem", sorted(CALLBACK_BACKENDS | FUNCTION_BACKENDS))
def test_every_backend_applies_the_naming_it_was_given(stem: str) -> None:
    """Declaring the slot is half of it — what the backend reports has to go THROUGH it."""
    source = _backends()[stem]
    if built_on_shared_base(source):
        assert_does_not_shadow_the_shared_wiring(stem, source)
        return
    assert "as_metric_names" in source, (
        f"{stem} declares `metric_names` but never applies it: call "
        "`as_metric_names(self.metric_names)` where metrics reach a tracker or the sidecar."
    )


@pytest.mark.parametrize("stem", sorted(CALLBACK_BACKENDS | FUNCTION_BACKENDS))
def test_no_backend_has_a_logger_slot(stem: str) -> None:
    """`trackers:` is the whole of a run's output wiring. A `logger:` knob named two different
    things — real metric sinks on one backend, framework callbacks on another."""
    tree = ast.parse(_backends()[stem])
    init = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    params = {a.arg for a in (*init.args.posonlyargs, *init.args.args, *init.args.kwonlyargs)}
    assert "logger" not in params, f"{stem} still has a `logger:` slot"


@pytest.mark.parametrize("stem", sorted(CALLBACK_BACKENDS | FUNCTION_BACKENDS))
def test_every_backend_warms_its_sources_before_forking(stem: str) -> None:
    """A `DataLoader` worker may be a forked child; if it is the FIRST to touch a lazy source,
    the download runs there and the worker dies with a SIGSEGV and no traceback.

    Checked fleet-wide rather than in the fastai file alone, because `import fastai` sets the
    start method to `fork` PROCESS-WIDE — so any backend sharing that interpreter inherits the
    hazard, not just the one that imported it.
    """
    source = _backends()[stem]
    if built_on_shared_base(source):
        # The fork warm lives in the task base's `initialize` (via `_prepare` →
        # recordstream's `prepare_record_dataset`, pinned upstream); no shadow = it applies.
        assert "def initialize" not in source, f"{stem} shadows the shared (warming) initialize"
        return
    assert "ensure_materialized" in source


#: How each backend gets its METRICS onto the wired trackers — a DIFFERENT question from the
#: provenance hookup above, and the distinction is what let a real bug through: raidar's
#: `rfdetr_detector` had `log_run_context` (so the provenance table passed) while never calling
#: `tracker.log_metrics` at all, so a tracked run recorded its git state and its log file and NOT
#: its mAP. Its evaluate path had no tracker wiring whatsoever.
#:
#: The sibling restoration project found the same shape on its `transformers.Trainer` backend, and
#: only because a backend SWEEP put eleven runs in one table and one came out blank. A backend read
#: ALONE looks perfect — it prints its score and writes its metrics sidecar.
METRIC_BRIDGES = {
    "fastai_segmenter.py": "log_metrics",
    "huggingface_segmenter.py": "_log_trainer_history",
    "jax_segmenter.py": "log_metrics",
    "keras_segmenter.py": "keras_tracking_callback",
    "lightning_segmenter.py": "TrackerLogger",
    "mlx_segmenter.py": "log_metrics",
    "tensorflow_segmenter.py": "keras_tracking_callback",
    "torch_segmenter.py": "log_metrics",
}


def test_every_backend_puts_its_METRICS_on_the_trackers() -> None:
    """Provenance is not enough: a run whose numbers reach only the log is invisible to every
    comparison, every tracker UI and every sweep."""
    for name, marker in METRIC_BRIDGES.items():
        source = (SOURCE_DIR / name).read_text()
        if built_on_shared_base(source):
            # The bridge lives in the shared machinery base (the `TrackerLogger` adapters in
            # `_build_trainer` / the `tracker.log_metrics` calls in the loop's `fit` and
            # `_score`) — pinned in matrainer's own suite; no shadow = it applies.
            assert_does_not_shadow_the_shared_wiring(name, source)
            continue
        assert marker in source, f"{name} never reaches a tracker's metric channel via {marker}"


def test_the_metric_bridge_table_covers_every_backend_on_disk() -> None:
    """The converse — a new backend must not be added without a metric channel."""
    on_disk = {p.name for p in SOURCE_DIR.glob("*_segmenter.py")}
    assert on_disk == set(METRIC_BRIDGES)

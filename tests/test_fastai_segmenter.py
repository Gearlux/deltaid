"""``FastaiSegmenter`` — swap the LOOP and nothing else.

What is pinned here is mostly the handful of fastai FACTS that are silent or fatal when wrong: the
loss axis, which metrics already know about segmentation, the `@patch`-installed `fit_one_cycle`,
and the fork hazard that makes a worker die with no traceback. The rest — num_classes, the metric
merge, the tracking contract — is parity with the other backends, checked because this backend
duplicates that logic rather than sharing it.
"""

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from confluid import LazyClass, lazy_param_names, to_pydantic
from recordstream import Mask

fastai = pytest.importorskip("fastai")

from deltaid.fastai_segmenter import DEFAULT_METRICS, FastaiSegmenter, _as_metric, _as_opt_func  # noqa: E402

NUM_CLASSES = 3
SIDE = 32


class _TinyNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(x))


def _records(n: int = 4, classes: int = NUM_CLASSES) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    return [
        {
            "image": torch.rand(3, SIDE, SIDE),
            "mask": Mask(rng.integers(0, classes, (SIDE, SIDE)).astype(np.int64)),
            "record_id": i,
        }
        for i in range(n)
    ]


def _segmenter(**kwargs: Any) -> FastaiSegmenter:
    params: Dict[str, Any] = {
        "model": _TinyNet(),
        "num_classes": NUM_CLASSES,
        "max_epochs": 1,
        "batch_size": 2,
        "num_workers": 0,
    }
    params.update(kwargs)
    return FastaiSegmenter(**params)


# --------------------------------------------------------------------------- #
# The fastai facts — measured, and silent or fatal when wrong
# --------------------------------------------------------------------------- #
class TestTheFastaiFacts:
    def test_the_default_loss_uses_axis_1(self) -> None:
        """fastai's `CrossEntropyLossFlat` defaults to `axis=-1`, which flattens the CHANNEL
        dimension of an [N, C, H, W] prediction into the batch. That is not a subtle degradation
        — it RAISES — but nothing in the config hints at it, which is why the default states it.
        """
        seg = _segmenter()
        seg.initialize()
        assert cast(Any, seg.loss).axis == 1

    def test_the_default_axis_would_actually_fail(self) -> None:
        """The counterfactual, so the assertion above is not folklore."""
        from fastai.losses import CrossEntropyLossFlat

        logits, target = torch.randn(2, NUM_CLASSES, 8, 8), torch.randint(0, NUM_CLASSES, (2, 8, 8))
        with pytest.raises(ValueError, match="batch_size"):
            CrossEntropyLossFlat(axis=-1)(logits, target)

    def test_the_configured_loss_agrees_with_plain_torch(self) -> None:
        """`axis=1` is the per-pixel reading — the same number torch.nn.CrossEntropyLoss gives."""
        seg = _segmenter()
        seg.initialize()
        logits, target = torch.randn(2, NUM_CLASSES, 8, 8), torch.randint(0, NUM_CLASSES, (2, 8, 8))
        loss = cast(Any, seg.loss)
        assert float(loss(logits, target)) == pytest.approx(float(nn.CrossEntropyLoss()(logits, target)), abs=1e-6)

    def test_the_default_metrics_already_know_about_segmentation(self) -> None:
        """The asymmetry worth knowing: only the LOSS carries the wrong default, because it is
        shared with classification. `DiceMulti` / `JaccardCoeffMulti` exist FOR segmentation and
        already default to axis=1, so they need no fix."""
        seg = _segmenter()
        seg.initialize()
        assert seg.metrics == list(DEFAULT_METRICS)
        for name in DEFAULT_METRICS:
            assert _as_metric(name)().axis == 1

    def test_fit_one_cycle_is_installed_by_the_schedule_IMPORT(self) -> None:
        """`Learner.fit_one_cycle` is a `@patch` applied as an import SIDE EFFECT, and fastai's
        `@patch` returns None — so importing the function by name binds None. Without the
        module import, `learn.fit_one_cycle` falls through fastcore's attribute delegation to
        the MODEL and raises `'SegmentationModelPytorch' object has no attribute ...`.
        """
        from fastai.learner import Learner

        assert hasattr(Learner, "fit_one_cycle")
        source = Path(inspect.getfile(FastaiSegmenter)).read_text()
        assert "from fastai.callback import schedule" in source

    def test_a_metric_name_that_does_not_exist_says_where_names_come_from(self) -> None:
        with pytest.raises(ValueError, match="fastai.metrics"):
            _as_metric("NotAMetric")

    def test_an_optimizer_name_that_does_not_exist_says_where_names_come_from(self) -> None:
        with pytest.raises(ValueError, match="fastai.optimizer"):
            _as_opt_func("NotAnOptimizer")

    def test_metrics_are_names_because_they_are_not_a_uniform_kind(self) -> None:
        """`foreground_acc` is a function, `DiceMulti` a class fastai instantiates itself — so
        there is nothing uniform for a `!class:` tag to name."""
        assert callable(_as_metric("foreground_acc"))
        assert isinstance(_as_metric("DiceMulti"), type)


# --------------------------------------------------------------------------- #
# The fork hazard — this backend is the reason it was found
# --------------------------------------------------------------------------- #
class TestForkSafety:
    def test_every_dataset_is_materialized_in_the_parent(self) -> None:
        """`import fastai` sets the multiprocessing start method to `fork`, so a DataLoader
        worker is a forked child. If that child is the FIRST to touch a lazy source, the
        download runs there and reaches `_scproxy` / CoreFoundation — not fork-safe — and the
        worker dies with a SIGSEGV and no traceback (`DataLoader worker exited unexpectedly`).
        """
        source = Path(inspect.getfile(FastaiSegmenter)).read_text()
        assert "ensure_materialized" in source
        # And no path resolves a dataset WITHOUT warming it: every call site goes through
        # `_prepare`, so a new entry point cannot quietly skip the guard.
        assert "ensure_record_dataset(self." not in source

    def test_prepare_warms_the_source(self) -> None:
        reads: List[int] = []

        class _Lazy:
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int) -> Dict[str, Any]:
                reads.append(index)
                return _records(1)[0]

        _segmenter()._prepare(_Lazy())
        assert reads == [0], "the source must be read ONCE, in the parent"

    def test_a_missing_split_needs_no_guard_at_the_call_site(self) -> None:
        assert _segmenter()._prepare(None) is None


# --------------------------------------------------------------------------- #
# What it shares with the other torch backends
# --------------------------------------------------------------------------- #
class TestParity:
    def test_num_classes_resolution_agrees(self) -> None:
        assert _resolved(_segmenter(model=_TinyNet(num_classes=5), num_classes=9)) == 5
        assert _resolved(_segmenter(model=LazyClass(_TinyNet, num_classes=4), num_classes=None)) == 4
        assert _resolved(_segmenter(model=LazyClass(_TinyNet), num_classes=None, train_set=_records())) == 3

    def test_it_reuses_the_same_model_wrapper_as_every_torch_backend(self) -> None:
        """fastai runs on torch, so the model slot takes any nn.Module — the config's `model:`
        line is unchanged between the three torch backends, which is the whole claim."""
        seg = _segmenter(model=_TinyNet())
        seg.initialize()
        assert isinstance(seg.model, nn.Module)

    def test_a_non_module_model_fails_during_SETUP(self) -> None:
        """Not minutes into a fit."""
        seg = _segmenter(model="not a model")
        with pytest.raises(TypeError, match="must be a torch.nn.Module"):
            seg.initialize()

    def test_there_is_no_train_metrics_slot(self) -> None:
        """A `Learner` computes its metrics on the VALIDATION pass only, so a `train_metrics`
        param would accept a wired set and silently never call it — the exact failure the
        per-split slots exist to remove."""
        params = inspect.signature(FastaiSegmenter.__init__).parameters
        assert "train_metrics" not in params
        assert {"metrics", "val_metrics", "test_metrics"} <= set(params)

    def test_it_uses_the_shared_combination_rule(self) -> None:
        assert "from matrainer.metric_slots import" in Path(inspect.getfile(FastaiSegmenter)).read_text()

    def test_the_loaders_are_the_lightning_backends_construction(self) -> None:
        """Same names, same deferral, same overridability — so ONE config idiom drives both. The
        divergence is exactly one kwarg: `collate_fn=self._to_tuple`, because fastai unpacks a
        batch POSITIONALLY."""
        seg = _segmenter()
        assert {"train_loader", "val_loader", "test_loader"} <= lazy_param_names(FastaiSegmenter)
        assert cast(Any, seg.train_loader).kwargs["collate_fn"] == seg._to_tuple

    def test_fastais_own_objects_are_deferred_slots_too(self) -> None:
        """Everything this backend CONSTRUCTS should be reachable from config — otherwise every
        knob it does not restate (a DataLoaders device, a Learner's `moms` / `splitter`) is
        unreachable from YAML."""
        assert {"dataloaders", "learner"} <= lazy_param_names(FastaiSegmenter)

    def test_trackers_and_callbacks_are_constructor_params(self) -> None:
        fields = to_pydantic(FastaiSegmenter).model_fields
        assert {"trackers", "callbacks"} <= set(fields)

    def test_no_local_hyperparameter_derivation_survives(self) -> None:
        source = Path(inspect.getfile(FastaiSegmenter)).read_text()
        assert "_config_params" not in source
        assert "runnable=self" in source

    def test_the_terminal_call_is_the_one_that_ends_the_run(self) -> None:
        source = inspect.getsource(FastaiSegmenter.fit)
        assert "capture_log_file" in source
        assert ".finalize(" not in source

    def test_the_entrypoint_markers_are_the_dispatch_table(self) -> None:
        from recordstream.runnable import runnable_entrypoints

        assert set(runnable_entrypoints(FastaiSegmenter)) == {"fit", "evaluate", "test", "predict"}

    def test_the_model_slot_is_not_lazy_any(self) -> None:
        tree = ast.parse(Path(inspect.getfile(FastaiSegmenter)).read_text())
        init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        model = next(a for a in init.args.args if a.arg == "model")
        assert model.annotation is not None and "Lazy[Any]" not in ast.unparse(model.annotation)


def _resolved(seg: FastaiSegmenter) -> int:
    seg.initialize()
    assert seg.num_classes is not None
    return seg.num_classes


# --------------------------------------------------------------------------- #
# The batch boundary
# --------------------------------------------------------------------------- #
class TestTheBatchBoundary:
    def test_a_batch_is_a_TUPLE_not_a_record_dict(self) -> None:
        """fastai unpacks positionally, which is the one real difference from Lightning."""
        x, y = _segmenter()._to_tuple(_records(3))
        assert x.shape == (3, 3, SIDE, SIDE)
        assert y.shape == (3, SIDE, SIDE)

    def test_the_target_is_int64(self) -> None:
        """albumentations returns int32 and the cross-entropy family rejects it."""
        _, y = _segmenter()._to_tuple(_records(2))
        assert y.dtype == torch.int64

    def test_with_targets_false_yields_the_input_alone(self) -> None:
        assert len(_segmenter()._to_tuple(_records(2), with_targets=False)) == 1


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
class TestEndToEnd:
    def test_fit_trains_and_reports_the_metrics_by_name(self) -> None:
        seg = _segmenter(train_set=_records(8), model=LazyClass(_TinyNet), max_epochs=2)
        seg.val_set = _records(4)
        seg.run()
        # The Recorder's columns, translated: `valid_loss` is the val loss and its unprefixed
        # metrics are VALIDATION ones, which is what fastai's vocabulary encodes.
        assert {"train/loss", "val/loss", "val/dice_multi", "val/jaccard_coeff_multi"} <= set(seg.computed_metrics)

    def test_a_missing_val_set_warns_rather_than_failing(self) -> None:
        """A `Learner` always validates — there is no "skip the val loop" mode — so the training
        data is reused and the metrics measure fit, not generalization. Worth saying out loud."""
        seg = _segmenter(train_set=_records(4), model=LazyClass(_TinyNet))
        seg.run()  # no val_set wired
        assert seg.computed_metrics

    def test_predict_streams_every_record_to_the_sink_with_its_metadata(self) -> None:
        """A plain no_grad forward over a RECORD loader, not `Learner.get_preds`: get_preds
        returns only the stacked predictions, and the sink needs each row's metadata."""
        written: List[Any] = []

        class _Sink:
            def write(self, prediction: Any, metadata: Any) -> None:
                written.append((prediction, metadata))

            def close(self) -> None:
                pass

        seg = _segmenter(task="predict", model=_TinyNet())
        seg.test_set = _records(5)
        seg.predictions_sink = _Sink()
        seg.run()

        assert len(written) == 5
        assert [m["record_id"] for _, m in written] == [0, 1, 2, 3, 4]
        assert set(written[0][0]) == {"logits", "probs", "mask"}

    def test_test_writes_a_metrics_sidecar(self, tmp_path: Path) -> None:
        import json

        seg = _segmenter(task="test", model=_TinyNet())
        seg.test_set = _records(4)
        seg.metrics_sidecar_path = str(tmp_path / "metrics.json")
        seg.run()
        assert "metrics" in json.loads((tmp_path / "metrics.json").read_text())

"""``MlxSegmenter`` — the eighth backend, on Apple's MLX.

Two groups. The ones shared with every other backend (num_classes resolution, the sentinels, the
metric slots, the tracking contract) are checked again because this backend DUPLICATES that logic
rather than inheriting it, and a divergence between copies is exactly what goes unnoticed. The
rest is this engine's alone, and it is about four things — three of which have no analogue
anywhere else in the fleet:

* the loss is masked by ``ignore_index`` HERE, because ``mlx.nn.losses`` takes no such argument;
* targets are **int32**, not the int64 every other backend names — MLX indexes with int32;
* train/eval is module STATE, so the mode is set once per phase rather than passed per call;
* arrays are LAZY, so ``mx.eval`` is what forces the work and omitting it does not raise.

Apple silicon ONLY: `mlx` is importable nowhere else, so this whole file is SKIPPED where the
`mlx` extra is absent — the module-level `importorskip` below is what enforces that, because the
segmenter import at the bottom of this block pulls `mlx.core` in at collection time.
"""

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

pytest.importorskip("mlx", reason="the mlx backend needs deltaid[mlx] (Apple silicon only)")

from confluid import PartialClass, partial_param_names, to_pydantic  # noqa: E402
from recordstream import Mask, collate_records  # noqa: E402

from deltaid.mlx_segmenter import _METRICS, MlxSegmenter  # noqa: E402
from deltaid.unet.mlx_unet import MlxUNet  # noqa: E402

NUM_CLASSES = 3
SIDE = 16


def _records(n: int = 4, classes: int = NUM_CLASSES, learnable: bool = False) -> List[Dict[str, Any]]:
    """Synthetic records in the shape a CHANNELS-LAST config produces.

    The image is ``[H, W, 3]`` and not ``[3, H, W]``: that IS the difference between an MLX config's
    preprocess chain and a torch one, so building the fixture the other way would test a pipeline
    no config produces.
    """
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        image = rng.random((SIDE, SIDE, 3)).astype("float32")
        mask = image.argmax(axis=-1) if learnable else rng.integers(0, classes, (SIDE, SIDE))
        out.append({"image": image, "mask": Mask(mask.astype(np.int64)), "record_id": i})
    return out


def _segmenter(**kwargs: Any) -> MlxSegmenter:
    params: Dict[str, Any] = {
        "model": PartialClass(MlxUNet),
        "num_classes": NUM_CLASSES,
        "max_epochs": 1,
        "batch_size": 2,
    }
    params.update(kwargs)
    return MlxSegmenter(**params)


# --------------------------------------------------------------------------- #
# What this engine changes
# --------------------------------------------------------------------------- #
class TestWhatTheEngineChanges:
    def test_the_batch_is_channels_last_with_an_INT32_target(self) -> None:
        """int32, not the int64 every other backend names: MLX indexes with int32 and its
        `cross_entropy` expects it. The one dtype in the fleet that differs, and it differs
        because of the framework rather than the task."""
        x, y = _segmenter()._to_arrays(collate_records(_records(2)))
        assert x.shape == (2, SIDE, SIDE, 3)
        assert y.shape == (2, SIDE, SIDE), "a class-id map has no channel axis in ANY engine"
        assert y.dtype == np.int32

    def test_predict_shaped_batches_carry_no_target(self) -> None:
        assert len(_segmenter()._to_arrays(collate_records(_records(2)), with_targets=False)) == 1

    def test_the_default_loss_agrees_with_torch(self) -> None:
        """`mlx.nn.losses.cross_entropy` takes `[..., C]` logits against `[...]` integer labels, so
        the `[N, H, W, C]` / `[N, H, W]` pair needs no reshape. Pinned numerically rather than by
        name because "the same loss" is the claim, and the shapes make it non-obvious."""
        import mlx.core as mx
        import torch

        rng = np.random.default_rng(0)
        logits = rng.standard_normal((2, 8, 8, NUM_CLASSES)).astype("float32")
        labels = rng.integers(0, NUM_CLASSES, (2, 8, 8)).astype("int32")

        ours = float(_segmenter()._loss_fn()(mx.array(logits), mx.array(labels)))
        theirs = float(
            torch.nn.functional.cross_entropy(torch.tensor(logits).permute(0, 3, 1, 2), torch.tensor(labels).long())
        )
        assert ours == pytest.approx(theirs, abs=1e-5)

    def test_the_ignore_index_is_applied_HERE_not_passed_to_mlx(self) -> None:
        """`mlx.nn.losses` takes no `ignore_index`, so the per-pixel loss is masked and the mean is
        over the VALID pixels only. Dividing by the full pixel count instead would shrink the loss
        in proportion to how much of the image is void — a silent, image-dependent LR change."""
        import mlx.core as mx

        logits = np.zeros((1, 4, 4, NUM_CLASSES), dtype="float32")
        labels = np.zeros((1, 4, 4), dtype="int32")
        labels[0, :2] = 100  # half the image is void

        masked = float(_segmenter(ignore_index=100)._loss_fn()(mx.array(logits), mx.array(labels)))
        # The same loss over ONLY the valid half — which is what masking must give.
        kept = float(_segmenter()._loss_fn()(mx.array(logits[:, 2:]), mx.array(labels[:, 2:])))
        assert masked == pytest.approx(kept, abs=1e-6), "the ignored half must not dilute the mean"

    def test_an_ignored_label_never_reaches_the_loss_as_a_class_id(self) -> None:
        """100 is not a valid class id, so it would index out of range inside the loss — clamped
        to 0 first and then masked out of the mean."""
        import mlx.core as mx

        logits = np.zeros((1, 2, 2, NUM_CLASSES), dtype="float32")
        labels = np.full((1, 2, 2), 100, dtype="int32")
        assert np.isfinite(float(_segmenter(ignore_index=100)._loss_fn()(mx.array(logits), mx.array(labels))))

    def test_the_loop_forces_the_lazy_graph_once_per_step(self) -> None:
        """MLX arrays are LAZY and omitting `mx.eval` does NOT raise: the graph grows every step
        until memory runs out, so the run just gets slower and then dies. The eval covers the
        parameters AND the optimizer state, not only the loss — the state is what carries the
        momentum buffers."""
        from matrainer.mlx.loop import MlxLoopRunnable

        source = inspect.getsource(MlxLoopRunnable._train_epoch)
        assert "mx.eval(network.parameters(), optimizer.state, loss)" in source

    def test_the_train_mode_is_set_PER_PHASE_not_per_call(self) -> None:
        """MLX carries train/eval as module STATE, so a `training=False` default on `__call__`
        would flip the network to eval on every forward — including the one inside the training
        step, where `nn.value_and_grad` calls `model(x)` with no flag to pass."""
        from matrainer.mlx.loop import MlxLoopRunnable

        assert "network.train(True)" in inspect.getsource(MlxLoopRunnable._train_epoch)
        assert "network.train(False)" in inspect.getsource(MlxLoopRunnable._eval_pass)

    def test_the_optimizer_name_is_INSTANTIATED_not_called(self) -> None:
        """The asymmetry with JAX: optax ships FACTORIES (`optax.adam(lr)`), `mlx.optimizers`
        ships CLASSES — so a name resolves and is constructed with the learning rate."""
        import mlx.optimizers as optim
        from matrainer.mlx.loop import as_mlx_optimizer as _as_optimizer

        built = _as_optimizer("Adam", 0.005)
        assert isinstance(built, optim.Adam)
        assert float(built.learning_rate) == pytest.approx(0.005)

    def test_the_losses_and_metrics_are_NAMES_so_there_is_no_registrations_module(self) -> None:
        """`mlx.nn.losses` ships plain FUNCTIONS, so there is no class for a `!class:` tag to
        resolve or for confluid's registry to tag — the fastai and JAX situation."""
        from deltaid.cli import FRAMEWORKS

        assert FRAMEWORKS["mlx"].registrations is None
        assert not (Path(inspect.getfile(MlxSegmenter)).parent / "mlx_segmentation.py").exists()

    def test_an_unknown_name_says_where_names_come_from(self) -> None:
        from matrainer.mlx.loop import as_mlx_optimizer as _as_optimizer

        with pytest.raises(ValueError, match="mlx.nn.losses"):
            _segmenter(loss="not_a_loss")._loss_fn()
        with pytest.raises(ValueError, match="Known names"):
            from deltaid.mlx_segmenter import _as_metric

            _as_metric("not_a_metric")
        with pytest.raises(ValueError, match="mlx.optimizers"):
            _as_optimizer("NotAnOptimizer", 1e-3)


# --------------------------------------------------------------------------- #
# The metrics — resolved by NAME, accumulated over the split
# --------------------------------------------------------------------------- #
class TestTheMetrics:
    """The ARITHMETIC lives in `matrainer.metric_functions`, pinned there against torchmetrics.

    What is this backend's own — and what is checked here — is the WIRING: which names reach the
    accumulator, and that `ignore_index` travels with them.
    """

    def test_the_named_metrics_are_matrainers_confusion_functions(self) -> None:
        """Not a re-implementation: the same objects the torch backends' torchmetrics agree with,
        so a number from this loop is comparable with one from theirs."""
        from matrainer.metric_functions import iou_per_class, mean_iou, overall_accuracy

        assert _METRICS == {
            "mean_iou": mean_iou,
            "pixel_accuracy": overall_accuracy,
            "iou_per_class": iou_per_class,
        }

    def test_a_named_metric_is_ACCUMULATED_and_a_callable_is_averaged(self) -> None:
        """Partitioned by NAME, before resolution — nothing here can know what a user-supplied
        callable would want accumulated."""
        seg = _segmenter(metrics=["mean_iou", lambda logits, labels, ignore: 1.0])
        confusion, batchwise = seg._partition_metrics("val")

        assert list(confusion) == ["mean_iou"]
        assert len(batchwise) == 1 and callable(batchwise[0])

    def test_no_accumulator_is_built_when_no_named_metric_was_wired(self) -> None:
        seg = _segmenter(metrics=[lambda logits, labels, ignore: 1.0])
        assert seg._new_split_state("val")["matrix"] is None

    def test_the_metric_does_not_depend_on_the_BATCH_SIZE(self) -> None:
        """The property the whole change exists for, exercised through the real epoch loop.

        A per-batch average cannot have it — a class absent from a batch is skipped by that
        batch's mean, so the value moves with the batch size.
        """
        records = _records(8)
        scores = {}
        for size in (2, 4, 8):
            seg = _segmenter(batch_size=size)
            seg.initialize()
            scores[size] = float(seg._eval_pass(seg._sequence(records), "val")["val/mean_iou"])

        assert len({round(value, 10) for value in scores.values()}) == 1, scores

    def test_the_ignore_index_reaches_the_accumulator(self) -> None:
        """The same knob must drop the void class from the SCORE as well as the loss, or the model
        is graded on pixels it was told to ignore."""
        seg = _segmenter(ignore_index=100)
        seg.initialize()
        records = _records(2)
        for record in records:
            record["mask"] = Mask(np.full((SIDE, SIDE), 100, dtype=np.int64))

        values = seg._eval_pass(seg._sequence(records), "val")
        assert float(values["val/mean_iou"]) == 0.0
        assert float(values["val/pixel_accuracy"]) == 0.0


# --------------------------------------------------------------------------- #
class TestTheSharedContract:
    def test_num_classes_resolution_agrees(self) -> None:
        assert _resolved(_segmenter(model=PartialClass(MlxUNet, num_classes=4), num_classes=None)) == 4
        assert _resolved(_segmenter(model=PartialClass(MlxUNet), num_classes=None, train_set=_records())) == 3

    def test_it_uses_the_shared_combination_rule(self) -> None:
        assert "from matrainer.metric_slots import" in Path(inspect.getfile(MlxSegmenter)).read_text()

    def test_a_split_slot_adds_to_the_shared_set_for_that_split_only(self) -> None:
        seg = _segmenter(metrics=["mean_iou"], val_metrics=["pixel_accuracy"])
        confusion, _ = seg._partition_metrics("val")
        assert list(confusion) == ["mean_iou", "pixel_accuracy"]
        test_confusion, _ = seg._partition_metrics("test")
        assert list(test_confusion) == ["mean_iou"]

    def test_all_four_metric_slots_are_declared(self) -> None:
        """The training pass scores too. What used to be described as a property of MLX was a
        property of the loop as written: the step returns `(loss, logits)` and MLX differentiates
        the FIRST element of a returned tuple, so the logits ride out for free."""
        params = inspect.signature(MlxSegmenter.__init__).parameters
        assert {"metrics", "train_metrics", "val_metrics", "test_metrics"} <= set(params)
        confusion, _ = _segmenter(metrics=["mean_iou"], train_metrics=["pixel_accuracy"])._partition_metrics("train")
        assert list(confusion) == ["mean_iou", "pixel_accuracy"]

    def test_the_step_returns_the_logits_it_took_the_gradient_through(self) -> None:
        """`nn.value_and_grad` differentiates the FIRST element of the returned tuple, so the
        logits come back beside the loss and the training metrics measure the network AS IT
        TRAINED — no second forward pass, and no eval-mode network standing in for a train-mode
        one."""
        from matrainer.mlx.loop import MlxLoopRunnable

        assert "return loss_fn(outputs, y), outputs" in inspect.getsource(MlxLoopRunnable._make_step)

    def test_the_entrypoint_markers_are_the_dispatch_table(self) -> None:
        from recordstream.runnable import runnable_entrypoints

        assert set(runnable_entrypoints(MlxSegmenter)) == {"fit", "evaluate", "test", "predict"}
        assert "run_entrypoint(self, self.task)" in inspect.getsource(MlxSegmenter.run)

    def test_trackers_and_the_metric_slots_are_constructor_params(self) -> None:
        fields = to_pydantic(MlxSegmenter).model_fields
        assert {"trackers", "metrics", "val_metrics", "test_metrics"} <= set(fields)

    def test_the_deferred_slots_are_declared_lazy(self) -> None:
        """`callbacks` is deliberately NOT here: `Optional[List[Partial[EpochCallback]]]` declares a
        LIST of deferred entries, which is what gates confluid's list broadcast — the slot itself
        is not the deferred thing, each entry is."""
        assert {"model", "train_set", "val_set", "test_set", "trackers"} <= partial_param_names(MlxSegmenter)

    def test_a_function_driven_backend_finishes_its_own_manifest(self) -> None:
        source = inspect.getsource(MlxSegmenter.fit)
        assert "log_run_context" in source and "capture_log_file" in source
        assert ".finalize(" not in source


def _resolved(seg: MlxSegmenter) -> int:
    seg.initialize()
    assert seg.num_classes is not None
    return seg.num_classes


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class TestTheLoop:
    def test_fit_trains_and_reports_both_splits(self) -> None:
        seg = _segmenter(train_set=_records(8), val_set=_records(4))
        seg.run()
        assert {"train/loss", "val/loss", "val/mean_iou", "val/pixel_accuracy"} <= set(seg.computed_metrics)

    def test_the_loss_actually_descends(self) -> None:
        """A finite loss is not evidence of training. Learnable data on purpose: over random masks
        the loss would sit at ln(C) however well the run was wired."""
        seg = _segmenter(train_set=_records(8, learnable=True), max_epochs=6)
        seg.run()
        losses = seg.history["train/loss"]
        assert len(losses) == 6
        assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_the_row_order_is_reshuffled_between_epochs(self) -> None:
        """A `fit()` would have done this; a hand-written loop must. Forgetting it is INVISIBLE —
        training silently runs on one fixed row order forever."""
        assert "train_seq.on_epoch_end()" in inspect.getsource(MlxSegmenter.fit)

    def test_a_per_class_metric_goes_to_the_TEXT_channel_not_log_metrics(self, tmp_path: Path) -> None:
        """`log_metrics` is a `{name: float}` channel, so a `[C]` vector has to go out as a table
        or be dropped — the same split every other backend makes."""

        class _Tracker:
            """Records what it was handed. `log_dir` is not decoration: `log_run_context` reads
            the FIRST tracker's for where the manifest lands, so a fake without one fails inside
            matrainer rather than in anything this test is about."""

            def __init__(self, log_dir: Path) -> None:
                self.log_dir = str(log_dir)
                self.texts: List[Any] = []
                self.metrics: List[Any] = []

            def __getattr__(self, name: str) -> Any:
                return lambda *a, **k: None  # every other Tracker method is a no-op here

            def log_metrics(self, values: Any, step: Any = None) -> None:
                self.metrics.append(values)

            def log_text(self, name: str, text: str, step: Any = None) -> None:
                self.texts.append((name, text))

        tracker = _Tracker(tmp_path)
        seg = _segmenter(
            train_set=_records(4),
            val_set=_records(4),
            val_metrics=["mean_iou", "iou_per_class"],
            class_names=["pet", "background", "boundary"],
            trackers=[tracker],
        )
        seg.run()

        assert [name for name, _ in tracker.texts] == ["val/iou_per_class"]
        assert "boundary" in tracker.texts[0][1]
        assert all(isinstance(v, float) for values in tracker.metrics for v in values.values())
        assert "val/iou_per_class" not in seg.computed_metrics, "a vector must not reach the scalars"

    def test_test_writes_a_metrics_sidecar(self, tmp_path: Path) -> None:
        seg = _segmenter(task="test", test_set=_records(4))
        seg.metrics_sidecar_path = str(tmp_path / "metrics.json")
        seg.run()
        assert "test/mean_iou" in json.loads((tmp_path / "metrics.json").read_text())["metrics"]

    def test_evaluate_scores_the_validation_split(self) -> None:
        """The fleet harmonization: `evaluate` used to delegate to `test()` in this backend."""
        seg = _segmenter(task="evaluate", val_set=_records(4))
        seg.run()
        assert "val/mean_iou" in seg.computed_metrics

    def test_predict_streams_every_record_to_the_sink_and_closes_it(self) -> None:
        written: List[Any] = []
        closed: List[bool] = []

        class _Sink:
            def write(self, prediction: Any, metadata: Any) -> None:
                written.append((prediction, metadata))

            def close(self) -> None:
                closed.append(True)

        seg = _segmenter(task="predict", test_set=_records(6))
        seg.predictions_sink = _Sink()
        seg.run()

        assert len(written) == 6
        assert [m["record_id"] for _, m in written] == [0, 1, 2, 3, 4, 5]
        assert set(written[0][0]) == {"logits", "probs", "mask"}
        assert written[0][0]["mask"].shape == (SIDE, SIDE)
        assert np.allclose(written[0][0]["probs"].sum(axis=-1), 1.0), "softmax over the CHANNEL axis"
        assert closed == [True]

    def test_a_callback_sees_each_epoch(self) -> None:
        """JAX ships no callback protocol, so a callback is a plain callable — the plainest thing
        that works, and the slot's signature is what makes it checkable."""
        seen: List[int] = []
        seg = _segmenter(train_set=_records(4), max_epochs=2, callbacks=[lambda epoch, logs: seen.append(epoch)])
        seg.run()
        assert seen == [0, 1]

    def test_the_outputs_raise_before_a_run(self) -> None:
        with pytest.raises(RuntimeError, match="only available after run"):
            _ = _segmenter().computed_metrics

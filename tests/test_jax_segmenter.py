"""``JaxSegmenter`` — the loop written out and jitted, over a Flax model.

Two groups. The ones shared with every other backend (num_classes resolution, the sentinels, the
metric slots, the tracking contract) are checked again because this backend DUPLICATES that logic
rather than inheriting it, and a divergence between copies is exactly what goes unnoticed. The
rest is this engine's alone, and it is about three things:

* the loss is masked by ``ignore_index`` HERE, because optax's losses take no such argument;
* the metrics are plain ``jnp`` functions resolved by NAME, because optax ships functions and
  there is no class for a `!class:` tag to resolve;
* the step functions are jitted ONCE and reused, because a fresh ``jit`` per epoch retraces.
"""

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from confluid import LazyClass, lazy_param_names, to_pydantic
from recordstream import Mask, collate_records

from deltaid.jax_segmenter import JaxSegmenter, _iou_per_class, _mean_iou, _pixel_accuracy
from deltaid.unet.flax_unet import FlaxUNet

NUM_CLASSES = 3
SIDE = 16


def _records(n: int = 4, classes: int = NUM_CLASSES, learnable: bool = False) -> List[Dict[str, Any]]:
    """Synthetic records in the shape a CHANNELS-LAST config produces.

    The image is ``[H, W, 3]`` and not ``[3, H, W]``: that IS the difference between a JAX config's
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


def _segmenter(**kwargs: Any) -> JaxSegmenter:
    params: Dict[str, Any] = {
        "model": LazyClass(FlaxUNet),
        "num_classes": NUM_CLASSES,
        "max_epochs": 1,
        "batch_size": 2,
    }
    params.update(kwargs)
    return JaxSegmenter(**params)


# --------------------------------------------------------------------------- #
# What this engine changes
# --------------------------------------------------------------------------- #
class TestWhatTheEngineChanges:
    def test_the_batch_is_channels_last_with_an_int64_target(self) -> None:
        x, y = _segmenter()._to_arrays(collate_records(_records(2)))
        assert x.shape == (2, SIDE, SIDE, 3)
        assert y.shape == (2, SIDE, SIDE), "a class-id map has no channel axis in ANY engine"
        assert y.dtype == np.int64

    def test_predict_shaped_batches_carry_no_target(self) -> None:
        assert len(_segmenter()._to_arrays(collate_records(_records(2)), with_targets=False)) == 1

    def test_the_default_loss_agrees_with_torch(self) -> None:
        """`softmax_cross_entropy_with_integer_labels` is the per-pixel twin of
        `torch.nn.CrossEntropyLoss` — it takes `[..., C]` logits against `[...]` integer labels, so
        the `[N, H, W, C]` / `[N, H, W]` pair needs no reshape. Pinned numerically rather than by
        name because "the same loss" is the claim, and the shapes make it non-obvious."""
        import jax.numpy as jnp
        import torch

        rng = np.random.default_rng(0)
        logits = rng.standard_normal((2, 8, 8, NUM_CLASSES)).astype("float32")
        labels = rng.integers(0, NUM_CLASSES, (2, 8, 8)).astype("int64")

        ours = float(_segmenter()._loss_fn()(jnp.asarray(logits), jnp.asarray(labels)))
        theirs = float(
            torch.nn.functional.cross_entropy(torch.tensor(logits).permute(0, 3, 1, 2), torch.tensor(labels))
        )
        assert ours == pytest.approx(theirs, abs=1e-5)

    def test_the_ignore_index_is_applied_HERE_not_passed_to_optax(self) -> None:
        """optax's losses take no `ignore_index`, so the per-pixel loss is masked and the mean is
        over the VALID pixels only. Dividing by the full pixel count instead would shrink the loss
        in proportion to how much of the image is void — a silent, image-dependent LR change."""
        import jax.numpy as jnp

        logits = np.zeros((1, 4, 4, NUM_CLASSES), dtype="float32")
        labels = np.zeros((1, 4, 4), dtype="int64")
        labels[0, :2] = 255  # half the image is void

        masked = float(_segmenter(ignore_index=255)._loss_fn()(jnp.asarray(logits), jnp.asarray(labels)))
        # The same loss over ONLY the valid half — which is what masking must give.
        kept = float(_segmenter()._loss_fn()(jnp.asarray(logits[:, 2:]), jnp.asarray(labels[:, 2:])))
        assert masked == pytest.approx(kept, abs=1e-6), "the ignored half must not dilute the mean"

    def test_an_ignored_label_never_reaches_the_loss_as_a_class_id(self) -> None:
        """255 is not a valid class id, so it would index out of range inside the loss — clamped
        to 0 first and then masked out of the mean."""
        import jax.numpy as jnp

        logits = np.zeros((1, 2, 2, NUM_CLASSES), dtype="float32")
        labels = np.full((1, 2, 2), 255, dtype="int64")
        assert np.isfinite(float(_segmenter(ignore_index=255)._loss_fn()(jnp.asarray(logits), jnp.asarray(labels))))

    def test_the_steps_are_jitted_ONCE_and_reused(self) -> None:
        """A fresh `nnx.jit` each epoch retraces the whole graph every time, which is slower than
        the eager loop it replaced."""
        source = inspect.getsource(JaxSegmenter.fit)
        assert source.count("_make_steps()") == 1
        assert "nnx.jit" not in source, "jitting belongs in `_make_steps`, not per epoch"

    def test_jit_can_be_turned_off_to_read_a_traceback(self) -> None:
        train, _ = _segmenter(jit=False)._make_steps()
        assert not type(train).__name__.startswith("Jitted")

    def test_the_losses_and_metrics_are_NAMES_so_there_is_no_registrations_module(self) -> None:
        """optax ships plain FUNCTIONS, so there is no class for a `!class:` tag to resolve or for
        confluid's registry to tag — the same situation the fastai metrics are in."""
        from deltaid.cli import FRAMEWORKS

        assert FRAMEWORKS["jax"].registrations is None
        assert not (Path(inspect.getfile(JaxSegmenter)).parent / "jax_segmentation.py").exists()

    def test_an_unknown_name_says_where_names_come_from(self) -> None:
        with pytest.raises(ValueError, match="optax.losses"):
            _segmenter(loss="not_a_loss")._loss_fn()
        with pytest.raises(ValueError, match="Known names"):
            _segmenter(metrics=["not_a_metric"])._metric_values(None, None, ["not_a_metric"])


# --------------------------------------------------------------------------- #
# The metrics — plain jnp functions, so they are checkable directly
# --------------------------------------------------------------------------- #
class TestTheMetrics:
    def _perfect(self) -> Any:
        labels = np.array([[[0, 1], [2, 0]]])
        logits = np.zeros((1, 2, 2, NUM_CLASSES), dtype="float32")
        for y in range(2):
            for x in range(2):
                logits[0, y, x, labels[0, y, x]] = 10.0
        return logits, labels

    def test_a_perfect_prediction_scores_one(self) -> None:
        logits, labels = self._perfect()
        assert float(_mean_iou(logits, labels, -100)) == pytest.approx(1.0)
        assert float(_pixel_accuracy(logits, labels, -100)) == pytest.approx(1.0)

    def test_ignored_pixels_are_excluded_from_both(self) -> None:
        """The same knob must drop the void class from the SCORE as well as the loss, or the model
        is graded on pixels it was told to ignore."""
        labels = np.array([[[0, 255], [255, 0]]])
        logits = np.zeros((1, 2, 2, NUM_CLASSES), dtype="float32")
        logits[..., 0] = 10.0
        assert float(_pixel_accuracy(logits, labels, 255)) == pytest.approx(1.0)
        assert float(_mean_iou(logits, labels, 255)) == pytest.approx(1.0)

    def test_a_class_absent_from_the_batch_is_SKIPPED_not_scored_zero(self) -> None:
        """Scoring it 0 would drag the mean down with a class the batch never had a chance to get
        right — which is what `MulticlassJaccardIndex` avoids too."""
        labels = np.zeros((1, 2, 2), dtype="int64")
        logits = np.zeros((1, 2, 2, NUM_CLASSES), dtype="float32")
        logits[..., 0] = 10.0
        assert float(_mean_iou(logits, labels, -100)) == pytest.approx(1.0)

    def test_the_per_class_vector_reports_nan_for_an_absent_class(self) -> None:
        """`nan` says "not present in this batch"; 0 would say "the model got this wrong"."""
        labels = np.zeros((1, 2, 2), dtype="int64")
        logits = np.zeros((1, 2, 2, NUM_CLASSES), dtype="float32")
        logits[..., 0] = 10.0
        values = np.asarray(_iou_per_class(logits, labels, -100))
        assert values.shape == (NUM_CLASSES,)
        assert values[0] == pytest.approx(1.0)
        assert np.isnan(values[1]) and np.isnan(values[2])

    def test_a_fully_ignored_batch_scores_zero_rather_than_dividing_by_zero(self) -> None:
        labels = np.full((1, 2, 2), 255, dtype="int64")
        logits = np.zeros((1, 2, 2, NUM_CLASSES), dtype="float32")
        assert float(_pixel_accuracy(logits, labels, 255)) == 0.0
        assert float(_mean_iou(logits, labels, 255)) == 0.0


# --------------------------------------------------------------------------- #
# The contract every backend shares
# --------------------------------------------------------------------------- #
class TestTheSharedContract:
    def test_num_classes_resolution_agrees(self) -> None:
        assert _resolved(_segmenter(model=LazyClass(FlaxUNet, num_classes=4), num_classes=None)) == 4
        assert _resolved(_segmenter(model=LazyClass(FlaxUNet), num_classes=None, train_set=_records())) == 3

    def test_it_uses_the_shared_combination_rule(self) -> None:
        assert "from matrainer.metric_slots import" in Path(inspect.getfile(JaxSegmenter)).read_text()

    def test_a_split_slot_adds_to_the_shared_set_for_that_split_only(self) -> None:
        seg = _segmenter(metrics=["mean_iou"], val_metrics=["pixel_accuracy"])
        assert list(seg._split_metrics("val")) == ["mean_iou", "pixel_accuracy"]
        assert list(seg._split_metrics("test")) == ["mean_iou"]

    def test_there_is_no_train_metrics_slot(self) -> None:
        """A property of this loop rather than an omission: the training pass computes the LOSS and
        nothing else, so a train-only metric would have nowhere to be computed."""
        assert "train_metrics" not in inspect.signature(JaxSegmenter.__init__).parameters

    def test_the_entrypoint_markers_are_the_dispatch_table(self) -> None:
        from recordstream.runnable import runnable_entrypoints

        assert set(runnable_entrypoints(JaxSegmenter)) == {"fit", "evaluate", "test", "predict"}
        assert "run_entrypoint(self, self.task)" in inspect.getsource(JaxSegmenter.run)

    def test_trackers_and_the_metric_slots_are_constructor_params(self) -> None:
        fields = to_pydantic(JaxSegmenter).model_fields
        assert {"trackers", "metrics", "val_metrics", "test_metrics"} <= set(fields)

    def test_the_deferred_slots_are_declared_lazy(self) -> None:
        """`callbacks` is deliberately NOT here: `Optional[List[Lazy[EpochCallback]]]` declares a
        LIST of deferred entries, which is what gates confluid's list broadcast — the slot itself
        is not the deferred thing, each entry is."""
        assert {"model", "train_set", "val_set", "test_set", "trackers"} <= lazy_param_names(JaxSegmenter)

    def test_a_function_driven_backend_finishes_its_own_manifest(self) -> None:
        source = inspect.getsource(JaxSegmenter.fit)
        assert "log_run_context" in source and "capture_log_file" in source
        assert ".finalize(" not in source


def _resolved(seg: JaxSegmenter) -> int:
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
        assert {"loss", "val_loss", "val_mean_iou", "val_pixel_accuracy"} <= set(seg.computed_metrics)

    def test_the_loss_actually_descends(self) -> None:
        """A finite loss is not evidence of training. Learnable data on purpose: over random masks
        the loss would sit at ln(C) however well the run was wired."""
        seg = _segmenter(train_set=_records(8, learnable=True), max_epochs=6)
        seg.run()
        losses = seg.history["loss"]
        assert len(losses) == 6
        assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_the_row_order_is_reshuffled_between_epochs(self) -> None:
        """A `fit()` would have done this; a hand-written loop must. Forgetting it is INVISIBLE —
        training silently runs on one fixed row order forever."""
        assert "train_seq.on_epoch_end()" in inspect.getsource(JaxSegmenter.fit)

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

        assert [name for name, _ in tracker.texts] == ["val_iou_per_class"]
        assert "boundary" in tracker.texts[0][1]
        assert all(isinstance(v, float) for values in tracker.metrics for v in values.values())
        assert "val_iou_per_class" not in seg.computed_metrics, "a vector must not reach the scalars"

    def test_test_writes_a_metrics_sidecar(self, tmp_path: Path) -> None:
        seg = _segmenter(task="test", test_set=_records(4))
        seg.metrics_sidecar_path = str(tmp_path / "metrics.json")
        seg.run()
        assert "test_mean_iou" in json.loads((tmp_path / "metrics.json").read_text())["metrics"]

    def test_evaluate_and_test_are_the_same_class(self) -> None:
        seg = _segmenter(task="evaluate", test_set=_records(4))
        seg.run()
        assert "test_mean_iou" in seg.computed_metrics

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

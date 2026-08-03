"""``HuggingFaceSegmenter`` — a ``transformers.Trainer`` fine-tuning a Hub checkpoint.

The backend that swaps the MODEL as well as the loop, and the tests are shaped by the three facts
that makes true — none of which any other backend has to deal with:

1. **The model computes its own loss** from ``labels=``, so there is no loss slot and no loss
   registration module. What the config CAN set is the ignore index, and it is not a loss
   argument here but a value on ``model.config.semantic_loss_ignore_index``, whose default is
   **255** rather than torch's ``-100``.
2. **The logits come back at 1/4 resolution.** SegFormer emits ``[N, C, H/4, W/4]`` and upsamples
   internally only for its own loss, so every metric and the sink must be handed aligned arrays.
3. **The batch keys are transformers'**, not this project's: ``pixel_values`` / ``labels``.

The behavioural group builds a tiny local model rather than downloading a checkpoint — the
structure under test is this class's, and a Hub round trip would make the suite need a network.
"""

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from confluid import lazy_param_names, to_pydantic
from recordstream import Mask

from deltaid.huggingface_segmenter import HuggingFaceSegmenter, _mean_iou, _pixel_accuracy, _upsample_logits

NUM_CLASSES = 3
SIDE = 32


class _TinySegformerLike(torch.nn.Module):
    """The engine's contract in miniature: ``forward(pixel_values, labels)`` returning
    ``(loss, logits)``, with the logits at a REDUCED resolution — the one structural fact about
    this backend that a metric or a sink has to cope with."""

    def __init__(self, num_classes: int = NUM_CLASSES, stride: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride
        self.head = torch.nn.Conv2d(3, num_classes, kernel_size=1, stride=stride)
        self.config = type("Cfg", (), {"semantic_loss_ignore_index": 255, "num_labels": num_classes})()

    def forward(self, pixel_values: torch.Tensor, labels: Any = None, **kwargs: Any) -> Any:
        logits = self.head(pixel_values)
        loss = None
        if labels is not None:
            upsampled = torch.nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = torch.nn.functional.cross_entropy(
                upsampled, labels, ignore_index=self.config.semantic_loss_ignore_index
            )
        return type("Out", (), {"loss": loss, "logits": logits, "__getitem__": lambda s, i: (loss, logits)[i]})()


def _records(n: int = 4, classes: int = NUM_CLASSES) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        image = torch.rand(3, SIDE, SIDE)
        out.append(
            {
                "image": image,
                "mask": Mask(rng.integers(0, classes, (SIDE, SIDE)).astype(np.int64)),
                "record_id": i,
            }
        )
    return out


def _segmenter(**kwargs: Any) -> HuggingFaceSegmenter:
    params: Dict[str, Any] = {
        "model": _TinySegformerLike(),
        "num_classes": NUM_CLASSES,
        "max_epochs": 1,
        "batch_size": 2,
        "report_to": "none",
    }
    params.update(kwargs)
    return HuggingFaceSegmenter(**params)


# --------------------------------------------------------------------------- #
# What this engine changes
# --------------------------------------------------------------------------- #
class TestWhatTheEngineChanges:
    def test_there_is_no_loss_slot_and_no_loss_registration_module(self) -> None:
        """An `AutoModelForSemanticSegmentation` computes its own loss from `labels=`, so there is
        nothing a config could wire — which is also why `FRAMEWORKS["huggingface"]` names no
        registrations module rather than naming an empty one."""
        from deltaid.cli import FRAMEWORKS

        assert "loss" not in inspect.signature(HuggingFaceSegmenter.__init__).parameters
        assert FRAMEWORKS["huggingface"].registrations is None
        assert not (Path(inspect.getfile(HuggingFaceSegmenter)).parent / "huggingface_segmentation.py").exists()

    def test_the_ignore_index_is_written_onto_the_MODEL_CONFIG(self) -> None:
        """The one backend where it is not a loss argument. The model reads
        `config.semantic_loss_ignore_index`, whose default is 255 — NOT torch's -100 — so leaving
        it alone would send a deltaid-default mask into `cross_entropy` as an out-of-range id."""
        seg = _segmenter(ignore_index=250, train_set=_records(2))
        seg.initialize()
        assert seg._network().config.semantic_loss_ignore_index == 250

    def test_the_batch_keys_are_transformers_own(self) -> None:
        """`pixel_values` is what an image model's forward takes; `labels` is what makes it
        return a loss. Neither is negotiable, and neither matches this project's record keys."""
        batch = _segmenter()._to_hf_batch(_records(2))
        assert set(batch) == {"pixel_values", "labels"}
        assert batch["pixel_values"].shape == (2, 3, SIDE, SIDE)
        assert batch["labels"].dtype == torch.int64

    def test_the_metrics_hook_upsamples_before_any_metric_sees_the_logits(self) -> None:
        """Placed in the hook rather than in a metric, so EVERY metric — built-in or
        user-supplied — sees arrays that already agree and none of them has to know that this
        model emits at a reduced resolution."""
        seen: Dict[str, Any] = {}

        def spy(logits: np.ndarray, labels: np.ndarray, ignore_index: int) -> float:
            seen["logits"], seen["labels"] = logits.shape, labels.shape
            return 0.0

        hook = _segmenter(metrics=[spy])._metrics_hook("val")
        pred = type("P", (), {"predictions": np.zeros((2, NUM_CLASSES, 8, 8)), "label_ids": np.zeros((2, 32, 32))})()
        hook(pred)
        assert seen["logits"] == (2, NUM_CLASSES, 32, 32) == (2, NUM_CLASSES, *seen["labels"][-2:])

    def test_a_tuple_prediction_is_unwrapped(self) -> None:
        """`Trainer` hands back a tuple when the model returns more than logits, and indexing the
        tuple as an array would silently score the wrong thing."""
        hook = _segmenter()._metrics_hook("val")
        logits = np.zeros((2, NUM_CLASSES, 8, 8))
        out = hook(type("P", (), {"predictions": (logits, None), "label_ids": np.zeros((2, 32, 32))})())
        assert set(out) == {"mean_iou", "pixel_accuracy"}

    def test_the_split_slots_merge_per_split(self) -> None:
        seg = _segmenter(metrics=["pixel_accuracy"], val_metrics=["mean_iou"])
        pred = type("P", (), {"predictions": np.zeros((2, NUM_CLASSES, 8, 8)), "label_ids": np.zeros((2, 32, 32))})()
        assert set(seg._metrics_hook("val")(pred)) == {"pixel_accuracy", "mean_iou"}
        assert set(seg._metrics_hook("test")(pred)) == {"pixel_accuracy"}


# --------------------------------------------------------------------------- #
# The metric functions — plain numpy, so they are checkable directly
# --------------------------------------------------------------------------- #
class TestTheBuiltInMetrics:
    def test_a_perfect_prediction_scores_one(self) -> None:
        labels = np.array([[[0, 1], [2, 0]]])
        logits = np.zeros((1, NUM_CLASSES, 2, 2))
        for y in range(2):
            for x in range(2):
                logits[0, labels[0, y, x], y, x] = 10.0
        assert _mean_iou(logits, labels, -100) == pytest.approx(1.0)
        assert _pixel_accuracy(logits, labels, -100) == pytest.approx(1.0)

    def test_ignored_pixels_are_excluded_from_both(self) -> None:
        """The same knob must drop the void class from the SCORE as well as the loss, or the
        model is graded on pixels it was told to ignore."""
        labels = np.array([[[0, 255], [255, 0]]])
        logits = np.zeros((1, NUM_CLASSES, 2, 2))
        logits[0, 0] = 10.0  # every pixel predicted class 0 — correct on both non-ignored ones
        assert _pixel_accuracy(logits, labels, 255) == pytest.approx(1.0)
        assert _mean_iou(logits, labels, 255) == pytest.approx(1.0)

    def test_a_fully_ignored_batch_scores_zero_rather_than_dividing_by_zero(self) -> None:
        labels = np.full((1, 2, 2), 255)
        assert _pixel_accuracy(np.zeros((1, NUM_CLASSES, 2, 2)), labels, 255) == 0.0
        assert _mean_iou(np.zeros((1, NUM_CLASSES, 2, 2)), labels, 255) == 0.0

    def test_the_upsample_preserves_the_class_count_and_reaches_the_label_size(self) -> None:
        out = _upsample_logits(np.zeros((2, NUM_CLASSES, 8, 8)), (32, 32))
        assert out.shape == (2, NUM_CLASSES, 32, 32)

    def test_an_already_aligned_logit_map_is_returned_untouched(self) -> None:
        logits = np.random.default_rng(0).random((2, NUM_CLASSES, 32, 32))
        assert _upsample_logits(logits, (32, 32)) is logits


# --------------------------------------------------------------------------- #
# The contract every backend shares
# --------------------------------------------------------------------------- #
class TestTheSharedContract:
    def test_the_entrypoint_markers_are_the_dispatch_table(self) -> None:
        from recordstream.runnable import runnable_entrypoints

        assert set(runnable_entrypoints(HuggingFaceSegmenter)) == {"fit", "evaluate", "test", "predict"}
        assert "run_entrypoint(self, self.task)" in inspect.getsource(HuggingFaceSegmenter.run)

    def test_trackers_is_a_constructor_param_that_admits_a_list(self) -> None:
        assert "trackers" in to_pydantic(HuggingFaceSegmenter).model_fields

    def test_the_deferred_slots_are_declared_lazy(self) -> None:
        """`Trainer` and `TrainingArguments` are DEFERRED slots, so a config can replace either
        without this class growing a knob per `TrainingArguments` field."""
        assert {"model", "train_set", "trainer", "training_arguments"} <= lazy_param_names(HuggingFaceSegmenter)

    def test_no_local_hyperparameter_derivation_survives(self) -> None:
        source = Path(inspect.getfile(HuggingFaceSegmenter)).read_text()
        assert "_config_params" not in source
        assert "runnable=self" in source

    def test_overriding_the_training_arguments_slot_ADDS_to_the_derived_floor(self) -> None:
        """A config reaching for one of `TrainingArguments`' ~100 fields must not lose the rest.

        The slot is a whole-value override (that is what a YAML key does), so the derived knobs
        live in `_training_arguments()` rather than pinned on the constructor's `LazyClass`.
        Pinned there, `training_arguments: {use_cpu: true}` would silently drop
        `remove_unused_columns=False` — which surfaces as a `KeyError: 'image'` three frames
        inside Trainer's own collator, saying nothing about the override.
        """
        from confluid import LazyClass

        seg = _segmenter(max_epochs=7, batch_size=4)
        seg.training_arguments = LazyClass("transformers.TrainingArguments", use_cpu=True)
        args = seg._training_arguments()
        assert args.use_cpu is True, "the override's own key must survive"
        assert args.remove_unused_columns is False, "the derived floor must survive the override"
        assert args.num_train_epochs == 7 and args.per_device_train_batch_size == 4

    def test_a_key_the_config_pinned_WINS_over_the_derived_one(self) -> None:
        """The derivation is a FLOOR, not an override — otherwise the slot could not be used to
        change anything the segmenter also has an opinion about."""
        from confluid import LazyClass

        seg = _segmenter(max_epochs=7)
        seg.training_arguments = LazyClass("transformers.TrainingArguments", num_train_epochs=2)
        assert seg._training_arguments().num_train_epochs == 2

    def test_a_function_driven_backend_finishes_its_own_manifest(self) -> None:
        """No tracking CALLBACK here (`report_to` is transformers' own integration surface), so
        the run's terminal status is this class's to stamp — left unfinished the manifest stays
        `running` and matrainer's atexit hook labels a successful run `interrupted`."""
        source = inspect.getsource(HuggingFaceSegmenter.fit)
        assert "log_run_context" in source
        assert "capture_log_file" in source
        assert ".finalize(" not in source


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class TestTheLoop:
    def test_fit_trains_and_reports_metrics(self, tmp_path: Path) -> None:
        seg = _segmenter(train_set=_records(8), output_dir=str(tmp_path / "run"))
        seg.val_set = _records(4)
        seg.run()
        assert {"eval_loss", "eval_mean_iou", "eval_pixel_accuracy"} <= set(seg.computed_metrics)

    def test_test_writes_a_metrics_sidecar(self, tmp_path: Path) -> None:
        seg = _segmenter(task="test", test_set=_records(4), output_dir=str(tmp_path / "run"))
        seg.metrics_sidecar_path = str(tmp_path / "metrics.json")
        seg.run()
        assert "eval_mean_iou" in json.loads((tmp_path / "metrics.json").read_text())["metrics"]

    def test_predict_streams_every_record_at_the_INPUT_resolution(self, tmp_path: Path) -> None:
        """The sink's consumer expects a prediction the size of the image it came from, and this
        engine is the only one that does not already produce that."""
        written: List[Any] = []
        closed: List[bool] = []

        class _Sink:
            def write(self, prediction: Any, metadata: Any) -> None:
                written.append((prediction, metadata))

            def close(self) -> None:
                closed.append(True)

        seg = _segmenter(task="predict", test_set=_records(6), output_dir=str(tmp_path / "run"))
        seg.predictions_sink = _Sink()
        seg.run()

        assert len(written) == 6
        assert [m["record_id"] for _, m in written] == [0, 1, 2, 3, 4, 5]
        assert set(written[0][0]) == {"logits", "probs", "mask"}
        assert tuple(written[0][0]["mask"].shape) == (SIDE, SIDE)
        assert closed == [True]

    def test_the_outputs_raise_before_a_run(self) -> None:
        with pytest.raises(RuntimeError, match="only available after run"):
            _ = _segmenter().computed_metrics

"""``LightningSegmenter`` — the merged train / evaluate / test / predict segmentation runnable.

Grouped by the decision each block pins, because most of them are silent when wrong: a slot that
accepts a value and ignores it, a default that cannot be told apart from a user's choice, a metric
that reports nothing. The end-to-end group at the bottom drives a real ``run()`` over synthetic
records, which is what catches the wiring the unit groups each half-cover.
"""

from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from confluid import Fluid, PartialClass, partial_param_names, to_pydantic
from recordstream import Mask, collate_records, per_record_predictions

from deltaid.lightning_segmenter import LightningSegmenter

NUM_CLASSES = 3
SIDE = 32


class _TinyNet(nn.Module):
    """A 1x1-conv 'segmenter': [N, 3, H, W] -> [N, C, H, W], trainable, instant."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(x))


def _records(n: int = 4, classes: int = NUM_CLASSES) -> List[Dict[str, Any]]:
    """Synthetic segmentation records in the shape the config pipeline produces."""
    rng = np.random.default_rng(0)
    return [
        {
            "image": torch.rand(3, SIDE, SIDE),
            "mask": Mask(rng.integers(0, classes, (SIDE, SIDE)).astype(np.int64)),
            "record_id": i,
        }
        for i in range(n)
    ]


def _target_masks(n: int = 2) -> torch.Tensor:
    """An int64 ``[N, H, W]`` target — read back exactly as the segmenter reads it.

    Through `collate_records` + `batch_tensor` rather than stacking the `Mask`es directly,
    because a `Mask` is an ndarray SUBCLASS and `torch.as_tensor` on one takes the memoryview
    path and raises — the read-back is the thing that knows how to do this.
    """
    from recordstream import batch_tensor

    return cast(torch.Tensor, batch_tensor(collate_records(_records(n)), "mask", dtype=torch.int64))


def _segmenter(**kwargs: Any) -> LightningSegmenter:
    params: Dict[str, Any] = {"model": _TinyNet(), "num_classes": NUM_CLASSES, "max_epochs": 1}
    params.update(kwargs)
    seg = LightningSegmenter(**params)
    for slot in (seg.train_loader, seg.val_loader, seg.test_loader):
        cast(Any, slot).kwargs.update(batch_size=2, num_workers=0, persistent_workers=False)
    cast(Any, seg.lightning).kwargs.update(
        max_epochs=1, enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False, logger=False
    )
    return seg


# --------------------------------------------------------------------------- #
# num_classes resolution
# --------------------------------------------------------------------------- #
class TestNumClassesResolution:
    def test_an_explicit_count_is_used(self) -> None:
        seg = _segmenter(model=nn.Identity(), num_classes=7)
        seg.initialize()
        assert seg.num_classes == 7

    def test_a_concrete_models_head_wins_over_the_segmenters_arg(self) -> None:
        """The head is authoritative — metrics sized against anything else would mis-score it."""
        seg = _segmenter(model=_TinyNet(num_classes=5), num_classes=9)
        seg.initialize()
        assert seg.num_classes == 5

    def test_a_lazy_models_pinned_kwarg_is_read_without_touching_the_data(self) -> None:
        seg = _segmenter(model=PartialClass(_TinyNet, num_classes=4), num_classes=None)
        seg.initialize()
        assert seg.num_classes == 4

    def test_it_falls_back_to_walking_the_mask_column(self) -> None:
        """The last resort, and the reason `num_mask_classes` exists: a per-pixel target is an
        array, so `recordstream.num_classes` (scalar-only, deliberately) cannot answer this."""
        seg = _segmenter(model=PartialClass(_TinyNet), num_classes=None, train_set=_records(classes=3))
        seg.initialize()
        assert seg.num_classes == 3

    def test_the_derived_count_is_injected_into_the_lazy_model(self) -> None:
        seg = _segmenter(model=PartialClass(_TinyNet), num_classes=None, train_set=_records(classes=3))
        seg.initialize()
        assert not isinstance(seg.model, Fluid)
        assert seg.model.num_classes == 3

    def test_it_raises_when_nothing_can_supply_a_count(self) -> None:
        seg = _segmenter(model=PartialClass(_TinyNet), num_classes=None, train_set=None)
        with pytest.raises(RuntimeError, match="could not determine num_classes"):
            seg.initialize()

    def test_eval_reads_the_count_off_the_trained_model_not_the_dataset(self) -> None:
        """An eval split may be a subset whose masks never show the last class; sizing the
        metrics from it would score the model against a head it does not have."""
        seg = _segmenter(model=_TinyNet(num_classes=5), num_classes=None, task="test")
        seg.test_set = _records(classes=2)
        seg.initialize()
        assert seg.num_classes == 5


# --------------------------------------------------------------------------- #
# Task-shaped defaults are SENTINELS
# --------------------------------------------------------------------------- #
class TestSentinelDefaults:
    def test_loss_defaults_to_cross_entropy_with_the_ignore_index(self) -> None:
        seg = _segmenter(ignore_index=250)
        seg.initialize()
        assert isinstance(seg.loss_fn, nn.CrossEntropyLoss)
        assert seg.loss_fn.ignore_index == 250

    def test_an_explicitly_wired_loss_is_never_replaced(self) -> None:
        """Why the default is a `None` sentinel rather than a live constructor default: a live
        one makes "the user chose CrossEntropy" indistinguishable from "the user said nothing",
        so a deliberately-chosen Dice loss would be silently discarded."""
        chosen = nn.NLLLoss()
        seg = _segmenter()
        seg.loss_fn = chosen
        seg.initialize()
        assert seg.loss_fn is chosen

    def test_the_metric_default_is_the_segmentation_pair(self) -> None:
        seg = _segmenter()
        seg.initialize()
        assert set(seg.val_metrics.keys()) == {"val/mean_iou", "val/dice_macro"}

    def test_the_default_metrics_honour_the_ignore_index(self) -> None:
        """The same knob must drop the void class from the SCORE as well as the loss —
        otherwise the model is graded on pixels it was told to ignore."""
        seg = _segmenter(ignore_index=250)
        seg.initialize()
        assert seg.val_metrics["val/mean_iou"].ignore_index == 250

    def test_resolution_is_idempotent_so_a_direct_step_call_works(self) -> None:
        """`training_step` resolves the sentinels itself, so a test or a custom loop can call it
        without `initialize()` — and calling it twice must not replace anything."""
        seg = _segmenter()
        seg.initialize()
        loss = seg.loss_fn
        seg._resolve_task_defaults()
        assert seg.loss_fn is loss


# --------------------------------------------------------------------------- #
# Per-split metric slots
# --------------------------------------------------------------------------- #
class TestPerSplitMetrics:
    def test_a_split_slot_ADDS_to_the_shared_set_for_that_split_only(self) -> None:
        from torchmetrics.classification import MulticlassAccuracy

        seg = _segmenter(val_metrics={"accuracy": PartialClass(MulticlassAccuracy)})
        seg.initialize()
        assert set(seg.val_metrics.keys()) == {"val/mean_iou", "val/dice_macro", "val/accuracy"}
        assert set(seg.train_metrics.keys()) == {"train/mean_iou", "train/dice_macro"}

    def test_all_four_slots_are_declared_constructor_params(self) -> None:
        """A post-construction-only slot is invisible to `to_pydantic`, so navigaitor's form-spec
        and StreamStudio would never offer it."""
        fields = to_pydantic(LightningSegmenter).model_fields
        assert {"metrics", "train_metrics", "val_metrics", "test_metrics"} <= set(fields)

    def test_metrics_are_built_once_even_though_run_calls_initialize_every_time(self) -> None:
        """`_metrics_built` is not tidiness: `is None` cannot tell "not built yet" from "wired
        with a spec", and a second build would re-prefix `val/` into `val/val/`."""
        seg = _segmenter()
        seg.initialize()
        seg.initialize()
        assert all(name.startswith("val/") and not name.startswith("val/val/") for name in seg.val_metrics)

    def test_the_shared_rule_comes_from_matrainer(self) -> None:
        """One combination rule across every backend AND both tasks — never re-derived here.

        Since the 2026-08-06 consolidation the rule is applied by the shared family base
        (`matrainer.torch.runnable._build_split_metrics`, pinned in matrainer's suite); what
        this module must not do is SHADOW the builder with an inline re-derivation.
        """
        import inspect

        import matrainer.torch.runnable as family

        import deltaid.lightning_segmenter as module

        source = inspect.getsource(module)
        assert "def _build_split_metrics" not in source, "the shared metric builder is shadowed"
        assert "from matrainer.metric_slots import" in inspect.getsource(family)


# --------------------------------------------------------------------------- #
# The TRAIN split reduces once per epoch, like the eval splits
# --------------------------------------------------------------------------- #
class TestTheTrainingSplitReducesOncePerEpoch:
    """It used to log the per-BATCH values with ``on_step=True``, which lost them two ways.

    Lightning's ``log_every_n_steps`` (50 by default) DROPS a per-step series shorter than the
    interval, so a 32-step run recorded no training metric at all — measured on the backend sweep
    at ``--epochs 1``, where `lightning_zoo` wrote `train/loss` and nothing else beside a
    `torch_zoo` row carrying `train/mean_iou` + `train/pixel_accuracy` from the SAME config. And
    what survived a longer run was one batch's value where every other backend reports an epoch
    aggregate, so two columns of a cross-framework table were not measuring the same thing.
    """

    def _fed(self, **kwargs: Any) -> LightningSegmenter:
        seg = _segmenter(**kwargs)
        seg.initialize()
        seg.log = lambda *a, **k: None  # type: ignore[method-assign]
        return seg

    def test_the_training_step_accumulates_and_the_epoch_end_reports(self) -> None:
        logged: List[Any] = []
        seg = self._fed()
        seg.log_dict = lambda values, *a, **k: logged.append(values)  # type: ignore[method-assign]

        seg.training_step(collate_records(_records(2)), 0)
        assert logged == [], "the training step must accumulate, not report"

        seg.on_train_epoch_end()
        assert set(logged[0]) == {"train/mean_iou", "train/dice_macro"}

    def test_the_reset_is_at_epoch_START_so_the_last_epoch_survives_the_run(self) -> None:
        """`_write_metrics_sidecar` and the `computed_metrics` @output both `compute()` AFTER
        `trainer.fit()` has returned, so resetting at the end would hand them an emptied metric."""
        seg = self._fed()
        seg.log_dict = lambda *a, **k: None  # type: ignore[method-assign]

        seg.training_step(collate_records(_records(2)), 0)
        metric = seg.train_metrics["train/mean_iou"]
        during_epoch = float(metric.compute())

        seg.on_train_epoch_end()
        assert float(metric.compute()) == pytest.approx(during_epoch), "the epoch END must not reset"

        seg.on_train_epoch_start()
        assert metric.update_count == 0, "the epoch START is what resets"

    def test_a_per_class_vector_on_the_TRAIN_split_is_reported_rather_than_dropped(self) -> None:
        """The second thing the old shape cost: it ran everything through `scalar_metrics`, so a
        `[C]` vector wired to `train_metrics` vanished with no error at all."""
        from torchmetrics.classification import MulticlassJaccardIndex

        tracker = _RecordingTracker()
        seg = self._fed(
            train_metrics={"iou_per_class": PartialClass(MulticlassJaccardIndex, average=None)},
            trackers=[tracker],
        )
        seg.log_dict = lambda *a, **k: None  # type: ignore[method-assign]

        seg.training_step(collate_records(_records(2)), 0)
        seg.on_train_epoch_end()

        assert [name for name, _, _ in tracker.texts] == ["train/iou_per_class"]

    def test_the_two_loss_series_are_unchanged(self) -> None:
        """`train/loss` (epoch) and `train/loss_step` (batch) stay two explicit names — Lightning
        appends `_step`/`_epoch` when ONE name carries both, and `train/loss_epoch` is a name no
        other backend writes."""
        names: List[str] = []
        seg = self._fed()
        seg.log = lambda name, *a, **k: names.append(name)  # type: ignore[method-assign]
        seg.log_dict = lambda *a, **k: None  # type: ignore[method-assign]

        seg.training_step(collate_records(_records(2)), 0)
        assert names == ["train/loss", "train/loss_step"]


# --------------------------------------------------------------------------- #
# trackers: — the only output wiring, and a CONSTRUCTOR param
# --------------------------------------------------------------------------- #
class TestTrackers:
    def test_trackers_is_a_constructor_param_that_admits_a_list(self) -> None:
        """confluid gates LIST broadcast on the receiving slot's ANNOTATION, so as a
        post-construction attribute typed `Optional[List[Tracker]]` a `trackers: [...]` YAML key
        silently did NOTHING — the whole tracking chain unreachable from config while every unit
        test passed, because tests assign `.trackers` in Python."""
        assert "trackers" in to_pydantic(LightningSegmenter).model_fields

    def test_the_constructor_stores_them_verbatim(self) -> None:
        """Normalizing in the constructor would iterate the value, and a deferred `Fluid` is not
        iterable (`'Class' object is not iterable`) — plus it would be functional work."""
        marker = PartialClass(_TinyNet)
        seg = _segmenter(trackers=marker)
        assert seg.trackers is marker

    def test_flowing_them_writes_back_so_a_fluid_is_not_rebuilt(self) -> None:
        """`flow()` on a live object is idempotent, but a `Fluid` would be REBUILT on every
        call — two MLflow runs for one training run."""
        from matrainer import CsvTracker

        seg = _segmenter(trackers=PartialClass(CsvTracker))
        first = seg._live_trackers()
        assert seg._live_trackers()[0] is first[0]

    def test_there_is_no_logger_slot(self) -> None:
        """It named two different things and neither survived; `trackers:` is the one wiring."""
        import inspect

        assert "logger" not in inspect.signature(LightningSegmenter.__init__).parameters

    def test_a_metric_names_key_broadcasts_from_a_flat_config(self) -> None:
        """The naming knob has to be reachable the way every other one is — a top-level key.

        Wired in a CONSTRUCTOR slot it arrives as a deferred marker (only `!class:X()` builds
        eagerly), which is why the slot stores it verbatim and `as_metric_names` flows it at use.
        """
        from confluid import flow, load
        from confluid.fluid import Fluid
        from matrainer.metric_names import as_metric_names

        yaml = """
runnable: !class:deltaid.LightningSegmenter
  run_name: x
model: "MODEL_SENTINEL"
metric_names: !class:matrainer.MetricNames
  target: keras
"""
        seg = load(yaml)["runnable"]
        seg = flow(seg) if isinstance(seg, Fluid) else seg

        assert seg.metric_names is not None, "a `metric_names:` key did not reach the slot"
        assert as_metric_names(seg.metric_names).target == "keras"

    def test_the_naming_knob_reaches_the_adapters(self) -> None:
        """One key, and every tracker this run writes to records that vocabulary."""
        from matrainer import CsvTracker, MetricNames
        from matrainer.lightning import TrackerLogger

        names = MetricNames(target="keras")
        seg = _segmenter(trackers=[CsvTracker(save_dir="./_deltaid_test_logs")], metric_names=names)

        adapters = [lg for lg in seg._build_trainer().loggers if isinstance(lg, TrackerLogger)]
        assert adapters and adapters[0].metric_names is names

    def test_no_local_hyperparameter_derivation_survives(self) -> None:
        """matrainer owns the ONE derivation (`runnable=self`), so two backends cannot answer
        "what were the hyperparameters?" differently."""
        import inspect

        import deltaid.lightning_segmenter as module

        source = inspect.getsource(module)
        assert "_config_params" not in source
        # `runnable=self` lives in the shared machinery base's `_build_trainer` since
        # 2026-08-06 (pinned in matrainer's suite); the module must not shadow it.
        assert "def _build_trainer" not in source


# --------------------------------------------------------------------------- #
# The batch boundary
# --------------------------------------------------------------------------- #
class TestTheBatchBoundary:
    def test_the_target_is_read_back_as_int64(self) -> None:
        """albumentations returns an int32 mask, and CrossEntropyLoss rejects it with
        *"expected target dtype to be Long or Byte, but got Int"*. The dtype is named by the
        CALLER — recordstream honours the contract rather than deciding one."""
        batch = collate_records(_records(2))
        seg = _segmenter()
        assert seg._targets(batch).dtype == torch.int64

    def test_the_collate_is_the_generic_one(self) -> None:
        """No bespoke deltaid collate — and the two keys arrive by DIFFERENT routes.

        The mask is a typed `Mask` item, so `collate_records` stacks it to `[N, H, W]` itself.
        The image is a live `torch.Tensor` (what `ToTensor` emits — an `NDArrayItem` cannot hold
        one), which is a PLAIN record value, so the collate gathers it as a per-record LIST and
        `batch_tensor` does the stacking at the model boundary. Both are recordstream's; deltaid
        writes neither.
        """
        batch = collate_records(_records(3))
        assert batch["mask"].shape == (3, SIDE, SIDE)  # stacked by the collate
        assert isinstance(batch["image"], list) and len(batch["image"]) == 3  # gathered

        from recordstream import batch_tensor

        assert batch_tensor(batch, "image").shape == (3, 3, SIDE, SIDE)
        assert batch_tensor(batch, "mask", dtype=torch.int64).shape == (3, SIDE, SIDE)

    def test_predict_step_emits_the_contract_not_raw_logits(self) -> None:
        """A sink reads `mask` / `probs` by NAME and cannot softmax on the caller's behalf —
        only the model knows whether its head already did."""
        seg = _segmenter()
        seg.initialize()
        out = seg.predict_step(collate_records(_records(2)), 0)
        assert set(out) == {"logits", "probs", "mask"}
        assert out["mask"].shape == (2, SIDE, SIDE)
        assert out["mask"].dtype == torch.int64

    def test_a_batched_output_is_sliced_to_one_entry_per_record(self) -> None:
        """The sink contract is per-record while the model emits `[N, ...]`; handing the batch
        over whole writes ONE entry for an N-record batch."""
        preds = {"mask": torch.zeros(4, 8, 8), "probs": torch.zeros(4, 3, 8, 8)}
        assert len(per_record_predictions(preds)) == 4

    def test_predictions_reach_a_sink_paired_with_their_metadata(self) -> None:
        written: List[Any] = []

        class _Sink:
            def write(self, prediction: Any, metadata: Any) -> None:
                written.append((prediction, metadata))

            def close(self) -> None:
                pass

        seg = _segmenter()
        seg.initialize()
        seg.predictions_sink = _Sink()
        seg.predict_step(collate_records(_records(3)), 0)
        assert len(written) == 3
        assert [m["record_id"] for _, m in written] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Dispatch + the I/O contract
# --------------------------------------------------------------------------- #
class TestDispatchAndContract:
    def test_the_entrypoint_markers_are_the_dispatch_table(self) -> None:
        """A hand-written {task: method} dict restates them and drifts."""
        import inspect

        from recordstream.runnable import entrypoint_tasks, runnable_entrypoints

        assert set(runnable_entrypoints(LightningSegmenter)) == {"fit", "evaluate", "test", "predict"}
        assert "run_entrypoint(self, self.task)" in inspect.getsource(LightningSegmenter.run)
        # The roles are what a discovery service reads to learn that ONE class both trains and
        # evaluates — which is why there is no separate evaluator class to tag.
        assert entrypoint_tasks(LightningSegmenter, "trainer") == ["fit"]
        assert set(entrypoint_tasks(LightningSegmenter, "evaluator")) == {"evaluate", "test"}
        assert entrypoint_tasks(LightningSegmenter, "predictor") == ["predict"]

    def test_autograd_is_claimed_only_for_fit(self) -> None:
        assert _segmenter(task="fit").__needs_autograd__ is True
        assert _segmenter(task="predict").__needs_autograd__ is False

    def test_the_outputs_raise_before_a_run(self) -> None:
        seg = _segmenter(model=PartialClass(_TinyNet))
        with pytest.raises(RuntimeError, match="only available after run"):
            _ = seg.trained_model

    def test_the_model_input_is_mandatory(self) -> None:
        with pytest.raises(TypeError):
            LightningSegmenter()  # type: ignore[call-arg]

    def test_every_flowed_slot_is_lazy_declared(self) -> None:
        assert {"model", "train_set", "optimizer", "lightning"} <= partial_param_names(LightningSegmenter)


# --------------------------------------------------------------------------- #
# End to end — the wiring the unit groups each half-cover
# --------------------------------------------------------------------------- #
class TestEndToEnd:
    def test_fit_trains_computes_metrics_and_exposes_them(self) -> None:
        seg = _segmenter(train_set=_records(8), model=PartialClass(_TinyNet))
        seg.val_set = _records(4)
        seg.run()

        assert seg.num_classes == NUM_CLASSES
        computed = seg.computed_metrics
        assert "val/mean_iou" in computed and "val/loss" in computed
        assert 0.0 <= computed["val/mean_iou"] <= 1.0
        assert isinstance(seg.trained_model, nn.Module)

    def test_the_loss_actually_descends(self) -> None:
        """A finite loss is not evidence of training — a badly wired one stays finite too.

        Learnable data on purpose: the mask is the argmax over the input's three channels, which
        a 1x1 conv can fit exactly. Over random masks the loss would sit at ``ln(C)`` however
        well the optimizer was wired, so the test would be measuring nothing. What is asserted is
        the DESCENT rather than an absolute value — that is the claim, and it is what says the
        optimizer reached the parameters at all.
        """
        learnable = [
            {"image": img, "mask": Mask(img.argmax(dim=0).numpy().astype(np.int64))}
            for img in (torch.rand(3, SIDE, SIDE) for _ in range(8))
        ]
        losses: List[float] = []

        class _Recorder(pl.Callback):
            def on_train_epoch_end(self, trainer: Any, module: Any) -> None:
                # `train/loss`, not `train/loss_epoch`: the step and epoch series are logged
                # under names of their own so the canonical one is the plain quantity.
                losses.append(float(trainer.callback_metrics["train/loss"]))

        seg = _segmenter(train_set=learnable, model=PartialClass(_TinyNet))
        cast(Any, seg.lightning).kwargs.update(max_epochs=12, callbacks=[_Recorder()])
        seg.run()

        assert len(losses) == 12
        assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_test_writes_a_metrics_sidecar(self, tmp_path: Path) -> None:
        import json

        seg = _segmenter(task="test", model=_TinyNet())
        seg.test_set = _records(4)
        seg.metrics_sidecar_path = str(tmp_path / "metrics.json")
        seg.run()

        payload = json.loads((tmp_path / "metrics.json").read_text())
        assert payload["experiment_name"] == "deltaid_segmentation"
        assert "test/mean_iou" in payload["metrics"]

    def test_predict_streams_every_record_to_the_sink_and_closes_it(self) -> None:
        written: List[Any] = []
        closed: List[bool] = []

        class _Sink:
            def write(self, prediction: Any, metadata: Any) -> None:
                written.append(prediction)

            def close(self) -> None:
                closed.append(True)

        seg = _segmenter(task="predict", model=_TinyNet())
        seg.test_set = _records(6)
        seg.predictions_sink = _Sink()
        seg.run()

        assert len(written) == 6
        assert closed == [True]

    def test_evaluate_and_test_are_the_same_class(self) -> None:
        """One runnable, four entry points — there is no separate evaluator class."""
        seg = _segmenter(task="evaluate", model=_TinyNet())
        seg.val_set = _records(4)
        seg.run()
        assert "val/mean_iou" in seg.computed_metrics


# --------------------------------------------------------------------------- #
# The per-class metric — computed, and no longer dropped
# --------------------------------------------------------------------------- #
class _RecordingTracker:
    """A tracker that keeps what it was handed, so the report can be asserted on."""

    def __init__(self) -> None:
        self.texts: List[Any] = []
        self.figures: List[Any] = []
        self.metrics: List[Any] = []

    def log_metrics(self, values: Any, step: Any = None) -> None:
        self.metrics.append((values, step))

    def log_text(self, name: str, text: str, step: Any = None) -> None:
        self.texts.append((name, text, step))

    def log_figure(self, name: str, figure: Any, step: Any = None) -> None:
        self.figures.append((name, figure, step))


class TestThePerClassReport:
    """A `[C]` vector cannot go through `log_dict` — Lightning's `convert_tensors_to_scalars`
    RAISES on it — so it goes out through the tracker TEXT channel instead of being dropped."""

    def _segmenter(self, **kwargs: Any) -> LightningSegmenter:
        from torchmetrics.classification import MulticlassJaccardIndex

        seg = _segmenter(
            val_metrics={"iou_per_class": PartialClass(MulticlassJaccardIndex, average=None)},
            **kwargs,
        )
        seg.initialize()
        return seg

    def test_the_scalars_still_go_to_log_dict_and_the_vector_does_not(self) -> None:
        """The whole reason the split exists: one `log_dict` call carrying a `[C]` tensor kills
        the validation loop on the first epoch."""
        seg = self._segmenter()
        logged: List[Any] = []
        seg.log_dict = lambda values, *a, **k: logged.append(values)  # type: ignore[method-assign]
        seg.val_metrics.update(torch.randn(2, NUM_CLASSES, SIDE, SIDE), _target_masks(2))
        seg._report_epoch_metrics(seg.val_metrics)

        assert logged and set(logged[0]) == {"val/mean_iou", "val/dice_macro"}
        assert all(isinstance(v, float) for v in logged[0].values()), "a vector reached log_dict"

    def test_the_vector_reaches_every_trackers_text_channel(self) -> None:
        tracker = _RecordingTracker()
        seg = self._segmenter(trackers=[tracker])
        seg.log_dict = lambda values, *a, **k: None  # type: ignore[method-assign]
        seg.val_metrics.update(torch.randn(2, NUM_CLASSES, SIDE, SIDE), _target_masks(2))
        seg._report_epoch_metrics(seg.val_metrics)

        assert [name for name, _, _ in tracker.texts] == ["val/iou_per_class"]
        table = tracker.texts[0][1]
        assert "class" in table and "iou_per_class" in table and f"{NUM_CLASSES} " not in table.splitlines()[0]

    def test_the_class_names_label_the_rows(self) -> None:
        """Unlike classification there is no label map to derive them from — a mask carries ids,
        not names — so they are a config knob and default to indices."""
        tracker = _RecordingTracker()
        seg = self._segmenter(trackers=[tracker], class_names=["pet", "background", "boundary"])
        seg.log_dict = lambda values, *a, **k: None  # type: ignore[method-assign]
        seg.val_metrics.update(torch.randn(2, NUM_CLASSES, SIDE, SIDE), _target_masks(2))
        seg._report_epoch_metrics(seg.val_metrics)

        assert "boundary" in tracker.texts[0][1]

    def test_no_figure_unless_plot_metrics_asks_for_one(self) -> None:
        """matplotlib is the `matrainer[plot]` extra and a figure is real work every eval epoch;
        the table costs nothing, so it is the default and the chart is opt-in."""
        tracker = _RecordingTracker()
        seg = self._segmenter(trackers=[tracker])
        seg.log_dict = lambda values, *a, **k: None  # type: ignore[method-assign]
        seg.val_metrics.update(torch.randn(2, NUM_CLASSES, SIDE, SIDE), _target_masks(2))
        seg._report_epoch_metrics(seg.val_metrics)
        assert tracker.figures == []

        plotting = self._segmenter(trackers=[tracker], plot_metrics=True)
        plotting.log_dict = lambda values, *a, **k: None  # type: ignore[method-assign]
        plotting.val_metrics.update(torch.randn(2, NUM_CLASSES, SIDE, SIDE), _target_masks(2))
        plotting._report_epoch_metrics(plotting.val_metrics)
        assert [name for name, _, _ in tracker.figures] == ["val/iou_per_class"]

    def test_a_run_with_only_scalar_metrics_reports_nothing_extra(self) -> None:
        """The selection is by SHAPE, so the default metric pair must produce no table at all."""
        tracker = _RecordingTracker()
        seg = _segmenter(trackers=[tracker])
        seg.initialize()
        seg.log_dict = lambda values, *a, **k: None  # type: ignore[method-assign]
        seg.val_metrics.update(torch.randn(2, NUM_CLASSES, SIDE, SIDE), _target_masks(2))
        seg._report_epoch_metrics(seg.val_metrics)
        assert tracker.texts == []


def test_persistent_workers_can_be_named_instead_of_derived() -> None:
    """The HOST-scoped knob: a config says `persistent_workers: false` and keeps its workers.

    macOS terminates persistent workers slowly enough that a short run spends longer stopping
    than training, which no derivation from `num_workers` can see."""
    runnable = LightningSegmenter(model=_TinyNet(), num_classes=NUM_CLASSES, num_workers=2, persistent_workers=False)
    for split in ("train_loader", "val_loader", "test_loader"):
        marker = getattr(runnable, split)
        assert marker.kwargs["persistent_workers"] is False, split
        assert marker.kwargs["num_workers"] == 2, split

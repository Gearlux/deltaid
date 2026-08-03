"""``KerasSegmenter`` — the first backend that is not torch, and what that actually costs.

Most of what is checked here is shared with every other backend (num_classes resolution, the
sentinels, the metric slots, the tracking contract) and is checked again because this backend
DUPLICATES that logic rather than inheriting it — a divergence between two copies is exactly what
goes unnoticed. The rest is this engine's alone, and all of it is about two facts:

* the image is channels-LAST, so a Keras config's preprocess chain stops before ``ToTensor``;
* ``from_logits=True`` is mandatory, because Keras's default silently normalizes raw logits by
  the row sum instead of raising.
"""

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pytest
from confluid import LazyClass, lazy_param_names, to_pydantic
from recordstream import Mask, collate_records

from deltaid._keras import keras
from deltaid.keras_segmenter import KerasSegmenter

NUM_CLASSES = 3
SIDE = 32


class _TinyNet(keras.Model):
    """A 1x1-conv 'segmenter': ``[N, H, W, 3] -> [N, H, W, C]`` — trainable, and instant."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        # `name=` explicitly: Keras derives one from the CLASS name otherwise, and on the
        # TensorFlow engine a leading underscore makes it an invalid root scope
        # (*"'__tiny_net_1' is not a valid root scope name"*). The torch engine does not care, so
        # without this the file passes on one engine and fails on the other.
        super().__init__(name="tiny_net")
        self.num_classes = num_classes
        self.head = keras.layers.Conv2D(num_classes, 1)

    def call(self, x: Any, training: bool = False) -> Any:
        return self.head(x)


def _records(n: int = 4, classes: int = NUM_CLASSES, learnable: bool = False) -> List[Dict[str, Any]]:
    """Synthetic records in the shape a CHANNELS-LAST config produces.

    Note the image is ``[H, W, 3]`` and not ``[3, H, W]``: that IS the difference between a Keras
    config's preprocess chain and a torch one, so building the fixture the other way would test
    a pipeline no config produces.
    """
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        image = rng.random((SIDE, SIDE, 3)).astype("float32")
        mask = image.argmax(axis=-1) if learnable else rng.integers(0, classes, (SIDE, SIDE))
        out.append({"image": image, "mask": Mask(mask.astype(np.int64)), "record_id": i})
    return out


def _segmenter(**kwargs: Any) -> KerasSegmenter:
    params: Dict[str, Any] = {
        "model": _TinyNet(),
        "num_classes": NUM_CLASSES,
        "max_epochs": 1,
        "batch_size": 2,
    }
    params.update(kwargs)
    return KerasSegmenter(**params)


def _default_miou(seg: KerasSegmenter) -> Any:
    """The live `MeanIoU` a resolved segmenter holds — the slot is `Optional[List[...]]`, so the
    narrowing is stated once here rather than at each of the three assertions below."""
    seg.initialize()
    assert seg.metrics is not None
    return next(m for m in seg.metrics if isinstance(m, keras.metrics.MeanIoU))


def _resolved(seg: KerasSegmenter) -> int:
    seg.initialize()
    assert seg.num_classes is not None
    return seg.num_classes


# --------------------------------------------------------------------------- #
# What is genuinely different about this engine
# --------------------------------------------------------------------------- #
class TestWhatTheEngineChanges:
    def test_the_batch_is_channels_last(self) -> None:
        """The one shape difference from the torch backends, and the reason a Keras config's
        preprocess chain stops after the joint resize."""
        x, y = _segmenter()._to_arrays(collate_records(_records(2)))
        assert x.shape == (2, SIDE, SIDE, 3)
        assert y.shape == (2, SIDE, SIDE), "a class-id map has no channel axis in ANY engine"

    def test_the_target_is_read_back_as_int64(self) -> None:
        """Named explicitly for the same reason the torch backends name it: albumentations
        returns an int32 mask, and the sparse losses want integer ids."""
        _, y = _segmenter()._to_arrays(collate_records(_records(2)))
        assert y.dtype == np.int64

    def test_predict_shaped_batches_carry_no_target(self) -> None:
        arrays = _segmenter()._to_arrays(collate_records(_records(2)), with_targets=False)
        assert len(arrays) == 1

    def test_the_default_loss_reads_raw_logits(self) -> None:
        """NOT optional: the head is a bare 1x1 conv, and Keras's default expects PROBABILITIES.
        Handed logits it normalizes by the row sum and clips rather than raising, so the loss
        stays finite and is merely wrong (measured: 4.6293 where the correct value is 1.4546)."""
        seg = _segmenter()
        seg.initialize()
        assert isinstance(seg.loss, keras.losses.SparseCategoricalCrossentropy)
        assert seg.loss.from_logits is True

    def test_the_default_iou_metric_accepts_logits(self) -> None:
        """`MeanIoU` defaults to `sparse_y_pred=True`, which expects an ALREADY-argmaxed
        prediction and raises a shape error on this head's output (measured)."""
        miou = _default_miou(_segmenter())
        assert miou.sparse_y_pred is False
        assert miou.num_classes == NUM_CLASSES

    def test_the_ignore_index_reaches_the_metric_as_keras_spells_it(self) -> None:
        assert _default_miou(_segmenter(ignore_index=2)).ignore_class == 2

    def test_the_default_sentinel_excludes_nothing(self) -> None:
        """`-100` matches the torch backends and excludes nothing, since a class-id mask never
        contains it — but Keras would treat the sentinel as a real class id, so it becomes None."""
        assert _default_miou(_segmenter()).ignore_class is None

    def test_the_split_metric_slots_are_UNIONED_not_narrowed(self) -> None:
        """`compile()` takes ONE metric list and reports it on both splits; there is no hook that
        narrows a metric to one split short of recompiling between phases, which would discard
        the optimizer state a fit is using.

        Widening is the honest failure: a metric appearing where it was not asked for is noise a
        reader ignores, where DROPPING it would be a wired metric that silently reports nothing.
        """
        extra = keras.metrics.SparseCategoricalAccuracy(name="only_val")
        seg = _segmenter(metrics=[keras.metrics.SparseCategoricalAccuracy(name="shared")], val_metrics=[extra])
        assert [m.name for m in seg._compiled_metrics()] == ["shared", "only_val"]  # type: ignore[union-attr]

    def test_the_iou_metric_on_the_torch_engine_is_WARNED_about_not_substituted(self) -> None:
        """`keras.ops.scatter` is a Python loop on the torch backend, and `MeanIoU` scatters once
        per PIXEL — 401,408 iterations for one 8x224x224 batch. Measured on that batch:
        tensorflow 0.009 s, jax 0.194 s, torch ~160 s, against a 0.37 s model step. So the run
        looks HUNG rather than slow.

        Warned rather than fixed by substituting a cheaper metric: the default must not change
        with the engine, or the project's one claim — that the same config gives the same run —
        stops being true. Classification never hit it because its metrics are per-IMAGE, so the
        same scatter loops over 8 indices instead of 401,408.
        """
        import deltaid.keras_segmenter as module

        seen: List[str] = []
        original_backend, original_warning = module.keras_backend, module.logger.warning
        module.keras_backend = lambda: "torch"  # type: ignore[assignment]
        module.logger.warning = lambda msg, *a, **k: seen.append(str(msg))  # type: ignore[assignment]
        try:
            module.KerasSegmenter._warn_if_the_engine_cannot_afford_these_metrics(
                [keras.metrics.MeanIoU(num_classes=3), keras.metrics.SparseCategoricalAccuracy()]
            )
        finally:
            module.keras_backend, module.logger.warning = original_backend, original_warning  # type: ignore[assignment]

        assert len(seen) == 1 and "MeanIoU" in seen[0]
        assert "KERAS_BACKEND=tensorflow" in seen[0], "a warning without the fix is just noise"

    def test_no_warning_when_the_engine_can_afford_them(self) -> None:
        """Both halves matter: another engine, or a metric that does not scatter per pixel."""
        import deltaid.keras_segmenter as module

        seen: List[str] = []
        original_backend, original_warning = module.keras_backend, module.logger.warning
        module.logger.warning = lambda msg, *a, **k: seen.append(str(msg))  # type: ignore[assignment]
        try:
            module.keras_backend = lambda: "tensorflow"  # type: ignore[assignment]
            module.KerasSegmenter._warn_if_the_engine_cannot_afford_these_metrics(
                [keras.metrics.MeanIoU(num_classes=3)]
            )
            module.keras_backend = lambda: "torch"  # type: ignore[assignment]
            module.KerasSegmenter._warn_if_the_engine_cannot_afford_these_metrics(
                [keras.metrics.SparseCategoricalAccuracy()]
            )
        finally:
            module.keras_backend, module.logger.warning = original_backend, original_warning  # type: ignore[assignment]

        assert seen == []

    def test_keras_STILL_ships_no_phase_scoped_metric_hook(self) -> None:
        """The REVISIT TRIGGER for the union above, not a restatement of it.

        The union is the engine's answer rather than a shortcut: verified against Keras 3.15.1,
        `compile` takes `metrics` and `weighted_metrics` and nothing phase-scoped
        (`weighted_metrics` is about sample WEIGHTS), the set lives in one `CompileMetrics`
        container, and `Model.metrics` is read-only. If a future Keras adds a public hook this
        FAILS — which is the point: the backlog item was closed on this evidence, so the evidence
        has to keep being checked.
        """
        import inspect

        params = set(inspect.signature(keras.Model.compile).parameters)
        assert "metrics" in params, "the check is against the real signature, so this is not vacuous"
        assert not (params - {"self", "optimizer", "loss", "loss_weights", "metrics", "weighted_metrics"}) & {
            "val_metrics",
            "validation_metrics",
            "eval_metrics",
            "test_metrics",
        }
        assert type(_TinyNet()).metrics.fset is None, "the metric set is not swappable through public API"

    def test_an_explicitly_wired_loss_is_never_replaced(self) -> None:
        chosen = keras.losses.CategoricalFocalCrossentropy()
        seg = _segmenter(loss=chosen)
        seg.initialize()
        assert seg.loss is chosen


# --------------------------------------------------------------------------- #
# The logic it duplicates from the reference backend — so it cannot drift
# --------------------------------------------------------------------------- #
class TestParityWithTheReferenceBackend:
    def test_num_classes_resolution_agrees(self) -> None:
        """Model head wins; explicit next; a `!lazy:` model's pinned kwarg next; the mask walk last."""
        assert _resolved(_segmenter(model=_TinyNet(num_classes=5), num_classes=9)) == 5
        assert _resolved(_segmenter(model=LazyClass(_TinyNet, num_classes=4), num_classes=None)) == 4
        assert _resolved(_segmenter(model=LazyClass(_TinyNet), num_classes=None, train_set=_records())) == 3

    def test_it_uses_the_shared_combination_rule(self) -> None:
        """One rule across every backend AND both tasks — never re-derived per backend."""
        assert "from matrainer.metric_slots import" in Path(inspect.getfile(KerasSegmenter)).read_text()

    def test_the_entrypoint_markers_are_the_dispatch_table(self) -> None:
        from recordstream.runnable import runnable_entrypoints

        assert set(runnable_entrypoints(KerasSegmenter)) == {"fit", "evaluate", "test", "predict"}
        assert "run_entrypoint(self, self.task)" in inspect.getsource(KerasSegmenter.run)

    def test_trackers_is_a_constructor_param_that_admits_a_list(self) -> None:
        assert "trackers" in to_pydantic(KerasSegmenter).model_fields

    def test_all_four_metric_slots_are_declared(self) -> None:
        fields = to_pydantic(KerasSegmenter).model_fields
        assert {"metrics", "train_metrics", "val_metrics", "test_metrics"} <= set(fields)

    def test_the_deferred_slots_are_declared_lazy(self) -> None:
        assert {"model", "train_set", "optimizer", "loss"} <= lazy_param_names(KerasSegmenter)

    def test_no_local_hyperparameter_derivation_survives(self) -> None:
        source = Path(inspect.getfile(KerasSegmenter)).read_text()
        assert "_config_params" not in source
        assert "runnable=self" in source


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class TestTheLoop:
    def test_fit_trains_and_reports_both_splits(self) -> None:
        seg = _segmenter(train_set=_records(8), model=LazyClass(_TinyNet))
        seg.val_set = _records(4)
        seg.run()
        assert {"loss", "miou", "val_loss", "val_miou"} <= set(seg.computed_metrics)

    def test_the_loss_actually_descends(self) -> None:
        """A finite loss is not evidence of training — a badly wired one stays finite too, which
        is precisely the `from_logits` failure mode. Learnable data, or the loss would sit at
        ln(C) however well the run was wired."""
        seg = _segmenter(train_set=_records(8, learnable=True), model=LazyClass(_TinyNet), max_epochs=12)
        seg.run()
        losses = seg.history["loss"]
        assert len(losses) == 12
        assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_test_writes_a_metrics_sidecar(self, tmp_path: Path) -> None:
        seg = _segmenter(task="test", test_set=_records(4))
        seg.metrics_sidecar_path = str(tmp_path / "metrics.json")
        seg.run()
        assert "miou" in json.loads((tmp_path / "metrics.json").read_text())["metrics"]

    def test_evaluate_and_test_are_the_same_class(self) -> None:
        seg = _segmenter(task="evaluate", test_set=_records(4))
        seg.run()
        assert "miou" in seg.computed_metrics

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
        assert closed == [True]

    def test_the_emitted_mask_is_a_plain_class_id_map(self) -> None:
        """The sink contract is recordstream's and framework-NEUTRAL — generic in its array type,
        so the same sink consumes this numpy output and the torch backends' tensors. Only the
        channel AXIS differs (last here), and the emitted mask is `[H, W]` either way."""
        written: List[Any] = []

        class _Sink:
            def write(self, prediction: Any, metadata: Any) -> None:
                written.append(prediction)

        seg = _segmenter(task="predict", test_set=_records(2))
        seg.predictions_sink = cast(Any, _Sink())
        seg.run()

        mask = written[0]["mask"]
        assert mask.shape == (SIDE, SIDE)
        assert mask.dtype == np.int64
        assert np.allclose(written[0]["probs"].sum(axis=-1), 1.0), "probs must be a softmax over the CHANNEL axis"

    def test_a_file_writing_callback_gets_its_directory_created(self, tmp_path: Path) -> None:
        """Keras opens the path without creating its parent, so a perfectly reasonable
        `./keras_logs/run.csv` dies with FileNotFoundError minutes into a fit."""
        target = tmp_path / "nested" / "history.csv"
        seg = _segmenter(
            train_set=_records(4),
            model=LazyClass(_TinyNet),
            callbacks=[keras.callbacks.CSVLogger(filename=str(target))],
        )
        seg.run()
        assert target.exists()

    def test_the_outputs_raise_before_a_run(self) -> None:
        seg = _segmenter(model=LazyClass(_TinyNet))
        with pytest.raises(RuntimeError, match="only available after run"):
            _ = seg.computed_metrics

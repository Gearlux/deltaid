# mypy: disable-error-code="attr-defined"
"""``TensorFlowSegmenter`` — the native ``tf.data`` + ``GradientTape`` backend.

Two groups, split by what they need:

* **Source-only** — the import ordering and the shared contract. These read the module as TEXT
  and run EVERYWHERE, including on an install with no TensorFlow at all. That is the point: the
  ordering they guard decides which engine keras binds, and getting it wrong produces a symptom
  (``tf.GradientTape`` failing on a torch-backed model) very far from its cause — so a guard that
  ran only where TensorFlow is installed would not be watching the machines that can break it.
* **Behavioural** — everything that imports the module, skipped unless keras is on the TensorFlow
  engine in THIS process.

That condition is not a hedge, it is Keras's own rule: **Keras 3 binds exactly one engine per
process, at first import.** A full-suite run imports ``tests/test_keras_segmenter.py`` first
(alphabetically), which loads keras on torch — so this loop genuinely cannot run in the same
process, and a marker pretending otherwise would report coverage that never executed. Run them as
their own invocation::

    KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_segmenter.py

Running this file alone needs no environment variable: it imports ``deltaid._tensorflow`` before
anything else can claim the engine.
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parent.parent / "deltaid" / "tensorflow_segmenter.py"

# Claim the engine before any other import can — this is what lets `pytest
# tests/test_tensorflow_segmenter.py` work with no environment variable set. Guarded on keras
# being unimported, because if something already bound it this would be a no-op that merely costs
# a TensorFlow load.
if importlib.util.find_spec("tensorflow") is not None and "keras" not in sys.modules:
    import deltaid._tensorflow  # noqa: F401


def _keras_is_on_tensorflow() -> bool:
    """Whether keras is bound to the TensorFlow engine in this process (never importing it)."""
    keras = sys.modules.get("keras")
    return keras is not None and keras.backend.backend() == "tensorflow"


#: Applied per-test rather than to the module, so the source-only group still runs where
#: TensorFlow is missing — see the module docstring for why that matters.
requires_tensorflow = pytest.mark.skipif(
    not _keras_is_on_tensorflow(),
    reason="needs keras on the TensorFlow engine: KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_segmenter.py",
)

NUM_CLASSES = 3
SIDE = 32


# --------------------------------------------------------------------------- #
# Source-only — runs without TensorFlow too
# --------------------------------------------------------------------------- #
def test_keras_is_reached_through_the_module_that_pins_the_engine() -> None:
    """``from deltaid._tensorflow import …``, NEVER ``from recordstream.keras import …``.

    ``recordstream.keras`` defaults ``KERAS_BACKEND`` to the first engine INSTALLED (torch ->
    tensorflow -> jax), so on this workspace's own venv it picks torch. Importing anything from
    there first would beat the ``setdefault`` in ``deltaid/_tensorflow.py`` into place — and
    isort would sort that ``recordstream`` line into the third-party group, ABOVE every deltaid
    line, so the mistake is one an autoformatter makes for you.
    """
    source = SOURCE.read_text()
    assert "from deltaid._tensorflow import" in source
    # AST rather than a substring scan: `from matrainer.keras import keras_tracking_callback` is
    # legitimate and CONTAINS "import keras", so a text check would fail on correct code.
    roots = {(node.module or "") for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)}
    roots |= {a.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import) for a in node.names}
    assert "recordstream.keras" not in roots
    assert "keras" not in roots and "tensorflow" not in roots


def test_the_environment_default_precedes_every_import_that_reads_it() -> None:
    """Keras reads ``KERAS_BACKEND`` ONCE, at import time, so the order in `_tensorflow.py` is
    load-bearing rather than stylistic."""
    text = (SOURCE.parent / "_tensorflow.py").read_text()
    setdefault = text.index('os.environ.setdefault("KERAS_BACKEND"')
    assert setdefault < text.index("import tensorflow as tf")
    assert setdefault < text.index("from recordstream.keras import")


def test_it_is_a_sibling_of_the_keras_backend_not_a_subclass() -> None:
    """It NARROWS a precondition its would-be parent does not have — ``_require_tensorflow_backend``
    raises where a ``KerasSegmenter`` runs fine — so a picker offering "any KerasSegmenter" would
    hand out something that cannot run here."""
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "TensorFlowSegmenter")
    bases = [ast.unparse(b) for b in cls.bases]
    assert "KerasSegmenter" not in bases
    # The narrowed precondition (the engine guard) lives on the machinery base since P5.
    import inspect

    from matrainer.tensorflow.loop import TensorFlowRunnable

    assert "TensorFlowRunnable" in SOURCE.read_text()
    assert "_require_tensorflow_backend" in inspect.getsource(TensorFlowRunnable.run)


def test_there_is_no_second_loss_module_for_this_engine() -> None:
    """TF >= 2.16's ``tf.keras`` IS Keras 3, so a ``tensorflow_segmentation.py`` would register
    the same classes under a second tag and make every bare ``!class:SparseCategoricalCrossentropy``
    ambiguous for no gain."""
    from deltaid.cli import FRAMEWORKS

    assert not (SOURCE.parent / "tensorflow_segmentation.py").exists()
    assert FRAMEWORKS["tensorflow"].registrations == FRAMEWORKS["keras"].registrations


def test_it_warms_its_sources_and_derives_no_hyperparameters_of_its_own() -> None:
    """Both live on the shared bases since P5 — the module must not shadow the wiring."""
    source = SOURCE.read_text()
    assert "SegmentationKerasHooks" in source and "TensorFlowRunnable" in source
    assert "def initialize" not in source and "def _callbacks" not in source
    assert "_config_params" not in source


# --------------------------------------------------------------------------- #
# Behavioural — needs keras bound to TensorFlow in THIS process
# --------------------------------------------------------------------------- #
def _tiny_net(num_classes: int = NUM_CLASSES) -> Any:
    """A 1x1-conv 'segmenter' as a raw ``tf.Module`` — what ``TensorFlowUNet`` is, in miniature."""
    from deltaid._tensorflow import tf

    class _TinyNet(tf.Module):
        def __init__(self) -> None:
            super().__init__(name="tiny")
            self.num_classes = num_classes
            self.kernel = tf.Variable(
                tf.random.stateless_uniform((1, 1, 3, num_classes), seed=(0, 0), minval=-0.5, maxval=0.5),
                name="kernel",
            )
            self.bias = tf.Variable(tf.zeros((num_classes,)), name="bias")

        def __call__(self, x: Any, training: bool = False) -> Any:
            return tf.nn.conv2d(x, self.kernel, strides=1, padding="SAME") + self.bias

    return _TinyNet()


def _records(n: int = 4, classes: int = NUM_CLASSES, learnable: bool = False) -> List[Dict[str, Any]]:
    """Synthetic records in the shape a CHANNELS-LAST config produces."""
    from recordstream import Mask

    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        image = rng.random((SIDE, SIDE, 3)).astype("float32")
        mask = image.argmax(axis=-1) if learnable else rng.integers(0, classes, (SIDE, SIDE))
        out.append({"image": image, "mask": Mask(mask.astype(np.int64)), "record_id": i})
    return out


def _segmenter(**kwargs: Any) -> Any:
    from deltaid.tensorflow_segmenter import TensorFlowSegmenter

    params: Dict[str, Any] = {
        "model": _tiny_net(),
        "num_classes": NUM_CLASSES,
        "max_epochs": 1,
        "batch_size": 2,
    }
    params.update(kwargs)
    return TensorFlowSegmenter(**params)


@requires_tensorflow
class TestWhatTheEngineChanges:
    def test_the_wrong_engine_is_refused_with_the_fix_in_the_message(self) -> None:
        """``tf.GradientTape`` cannot differentiate a torch-backed ``keras.Model``, and the
        failure deep inside the tape says nothing useful."""
        import deltaid.tensorflow_segmenter as module

        original = module.keras_backend
        module.keras_backend = lambda: "torch"  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match="KERAS_BACKEND"):
                module.TensorFlowSegmenter._require_tensorflow_backend()
        finally:
            module.keras_backend = original  # type: ignore[assignment]

    def test_each_split_gets_its_OWN_metric_objects(self) -> None:
        """A Keras metric is STATEFUL, so sharing one instance across splits would fold the
        validation pass's counts into the training numbers. This loop can honour a per-split set
        where the Keras backend cannot, because each traced step closes over its own list."""
        seg = _segmenter()
        seg.initialize()
        train, val = seg._metric_objects["train"], seg._metric_objects["val"]
        assert [m.name for m in train] == [m.name for m in val]
        assert all(a is not b for a, b in zip(train, val)), "the splits share metric instances"

    def test_a_split_slot_adds_to_the_shared_set_for_that_split_only(self) -> None:
        """The MERGE rule — this backend narrows where the Keras one can only widen."""
        from deltaid._tensorflow import keras

        seg = _segmenter(val_metrics=[keras.metrics.SparseCategoricalAccuracy(name="only_val")])
        seg.initialize()
        assert "only_val" in [m.name for m in seg._metric_objects["val"]]
        assert "only_val" not in [m.name for m in seg._metric_objects["train"]]

    def test_a_wired_metric_keeps_its_knobs_through_the_per_split_clone(self) -> None:
        """The clone goes through Keras's own `get_config()`/`from_config()`, so a metric wired
        with `sparse_y_pred: false` does not silently come back with the default."""
        from deltaid._tensorflow import keras

        wired = keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False, name="wired_iou")
        seg = _segmenter(metrics=[wired])
        seg.initialize()
        clone = next(m for m in seg._metric_objects["train"] if m.name == "wired_iou")
        assert clone is not wired
        assert clone.sparse_y_pred is False and clone.num_classes == NUM_CLASSES

    def test_the_defaults_match_the_keras_backend_exactly(self) -> None:
        """A config moved between the two must not silently change its loss."""
        from deltaid._tensorflow import keras

        seg = _segmenter()
        seg.initialize()
        assert isinstance(seg._loss_fn, keras.losses.SparseCategoricalCrossentropy)
        assert seg._loss_fn.from_logits is True
        mean_iou = next(m for m in seg._metric_objects["train"] if m.name == "mean_iou")
        assert mean_iou.sparse_y_pred is False

    def test_the_batch_is_channels_last(self) -> None:
        from recordstream import collate_records

        x, y = _segmenter()._to_arrays(collate_records(_records(2)))
        assert x.shape == (2, SIDE, SIDE, 3)
        assert y.shape == (2, SIDE, SIDE)
        assert y.dtype == np.int64


@requires_tensorflow
class TestTheLoop:
    def test_fit_trains_and_reports_both_splits(self) -> None:
        seg = _segmenter(train_set=_records(8))
        seg.val_set = _records(4)
        seg.run()
        # What LEAVES the run is canonical; `_history` keeps the loop's Keras-shaped columns.
        assert {"train/loss", "train/mean_iou", "val/loss", "val/mean_iou"} <= set(seg.computed_metrics)
        assert {"loss", "mean_iou", "val_loss", "val_mean_iou"} <= set(seg._history)

    def test_the_loss_actually_descends(self) -> None:
        """A finite loss is not evidence of training — a badly wired one stays finite too."""
        seg = _segmenter(train_set=_records(8, learnable=True), max_epochs=10)
        seg.run()
        losses = seg.history["loss"]
        assert len(losses) == 10
        assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_a_short_final_batch_does_not_retrace_into_a_variable_creation(self) -> None:
        """The optimizer's slot variables are built OUTSIDE the graph on purpose: Keras builds
        them on the first `apply_gradients`, and creating variables during tracing is legal once
        and an ERROR on any retrace — which a short final batch is enough to trigger."""
        seg = _segmenter(train_set=_records(7), batch_size=2)  # 3 full batches + a 1-row tail
        seg.run()
        assert "train/loss" in seg.computed_metrics

    def test_test_writes_a_metrics_sidecar(self, tmp_path: Path) -> None:
        seg = _segmenter(task="test", test_set=_records(4))
        seg.metrics_sidecar_path = str(tmp_path / "metrics.json")
        seg.run()
        assert "test/mean_iou" in json.loads((tmp_path / "metrics.json").read_text())["metrics"]

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
        assert written[0][0]["mask"].shape == (SIDE, SIDE)
        assert closed == [True]


@requires_tensorflow
class TestTheSavedModel:
    """The one run artifact that is TensorFlow's and not Keras's."""

    def test_a_raw_tf_module_is_exported_with_an_explicitly_traced_signature(self, tmp_path: Path) -> None:
        """A ``tf.Module`` has no ``export``, and a bare ``__call__`` is untraced Python that
        cannot be serialized — so this branch traces the signature itself."""
        from deltaid._tensorflow import tf

        target = tmp_path / "saved"
        seg = _segmenter(train_set=_records(4), export_dir=str(target))
        seg.run()

        assert (target / "saved_model.pb").exists()
        loaded = tf.saved_model.load(str(target))
        out = loaded.signatures["serving_default"](tf.zeros((1, SIDE, SIDE, 3)))
        assert list(out.values())[0].shape == (1, SIDE, SIDE, NUM_CLASSES)

    def test_no_export_dir_writes_nothing(self, tmp_path: Path) -> None:
        seg = _segmenter(train_set=_records(4))
        seg.run()
        assert list(tmp_path.iterdir()) == []

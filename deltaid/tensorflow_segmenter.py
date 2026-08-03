"""``TensorFlowSegmenter`` — the same segmentation run on a native ``tf.data`` + ``GradientTape`` loop.

The fifth backend, and the one that answers a fair question: *does this need its own runnable,
when ``KERAS_BACKEND=tensorflow deltaid train --framework keras`` already runs the identical
config?* Yes — but ONLY for what Keras 3 does not expose:

* ``_dataset`` — a ``tf.data.Dataset`` over recordstream batches, with ``prefetch(AUTOTUNE)``;
* ``_train_step_fn`` / ``_eval_step_fn`` / ``_predict_step_fn`` — ``tf.GradientTape`` under
  ``@tf.function``, which is what makes ``jit_compile`` (XLA) a knob at all;
* ``_run_epoch`` — the loop `fit()` would have run;
* ``_export`` — the SavedModel, the one artifact that is TensorFlow's and not Keras's.

It is a SIBLING of :class:`~deltaid.keras_segmenter.KerasSegmenter`, not a subclass. The
duplication is deliberate and temporary (an engine-neutral base is deferred until the detection
project has been through the same treatment), and it is preferable to inheritance here for a
reason that is a property of the code: this class **narrows a precondition** its would-be parent
does not have — ``_require_tensorflow_backend()`` raises where a ``KerasSegmenter`` runs fine — so
a picker offering "any KerasSegmenter" would hand out something that cannot run.

It still declares ``framework="keras"``, NOT ``"tensorflow"``: the axis names the API a class is
WIRED TO, and everything in its slots is a ``keras.losses`` / ``keras.metrics`` / ``keras.Model``.

**There is deliberately no `tensorflow_segmentation.py`** — TF >= 2.16's ``tf.keras`` IS Keras 3,
so a second loss module would register the same classes under a second tag and make every bare
``!class:SparseCategoricalCrossentropy`` ambiguous. ``FRAMEWORKS["tensorflow"]`` names
:mod:`deltaid.keras_segmentation` on purpose.

Run it with ``deltaid train cfg.yaml --framework tensorflow``.
"""

from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_slots import merge_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, as_tracker_list
from recordstream import (
    PredictionsSink,
    RecordSource,
    SegmentationOutput,
    batch_metadata,
    batch_values,
    ensure_materialized,
    ensure_record_dataset,
    num_mask_classes,
)
from recordstream.runnable import ProgressReporting, entrypoint, run_entrypoint

# Everything TF- and Keras-shaped comes from ONE module, which pins KERAS_BACKEND=tensorflow
# before importing keras. Importing `RecordSequence` from `recordstream.keras` directly would let
# isort sort that line above every deltaid line — and that module carries the competing default.
from deltaid._tensorflow import RecordSequence, keras, keras_backend, tf

logger = get_logger(__name__)

#: Which entry point :meth:`TensorFlowSegmenter.run` dispatches to.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]

#: What each compile-shaped slot admits — see the Keras backend for why these are spelled out.
KerasLoss = Union[str, keras.losses.Loss]
KerasMetric = Union[str, keras.metrics.Metric]
KerasOptimizer = Union[str, keras.optimizers.Optimizer]

#: The model slot takes two kinds of thing, and they share no useful ancestor: a ``keras.Model``
#: descends ``Layer -> KerasAutoTrackable``, so it is NOT a ``tf.Module``. Spelled as a union
#: rather than `Any` so `to_pydantic` still has something to build a widget from.
TensorFlowNetwork = Union[keras.Model, tf.Module]

__all__ = ["KerasLoss", "KerasMetric", "KerasOptimizer", "SegmenterTask", "TensorFlowNetwork", "TensorFlowSegmenter"]


@configurable(task="segmentation", role="trainer", framework="keras")
class TensorFlowSegmenter(ProgressReporting):
    """Train, evaluate or predict a semantic segmenter with a native TensorFlow loop.

    Args:
        model: The segmentation model — a ``keras.Model`` or a raw ``tf.Module``. Wire ``!lazy:``
            so ``initialize()`` can inject the resolved ``num_classes``.
        train_set: Training source (``task="fit"``).
        test_set: Evaluation / prediction source.
        num_classes: Class count, background included. Omitted, it is derived from the model or
            the mask column.
        ignore_index: Pixel value excluded from the default metrics (Keras's ``ignore_class``).
        task: Which entry point ``run()`` dispatches to.
        max_epochs / batch_size: the shared knobs.
        jit_compile: Trace the steps with XLA. Off by default — it costs a slow first step and
            makes a traceback harder to read, which is the wrong trade for a smoke run.
        prefetch: Overlap input preparation with the step (``tf.data.AUTOTUNE``).
        export_dir: Where to write the SavedModel after a fit. ``None`` writes none.
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring.
        callbacks: ``keras.callbacks.Callback`` instances, driven by hand through a ``CallbackList``.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the per-pixel mask.
        optimizer / loss: Keras objects, or their string aliases.
        metrics: Shared metrics; ``train_metrics`` / ``val_metrics`` / ``test_metrics`` are that
            split's OWN, merged over ``metrics``. Unlike the Keras backend this loop CAN narrow
            per split — each split gets its own metric OBJECTS, because they are stateful.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[TensorFlowNetwork]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        task: SegmenterTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        jit_compile: bool = False,
        prefetch: bool = True,
        export_dir: Optional[str] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy[keras.callbacks.Callback]]] = None,
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "mask",
        optimizer: Lazy[KerasOptimizer] = "adam",
        loss: Optional[Lazy[KerasLoss]] = None,
        metrics: Optional[List[Lazy[KerasMetric]]] = None,
        train_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        val_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        test_metrics: Optional[List[Lazy[KerasMetric]]] = None,
    ) -> None:
        super().__init__()
        self.model: TensorFlowNetwork = cast(TensorFlowNetwork, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: SegmenterTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.jit_compile = jit_compile
        self.prefetch = prefetch
        self.export_dir = export_dir
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.optimizer: Lazy[KerasOptimizer] = optimizer
        self.loss: Optional[Lazy[KerasLoss]] = loss
        self.metrics: Optional[List[Lazy[KerasMetric]]] = metrics
        self.train_metrics: Optional[List[Lazy[KerasMetric]]] = train_metrics
        self.val_metrics: Optional[List[Lazy[KerasMetric]]] = val_metrics
        self.test_metrics: Optional[List[Lazy[KerasMetric]]] = test_metrics
        self.callbacks: Optional[List[Lazy[keras.callbacks.Callback]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # The live objects `compile()` would have owned. Each SPLIT gets its own metric
        # instances: a metric object is STATEFUL, so splits with differing sets cannot share one.
        self._optimizer: Any = None
        self._loss_fn: Any = None
        self._metric_objects: Dict[str, List[Any]] = {"train": [], "val": [], "test": []}
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Check the engine, resolve ``num_classes``, then build the model and the live objects."""
        self._require_tensorflow_backend()
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        self._build_live_objects()

    @staticmethod
    def _require_tensorflow_backend() -> None:
        """Fail loudly when keras loaded some OTHER engine — the one way this backend goes wrong.

        ``tf.GradientTape`` cannot differentiate a torch-backed ``keras.Model``, and the failure
        deep inside the tape says nothing useful. This is not hypothetical: ``recordstream.keras``
        defaults ``KERAS_BACKEND`` to the first backend INSTALLED, in the order torch ->
        tensorflow -> jax, so any machine with both ``deltaid[lightning]`` and
        ``deltaid[tensorflow]`` lands on torch unless something set the variable first
        (:mod:`deltaid._tensorflow` does).

        This check is also why the class is not a ``KerasSegmenter`` subclass: it NARROWS a
        precondition its would-be parent does not have.
        """
        backend = keras_backend()
        if backend != "tensorflow":
            raise RuntimeError(
                f"TensorFlowSegmenter needs Keras on the TensorFlow engine, but keras loaded the "
                f"{backend!r} one. Keras reads KERAS_BACKEND ONCE, at import time. Export "
                f"KERAS_BACKEND=tensorflow before running, or use the `keras` backend — it runs "
                f"the identical config on whichever engine is loaded."
            )

    def _initialize_fit(self) -> None:
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)

        nc = self._num_classes_without_data()
        if nc is None:
            if self.train_set is None:
                raise RuntimeError(
                    "could not determine num_classes — set it on the segmenter or the model, or wire a train_set."
                )
            logger.info(f"Deriving num_classes from the {self.target_key!r} column (a full pass over the masks)...")
            nc = num_mask_classes(self.train_set, self.target_key)
            logger.info(f"Derived num_classes={nc} from the mask column.")
        self.num_classes = int(nc)

    def _initialize_eval(self) -> None:
        """Model-driven setup: the trained head's width is authoritative, not the eval split's."""
        self.model = flow(self.model)
        nc = getattr(self.model, "num_classes", None) or self.num_classes
        if not nc:
            raise RuntimeError("could not determine num_classes from the model (it must expose `num_classes`).")
        self.num_classes = int(nc)
        self.test_set = self._prepare(self.test_set)
        self.val_set = self._prepare(self.val_set)

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process (the fork guard)."""
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _num_classes_without_data(self) -> Optional[int]:
        """The class count from the model or the config — everything short of walking the masks."""
        if not isinstance(self.model, Fluid):
            model_nc = getattr(self.model, "num_classes", None)
            if model_nc:
                if self.num_classes is not None and int(self.num_classes) != int(model_nc):
                    logger.warning(
                        f"Segmenter num_classes={self.num_classes} != model num_classes={model_nc}; using model's."
                    )
                return int(model_nc)
        if self.num_classes is not None:
            return int(self.num_classes)
        if isinstance(self.model, Fluid):
            pinned = self.model.kwargs.get("num_classes")
            if pinned:
                return int(pinned)
        return None

    def _resolve_task_defaults(self) -> None:
        """Fill the unset (``None``) task-shaped slots. Idempotent; never overwrites.

        Same choices as the Keras backend, and that is the point: a config moved between the two
        must not silently change its loss. ``from_logits=True`` because the head emits raw logits,
        and ``sparse_y_pred=False`` because ``MeanIoU`` otherwise expects an already-argmaxed
        prediction and raises on raw logits.
        """
        if self.loss is None:
            self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if self.metrics is None and self.num_classes:
            # `ignore_class` is Keras's `ignore_index`. `None` when nothing is excluded, because
            # Keras treats the sentinel as a real class id rather than as "no exclusion".
            ignore = self.ignore_index if self.ignore_index >= 0 else None
            self.metrics = [
                keras.metrics.MeanIoU(
                    num_classes=self.num_classes, sparse_y_pred=False, ignore_class=ignore, name="miou"
                ),
                keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy"),
            ]

    def _build_live_objects(self) -> None:
        """Build the model at the resolved width, then the optimizer, loss and per-split metrics."""
        self._resolve_task_defaults()
        self.model = flow(self.model, num_classes=self.num_classes)
        self._optimizer = _as_optimizer(self.optimizer)
        self._loss_fn = flow(self.loss)
        for split in ("train", "val", "test"):
            self._metric_objects[split] = self._split_metrics(split)

    def _split_metrics(self, split: str) -> List[Any]:
        """One split's live metric objects — its own spec merged over the shared one.

        The combination rule is :func:`~matrainer.metric_slots.merge_metric_specs`, shared with
        every other backend. Unlike the Keras backend this loop CAN honour a per-split set,
        because each traced step closes over its own list.
        """
        spec = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or []
        return [_as_metric(m) for m in spec]

    # ---- the batch boundary -------------------------------------------------- #
    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        """recordstream's batching adapter, wearing THIS task's batch shape."""
        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=partial(self._to_arrays, with_targets=with_targets),
        )

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy ``(x, y)`` the traced step consumes.

        Channels LAST (``[N, H, W, C]``) like every TensorFlow model, and the mask ``[N, H, W]``
        int64 — the same shape every backend's target has, since a class-id map has no channel
        axis.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int64")

    def _dataset(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> Tuple[Any, Any]:
        """A ``tf.data.Dataset`` over recordstream batches, plus the sequence that feeds it.

        The sequence comes back too because the caller owns one thing ``fit()`` would have done
        for it: :meth:`RecordSequence.on_epoch_end`, the per-epoch reshuffle. Keras calls that
        itself; a hand-written loop that forgets it trains every epoch on the same row order.

        ``output_signature`` is derived from a real first batch rather than guessed — the batch
        shape is :meth:`_to_arrays`'s, and hard-coding it here would put the same fact twice.
        """
        sequence = self._sequence(source, shuffle=shuffle, with_targets=with_targets)
        dataset = tf.data.Dataset.from_generator(
            lambda: (sequence[i] for i in range(len(sequence))),
            output_signature=_output_signature(sequence[0]),
        )
        return (dataset.prefetch(tf.data.AUTOTUNE) if self.prefetch else dataset), sequence

    # ---- the graph steps ----------------------------------------------------- #
    def _train_step_fn(self) -> Any:
        """The forward/backward pair, traced into a graph.

        Built once and reused across epochs: a fresh ``@tf.function`` each epoch would retrace
        the whole graph every time, which is slower than the eager loop it replaced.
        """
        network, optimizer, loss_fn = self.model, self._optimizer, self._loss_fn
        metrics = self._metric_objects["train"]

        @tf.function(jit_compile=True if self.jit_compile else None)
        def train_step(x: Any, y: Any) -> Any:
            with tf.GradientTape() as tape:
                logits = network(x, training=True)
                loss = loss_fn(y, logits)
            variables = network.trainable_variables
            optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
            for metric in metrics:
                metric.update_state(y, logits)
            return loss

        return train_step

    def _eval_step_fn(self, split: str = "val") -> Any:
        """The forward pass plus metric update — no tape, and ``training=False``.

        ``split`` selects which metric OBJECTS the traced step updates, so a metric wired to
        ``val_metrics`` is not also computed on the test pass.
        """
        network, loss_fn = self.model, self._loss_fn
        metrics = self._metric_objects[split]

        @tf.function(jit_compile=True if self.jit_compile else None)
        def eval_step(x: Any, y: Any) -> Any:
            logits = network(x, training=False)
            for metric in metrics:
                metric.update_state(y, logits)
            return loss_fn(y, logits)

        return eval_step

    def _predict_step_fn(self) -> Any:
        """Logits only — the graph half of :meth:`predict`."""
        network = self.model

        @tf.function(jit_compile=True if self.jit_compile else None)
        def predict_step(x: Any) -> Any:
            return network(x, training=False)

        return predict_step

    # ---- metric bookkeeping (what `compile()` would have owned) -------------- #
    def _reset_metrics(self, split: str) -> None:
        for metric in self._metric_objects[split]:
            metric.reset_state()

    def _metric_results(self, split: str) -> Dict[str, float]:
        return {metric.name: float(metric.result()) for metric in self._metric_objects[split]}

    def _run_epoch(self, dataset: Any, step: Any, callbacks: Any = None, split: str = "train") -> Dict[str, float]:
        """One pass over ``dataset``, returning ``{"loss": …, **metrics}``."""
        self._reset_metrics(split)
        total, batches = 0.0, 0
        for index, (x, y) in enumerate(dataset):
            total += float(step(x, y))
            batches += 1
            if callbacks is not None:
                callbacks.on_train_batch_end(index, {"loss": total / batches})
        return {"loss": total / batches if batches else float("nan"), **self._metric_results(split)}

    # ---- callbacks + tracking ------------------------------------------------ #
    def _callbacks(self) -> List[Any]:
        """Wired Keras callbacks plus OPTIONAL matrainer tracking (a few lines; removable)."""
        callbacks = [flow(cb) for cb in (self.callbacks or [])]
        for callback in callbacks:
            target = getattr(callback, "filename", None) or getattr(callback, "filepath", None)
            if isinstance(target, (str, Path)):
                Path(target).expanduser().parent.mkdir(parents=True, exist_ok=True)
        trackers = self._live_trackers()
        if trackers:
            from matrainer.keras import keras_tracking_callback

            callbacks.append(keras_tracking_callback(trackers, runnable=self))
        return callbacks

    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot."""
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        """The custom loop: epochs, ``tf.GradientTape`` steps, callbacks, then the SavedModel."""
        logger.info(f"deltaid tensorflow training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("TensorFlowSegmenter.fit() requires a 'train_set'.")
        train_data, sequence = self._dataset(self.train_set, shuffle=True)
        val_data = self._dataset(self.val_set)[0] if self.val_set is not None else None

        # Build the optimizer's slot variables OUTSIDE the graph. Keras builds them on the first
        # `apply_gradients`, and doing that inside a `tf.function` creates variables during
        # tracing — legal on the first trace, an error on any retrace (a short final batch is
        # enough to trigger one).
        self._optimizer.build(self.model.trainable_variables)
        train_step, eval_step = self._train_step_fn(), self._eval_step_fn()

        callbacks = keras.callbacks.CallbackList(self._callbacks(), add_history=True, model=self.model)
        callbacks.set_params({"epochs": self.max_epochs, "steps": len(sequence), "verbose": 0})
        callbacks.on_train_begin()

        history: Dict[str, List[float]] = {}
        logs: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            callbacks.on_epoch_begin(epoch)
            logs = self._run_epoch(train_data, train_step, callbacks)
            if val_data is not None:
                logs.update({f"val_{k}": v for k, v in self._run_epoch(val_data, eval_step, split="val").items()})
            callbacks.on_epoch_end(epoch, logs)
            # Keras's `fit()` calls this between epochs; a hand-written loop has to.
            sequence.on_epoch_end()
            for key, value in logs.items():
                history.setdefault(key, []).append(value)
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
        callbacks.on_train_end(logs)

        self._history = {k: list(v) for k, v in history.items()}
        self._computed_metrics = {k: v[-1] for k, v in history.items()}
        logger.info("Training complete.")
        # Both write LOCAL files and touch no tracker, so they are safe after the run has ended.
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        self._export()

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        """Score the held-out split with the same traced step the training loop validates with."""
        if self.test_set is None:
            raise ValueError("TensorFlowSegmenter.test() requires a 'test_set'.")
        dataset, _ = self._dataset(self.test_set)
        self._computed_metrics = self._run_epoch(dataset, self._eval_step_fn("test"), split="test")
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each batch with its metadata.

        Walks ``sequence.batches()`` — the COLLATED record batches — rather than the ``tf.data``
        pipeline, because the sink needs the metadata that lives beside the image and
        ``_to_arrays`` has already dropped it. One walk, so output and metadata cannot desync.
        """
        if self.test_set is None:
            raise ValueError("TensorFlowSegmenter.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        predict_step = self._predict_step_fn()
        sequence = self._sequence(self.test_set, with_targets=False)
        for batch in sequence.batches():
            x = self._to_arrays(batch, with_targets=False)[0]
            logits = np.asarray(predict_step(tf.convert_to_tensor(x)))
            if sink is not None:
                self._write_predictions(logits, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, logits: np.ndarray, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.SegmentationOutput` per record, paired with its metadata."""
        probs = _softmax(logits)
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        for i in range(len(logits)):
            out: SegmentationOutput[np.ndarray] = {
                "logits": logits[i],
                "probs": probs[i],
                "mask": np.argmax(logits[i], axis=-1).astype("int64"),
            }
            meta = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            sink.write(out, meta)

    # ---- the SavedModel ------------------------------------------------------ #
    def _export(self) -> None:
        """Write the SavedModel — the one run artifact that is TensorFlow's and not Keras's.

        Two writers, because the model slot takes two kinds of thing. A ``keras.Model`` has
        ``export``, which builds the serving signature for you. A raw ``tf.Module`` — which is
        what :class:`deltaid.unet.tensorflow_unet.TensorFlowUNet` is — does NOT, so it goes
        through ``tf.saved_model.save`` with an explicitly traced signature: a bare
        ``tf.Module.__call__`` is untraced Python and cannot be serialized. Branching on the
        CAPABILITY rather than on the class keeps anything else callable working too.
        """
        if not self.export_dir:
            return
        path = Path(self.export_dir).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "export"):
            self.model.export(str(path))
        else:
            channels = 3
            signature = tf.function(
                lambda x: self.model(x, training=False),
                input_signature=[tf.TensorSpec(shape=(None, None, None, channels), dtype=tf.float32)],
            )
            tf.saved_model.save(self.model, str(path), signatures={"serving_default": signature})
        logger.info(f"Exported SavedModel to {path}")

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> Any:
        """The live trained model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run() — the model is still deferred.")
        return self.model

    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if not self._computed_metrics:
            raise RuntimeError("computed_metrics is only available after run().")
        return dict(self._computed_metrics)

    @property
    @output
    def history(self) -> Dict[str, List[float]]:
        """Per-epoch metric history — what a ``fit()``'s ``History`` would have accumulated."""
        return {key: list(values) for key, values in self._history.items()}


def _output_signature(batch: Any) -> Tuple[Any, ...]:
    """``tf.TensorSpec``s matching a real batch, with the row count left dynamic.

    ``from_generator`` needs the element spec up front. Deriving it from batch zero keeps the
    batch SHAPE defined in exactly one place — :meth:`TensorFlowSegmenter._to_arrays` — instead of
    restating "float32 [N, H, W, C] and int64 [N, H, W]" here where it could drift.
    """
    arrays = [np.asarray(item) for item in batch]
    return tuple(tf.TensorSpec(shape=(None, *a.shape[1:]), dtype=tf.as_dtype(a.dtype)) for a in arrays)


def _as_optimizer(value: Any) -> Any:
    """A live ``keras.optimizers.Optimizer`` from an instance, a class, or a string alias.

    The bare-CLASS branch is tested BEFORE ``flow()`` rather than after, which is the one place
    here the order matters: ``flow()`` INSTANTIATES a bare type on the way past, and for a
    confluid-registered class it does so through ``materialize()``, where sibling config keys
    broadcast into the constructor.
    """
    if isinstance(value, type):
        return value()
    built = flow(value)
    if isinstance(built, str):
        return keras.optimizers.get(built)
    return built


def _as_metric(value: Any) -> Any:
    """A FRESH ``keras.metrics.Metric`` from an instance, a class, a string alias or a marker.

    Fresh is the whole contract, and it is why this exists rather than a `flow()` call. A Keras
    metric is an ACCUMULATOR: it holds `tf.Variable` state across `update_state` calls, so handing
    the same object to the training and validation steps would fold one split's counts into the
    other's numbers. :meth:`TensorFlowSegmenter._split_metrics` therefore calls this once per
    split, and every route below has to yield a new object.

    A LIVE instance is cloned through Keras's own `get_config()` / `from_config()` — the same
    protocol `keras.Model.compile` uses to reconstruct metrics, so a wired
    `!class:keras.metrics.MeanIoU {num_classes: 3, sparse_y_pred: false}` keeps every knob it was
    given (measured: the clone reports 0.0 where the original, after an update, reports 1.0).
    Metrics that do not implement it fall back to being shared, which is the only honest answer:
    a config that wires such a metric gets one accumulator, not a silently wrong copy.
    """
    if isinstance(value, type):
        return value()
    built = flow(value)
    if isinstance(built, str):
        return keras.metrics.get(built)
    if isinstance(built, keras.metrics.Metric):
        try:
            return type(built).from_config(built.get_config())
        except (TypeError, ValueError, NotImplementedError):  # pragma: no cover - exotic metric
            logger.debug(f"{type(built).__name__} is not clonable; the splits will share it.")
    return built


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the CHANNEL axis, which is LAST in TensorFlow."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))

"""``KerasSegmenter`` — the same segmentation run, on Keras 3 instead of Lightning.

The fourth backend, and its job is to make one claim checkable: **the config is the program**. A
Keras run and a Lightning run share the dataset, the batch/epoch knobs, the run identity, the
metrics sidecar and the tracking. What differs is the training loop, the model and — uniquely for
this engine — the *layout* of the image tensor.

Almost none of it is duplicated:

* **Data** comes from recordstream unchanged, and so does the BATCHING:
  ``recordstream.keras.RecordSequence`` is the ``keras.utils.PyDataset`` a ``DataLoader`` would
  have been. All this backend adds is :meth:`_to_arrays`, the few lines that say what a
  segmentation batch looks like — handed in exactly as a ``collate_fn`` is handed to a
  ``DataLoader``.
* **Run artifacts** come from matrainer; **tracking** is ``keras_tracking_callback``.

So what is written here is a batch-shaping function, ``compile()`` + ``fit()``, and the dispatch.

**TWO real differences from the torch backends, both about the ENGINE and both living in the
config where a reader can see them:**

1. **Keras is channels-LAST.** The image arrives ``[N, H, W, C]``, so the ``preprocess`` chain for
   a Keras run simply STOPS after the joint albumentations resize rather than adding the CHW
   ``ToTensor`` the torch chain ends with. The MASK is ``[N, H, W]`` either way — a class-id map
   has no channel axis — so only the image line changes.
2. **`from_logits=True` is mandatory on the loss.** The head is a bare 1x1 conv emitting raw
   logits; Keras's default expects probabilities and, handed logits, normalizes by the row sum and
   CLIPS rather than raising. Measured: the correct loss is 1.4546 on a batch where the wrong one
   reports 4.6293 — finite, and simply wrong. See :mod:`deltaid.keras_segmentation`.

Keras 3 picks its engine from ``KERAS_BACKEND`` (``torch`` / ``tensorflow`` / ``jax``); this module
never sets it, so ``deltaid train --framework keras`` runs on whichever the operator chose.

Minimal flat YAML::

    runnable: !class:deltaid.KerasSegmenter
    model: !lazy:deltaid.KerasUNet
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess_keras }
    num_classes: 3
"""

from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import numpy as np
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_slots import union_metric_specs
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
from recordstream.keras import RecordSequence
from recordstream.runnable import ProgressReporting, entrypoint, run_entrypoint

from deltaid._keras import keras, keras_backend

logger = get_logger(__name__)

#: Which entry point :meth:`KerasSegmenter.run` dispatches to — the same vocabulary every other
#: backend uses, because the verb is a property of the TASK, not of the engine.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]

# What Keras accepts in each compile slot: its own class, or the string alias it resolves to.
# Spelled out rather than left as `Any` because `Any` types nothing and gives `to_pydantic`
# nothing to build a widget from. It does NOT change how a wired value materializes: a nested
# `!class:` in a CONSTRUCTOR slot stays a deferred `Class` marker regardless of annotation, so
# `_compile` flows these at USE — which is what the `Lazy[...]` marker declares.
KerasLoss = Union[str, keras.losses.Loss]
KerasMetric = Union[str, keras.metrics.Metric]
KerasOptimizer = Union[str, keras.optimizers.Optimizer]

__all__ = ["KerasLoss", "KerasMetric", "KerasOptimizer", "KerasSegmenter", "SegmenterTask"]


@configurable(task="segmentation", role="trainer", framework="keras")
class KerasSegmenter(ProgressReporting):
    """Train, evaluate or predict a semantic segmenter with Keras 3.

    Args:
        model: The segmentation model — wire ``!lazy:`` so ``initialize()`` can inject the
            resolved ``num_classes``.
        train_set: Training source (``task="fit"``).
        test_set: Evaluation / prediction source.
        num_classes: Class count, background included. Omitted, it is derived from the model or
            the mask column.
        ignore_index: Pixel value excluded from the default metrics (Keras's ``ignore_class``).
            The default ``-100`` matches the torch backends and excludes nothing, since a
            class-id mask never contains it.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring.
        callbacks: ``keras.callbacks.Callback`` instances. A file-writing one gets its parent
            directory created, because Keras opens the path without doing so.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the per-pixel mask.
        optimizer: Keras optimizer — a string alias (``"adam"``) or an instance.
        loss: Keras loss. ``None`` picks ``SparseCategoricalCrossentropy(from_logits=True)``.
        metrics / train_metrics / val_metrics / test_metrics: Keras metrics. This engine
            computes their UNION — see :meth:`_compiled_metrics`.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[keras.Model]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        task: SegmenterTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
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
        # `keras.Model`, not `Lazy[keras.Model]`: the DEFERRED-ness is declared once, on the ctor
        # param above, which is what `lazy_param_names` / `to_pydantic` read. Restating it here
        # would make the attribute a union no method could call through.
        self.model: keras.Model = cast(keras.Model, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: SegmenterTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.optimizer: Lazy[KerasOptimizer] = optimizer
        # Sentinels: the loss needs `from_logits`, the metrics need `num_classes` — and a live
        # default could not be told apart from a user's explicit choice.
        self.loss: Optional[Lazy[KerasLoss]] = loss
        self.metrics: Optional[List[Lazy[KerasMetric]]] = metrics
        self.train_metrics: Optional[List[Lazy[KerasMetric]]] = train_metrics
        self.val_metrics: Optional[List[Lazy[KerasMetric]]] = val_metrics
        self.test_metrics: Optional[List[Lazy[KerasMetric]]] = test_metrics
        # Stored VERBATIM (lazy-init): entries may be deferred markers, flowed where used.
        self.callbacks: Optional[List[Lazy[keras.callbacks.Callback]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve ``num_classes``, then build and compile the Keras model."""
        logger.info(f"Keras backend in effect: {keras_backend()}")
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        self._compile()

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
        """Normalize a wired source's TYPE and its STATE, in THIS process.

        ``ensure_materialized`` is the fork guard: a `PyDataset` may be read by worker processes,
        and on a machine where anything has set the `fork` start method a lazy source first built
        in the child dies with a SIGSEGV and no traceback.
        """
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

        ``from_logits=True`` is not optional here — see the module docstring. And the default
        metric needs ``sparse_y_pred=False``: ``MeanIoU`` otherwise expects an ALREADY-argmaxed
        prediction and raises a shape error on raw logits (measured).
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

    def _compile(self) -> None:
        """Build the model at the resolved width, then compile it.

        The compile slots are flowed at USE: a nested ``!class:`` in a constructor slot arrives
        as a deferred ``Class`` marker, and Keras takes an unbuilt marker for a custom callable
        and fails deep inside ``compile`` rather than saying what is wrong. A bare ``flow()`` is
        the whole guard — it returns a live object, a string alias and a list unchanged.

        The model included: it reaches here deferred (``!lazy:`` is skipped by confluid's
        auto-flow walkers precisely so the resolved width can be injected now), and an ALREADY
        live model comes back untouched with the runtime kwarg dropped.
        """
        self._resolve_task_defaults()
        self.model = flow(self.model, num_classes=self.num_classes)
        metrics = [flow(m) for m in self._compiled_metrics()]
        self._warn_if_the_engine_cannot_afford_these_metrics(metrics)
        self.model.compile(
            optimizer=flow(self.optimizer),
            loss=flow(self.loss),
            metrics=metrics,
        )

    @staticmethod
    def _warn_if_the_engine_cannot_afford_these_metrics(metrics: Sequence[Any]) -> None:
        """Say so when the IoU family meets the torch engine, because the run will look HUNG.

        ``keras.metrics.MeanIoU`` / ``IoU`` / ``OneHotMeanIoU`` accumulate a ``[C, C]`` confusion
        matrix through ``keras.ops.scatter``, and the torch backend implements that as a **Python
        loop over every index** (``keras/src/backend/torch/core.py``: ``for i in
        range(indices.shape[0]): zeros[tuple(index)] += values[i]``). Per-pixel that index count is
        the PIXEL count, so one 8x224x224 batch is 401,408 iterations. Measured on that batch:

        ============  =========================
        engine        one ``update_state`` call
        ============  =========================
        tensorflow    0.009 s
        jax           0.194 s
        **torch**     **~160 s**
        ============  =========================

        The model step itself is 0.37 s, so on torch the metric is ~99.8% of the run. There is
        nothing to fix in this project — the metric is upstream's and correct, and the engine is
        the operator's choice — so this warns instead of substituting a different metric: a
        default that silently changed with the engine would break the one claim the project makes,
        that the same config gives the same run.

        Classification never hit this, and the asymmetry is the tell: its metrics are per-IMAGE,
        so the same scatter loops over 8 indices rather than 401,408.
        """
        if keras_backend() != "torch":
            return
        slow = [m for m in metrics if isinstance(m, (keras.metrics.MeanIoU, keras.metrics.IoU))]
        if not slow:
            return
        logger.warning(
            f"{', '.join(type(m).__name__ for m in slow)} on the TORCH engine will make this run "
            f"appear to hang: keras.ops.scatter is a Python loop there, ~160 s per 8x224x224 batch "
            f"against 0.009 s on TensorFlow (measured). Re-run with KERAS_BACKEND=tensorflow (or "
            f"jax), or drop the IoU metric and keep pixel accuracy."
        )

    def _compiled_metrics(self) -> Sequence[Lazy[KerasMetric]]:
        """The ONE metric list this engine can compute — the UNION of the shared and split slots.

        ``keras.Model.compile`` takes a single list and reports it on training AND validation
        (prefixed ``val_``); there is no hook that narrows a metric to one split short of
        recompiling between phases, which would discard the optimizer state a fit is using. So a
        metric wired to ``val_metrics`` is computed on the training split too.

        Widening is the honest failure: a metric appearing where it was not asked for is noise a
        reader ignores, where DROPPING it would be a wired metric that silently reports nothing.
        """
        return union_metric_specs(self.metrics, self.train_metrics, self.val_metrics, self.test_metrics) or []

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
        """One collated record batch → the numpy ``(x, y)`` Keras consumes.

        The Keras twin of the Lightning backend's ``_targets``, and short for the same reason:
        recordstream hands over numpy already, so nothing is converted twice.

        **Keras is channels-last**, so the image arrives ``[N, H, W, C]`` — a Keras config's
        preprocessing chain therefore stops after the joint resize rather than adding the CHW
        ``ToTensor`` the torch chain ends with. The MASK needs no such difference: a class-id map
        is ``[N, H, W]`` in every engine. ``int64`` is named explicitly for the same reason the
        torch backends name it — albumentations returns int32.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int64")

    def _callbacks(self) -> List[Any]:
        """Wired Keras callbacks plus OPTIONAL matrainer tracking (a few lines; removable)."""
        callbacks = [flow(cb) for cb in (self.callbacks or [])]
        for callback in callbacks:
            # Keras's file-writing callbacks open their path without creating its parent, so a
            # perfectly reasonable `./keras_logs/run.csv` dies with FileNotFoundError several
            # minutes into a fit — after the data walk and the model build.
            target = getattr(callback, "filename", None) or getattr(callback, "filepath", None)
            if isinstance(target, (str, Path)):
                Path(target).expanduser().parent.mkdir(parents=True, exist_ok=True)
        trackers = self._live_trackers()
        if trackers:
            from matrainer.keras import keras_tracking_callback

            # `runnable=self` rather than a params dict: matrainer derives the hyperparameters
            # from the flowed object graph AND dumps the resolved config.
            callbacks.append(keras_tracking_callback(trackers, runnable=self))
        return callbacks

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"deltaid keras training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("KerasSegmenter.fit() requires a 'train_set'.")
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None
        history = self.model.fit(
            train_seq, validation_data=val_seq, epochs=self.max_epochs, callbacks=self._callbacks()
        )
        self._history = {k: [float(x) for x in v] for k, v in history.history.items()}
        self._computed_metrics = {k: v[-1] for k, v in self._history.items()}
        logger.info(f"Training complete: {self._computed_metrics}")
        # Local file only — no tracker touch, so it is safe after `keras_tracking_callback`'s
        # `on_train_end` has closed the run out.
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("KerasSegmenter.test() requires a 'test_set'.")
        results = self.model.evaluate(self._sequence(self.test_set), return_dict=True, callbacks=self._callbacks())
        self._computed_metrics = {k: float(v) for k, v in results.items()}
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        if self.test_set is None:
            raise ValueError("KerasSegmenter.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            logits = self.model.predict(sequence[index][0], verbose=0)
            if sink is not None:
                self._write_predictions(np.asarray(logits), batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, logits: np.ndarray, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.SegmentationOutput` per record, paired with its metadata.

        The contract is recordstream's and framework-neutral — it is generic in its array type,
        so the SAME sink that consumes the torch backends' output consumes this numpy one with no
        adapter. The channel axis is LAST here, so the softmax and argmax run over ``-1`` where
        the torch backends use dim 1; the emitted ``mask`` is ``[H, W]`` either way.
        """
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

    # ---- tracking ------------------------------------------------------------ #
    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot.

        Written back rather than returned fresh each call: ``flow()`` on a LIVE object returns
        that same object (idempotent), but a deferred ``Fluid`` would be REBUILT on every call —
        two MLflow runs for one training run.
        """
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

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
        """Per-epoch metric history, as Keras's ``History`` recorded it."""
        return {key: list(values) for key, values in self._history.items()}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the CHANNEL axis, which is LAST in Keras.

    Written out rather than taken from the engine because this runs on numpy, after
    ``model.predict`` has already handed the arrays back — reaching for a backend op here would
    re-enter whichever engine Keras is bound to for no gain.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))

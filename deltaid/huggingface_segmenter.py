"""``HuggingFaceSegmenter`` — the segmentation run with ``transformers.Trainer`` driving it.

The sixth backend and the one that writes least while swapping most: ``Trainer`` owns the epoch,
the optimizer, the scheduler, gradient accumulation, mixed precision and distribution (via
Accelerate). What deltaid supplies is the data, the run identity and the artifacts.

**What it owns is three things, and TWO are facts about semantic segmentation on the Hub that
have no classification analogue:**

1. ``_to_hf_batch`` — one collated record batch to ``{"pixel_values", "labels"}``, handed to
   ``Trainer`` as its ``data_collator``, which is the same seat a ``collate_fn`` occupies
   elsewhere.
2. **The logits come back at a REDUCED resolution, and everything downstream must upsample.**
   SegFormer emits ``[N, C, H/4, W/4]`` — measured: a 224x224 input gives 56x56 — and upsamples
   internally *only* for its own loss. So the metric hook and the prediction path both resize the
   logits to the label resolution first; comparing them raw does not merely score badly, it
   RAISES (*"If `preds` have one dimension more than `target`, the shape of `preds`…"*).
3. **The ignore index is the MODEL's config knob, and its default is 255, not torch's -100.**
   ``initialize()`` writes the segmenter's ``ignore_index`` onto
   ``config.semantic_loss_ignore_index``. Without that a mask using deltaid's default would reach
   ``cross_entropy`` as an out-of-range class id and raise — which is the honest failure, but for
   the wrong reason.

**There is NO loss set and that is a fact about transformers, not an omission:** an
``AutoModelForSemanticSegmentation`` handed ``labels=`` computes its own loss inside ``forward``,
so there is nothing a config could wire and therefore no ``huggingface_segmentation.py``. Its
``FRAMEWORKS`` entry has ``registrations=None``.

``report_to`` defaults to ``"none"`` ON PURPOSE: Trainer auto-enables every tracker it can import,
and this workspace has wandb AND mlflow installed, so the default would silently start reporting
to both. Tracking goes through the ``trackers`` slot like every other backend.

Run it with ``deltaid train cfg.yaml --framework huggingface``.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from loggair import get_logger
from marainer.artifacts import write_metrics_sidecar
from marainer.metric_slots import merge_metric_specs
from marainer.run_names import generate_run_name, run_prefix
from marainer.tracking.base import Tracker, as_tracker_list
from marainer.tracking.provenance import capture_log_file, log_run_context
from recordstream import (
    PredictionsSink,
    RecordSource,
    batch_metadata,
    batch_tensor,
    collate_records,
    ensure_materialized,
    ensure_record_dataset,
    num_mask_classes,
    segmentation_output,
)
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torch.utils.data import DataLoader

if TYPE_CHECKING:  # pragma: no cover - typing only; the module imports transformers lazily
    from transformers import Trainer, TrainingArguments

logger = get_logger(__name__)

#: Which entry point :meth:`HuggingFaceSegmenter.run` dispatches to.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]

#: A metric: a callable over ``(logits, labels)`` numpy arrays, or the NAME of a built-in one.
HuggingFaceMetric = Union[str, Callable[[np.ndarray, np.ndarray], float]]

#: The model slot. ``torch.nn.Module`` rather than ``transformers.PreTrainedModel`` as the second
#: arm, to keep the module-level ``import transformers`` deferred — see the class docstring.
HuggingFaceNetwork = Union[Any, torch.nn.Module]

__all__ = ["HuggingFaceMetric", "HuggingFaceSegmenter", "SegmenterTask"]


@configurable(task="segmentation", role="trainer", framework="huggingface")
class HuggingFaceSegmenter(TorchRunner, ProgressReporting):
    """Train, evaluate or predict a semantic segmenter with ``transformers.Trainer``.

    Args:
        model: The segmentation model — wire ``!lazy:`` so ``initialize()`` can inject the
            resolved ``num_classes``.
        train_set: Training source (``task="fit"``).
        test_set: Evaluation / prediction source.
        num_classes: Class count, background included. Omitted, it is derived from the model or
            the mask column.
        ignore_index: Pixel value excluded from the loss. Written onto the model's
            ``semantic_loss_ignore_index`` — see the module docstring for why that matters.
        task: Which entry point ``run()`` dispatches to.
        max_epochs / batch_size / num_workers / learning_rate / weight_decay: the shared knobs,
            translated into ``TrainingArguments``.
        report_to: Trainer's own tracking integrations. ``"none"`` on purpose.
        output_dir: Where Trainer writes its state; defaults beside the sidecar.
        trackers: marainer trackers — one or a list.
        callbacks: ``transformers.TrainerCallback`` instances.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the per-pixel mask.
        metrics: Metrics computed on the evaluation pass; ``val_metrics`` / ``test_metrics`` are
            that split's own, merged over ``metrics``. There is no ``train_metrics``: a
            ``Trainer``'s ``compute_metrics`` is its EVAL hook, so the slot would accept a wired
            set and silently never call it.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``training_arguments``,
    ``trainer``, ``test_loader``, ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[HuggingFaceNetwork]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        task: SegmenterTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        learning_rate: float = 6e-5,
        weight_decay: float = 0.01,
        report_to: str = "none",
        output_dir: Optional[str] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy[Any]]] = None,
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "mask",
        metrics: Optional[List[HuggingFaceMetric]] = None,
        val_metrics: Optional[List[HuggingFaceMetric]] = None,
        test_metrics: Optional[List[HuggingFaceMetric]] = None,
    ) -> None:
        super().__init__()
        self.model: HuggingFaceNetwork = cast(HuggingFaceNetwork, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: SegmenterTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        # Stored rather than closed over, because `_training_arguments()` reads them at RUN time:
        # they are the floor it injects into whatever `training_arguments` the config left, so a
        # post-construction `seg.learning_rate = …` still reaches the Trainer.
        self.num_workers = int(num_workers)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.report_to = report_to
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.metrics: Optional[List[HuggingFaceMetric]] = metrics
        self.val_metrics: Optional[List[HuggingFaceMetric]] = val_metrics
        self.test_metrics: Optional[List[HuggingFaceMetric]] = test_metrics
        # All THREE transformers objects are DEFERRED SLOTS, and the targets are STRING dotted
        # paths: this module has no top-level `import transformers` (the methods that need it
        # import it in their bodies), and confluid resolves a dotted target at flow time — so the
        # slots stay declarable without dragging the library into `import deltaid`.
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(
            DataLoader,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_records,
        )
        # EMPTY on purpose — every derived knob is applied in `_training_arguments()` instead.
        # A config overriding this slot REPLACES it whole (that is what a YAML key does), so
        # pinning the defaults here would mean `training_arguments: {use_cpu: true}` silently
        # dropped `num_train_epochs`, the batch size AND `remove_unused_columns=False` — the last
        # of which fails with a `KeyError: 'image'` three frames inside Trainer's collator.
        self.training_arguments: Lazy["TrainingArguments"] = LazyClass("transformers.TrainingArguments")
        self.trainer: Lazy["Trainer"] = LazyClass("transformers.Trainer")
        self.callbacks: Optional[List[Lazy[Any]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._computed_metrics: Dict[str, float] = {}

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override
        """Only ``fit`` optimizes — an evaluate/predict run is happy under inference mode."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve ``num_classes``, build the model, and put the ignore index where it is read."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        self.model = flow(self.model, num_classes=self.num_classes)
        self._apply_ignore_index()

    def _apply_ignore_index(self) -> None:
        """Write the segmenter's ``ignore_index`` onto the model's own config.

        This backend is the ONE where the ignore index is not a loss argument: the model computes
        its loss internally and reads ``config.semantic_loss_ignore_index``, whose default is
        **255** (measured) rather than torch's ``-100``. Leaving it alone would mean a mask using
        deltaid's default reaches ``cross_entropy`` as an out-of-range class id.

        Set on the CONFIG rather than passed anywhere, because only the model's own forward sees
        it — and set here, in the segmenter, because only the run knows what the config asked for.
        """
        config = getattr(self._network(), "config", None)
        if config is not None and hasattr(config, "semantic_loss_ignore_index"):
            config.semantic_loss_ignore_index = self.ignore_index

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
        nc = self.num_classes or getattr(self.model, "num_classes", None)
        if not nc:
            raise RuntimeError("could not determine num_classes — set it, or wire a model that exposes it.")
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
                return int(model_nc)
        if self.num_classes is not None:
            return int(self.num_classes)
        if isinstance(self.model, Fluid):
            pinned = self.model.kwargs.get("num_classes")
            if pinned:
                return int(pinned)
        return None

    def _network(self) -> Any:
        """The live ``transformers`` model — the wrapper's, or a bare one wired directly."""
        return getattr(self.model, "model", self.model)

    # ---- data --------------------------------------------------------------- #
    def _to_hf_batch(self, records: Sequence[Any]) -> Dict[str, torch.Tensor]:
        """One list of records → the mapping ``Trainer`` feeds the model.

        The keys are transformers' contract, not ours: ``pixel_values`` is what an image model's
        ``forward`` takes and ``labels`` is what makes it return a loss. ``int64`` is named
        explicitly for the same reason as every other backend — albumentations returns int32 and
        the cross-entropy family rejects it.
        """
        batch = collate_records(list(records))
        return {
            "pixel_values": batch_tensor(batch, self.input_key),
            "labels": batch_tensor(batch, self.target_key, dtype=torch.int64),
        }

    def _record_loader(self, source: Any) -> DataLoader:
        """A loader of COLLATED RECORD BATCHES — the shape :meth:`predict` needs.

        Prediction pairs each output row with the metadata that travelled beside the image, and
        :meth:`_to_hf_batch` has already dropped it.
        """
        return cast(DataLoader, flow(self.test_loader, dataset=self._prepare(source)))

    # ---- the Trainer -------------------------------------------------------- #
    def _training_arguments(self) -> Any:
        """``TrainingArguments`` from the shared knobs — the flat config, in Trainer's words.

        Derived here rather than pinned on the constructor's ``LazyClass`` so that OVERRIDING the
        slot ADDS to these instead of replacing them. `TrainingArguments` has ~100 fields and a
        config reaching for one of them (``use_cpu``, gradient accumulation, a warmup schedule)
        would otherwise lose every knob below — including ``remove_unused_columns=False``, whose
        absence surfaces as a `KeyError: 'image'` inside Trainer's own collator rather than as
        anything about the override.

        A key the config pinned WINS: only the ones it did not mention are injected, so this
        derivation is a floor rather than an override.
        """
        # `Dict[str, Any]`, not the inferred `dict[str, object]`: `flow()` has a keyword-only
        # `solidify: bool`, so a `**` splat of an `object`-valued mapping is a type error there.
        derived: Dict[str, Any] = {
            "output_dir": self._output_dir(),
            "num_train_epochs": self.max_epochs,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dataloader_num_workers": self.num_workers,
            # A generic map-style source is not a `datasets.Dataset`, so Trainer must not try to
            # drop "unused" columns by inspecting the model signature — our collate already
            # produces exactly the two keys the model takes.
            "remove_unused_columns": False,
            "report_to": self.report_to,
            "run_name": self.run_name,
            "disable_tqdm": True,
            "save_strategy": "no",
        }
        pinned = getattr(self.training_arguments, "kwargs", None) or {}
        return flow(self.training_arguments, **{k: v for k, v in derived.items() if k not in pinned})

    def _build_trainer(self, train: Any = None, evaluate: Any = None, split: str = "val") -> Any:
        """Flow the ``Trainer`` with everything only the RUN knows."""
        return flow(
            self.trainer,
            model=self._network(),
            args=self._training_arguments(),
            train_dataset=train,
            eval_dataset=evaluate,
            data_collator=self._to_hf_batch,
            compute_metrics=self._metrics_hook(split),
            callbacks=[flow(cb) for cb in (self.callbacks or [])] or None,
        )

    def _metrics_hook(self, split: str) -> Any:
        """Trainer's ``compute_metrics``, bound to ONE split's metric list.

        A ``Trainer`` holds a single hook, so the split is decided where the Trainer is BUILT —
        ``fit`` evaluates the validation split, ``test`` the held-out one — rather than inside
        the hook, which cannot tell which dataset it was called on.

        **The hook is where the upsample happens**, and that placement is deliberate: the model's
        logits are ``[N, C, H/4, W/4]`` while the labels are ``[N, H, W]``, so aligning them is a
        property of THIS engine, not of any metric. Doing it here means every metric — built-in or
        user-supplied — sees arrays that already agree, and none of them has to know that SegFormer
        emits at a reduced resolution.
        """
        metrics = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or list(_DEFAULT_METRICS)
        ignore_index = self.ignore_index

        def compute_metrics(eval_pred: Any) -> Dict[str, float]:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
            logits = logits[0] if isinstance(logits, tuple) else logits
            upsampled = _upsample_logits(np.asarray(logits), np.asarray(labels).shape[-2:])
            out: Dict[str, float] = {}
            for metric in metrics:
                fn = _as_metric(metric)
                name = getattr(fn, "__name__", str(metric)).lstrip("_")
                out[name] = float(fn(upsampled, np.asarray(labels), ignore_index))
            return out

        return compute_metrics

    def _output_dir(self) -> str:
        """Where Trainer writes its state — beside the sidecar when one is configured."""
        if self.output_dir:
            return str(Path(self.output_dir).expanduser())
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent / self.run_name)
        return str(Path("./hf_runs") / self.run_name)

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"deltaid huggingface training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("HuggingFaceSegmenter.fit() requires a 'train_set'.")
        # Provenance through marainer's plain FUNCTIONS — `runnable=self` so the hyperparameters
        # and the resolved config come from marainer's ONE derivation.
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        trainer = self._build_trainer(train=self.train_set, evaluate=self.val_set)
        result = trainer.train()
        self._computed_metrics = {k: float(v) for k, v in result.metrics.items() if _is_number(v)}
        if self.val_set is not None:
            self._computed_metrics.update({k: float(v) for k, v in trainer.evaluate().items() if _is_number(v)})
        _log_trainer_history(trainer, trackers)
        logger.info(f"Training complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        # Uploads the log file, stamps the manifest's terminal status AND finalizes every tracker.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("HuggingFaceSegmenter.test() requires a 'test_set'.")
        results = self._build_trainer(evaluate=self.test_set, split="test").evaluate()
        self._computed_metrics = {k: float(v) for k, v in results.items() if _is_number(v)}
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        """Stream per-record predictions to the sink.

        A plain ``torch.no_grad()`` forward rather than ``Trainer.predict``: that returns only the
        stacked logits, and the sink contract needs each row paired with the metadata from the
        record it came from.
        """
        if self.test_set is None:
            raise ValueError("HuggingFaceSegmenter.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        network = self._network().eval()
        device = next(network.parameters()).device
        with torch.no_grad():
            for batch in self._record_loader(self.test_set):
                images = batch_tensor(batch, self.input_key, device)
                logits = network(pixel_values=images).logits
                # Back to the INPUT resolution before anything reads the mask — the sink's
                # consumer expects a prediction the size of the image it came from, and this
                # engine is the only one that does not already produce that.
                logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
                if sink is not None:
                    self._write_predictions(logits, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, logits: torch.Tensor, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.SegmentationOutput` per record, paired with its metadata."""
        out = segmentation_output(logits)
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        for i in range(len(out["mask"])):
            item = {"logits": out["logits"][i], "probs": out["probs"][i], "mask": out["mask"][i]}
            meta = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            sink.write(item, meta)

    # ---- tracking ------------------------------------------------------------ #
    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot."""
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


def _upsample_logits(logits: np.ndarray, size: Any) -> np.ndarray:
    """Resize ``[N, C, h, w]`` logits to ``[N, C, H, W]`` — the engine's one structural quirk.

    Bilinear, matching what the model does internally for its own loss, so the metric scores the
    same prediction the loss was computed against. A no-op when the sizes already agree, which
    keeps this correct for a Hub model that emits at full resolution.
    """
    if tuple(logits.shape[-2:]) == tuple(size):
        return logits
    resized = F.interpolate(torch.as_tensor(logits), size=tuple(size), mode="bilinear", align_corners=False)
    return cast(np.ndarray, resized.numpy())


def _mean_iou(logits: np.ndarray, labels: np.ndarray, ignore_index: int) -> float:
    """Mean intersection-over-union over the classes PRESENT in either prediction or target.

    A class absent from both is skipped rather than scored 0 — otherwise a split that happens not
    to contain one of the label scheme's classes drags the mean down for a reason that says
    nothing about the model.
    """
    predicted = np.argmax(logits, axis=1)
    valid = labels != ignore_index
    scores = []
    for class_id in range(logits.shape[1]):
        pred_c = (predicted == class_id) & valid
        true_c = (labels == class_id) & valid
        union = np.logical_or(pred_c, true_c).sum()
        if union == 0:
            continue
        scores.append(float(np.logical_and(pred_c, true_c).sum()) / float(union))
    return float(np.mean(scores)) if scores else 0.0


def _pixel_accuracy(logits: np.ndarray, labels: np.ndarray, ignore_index: int) -> float:
    """Fraction of non-ignored pixels whose argmax matches the label."""
    valid = labels != ignore_index
    total = int(valid.sum())
    if total == 0:
        return 0.0
    return float(((np.argmax(logits, axis=1) == labels) & valid).sum()) / float(total)


#: The metrics a config can name. Plain FUNCTIONS over numpy — there is no class for a `!class:`
#: tag to name or for confluid's registry to tag, the same situation the fastai names are in.
_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray, int], float]] = {
    "mean_iou": _mean_iou,
    "pixel_accuracy": _pixel_accuracy,
}

#: What an unset `metrics:` computes.
_DEFAULT_METRICS = ("mean_iou", "pixel_accuracy")


def _as_metric(value: HuggingFaceMetric) -> Callable[..., float]:
    """A metric callable from a callable or from its NAME."""
    built = flow(value)
    if callable(built):
        return cast(Callable[..., float], built)
    metric = _METRICS.get(str(built))
    if metric is None:
        raise ValueError(f"Unknown metric {built!r}. Known names: {', '.join(sorted(_METRICS))}.")
    return metric


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _log_trainer_history(trainer: Any, trackers: Sequence[Tracker]) -> None:
    """Forward EVERY logged step to the trackers, not just the final metrics.

    ``Trainer`` accumulates its own ``log_history`` and nothing else sees those rows, so without
    this a tracked run would show one point per RUN where every other backend shows a curve.
    """
    if not trackers:
        return
    for entry in getattr(trainer.state, "log_history", []) or []:
        values = {k: float(v) for k, v in entry.items() if _is_number(v) and k != "epoch"}
        if not values:
            continue
        step = int(entry.get("step", 0))
        for tracker in trackers:
            tracker.log_metrics(values, step=step)

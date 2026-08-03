"""``FastaiSegmenter`` — the same segmentation run with a fastai ``Learner``.

The third backend, and the one that swaps the LOOP and nothing else. fastai runs on torch, so it
reuses :class:`~deltaid.smp_model.SegmentationModelPytorch` unchanged, the same
``[N, 3, H, W]`` batch shape, the same ``collate_records``, and the same torch losses — the
``fastai`` block in the example config overrides ``runnable`` and NOTHING else, which makes it the
smallest block in the file and the project's strongest evidence for its own claim.

What it brings that the others do not: one-cycle scheduling, a ``Recorder`` that reports every
epoch, and fastai's own metric vocabulary.

**What it owns is three things**, and two of them are measured facts about fastai rather than
choices:

1. ``_to_tuple`` — the twin of the Lightning backend's ``_targets``. fastai unpacks a batch
   POSITIONALLY, so the batch must BE a tuple rather than the record dict a ``LightningModule``
   reads by key.
2. **The loss needs ``axis=1``, and the default is WRONG for segmentation.**
   ``CrossEntropyLossFlat`` defaults to ``axis=-1``, which flattens the CHANNEL dimension into the
   batch and dies with *"Expected input batch_size (48) to match target batch_size (128)"*
   (measured). With ``axis=1`` it agrees with ``torch.nn.CrossEntropyLoss`` to seven decimals. A
   classification config carried over unchanged would fail loudly here — which is the good case;
   the reason it is stated in the default rather than left to the user is that nothing else in the
   config hints at it.
3. **The metrics do NOT need the same fix**, and that asymmetry is worth knowing:
   ``DiceMulti`` and ``JaccardCoeffMulti`` already default to ``axis=1`` because they exist for
   segmentation, so only the loss — which is shared with classification — carries the wrong
   default.

**Metrics are NAMES, not ``!class:`` tags, and that is a fastai fact.** ``foreground_acc`` is a
plain function and ``Dice`` / ``DiceMulti`` / ``JaccardCoeffMulti`` are classes fastai instantiates
itself (``mk_metric`` does ``if isinstance(m, type): m = m()``), so there is nothing uniform for a
tag to resolve or for confluid's registry to tag. ``_as_metric`` resolves a string against
``fastai.metrics``, naming the module when it cannot. Do NOT add a metric-wrapper class to make
them registrable.

``from fastai.callback import schedule`` is load-bearing: ``Learner.fit_one_cycle`` is a ``@patch``
installed as that module's IMPORT SIDE EFFECT, and fastai's ``@patch`` returns ``None``, so
importing the function by name binds None. Without the import, ``learn.fit_one_cycle`` falls
through fastcore's attribute delegation to the MODEL and raises
``'SegmentationModelPytorch' object has no attribute 'fit_one_cycle'``.

Run it with ``deltaid train cfg.yaml --framework fastai``.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import torch
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from fastai import metrics as fastai_metrics
from fastai import optimizer as fastai_optimizer

# `Learner.fit_one_cycle` is a `@patch` installed as this module's import SIDE EFFECT — see the
# module docstring. Do not "tidy" it away.
from fastai.callback import schedule  # noqa: F401
from fastai.callback.core import Callback as FastaiCallback
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_slots import merge_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, as_tracker_list
from matrainer.tracking.provenance import capture_log_file, log_run_context
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

logger = get_logger(__name__)

#: Which entry point :meth:`FastaiSegmenter.run` dispatches to.
SegmenterTask = Literal["fit", "evaluate", "test", "predict"]

#: A fastai metric: a callable, a metric class, or the NAME of either in ``fastai.metrics``.
FastaiMetric = Union[str, Callable[..., Any]]

#: A fastai optimizer function, or its name in ``fastai.optimizer``.
FastaiOptimizer = Union[str, Callable[..., Any]]

#: The per-pixel default set — both already default to ``axis=1``, unlike the loss.
DEFAULT_METRICS: Tuple[str, ...] = ("DiceMulti", "JaccardCoeffMulti")

__all__ = ["DEFAULT_METRICS", "FastaiMetric", "FastaiOptimizer", "FastaiSegmenter", "SegmenterTask"]


@configurable(task="segmentation", role="trainer", framework="torch")
class FastaiSegmenter(TorchRunner, ProgressReporting):
    """Train, evaluate or predict a semantic segmenter with a fastai ``Learner``.

    Args:
        model: Segmentation model emitting ``[N, num_classes, H, W]`` logits — wire ``!lazy:`` so
            :meth:`initialize` can inject the resolved ``num_classes`` at run time.
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``).
        num_classes: Class count, background included. Omitted, it is derived from the model or
            the mask column.
        ignore_index: Pixel value skipped by the default loss.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        num_workers: DataLoader worker processes.
        learning_rate: Peak LR — ``lr_max`` for one-cycle, ``lr`` otherwise.
        weight_decay: fastai's ``wd``. ``None`` is fastai's own default.
        one_cycle: Use ``fit_one_cycle`` (fastai's headline scheduler) instead of plain ``fit``.
        opt_func: A ``fastai.optimizer`` function or its name (``"Adam"``, ``"SGD"``, …).
        loss: The loss. ``None`` picks ``CrossEntropyLossFlat(axis=1)`` — see the module
            docstring for why the axis is not fastai's default.
        metrics: fastai metrics (callables or names), computed on the validation pass.
        val_metrics / test_metrics: That split's OWN metrics, merged over ``metrics``. There is
            no ``train_metrics``: a fastai ``Learner`` computes its metrics on the validation
            pass only, so the slot would accept a set and silently never call it.
        callbacks: fastai callbacks.
        trackers: matrainer trackers — one or a list.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the per-pixel mask.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``train/val/test_loader``,
    ``dataloaders``, ``learner``, ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[torch.nn.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        task: SegmenterTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        learning_rate: float = 1e-3,
        weight_decay: Optional[float] = None,
        one_cycle: bool = True,
        opt_func: FastaiOptimizer = "Adam",
        loss: Optional[Lazy[Any]] = None,
        metrics: Optional[List[FastaiMetric]] = None,
        val_metrics: Optional[List[FastaiMetric]] = None,
        test_metrics: Optional[List[FastaiMetric]] = None,
        callbacks: Optional[List[Lazy[FastaiCallback]]] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "mask",
    ) -> None:
        super().__init__()
        # `torch.nn.Module`, not `Lazy[...]`: the DEFERRED-ness is declared once, on the ctor
        # param above, which is what `lazy_param_names` / `to_pydantic` read. Restating it here
        # would make the attribute a union no method could call through.
        self.model: torch.nn.Module = cast(torch.nn.Module, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: SegmenterTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.one_cycle = one_cycle
        self.opt_func: FastaiOptimizer = opt_func
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        # Stored VERBATIM (lazy-init): entries may be deferred `Fluid` markers, flowed where used.
        self.callbacks: Optional[List[Lazy[FastaiCallback]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        # The loaders are the Lightning backend's construction, not a private one — same names,
        # same deferral, same overridability, so ONE config idiom drives both backends. The
        # divergence is exactly ONE kwarg: `collate_fn=self._to_tuple` where Lightning bakes
        # `collate_records`, because fastai unpacks a batch positionally.
        loader_kw: Dict[str, Any] = dict(
            collate_fn=self._to_tuple,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        # fastai's own two objects, deferred for the same reason the loaders are: everything this
        # backend CONSTRUCTS should be reachable from config, or every knob it does not restate —
        # a `DataLoaders` device, a `Learner`'s `moms` / `splitter` / `model_dir` — is unreachable
        # from YAML. `DataLoaders(*loaders, …)` takes its loaders POSITIONALLY, so `_dataloaders()`
        # supplies them as `flow()`'s positional runtime args.
        self.dataloaders: Lazy[DataLoaders] = LazyClass(DataLoaders)
        self.learner: Lazy[Learner] = LazyClass(Learner, lr=learning_rate, wd=weight_decay)
        # Sentinels, for the same reason every other backend uses them: a live default could not
        # be told apart from a user's explicit choice.
        self.loss: Optional[Lazy[Any]] = loss
        self.metrics: Optional[List[FastaiMetric]] = metrics
        self.val_metrics: Optional[List[FastaiMetric]] = val_metrics
        self.test_metrics: Optional[List[FastaiMetric]] = test_metrics
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._computed_metrics: Dict[str, float] = {}
        self._learner: Optional[Learner] = None

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve ``num_classes``, then build the model and the loss with it."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError(
                f"FastaiSegmenter.model must be a torch.nn.Module, got {type(self.model).__name__}. "
                "Wire `!lazy:deltaid.SegmentationModelPytorch` (or any nn.Module) into the model slot."
            )
        self._resolve_task_defaults()

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
        self.model = flow(self.model, num_classes=self.num_classes)

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

        ``ensure_record_dataset`` does the first; ``ensure_materialized`` does the second, and for
        this backend it is not optional. **fastai sets the multiprocessing start method to ``fork``
        at import**, so a ``DataLoader`` worker is a forked child — and if that child is the first
        to touch a lazy source, ``load_dataset`` runs there, reaching ``_scproxy`` /
        CoreFoundation, which is not fork-safe. The result is a **SIGSEGV with no Python
        traceback**, surfacing only as ``DataLoader worker exited unexpectedly`` (measured on
        macOS, 2026-08-02). Reading one record HERE, in the parent, leaves the child a source that
        needs nothing.

        Setting ``num_workers: 0`` also avoids it, by not forking at all — which is why the crash
        looks intermittent and gets misread as a flaky test rather than a real ordering bug.
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
        """Fill the ``loss`` / ``metrics`` sentinels. Idempotent — a wired value is never replaced.

        **The loss axis is the one thing this backend must not get wrong.**
        ``CrossEntropyLossFlat`` defaults to ``axis=-1``, which flattens the CHANNEL dimension of
        an ``[N, C, H, W]`` prediction into the batch and raises *"Expected input batch_size (48)
        to match target batch_size (128)"*. ``axis=1`` is the per-pixel reading, and it agrees
        with ``torch.nn.CrossEntropyLoss`` to seven decimals (measured).
        """
        if self.loss is None:
            self.loss = CrossEntropyLossFlat(axis=1, ignore_index=self.ignore_index)
            logger.debug(f"loss defaulted to CrossEntropyLossFlat(axis=1, ignore_index={self.ignore_index}).")
        if self.metrics is None:
            # Both already default to `axis=1` — they exist for segmentation, so only the loss
            # (shared with classification) carries the wrong default.
            self.metrics = list(DEFAULT_METRICS)

    # ---- the batch boundary -------------------------------------------------- #
    def _to_tuple(self, records: Sequence[Any], with_targets: bool = True) -> Tuple[torch.Tensor, ...]:
        """One list of records → the ``(x, y)`` tuple fastai consumes.

        The fastai twin of the Lightning backend's ``_targets``, and short for the same reason:
        recordstream does the work. ``collate_records`` stacks the already-tensorized image to
        ``[N, 3, H, W]`` and the mask to ``[N, H, W]``; what is left is naming ``int64``, because
        albumentations returns an int32 mask and the cross-entropy family rejects it.
        """
        batch = collate_records(list(records))
        x = batch_tensor(batch, self.input_key)
        if not with_targets:
            return (x,)
        return x, batch_tensor(batch, self.target_key, dtype=torch.int64)

    def _dataloaders(self) -> DataLoaders:
        """fastai's ``DataLoaders`` — a train loader and a validation loader, always both.

        ``Learner`` validates at the end of every epoch and its ``Recorder`` reports from that
        pass, so unlike ``pytorch_lightning.Trainer`` it has no "no validation" mode. When no
        ``val_set`` is wired the training data is reused, which keeps a run working but makes the
        reported metrics training metrics — worth saying out loud, hence the warning.
        """
        train_ds = self._prepare(self.train_set)
        train = flow(self.train_loader, dataset=train_ds)
        if self.val_set is not None:
            valid = flow(self.val_loader, dataset=self._prepare(self.val_set))
            return cast(DataLoaders, flow(self.dataloaders, train, valid))
        logger.warning(
            "No 'val_set' wired: fastai's Learner always validates, so the TRAINING data is being "
            "used for validation and the reported metrics measure fit, not generalization. Wire a "
            "'val_set' to get a real held-out score."
        )
        return cast(DataLoaders, flow(self.dataloaders, train, flow(self.val_loader, dataset=train_ds)))

    # ---- the learner -------------------------------------------------------- #
    def _build_learner(self, dataloaders: DataLoaders, split: str = "val") -> Learner:
        """Assemble the ``Learner``: model, loss, metrics, callbacks, artifact path.

        ``split`` picks which metric slot is merged over the shared ``metrics`` — a ``Learner``
        holds ONE list and computes it on the validation pass, so the split is decided by which
        entry point is building it rather than by anything inside the Learner.
        """
        path = self._artifact_dir()
        learner = cast(
            Learner,
            flow(
                self.learner,
                dls=dataloaders,
                model=self.model,
                loss_func=flow(self.loss),
                opt_func=_as_opt_func(self.opt_func),
                metrics=[
                    _as_metric(m) for m in (merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or [])
                ],
                cbs=self._callbacks(),
                path=Path(path) if path else None,
            ),
        )
        self._learner = learner
        return learner

    def _callbacks(self) -> List[Any]:
        """The wired fastai callbacks, with any file destination's parent directory created.

        fastai's ``CSVLogger`` opens ``learn.path/fname`` without creating it, so a perfectly
        reasonable ``./fastai_logs/run.csv`` would die *after* the data walk and the model build.
        """
        callbacks = [flow(cb) for cb in (self.callbacks or [])]
        base = Path(self._artifact_dir() or ".")
        for callback in callbacks:
            fname = getattr(callback, "fname", None)
            if isinstance(fname, (str, Path)):
                (base / fname).expanduser().parent.mkdir(parents=True, exist_ok=True)
        return callbacks

    def _artifact_dir(self) -> Optional[str]:
        """Where run artifacts land. fastai has no checkpoint-callback convention to read, so
        this follows the sidecar's directory when one is configured."""
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent)
        return None

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"deltaid fastai training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("FastaiSegmenter.fit() requires a 'train_set'.")
        # Provenance through matrainer's plain FUNCTIONS — `runnable=self` so the hyperparameters
        # and the resolved config come from matrainer's ONE derivation rather than a dict this
        # backend transcribes for itself.
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        learner = self._build_learner(self._dataloaders())
        if self.one_cycle:
            learner.fit_one_cycle(self.max_epochs, lr_max=self.learning_rate)
        else:
            learner.fit(self.max_epochs, lr=self.learning_rate)
        self._computed_metrics = _recorded_metrics(learner)
        _log_recorded_epochs(learner, trackers)
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
            raise ValueError("FastaiSegmenter.test() requires a 'test_set'.")
        loader = flow(self.test_loader, dataset=self._prepare(self.test_set))
        learner = self._build_learner(DataLoaders(loader, loader), "test")
        values = learner.validate()
        self._computed_metrics = _named_values(learner, values)
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        """Stream per-record predictions to the sink.

        A plain ``torch.no_grad()`` forward rather than ``Learner.get_preds``: ``get_preds``
        returns only the stacked predictions, and the sink contract needs each row paired with the
        metadata from the record it came from.

        Same ``test_loader`` slot as :meth:`test`, flowed with ``collate_fn=collate_records`` — a
        runtime kwarg overrides the stored one, so this swaps the batch shape back to the record
        dict without a second loader. The default collate is ``_to_tuple``, which has already
        dropped the metadata the sink needs; every other knob the operator set still applies.
        """
        if self.test_set is None:
            raise ValueError("FastaiSegmenter.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        model = self.model.eval()
        device = next(model.parameters()).device
        records = flow(self.test_loader, dataset=self._prepare(self.test_set), collate_fn=collate_records)
        with torch.no_grad():
            for batch in records:
                logits = model(batch_tensor(batch, self.input_key, device))
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
    def trained_model(self) -> torch.nn.Module:
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


def _as_metric(value: FastaiMetric) -> Any:
    """A fastai metric from a callable or from its NAME in ``fastai.metrics``.

    The string form is not sugar. ``foreground_acc`` is a plain function while ``DiceMulti`` and
    ``JaccardCoeffMulti`` are classes fastai instantiates itself (``mk_metric`` does
    ``if isinstance(m, type): m = m()``), so none of them is uniform enough for a ``!class:`` tag
    to name or for confluid's registry to tag. A name is the only thing a YAML config can write,
    so this resolves it and says which names exist when it cannot.
    """
    built = flow(value)
    if not isinstance(built, str):
        return built
    metric = getattr(fastai_metrics, built, None)
    if metric is None:
        raise ValueError(
            f"Unknown fastai metric {built!r}. Names come from `fastai.metrics` — for segmentation: "
            f"DiceMulti, JaccardCoeffMulti, foreground_acc, Dice, JaccardCoeff."
        )
    return metric


def _as_opt_func(value: FastaiOptimizer) -> Any:
    """A fastai optimizer function from a callable or its name (``"Adam"``, ``"SGD"``, …)."""
    built = flow(value)
    if not isinstance(built, str):
        return built
    opt = getattr(fastai_optimizer, built, None)
    if opt is None:
        raise ValueError(
            f"Unknown fastai optimizer {built!r}. Names come from `fastai.optimizer` — e.g. Adam, "
            f"SGD, RMSProp, RAdam, QHAdam, Lamb, ranger."
        )
    return opt


def _metric_names(learner: Learner) -> List[str]:
    """The Recorder's column names, minus the bookkeeping ones."""
    names = list(getattr(learner.recorder, "metric_names", []) or [])
    return [n for n in names if n not in ("epoch", "time")]


def _named_values(learner: Learner, values: Sequence[Any]) -> Dict[str, float]:
    """Pair a Recorder row with its column names, tolerating a shorter row.

    ``Learner.validate`` returns one fewer value than a training epoch does (there is no
    ``train_loss``), so the names are matched from the END — where ``valid_loss`` and the metrics
    line up — rather than from the front.
    """
    names = _metric_names(learner)
    if len(names) > len(values):
        names = names[len(names) - len(values) :]
    return {name: float(value) for name, value in zip(names, values)}


def _recorded_metrics(learner: Learner) -> Dict[str, float]:
    """The last epoch's metrics as ``{name: value}``, or ``{}`` if nothing was recorded."""
    values = list(getattr(learner.recorder, "values", []) or [])
    return _named_values(learner, values[-1]) if values else {}


def _log_recorded_epochs(learner: Learner, trackers: Sequence[Tracker]) -> None:
    """Forward EVERY recorded epoch to the trackers, not just the final one.

    ``Learner.fit`` reports per epoch through its ``Recorder`` and nothing else sees those rows,
    so without this a tracked fastai run would show one point per RUN where every other backend
    shows a curve.
    """
    if not trackers:
        return
    for epoch, row in enumerate(list(getattr(learner.recorder, "values", []) or [])):
        metrics = _named_values(learner, row)
        for tracker in trackers:
            tracker.log_metrics(metrics, step=epoch)

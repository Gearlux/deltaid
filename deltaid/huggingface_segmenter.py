"""``HuggingFaceSegmenter`` — the segmentation run with ``transformers.Trainer`` driving it.

Since P6 (2026-08-08) this is ``(SegmentationBase, matrainer.huggingface.
HuggingFaceRunnable)``: the machinery — the deferred ``TrainingArguments`` / ``Trainer``
slots, the run-time argument FLOOR, the entry points with the plain-function tracking hookup,
the ``log_history`` replay, the record-loader predict walk — lives in
:class:`matrainer.huggingface.trainer.HuggingFaceRunnable`; the ``num_classes`` resolution
(model head → explicit → lazy kwarg → the announced mask walk) in
:class:`~deltaid.segmentation_base.SegmentationBase`. What stays HERE is segmentation's
huggingface vocabulary, and TWO of its three pieces are facts about semantic segmentation on
the Hub with no classification analogue:

1. ``_to_hf_batch`` — one collated record batch to ``{"pixel_values", "labels"}``, the mask
   read int64 (albumentations returns int32 and the cross-entropy family rejects it).
2. **The logits come back at a REDUCED resolution, and everything downstream must upsample.**
   SegFormer emits ``[N, C, H/4, W/4]`` — measured: a 224x224 input gives 56x56 — and
   upsamples internally *only* for its own loss. So the metric hook and the prediction path
   both resize the logits to the label resolution first; comparing them raw does not merely
   score badly, it RAISES.
3. **The ignore index is the MODEL's config knob, and its default is 255, not torch's -100.**
   ``initialize()`` writes the segmenter's ``ignore_index`` onto
   ``config.semantic_loss_ignore_index``. Without that a mask using deltaid's default would
   reach ``cross_entropy`` as an out-of-range class id and raise — the honest failure, for
   the wrong reason.

**There is NO loss set and that is a fact about transformers, not an omission:** an
``AutoModelForSemanticSegmentation`` handed ``labels=`` computes its own loss inside
``forward``, so there is nothing a config could wire and therefore no
``huggingface_segmentation.py``. Its ``FRAMEWORKS`` entry has ``registrations=None``.

Run it with ``deltaid train cfg.yaml --framework huggingface``.
"""

from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from confluid import Mandatory, Partial, configurable, flow
from loggair import get_logger
from matrainer.huggingface.trainer import HuggingFaceRunnable
from matrainer.metric_functions import accumulate_confusion, iou_per_class, mean_iou, overall_accuracy, zero_confusion
from matrainer.metric_names import MetricNames
from matrainer.metric_slots import merge_metric_specs
from matrainer.tracking.base import Tracker
from recordstream import RecordSource, batch_tensor, collate_records, segmentation_output

from deltaid.segmentation_base import SegmentationBase

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
class HuggingFaceSegmenter(SegmentationBase, HuggingFaceRunnable):
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
        trackers: matrainer trackers — one or a list.
        callbacks: ``transformers.TrainerCallback`` instances.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the per-pixel mask.
        metrics: Metrics computed on the evaluation pass; ``val_metrics`` / ``test_metrics`` are
            that split's own, merged over ``metrics``. There is no ``train_metrics``: a
            ``Trainer``'s ``compute_metrics`` is its EVAL hook, so the slot would accept a wired
            set and silently never call it.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. A ``Trainer`` writes ``train_*`` and
            ``eval_*``; the default translates those to the canonical ``train/*`` and ``val/*``
            for the trackers and the sidecar.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``training_arguments``,
    ``trainer``, ``test_loader``, ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Partial[HuggingFaceNetwork]],
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
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
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        callbacks: Optional[List[Partial[Any]]] = None,
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "mask",
        metrics: Optional[List[HuggingFaceMetric]] = None,
        val_metrics: Optional[List[HuggingFaceMetric]] = None,
        test_metrics: Optional[List[HuggingFaceMetric]] = None,
        metric_names: Optional[Partial[MetricNames]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            report_to=report_to,
            output_dir=output_dir,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metrics=metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)

    # ---- setup: num_classes is the task base's; the ignore index is ours -------------------- #
    def _resolve_task_defaults(self) -> None:
        """The shared resolution, plus this engine's one placement duty.

        Riding this hook rather than overriding ``initialize`` (the fleet rule: a concrete on
        a shared base extends via hooks) is safe because the task base calls it AFTER the
        model is flowed, on both the fit and the eval path — exactly when the model's config
        exists to write onto; ``_apply_ignore_index`` is idempotent like the rest of the hook.
        """
        super()._resolve_task_defaults()
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

    def _default_hf_metrics(self) -> Optional[List[Any]]:
        return list(_DEFAULT_METRICS)

    # ---- data ------------------------------------------------------------------------------- #
    def _to_hf_batch(self, records: Sequence[Any]) -> Dict[str, torch.Tensor]:
        """One list of records → the mapping ``Trainer`` feeds the model.

        ``int64`` is named explicitly for the same reason as every other backend —
        albumentations returns int32 and the cross-entropy family rejects it.
        """
        batch = collate_records(list(records))
        return {
            "pixel_values": batch_tensor(batch, self.input_key),
            "labels": batch_tensor(batch, self.target_key, dtype=torch.int64),
        }

    # ---- metrics ---------------------------------------------------------------------------- #
    def _metrics_hook(self, split: str) -> Any:
        """Trainer's ``compute_metrics``, bound to ONE split's metric list.

        **The hook is where the upsample happens**, and that placement is deliberate: the model's
        logits are ``[N, C, H/4, W/4]`` while the labels are ``[N, H, W]``, so aligning them is a
        property of THIS engine, not of any metric. Doing it here means every metric — built-in or
        user-supplied — sees arrays that already agree, and none of them has to know that SegFormer
        emits at a reduced resolution.
        """
        metrics = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or list(_DEFAULT_METRICS)
        ignore_index, num_classes = self.ignore_index, self.num_classes

        def compute_metrics(eval_pred: Any) -> Dict[str, Any]:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
            logits = logits[0] if isinstance(logits, tuple) else logits
            targets = np.asarray(labels)
            upsampled = _upsample_logits(np.asarray(logits), targets.shape[-2:])
            confusion, batchwise = _partition_metrics(metrics)
            out: Dict[str, Any] = {}
            if confusion:
                if not num_classes:
                    raise RuntimeError("HuggingFaceSegmenter: a confusion-matrix metric needs num_classes.")
                # The channel axis is SECOND here (`[N, C, H, W]`, transformers' contract) and the
                # accumulator argmaxes the LAST, so it moves — the one line of this that is about
                # THIS engine rather than about the metric.
                matrix = zero_confusion(num_classes)
                accumulate_confusion(matrix, np.moveaxis(upsampled, 1, -1), targets, ignore_index)
                out.update({name: fn(matrix) for name, fn in confusion.items()})
            for metric in batchwise:
                fn = _as_metric(metric)
                name = getattr(fn, "__name__", str(metric)).lstrip("_")
                out[name] = float(fn(upsampled, targets, ignore_index))
            return out

        return compute_metrics

    # ---- predictions ------------------------------------------------------------------------- #
    def _predict_batch(self, network: Any, batch: Any, device: Any) -> Any:
        """The forward plus the upsample back to the INPUT resolution.

        The sink's consumer expects a prediction the size of the image it came from, and this
        engine is the only one that does not already produce that.
        """
        images = batch_tensor(batch, self.input_key, device)
        logits = network(pixel_values=images).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

    def _prediction_output(self, preds: Any) -> Any:
        """The batched :class:`~recordstream.SegmentationOutput` — the machinery slices it."""
        return segmentation_output(preds)


def _upsample_logits(logits: np.ndarray, size: Any) -> np.ndarray:
    """Resize ``[N, C, h, w]`` logits to ``[N, C, H, W]`` — the engine's one structural quirk.

    Bilinear, matching what the model does internally for its own loss, so the metric scores the
    same prediction the loss was computed against. A no-op when the sizes already agree, which
    keeps this correct for a Hub model that emits at full resolution.
    """
    if tuple(logits.shape[-2:]) == tuple(size):
        return logits
    resized = F.interpolate(torch.as_tensor(logits), size=tuple(size), mode="bilinear", align_corners=False)
    return resized.numpy()


#: Metric NAMES this backend resolves — derived from the split's CONFUSION MATRIX.
#:
#: Unlike the JAX and MLX loops, this backend was never batch-averaging: `batch_eval_metrics` is
#: False by default, so a `Trainer` gathers the WHOLE split and calls `compute_metrics` ONCE
#: (verified) — the hand-written numpy that used to live here agreed with these to ten decimals on
#: real arrays. What the swap buys is therefore not a fix but the two things sharing an
#: implementation buys: agreement with the torch backends' torchmetrics BY CONSTRUCTION rather than
#: by two implementations happening to match, and `iou_per_class` — the per-class VECTOR the other
#: two engines already offered and this one could not.
#:
#: Plain functions over a matrix, so there is nothing for confluid to tag; see the module docstring.
_METRICS: Dict[str, Callable[[np.ndarray], Any]] = {
    "mean_iou": mean_iou,
    "pixel_accuracy": overall_accuracy,
    "iou_per_class": iou_per_class,
}

#: What an unset `metrics:` computes.
_DEFAULT_METRICS = ("mean_iou", "pixel_accuracy")


def _partition_metrics(metrics: Sequence[HuggingFaceMetric]) -> Any:
    """Split a metric list into the CONFUSION-based ones and the per-batch callables.

    By NAME, before resolution, which is why only a NAMED metric can be confusion-based: nothing
    here can know what a user-supplied callable would want accumulated, so one keeps the
    ``(logits, labels, ignore_index)`` protocol it was written against.
    """
    confusion: Dict[str, Callable[[np.ndarray], Any]] = {}
    batchwise: List[HuggingFaceMetric] = []
    for metric in metrics:
        name = flow(metric)
        if isinstance(name, str) and name in _METRICS:
            confusion[name] = _METRICS[name]
        else:
            batchwise.append(metric)
    return confusion, batchwise


def _as_metric(value: HuggingFaceMetric) -> Callable[..., float]:
    """A metric CALLABLE — the per-batch kind, over ``(logits, labels, ignore_index)``.

    A NAME in :data:`_METRICS` never reaches here: :func:`_partition_metrics` routes it to the
    confusion accumulator, whose functions take a MATRIX. What is left is resolving a user-supplied
    callable and refusing an unknown name, naming the ones that exist.
    """
    built = flow(value)
    if callable(built):
        return cast(Callable[..., float], built)
    metric = _METRICS.get(str(built))
    if metric is None:
        raise ValueError(f"Unknown metric {built!r}. Known names: {', '.join(sorted(_METRICS))}.")
    return metric

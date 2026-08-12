"""``FastaiSegmenter`` — the same segmentation run with a fastai ``Learner``.

The fastai member of the segmentation fleet, assembled from the workspace trainer bases
(2026-08-07): the Learner machinery — the ``DataLoaders``/``Learner`` assembly, the positional
``_to_tuple`` collate (built on the task's own int64-mask ``_targets`` hook), the Recorder
read-back into canonical names, the plain-function tracking hookup — is
``matrainer.fastai.FastaiRunnable``, and the segmentation task — ``num_classes`` resolution,
the ``ignore_index`` semantics — is the SAME :mod:`deltaid.segmentation_base` +
:mod:`deltaid.segmentation_torch` pair every torch-family backend runs on. What remains HERE is
the config surface (the full flat constructor — the confluid introspection reads THIS
signature) and the task's fastai VOCABULARY, which carries the two measured fastai facts:

1. **The loss needs ``axis=1``, and fastai's default is WRONG for segmentation.**
   ``CrossEntropyLossFlat`` defaults to ``axis=-1``, which flattens the CHANNEL dimension into
   the batch and dies with *"Expected input batch_size (48) to match target batch_size (128)"*
   (measured). With ``axis=1`` it agrees with ``torch.nn.CrossEntropyLoss`` to seven decimals.
   The DEFAULT states it because nothing else in a config hints at it.
2. **The metrics need no such fix** — ``DiceMulti`` and ``JaccardCoeffMulti`` already default
   to ``axis=1`` because they exist FOR segmentation; only the loss, shared with
   classification, carries the wrong default.

**One name is CONFIGURED rather than resolved — ``pixel_accuracy``** (:class:`PixelAccuracy`):
fastai's ``accuracy`` argmaxes over ``axis=-1`` (the WIDTH of an ``[N, C, H, W]`` prediction)
and ``foreground_acc`` scores one class, so the quantity every other engine reports was
unreachable by name. It wraps nothing — it is the ACCUMULATOR fastai does not ship.

Run it with ``deltaid train cfg.yaml --framework fastai``.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
from confluid import Mandatory, Partial, configurable
from fastai.callback.core import Callback as FastaiCallback
from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import Metric
from fastai.torch_core import flatten_check
from loggair import get_logger
from matrainer.fastai import FastaiLoss, FastaiMetric, FastaiOptimizer, FastaiRunnable
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker
from recordstream import RecordSource
from recordstream.runnable import RunnableTask

from deltaid.segmentation_torch import SegmentationTorchHooks

logger = get_logger(__name__)

#: Which entry point :meth:`FastaiSegmenter.run` dispatches to (the standard four).
SegmenterTask = RunnableTask

#: The per-pixel default set — both already default to ``axis=1``, unlike the loss.
DEFAULT_METRICS = ("DiceMulti", "JaccardCoeffMulti")

__all__ = ["DEFAULT_METRICS", "FastaiMetric", "FastaiOptimizer", "FastaiSegmenter", "PixelAccuracy", "SegmenterTask"]


@configurable(task="segmentation", role="trainer", framework="torch")
class FastaiSegmenter(SegmentationTorchHooks, FastaiRunnable):
    """Train, evaluate or predict a semantic segmenter with a fastai ``Learner``.

    Args:
        model: Segmentation model emitting ``[N, num_classes, H, W]`` logits — wire ``!lazy:``
            so :meth:`initialize` can inject the resolved ``num_classes`` at run time.
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``).
        num_classes: Class count, background included. Omitted, it is derived from the model or
            the mask column.
        ignore_index: Pixel value skipped by the default loss (and the ``pixel_accuracy``
            alias).
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
        metrics: fastai metrics (callables or names), computed on both passes of a fit.
        train_metrics / val_metrics / test_metrics: That split's OWN metrics. ``test_metrics``
            merges over ``metrics`` and NARROWS; ``train_metrics`` and ``val_metrics`` are
            UNIONED, because a fitting ``Learner`` holds ONE list and reports it on both passes.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. The ``Recorder``'s columns are
            ``train_loss`` / ``valid_loss`` / the metrics; the default translates them to the
            canonical ``train/loss`` / ``val/loss`` / ``val/<metric>``, an unprefixed column
            landing under ``val`` because that is the split a ``Learner`` measures.
        callbacks: fastai callbacks.
        trackers: matrainer trackers — one or a list.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the per-pixel mask.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__``, so nothing here NEEDS wiring; a same-named top-level YAML key replaces one
    via broadcasting when you want to: ``val_set``, ``train/val/test_loader``, ``dataloaders``,
    ``learner``, ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Partial[torch.nn.Module]],
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
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
        loss: Optional[Partial[FastaiLoss]] = None,
        metrics: Optional[List[FastaiMetric]] = None,
        train_metrics: Optional[List[FastaiMetric]] = None,
        val_metrics: Optional[List[FastaiMetric]] = None,
        test_metrics: Optional[List[FastaiMetric]] = None,
        metric_names: Optional[Partial[MetricNames]] = None,
        callbacks: Optional[List[Partial[FastaiCallback]]] = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "mask",
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            trackers=trackers,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            one_cycle=one_cycle,
            opt_func=opt_func,
            loss=loss,
            callbacks=callbacks,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.class_names: Optional[List[str]] = None
        self.plot_metrics = False

    # ---- the task's fastai vocabulary ------------------------------------------------------- #
    def _default_fastai_loss(self) -> Any:
        """``CrossEntropyLossFlat(axis=1)`` — fastai's loss with the axis segmentation needs.

        The wrapper's own default (``axis=-1``) flattens the CHANNEL dimension of an
        ``[N, C, H, W]`` prediction into the batch (measured; the module docstring records the
        failure); with ``axis=1`` it agrees with ``torch.nn.CrossEntropyLoss`` to seven
        decimals. Stated in the DEFAULT because nothing in a config hints at it.
        """
        return CrossEntropyLossFlat(axis=1, ignore_index=self.ignore_index)

    def _default_fastai_metrics(self) -> List[FastaiMetric]:
        # Both already default to `axis=1` — they exist for segmentation, so only the loss
        # (shared with classification) carries the wrong default.
        return list(DEFAULT_METRICS)

    def _metric_aliases(self) -> Dict[str, Callable[[], Any]]:
        return {"pixel_accuracy": lambda: PixelAccuracy(ignore_index=self.ignore_index)}

    def _metric_vocabulary_hint(self) -> str:
        return "For segmentation: DiceMulti, JaccardCoeffMulti, foreground_acc, Dice, JaccardCoeff."


class PixelAccuracy(Metric):
    """Per-pixel accuracy over the NON-IGNORED pixels — the one metric fastai has no name for.

    ``fastai.metrics.accuracy`` argmaxes over ``axis=-1``, which on an ``[N, C, H, W]`` prediction
    is the WIDTH, and ``foreground_acc`` scores one class rather than all pixels. So the quantity
    every other engine here reports as ``pixel_accuracy`` was simply unreachable by NAME — the one
    blank column in the backend sweep's table.

    **Why a CLASS when the module docstring says metrics are names.** That rule is about not
    wrapping an upstream metric to make it registrable, and this wraps nothing: it is the
    ACCUMULATOR fastai does not ship for this quantity. `AvgMetric(partial(accuracy, axis=1))`
    would have been one line and averages per batch WEIGHTED BY BATCH SIZE, which is the right
    answer only while every image contributes the same number of counted pixels — untrue the
    moment ``ignore_index`` drops a varying number of them. Accumulating counts and dividing once
    is exact, and it is precisely the shape fastai's own ``DiceMulti`` uses for the same reason.

    The name is ``pixel_accuracy`` because that is what the other seven engines call it; a metric
    reported under its own class name would open a column no other row fills.
    """

    def __init__(self, axis: int = 1, ignore_index: int = -100) -> None:
        self.axis = axis
        self.ignore_index = ignore_index

    def reset(self) -> None:
        self.correct, self.total = 0, 0

    def accumulate(self, learn: Any) -> None:
        predicted, targets = flatten_check(learn.pred.argmax(dim=self.axis), learn.y)
        valid = targets != self.ignore_index
        self.correct += int(((predicted == targets) & valid).sum().item())
        self.total += int(valid.sum().item())

    @property
    def value(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def name(self) -> str:
        return "pixel_accuracy"

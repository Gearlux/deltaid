"""LightningSegmenter: a standard PyTorch-Lightning semantic segmenter (train + eval in one class).

The Lightning member of the segmentation fleet, assembled from the workspace trainer bases
(2026-08-06): the Lightning machinery — loaders, the deferred ``pl.Trainer`` slot, the step
functions, tracking — is ``matrainer.lightning.LightningRunnable``, and the segmentation task —
``num_classes`` resolution (model head → explicit → lazy kwarg → the announced mask walk), the
``ignore_index``-aware loss + metric defaults, the per-class vector reporting — is
:mod:`deltaid.segmentation_base` + :mod:`deltaid.segmentation_torch`. What remains HERE is only
the config surface: the full flat constructor (the confluid introspection reads THIS signature)
storing the task knobs.

Minimal flat YAML (top-level keys wired by Confluid broadcasting)::

    runnable: !class:deltaid.LightningSegmenter
      task: fit
    model: !lazy:deltaid.SegmentationModelPytorch { arch: unet, encoder_name: resnet34 }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    val_set:   !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 8
    max_epochs: 20

    deltaid train train.yaml        # or: recordstream run train.yaml
"""

from typing import Iterable, List, Optional, Union

import torch.nn as nn
from confluid import Mandatory, Partial, configurable
from loggair import get_logger
from matrainer.lightning import LightningRunnable
from matrainer.metric_names import MetricNames
from matrainer.torch.metrics import MetricsSpec
from matrainer.tracking.base import Tracker
from recordstream import RecordSource
from recordstream.runnable import RunnableTask

from deltaid.segmentation_torch import SegmentationTorchHooks

logger = get_logger(__name__)

#: Which entry point :meth:`LightningSegmenter.run` dispatches to (the standard four).
SegmenterTask = RunnableTask


@configurable(task="segmentation", role="trainer", framework="torch")
class LightningSegmenter(SegmentationTorchHooks, LightningRunnable):
    """A standard Lightning semantic segmenter — train, evaluate, test, or predict via ``task``.

    Args:
        model: Segmentation model emitting ``[N, num_classes, H, W]`` logits (wire ``!lazy:`` so
            :meth:`initialize` flows it with the resolved num_classes at run time).
        train_set: Training dataset / Stream (for ``task="fit"``).
        test_set: Held-out eval dataset / Stream (for ``task="test"`` / ``"predict"``; also tested
            after ``fit`` when set).
        num_classes: Output class count, background included; omit to resolve from the model or
            the mask column.
        ignore_index: Pixel value skipped by the loss AND the metrics — the standard way to
            exclude an "unlabelled" / "void" class. ``-100`` (torch's own default) means no pixel
            is excluded, since a class-id mask never contains it.
        task: Entry point :meth:`run` dispatches to — ``"fit"`` / ``"evaluate"`` / ``"test"`` / ``"predict"``.
        max_epochs / batch_size / num_workers: ergonomic flat-config knobs.
        trackers: matrainer trackers — one or a list. They ARE this run's metric sinks: each is
            adapted to Lightning's logger interface, so there is no separate ``logger:`` slot of
            ``pytorch_lightning.loggers``.
        experiment_name / run_name: tracking identity.
        input_key: Record key carrying the model input image (default ``"image"``).
        target_key: Record key carrying the per-pixel mask (default ``"mask"`` — the key
            ``ConvertToMask`` writes, which is also albumentations' own vocabulary).
        metrics: Single-metric templates flowed per split (default mean IoU + macro Dice).
        train_metrics / val_metrics / test_metrics: That split's OWN templates, merged over
            ``metrics`` — how a metric is added to one split only (a per-class IoU on validation).
            Each slot holds the spec until :meth:`initialize` builds it, and the live prefixed
            ``MetricCollection`` after.
        class_names: Labels for the per-class metric report, in class-id order. Omitted, the
            classes are named by INDEX — which is readable but not informative, so a run that
            wires a ``MulticlassJaccardIndex(average=None)`` usually wants these too. Unlike
            classification there is no label map to derive them from: a mask carries ids, not
            names.
        plot_metrics: ALSO send each per-class metric to the trackers as a matplotlib bar chart,
            not only as a table. Off by default: it makes matplotlib (``matrainer[plot]``) a cost
            of every eval epoch, where the table costs nothing.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. The ``self.log`` calls already write
            the canonical ``<split>/<name>``, so the default translates nothing; it is handed to
            each ``TrackerLogger``, where a wired ``target:`` records the same run in another
            engine's spelling.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__`` (the task-shaped ones, ``loss_fn`` + the metric slots, resolve to this task's
    defaults in ``initialize()``), so nothing here NEEDS wiring; a same-named top-level YAML key
    replaces one via broadcasting when you want to. ``val_set`` is the one a config typically
    does wire (there is no sensible default validation split): ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``lightning``, ``loss_fn``, ``predictions_sink``,
    ``metrics_sidecar_path``, ``log_system_metrics``.
    """

    def __init__(
        self,
        model: Mandatory[Partial[nn.Module]],
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        ignore_index: int = -100,
        task: SegmenterTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "mask",
        metrics: MetricsSpec = None,
        train_metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        class_names: Optional[List[str]] = None,
        plot_metrics: bool = False,
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
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics


__all__ = ["LightningSegmenter", "MetricsSpec", "SegmenterTask"]

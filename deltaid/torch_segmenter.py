"""``TorchSegmenter`` — the same segmentation run as a PLAIN PyTorch training loop.

The control-experiment member of the segmentation fleet, assembled from the workspace trainer
bases (2026-08-06): the hand-written loop machinery — device bracketing, the epoch driver,
scoring, checkpointing, the plain-function tracking hookup — is
``matrainer.torch.TorchLoopRunnable``, and the segmentation task is the SAME
:mod:`deltaid.segmentation_base` + :mod:`deltaid.segmentation_torch` pair the Lightning backend
runs on — which is the control experiment executed structurally: the two backends now share the
task by construction, so "swap the runnable, keep the config" cannot drift. What remains HERE is
only the config surface: the full flat constructor (the confluid introspection reads THIS
signature) storing the task knobs.

Minimal flat YAML — the Lightning block with one key changed::

    runnable: !class:deltaid.TorchSegmenter
    model: !lazy:deltaid.SegmentationModelPytorch { arch: unet }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    num_classes: 3
    max_epochs: 5

Run it with ``deltaid train cfg.yaml --framework torch``.
"""

from typing import Iterable, List, Optional, Union

import torch.nn as nn
from confluid import Mandatory, Partial, configurable
from loggair import get_logger
from matrainer.metric_names import MetricNames
from matrainer.torch import Device, TorchLoopRunnable
from matrainer.torch.metrics import MetricsSpec
from matrainer.tracking.base import Tracker
from recordstream import RecordSource
from recordstream.runnable import RunnableTask

from deltaid.segmentation_torch import SegmentationTorchHooks

logger = get_logger(__name__)

#: Which entry point :meth:`TorchSegmenter.run` dispatches to (the standard four).
SegmenterTask = RunnableTask

__all__ = ["Device", "MetricsSpec", "SegmenterTask", "TorchSegmenter"]


@configurable(task="segmentation", role="trainer", framework="torch")
class TorchSegmenter(SegmentationTorchHooks, TorchLoopRunnable):
    """Train, evaluate or predict a semantic segmenter with a hand-written PyTorch loop.

    Args:
        model: Segmentation model emitting ``[N, num_classes, H, W]`` logits — wire ``!lazy:`` so
            :meth:`initialize` can inject the resolved ``num_classes`` at run time.
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``; also tested after a fit).
        num_classes: Class count, background included. Omitted, it is derived from the model or
            the mask column.
        ignore_index: Pixel value skipped by the loss AND the metrics — the standard way to
            exclude an "unlabelled" / "void" class.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        num_workers: DataLoader worker processes. Persistent workers follow from this, exactly as
            in the Lightning backend — there is no separate knob.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the per-pixel mask.
        metrics: torchmetrics templates, flowed per split. ``None`` picks mean IoU + macro Dice.
        train_metrics / val_metrics / test_metrics: That split's OWN templates, merged over
            ``metrics`` — how a metric is added to one split only (a per-class IoU on validation).
        device: Where to run — see :data:`Device`.
        checkpoint_dir: Directory for ``model.pt`` (a plain ``state_dict``). ``None`` writes no
            weights, which is what a smoke run wants.
        grad_clip: Clip gradients to this global norm before each ``step()``. ``None`` disables.
        trackers: matrainer trackers — one or a list.
        class_names: Labels for the per-class metric report, in class-id order; indices otherwise.
        plot_metrics: ALSO send each per-class metric to the trackers as a matplotlib bar chart.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. This loop already writes the canonical
            ``<split>/<name>``, so the default translates nothing; wiring ``target:`` is what
            makes it record in another engine's spelling.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__`` (the task-shaped ones, ``loss_fn`` + the metric slots, resolve to this task's
    defaults in ``initialize()``), so nothing here NEEDS wiring; a same-named top-level YAML key
    replaces one via broadcasting when you want to: ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``loss_fn``, ``predictions_sink``, ``metrics_sidecar_path``.
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
        experiment_name: str = "deltaid_segmentation",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "mask",
        metrics: MetricsSpec = None,
        train_metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        device: Device = "auto",
        checkpoint_dir: Optional[str] = None,
        grad_clip: Optional[float] = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
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
            device=device,
            checkpoint_dir=checkpoint_dir,
            grad_clip=grad_clip,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.ignore_index = int(ignore_index)
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics

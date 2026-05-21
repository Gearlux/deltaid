"""LightningSegmentationEvaluator: specialised semantic-segmentation evaluator.

Subclass of :class:`~marainer.evaluator.LightningEvaluator` that bakes in
segmentation-specific defaults so a minimal YAML config only needs to specify
the dataset, model, and number of classes — no adapter wiring for collate or
metrics required.

Baked-in defaults (all overridable via YAML ``@configurable`` attributes):

* ``loader`` ``collate_fn`` —
  :func:`~deltaid.segmentation.segmentation_collate_fn`.
* ``metrics`` — :class:`~torchmetrics.classification.MulticlassJaccardIndex`
  (mean IoU) + :class:`~torchmetrics.classification.MulticlassF1Score`
  (Dice-equivalent, macro). ``ignore_index`` propagates so don't-care pixels
  are excluded from accumulation.

No deltaid-owned metric adapter exists; override directly with any
``torchmetrics`` class or ``MetricCollection`` from YAML.

Minimal YAML example::

    evaluator: !class:deltaid.evaluator.LightningSegmentationEvaluator
      num_classes: 2
      model: !class:deltaid.deeplabv3plus.DeepLabV3PlusModel
        num_classes: !ref:num_classes
        checkpoint_path: ./checkpoints/last.ckpt
      dataset: !ref:val_set
"""

from typing import Any, Optional

import pytorch_lightning as L
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from confluid import Class, Lazy, configurable
from deltaid.segmentation import segmentation_collate_fn
from deltaid.trainer import _default_segmentation_metrics
from marainer.evaluator import LightningEvaluator


@configurable
class LightningSegmentationEvaluator(LightningEvaluator):
    """Specialised semantic-segmentation evaluator with baked-in defaults.

    Args:
        model: Segmentation model that emits ``[N, num_classes, H, W]`` logits.
            The model is responsible for loading its own weights — pass
            ``checkpoint_path`` to the model class.
        dataset: Evaluation dataset or Flux pipeline.
        num_classes: Number of output classes including background; drives
            metric construction.
        ignore_index: Pixel value in ``target`` that is skipped during metric
            accumulation. Propagates to every default metric. When the user
            overrides ``metrics`` via YAML, the upstream class is responsible
            for its own ``ignore_index``.
        loader: Deferred DataLoader (default includes
            ``segmentation_collate_fn``).
        predictions_sink: Optional sink with ``.write(prediction, metadata)``
            and ``.close()``.
        lightning: PyTorch Lightning Trainer config.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        metrics_sidecar_path: Optional path; when set, writes a logger-agnostic
            JSON dump of the final metrics after evaluation completes.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        num_classes: int,
        ignore_index: int = -100,
        loader: Lazy[Any] = Class(DataLoader, collate_fn=segmentation_collate_fn),
        predictions_sink: Any = None,
        lightning: Any = Class(L.Trainer),
        experiment_name: str = "deltaid_segmentation_eval",
        run_name: str = "evaluate",
        metrics_sidecar_path: Optional[str] = None,
    ) -> None:
        metrics: MetricCollection = _default_segmentation_metrics(
            num_classes, ignore_index
        ).clone(prefix="test/")
        super().__init__(
            model=model,
            dataset=dataset,
            loader=loader,
            metrics=metrics,
            predictions_sink=predictions_sink,
            lightning=lightning,
            experiment_name=experiment_name,
            run_name=run_name,
            metrics_sidecar_path=metrics_sidecar_path,
        )
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)

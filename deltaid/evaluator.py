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

from typing import Any

from confluid import LazyClass, configurable
from marainer.evaluator import LightningEvaluator
from torch.utils.data import DataLoader

from deltaid.segmentation import segmentation_collate_fn
from deltaid.trainer import _default_segmentation_metrics


@configurable(task="segmentation", role="evaluator")
class LightningSegmentationEvaluator(LightningEvaluator):
    """Specialised semantic-segmentation evaluator with baked-in defaults.

    Mirrors :class:`~marainer.evaluator.LightningEvaluator`'s minimal-ctor
    discipline: only ``model`` / ``dataset`` / ``num_classes`` (+ identity
    ``ignore_index`` / ``experiment_name`` / ``run_name``). The ``test/`` metric
    set and the ``segmentation_collate_fn`` loader are body attributes;
    ``predictions_sink`` / ``lightning`` / ``metrics_sidecar_path`` inherit the
    base defaults and are configured post-construction (YAML / broadcasting; a
    deferred slot uses ``!lazy:``).

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
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        num_classes: int,
        ignore_index: int = -100,
        experiment_name: str = "deltaid_segmentation_eval",
        run_name: str = "evaluate",
    ) -> None:
        super().__init__(model=model, dataset=dataset, experiment_name=experiment_name, run_name=run_name)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.metrics = _default_segmentation_metrics(self.num_classes, self.ignore_index).clone(prefix="test/")
        self.loader = LazyClass(DataLoader, collate_fn=segmentation_collate_fn)

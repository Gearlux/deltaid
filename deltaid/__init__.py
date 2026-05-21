"""Deltaid: semantic-segmentation models and adapters for the Marainer training framework."""

from deltaid.deeplabv3plus import DeepLabV3PlusModel
from deltaid.evaluator import LightningSegmentationEvaluator
from deltaid.segmentation import segmentation_collate_fn
from deltaid.trainer import LightningSegmentationTrainer

__version__ = "0.1.0"

__all__ = [
    "DeepLabV3PlusModel",
    "LightningSegmentationEvaluator",
    "LightningSegmentationTrainer",
    "segmentation_collate_fn",
]

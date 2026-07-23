"""Deltaid: a standard PyTorch-Lightning semantic segmenter, using marainer for optional tracking."""

from deltaid.deeplabv3plus import DeepLabV3PlusModel
from deltaid.segmentation import segmentation_collate_fn
from deltaid.segmenter import LightningSegmenter

__version__ = "0.2.0"

__all__ = [
    "DeepLabV3PlusModel",
    "LightningSegmenter",
    "segmentation_collate_fn",
]

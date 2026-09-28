"""``SegmentationBase`` — the task half of every deltaid backend, written once.

The per-pixel logic that used to exist per backend module: ``num_classes`` resolution (model
head → explicit → lazy kwarg → the announced mask walk), the ``ignore_index`` semantics, the
fork-safe split preparation, and the fit/eval initialization split.

It subclasses the trainer API base (`matrainer.runnable.TrainerRunnable`) and is deliberately
**torch-free**: a keras/jax/mlx backend imports this module too, and the core-is-backend-free
mandate forbids it dragging a framework in. The torch-family hook IMPLEMENTATIONS (mask dtype,
default loss/metrics, per-class reporting) live in :mod:`deltaid.segmentation_torch`; a
concrete backend composes ``(SegmentationTorchHooks, <Framework>Runnable)``.
"""

from typing import List, Optional

from confluid import Fluid, flow
from loggair import get_logger
from matrainer.runnable import TrainerRunnable
from recordstream import num_mask_classes

logger = get_logger(__name__)

__all__ = ["SegmentationBase"]


class SegmentationBase(TrainerRunnable):
    """The segmentation task base: ``num_classes`` resolution + the fit/eval initialization.

    The attributes below are STORED by the concrete backend's own ``__init__`` (they are its
    config surface); this base only declares their names/types for the methods that read them.
    """

    log_label = "deltaid"

    num_classes: Optional[int]
    #: Pixel value skipped by the loss AND the metrics — the standard way to exclude an
    #: "unlabelled" / "void" class. torch's own default (-100) means no pixel is excluded.
    ignore_index: int
    #: Labels for the per-class metric report, in class-id order. Unlike classification there
    #: is no label map to derive them from: a mask carries ids, not names.
    class_names: Optional[List[str]]
    plot_metrics: bool

    def initialize(self) -> None:
        """Resolve ``num_classes``, then flow the model + the loss + the metrics with it."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()

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
            # The last resort, and the expensive one: decode the mask column to find the
            # largest class id. Announced because it is a full pass over the targets — for
            # segmentation the count is usually a property of the LABEL SCHEME, so setting it
            # explicitly is both cheaper and more honest about a class that appears in no
            # training image.
            logger.info(f"Deriving num_classes from the {self.target_key!r} column (a full pass over the masks)...")
            nc = num_mask_classes(self.train_set, self.target_key)
            logger.info(f"Derived num_classes={nc} from the mask column.")
        self.num_classes = int(nc)

        self.model = flow(self.model, num_classes=self.num_classes)
        self._resolve_task_defaults()
        self._build_fit_metrics()

    def _initialize_eval(self) -> None:
        # The trained model's head is authoritative for eval — flow it first, read num_classes
        # off it. NOT the eval dataset, which may be a subset whose masks never show the last
        # class and would mis-size the metrics against the head.
        self.model = flow(self.model)
        nc = getattr(self.model, "num_classes", None) or self.num_classes
        if not nc:
            raise RuntimeError("could not determine num_classes from the model (it must expose `num_classes`).")
        self.num_classes = int(nc)
        self._resolve_task_defaults()
        eval_slot = "val_set" if self.task == "evaluate" else "test_set"
        setattr(self, eval_slot, self._prepare(getattr(self, eval_slot)))
        self._build_eval_metrics()

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

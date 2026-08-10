"""``HuggingFaceSegmentationModel`` — an ``AutoModelForSemanticSegmentation`` behind the model contract.

The Hub is the zoo here, so this is the same shape as :class:`deltaid.smp_model.SegmentationModelPytorch`
with a different builder underneath: ``__init__`` stores only, the network is built on first use by
:meth:`solidify` (confluid's ``flow()``-time hook), and required configuration is validated at
build time so zero-arg construction keeps working.

Three things are specific to the Hub and are why this class exists rather than a config wiring
``transformers`` directly:

* **``num_classes`` arrives at run time.** A pretrained checkpoint carries its own head — 150
  ADE20K classes for the default SegFormer — so pointing it at a 3-class dataset is a shape
  mismatch. The build passes ``num_labels`` plus ``ignore_mismatched_sizes=True``, which tells
  transformers to DISCARD the pretrained head and start a fresh one at the requested width.
  Without it the load raises; with it, fine-tuning works the way the Hub intends (measured: the
  load report shows ``decode_head.classifier`` reinitialised 150 -> 3).
* **The model computes its own loss**, from ``labels=``, using ``config.semantic_loss_ignore_index``
  — which defaults to **255**, NOT torch's ``-100``. So the segmenter writes its own
  ``ignore_index`` onto the config; see :mod:`deltaid.huggingface_segmenter`.
* **The logits come back at a REDUCED resolution.** SegFormer emits ``[N, C, H/4, W/4]``
  (measured: a 224x224 input gives 56x56), and it upsamples internally only for its own loss.
  Anything else that compares logits to the mask — a metric, a predictions sink — has to
  upsample first, which the segmenter does.

``model_name`` is a Hub id (``"nvidia/segformer-b0-finetuned-ade-512-512"``), the same string you
would pass to ``from_pretrained``.
"""

from typing import Any, List, Optional

from confluid import configurable
from loggair import get_logger
from matrainer.uri import ModelRef, hf_model_ref

logger = get_logger(__name__)

__all__ = ["HuggingFaceSegmentationModel"]


@configurable(task="segmentation", role="model", lazy=True, framework="huggingface")
class HuggingFaceSegmentationModel:
    """A Hub semantic-segmentation checkpoint with a head sized to your dataset.

    Args:
        model_name: A Hub model id, e.g. ``"nvidia/segformer-b0-finetuned-ade-512-512"`` or
            ``"nvidia/mit-b0"``. Sentinel ``""`` is invalid but permitted so the class is
            zero-arg constructible; it is rejected at build time with a clear error.
        num_classes: Width of the per-pixel head. Sentinel ``0`` likewise — a segmenter injects
            the resolved count via ``flow(model, num_classes=N)`` at run time, so it is not
            written in YAML.
        pretrained: Load the checkpoint's weights. ``False`` builds the same architecture from
            its config with fresh weights, which keeps a first run offline and reproducible.
        trust_remote_code: Allow a Hub repo to execute its own modelling code. Off by default
            because it runs code from the internet; turn it on only for a repo you trust.
        seed: Seed for torch's global RNG, set when the model is BUILT — a pretrained load
            still randomly reinitializes the ``num_labels`` head, and ``TrainingArguments.seed``
            applies only when training starts, too late to cover it. Applied WITHOUT restoring
            the RNG state (the workspace wrapper-seeding idiom).
    """

    def __init__(
        self,
        model_name: str = "",
        num_classes: int = 0,
        pretrained: bool = True,
        trust_remote_code: bool = False,
        seed: int = 0,
    ) -> None:
        # Lazy: store only. No Hub call, no download — the workspace zero-arg mandate.
        self.model_name = model_name
        self.num_classes = int(num_classes)
        self.pretrained = pretrained
        self.trust_remote_code = trust_remote_code
        self.seed = int(seed)
        self._model: Optional[Any] = None

    def solidify(self) -> Any:
        """Build (once) and return the live ``transformers`` model."""
        if self._model is not None:
            return self._model
        if not self.model_name:
            raise ValueError("HuggingFaceSegmentationModel.model_name is required (got empty).")
        if self.num_classes < 1:
            raise ValueError(
                f"HuggingFaceSegmentationModel.num_classes must be >= 1 to build (got {self.num_classes}) — "
                "set it explicitly, or let the segmenter derive it."
            )
        import torch
        from transformers import AutoConfig, AutoModelForSemanticSegmentation

        # Seeded BEFORE the build (the restoration project's 2026-08-06 idiom): a pretrained
        # load still randomly reinitializes the `num_labels` head (`ignore_mismatched_sizes`),
        # and `TrainingArguments.seed` applies only when TRAINING starts — too late to cover
        # it (the detection project measured a 3.4x metric spread from exactly this ordering).
        torch.manual_seed(self.seed)

        if self.pretrained:
            # `ignore_mismatched_sizes` is what makes fine-tuning possible at all: the checkpoint's
            # head is sized for ITS dataset, and without this the load raises on the mismatch
            # instead of reinitialising the head.
            self._model = AutoModelForSemanticSegmentation.from_pretrained(
                self.model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
                trust_remote_code=self.trust_remote_code,
            )
        else:
            config = AutoConfig.from_pretrained(
                self.model_name, num_labels=self.num_classes, trust_remote_code=self.trust_remote_code
            )
            self._model = AutoModelForSemanticSegmentation.from_config(config)
        logger.info(f"Built {self.model_name} with a {self.num_classes}-class head (pretrained={self.pretrained}).")
        return self._model

    @property
    def model(self) -> Any:
        """The live ``transformers`` model, built on first access."""
        return self.solidify()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the live model, so the wrapper is callable wherever the model is."""
        return self.model(*args, **kwargs)

    @property
    def model_refs(self) -> List[ModelRef]:
        """The weights this wrapper loads, for the run manifest's ``models`` section.

        Reads STORED CONFIG ONLY — never builds, downloads or opens anything, because the
        property is asked of a wrapper that may never run. An EMPTY list when
        ``pretrained=False``: random weights referenced no artifact, and the architecture is
        already recorded in the run's params, so naming one would name something nobody loaded.
        """
        if not self.model_name or not self.pretrained:
            return []
        return [hf_model_ref(self.model_name, role="pretrained")]

"""smp-backed semantic-segmentation wrapper — every architecture in the library, one class.

Wraps :func:`segmentation_models_pytorch.create_model` into a plain :class:`torch.nn.Module`
whose ``forward(images) -> logits`` signature plugs straight into
:class:`deltaid.lightning_segmenter.LightningSegmenter`, which calls ``preds = self(x)`` and
``loss = self.loss_fn(preds, y)`` in its ``training_step`` — for segmentation that is all the glue
needed. No bundled batch type, no step-output dict, no per-mode forward divergence.

**One wrapper, twelve architectures.** This class replaced a `DeepLabV3PlusModel` that named
``smp.DeepLabV3Plus`` directly (2026-08-02). The two differ in exactly one line of body —
``create_model(arch, ...)`` instead of ``DeepLabV3Plus(...)`` — and everything else about them
was identical, so a per-architecture wrapper would have been eleven copies of this file
distinguished by a string. ``arch`` is that string, and it is a closed
:data:`SegmentationArchitecture` Literal so a picker, a form-spec and a StreamStudio widget
enumerate the choices instead of offering free text.

The wrapper exists for the same three reasons its classification twin (``sonair.TimmModel``) does:

1. **Confluid configurability** — ``@configurable`` exposes every smp constructor knob (the
   architecture, the encoder, its pretrained weights, and any decoder-specific keyword) for YAML
   wiring, with defaults that train out of the box.
2. **Checkpoint loading** — ``checkpoint_path`` strips Lightning's ``model.`` prefix via
   :func:`matrainer.torch.load_lightning_checkpoint`, so a previously trained model resumes or
   evaluates without the caller knowing about Lightning's wrapping.
3. **Generic across the zoo** — ``encoder_name`` accepts any of smp's ~800 encoders (ResNet,
   EfficientNet, MiT/SegFormer, MobileNet, ConvNeXt, Swin …), so one wrapper covers the library.

Typical wiring (YAML) — declare the model ``!lazy:`` so the segmenter injects the resolved
``num_classes`` at flow time (omit it from the config)::

    model: !lazy:deltaid.SegmentationModelPytorch
      arch: unet
      encoder_name: resnet34
      encoder_weights: imagenet

(Pin ``num_classes`` explicitly only to override the value the segmenter derives.)
"""

from typing import Any, List, Literal, Optional, cast, get_args
from urllib.parse import quote, urlencode

import segmentation_models_pytorch as smp
import torch.nn as nn
from confluid import configurable
from loggair import get_logger
from matrainer.torch import load_lightning_checkpoint
from matrainer.uri import ModelRef, file_model_ref
from torch import Tensor

logger = get_logger(__name__)

#: Scheme naming the smp zoo in a model reference. smp resolves an encoder's weights through
#: several upstream registries depending on the encoder, so the honest identifier is the
#: architecture + encoder + weight-set smp itself was asked for — see
#: :attr:`SegmentationModelPytorch.model_refs`.
SMP_URI_PREFIX = "smp://"

#: The decoder architectures smp ships, as a CLOSED set — so `to_pydantic` gives navigaitor's
#: form-spec and StreamStudio an enumerable widget rather than a free-text box, and a typo fails
#: at the call site instead of inside smp.
#:
#: Written out rather than derived from ``smp.MODEL_ARCHITECTURES_MAPPING``, because a `Literal`
#: cannot be built at runtime and still be a type. `tests/test_smp_model.py` asserts the two agree
#: in BOTH directions, so an smp release that adds an architecture fails here loudly instead of
#: leaving it silently unofferable.
SegmentationArchitecture = Literal[
    "unet",
    "unetplusplus",
    "manet",
    "linknet",
    "fpn",
    "pspnet",
    "pan",
    "deeplabv3",
    "deeplabv3plus",
    "upernet",
    "segformer",
    "dpt",
]

#: The runtime tuple, DERIVED from the Literal — never a second hand-written list.
SEGMENTATION_ARCHITECTURES = get_args(SegmentationArchitecture)


@configurable(task="segmentation", role="model", lazy=True, framework="torch")
class SegmentationModelPytorch(nn.Module):
    """A ``segmentation_models_pytorch`` model wrapped for the deltaid segmenter flow.

    ``forward(images) -> logits`` only — a segmentation model computes its loss outside the
    model, so the segmenter's standard ``loss_fn(preds, target)`` call works with no batch wrapper.

    **Lazy construction (Lazy-Init / Zero-Arg mandate).** ``__init__`` does NO functional work — it
    only stores config. The smp model (and any pretrained-weight download) is built on first use by
    :meth:`solidify` — called by :meth:`forward`, by confluid's auto-build on ``flow()``, or
    explicitly — and cached in ``self.segmenter``, so ``SegmentationModelPytorch()`` is cheap and
    does no network I/O. Tagged ``lazy=True`` so a config wires it ``!lazy:`` and the segmenter
    flows it with the resolved ``num_classes``. ``num_classes`` defaults to the unset sentinel
    ``0`` and is validated when the model is BUILT, never in ``__init__``.

    Args:
        arch: Decoder architecture — one of :data:`SEGMENTATION_ARCHITECTURES`. Defaults to ``unet``.
        encoder_name: smp encoder id — ``resnet34``, ``efficientnet-b0``, ``mit_b0``, ``tu-convnext_tiny``, ….
        encoder_weights: Pretrained encoder weights tag (``imagenet``); ``None`` trains the encoder from scratch.
        num_classes: Output channel count, background included. **Required to build** — ``0`` (the default) means unset.
        in_channels: Input channel count. Defaults to 3 (RGB).
        checkpoint_path: Optional Lightning ``.ckpt`` / state-dict to load after the model is built; ``""`` = none.
        kwargs: Any further smp constructor knob, forwarded VERBATIM (``decoder_channels``, ``encoder_depth``, …).

    Attributes:
        checkpoint_strict: Whether ``load_state_dict`` enforces exact key matching. Defaults to
            ``False`` so a re-headed ``num_classes`` does not reject the whole load.
        segmenter: Lazy cache for the built smp model (``None`` until :meth:`solidify`). A PUBLIC
            submodule name, so a state_dict saved through Lightning — which prefixes ``model.``,
            stripped by ``load_lightning_checkpoint`` — round-trips as ``segmenter.*`` keys that
            match this wrapper's layout. Naming it ``self.model`` would make the strip chew through
            this wrapper's own attribute name too, producing keys that match nothing.
    """

    def __init__(
        self,
        arch: SegmentationArchitecture = "unet",
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        num_classes: int = 0,
        in_channels: int = 3,
        checkpoint_path: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Store config ONLY — no create_model, no weight download, no checkpoint load. The model
        # is built lazily on first use (via `solidify()`, the cached `backbone` property, or
        # `forward()` — all three share the one cache), so this constructor is cheap and zero-arg
        # construction works (Lazy-Init / Zero-Arg mandate).
        self.arch: SegmentationArchitecture = arch
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_strict: bool = False
        # Forwarded verbatim to `create_model` at build time (the pass-through mandate: this
        # wrapper must not shadow smp's names or silently rewrite its defaults).
        self.model_kwargs = dict(kwargs)
        self.segmenter: Optional[nn.Module] = None

    @property
    def model_refs(self) -> List[ModelRef]:
        """The weight artifacts this model loads — see :mod:`matrainer.uri`.

        smp routes an encoder's pretrained weights through whichever upstream registry owns it
        (timm, torchvision, the library's own release assets), and which one that is depends on
        the encoder name — so resolving a single repository URL would need smp's internal tables
        and would be wrong for most of its ~800 encoders. What IS unambiguous is the triple smp
        was asked for, so that is the reference, with no browsable link rather than a guessed one.

        ``encoder_weights=None`` builds the encoder from scratch and loads nothing.
        """
        refs: List[ModelRef] = []
        if self.encoder_weights:
            selection = urlencode({"weights": str(self.encoder_weights)})
            refs.append(ModelRef(uri=f"{SMP_URI_PREFIX}{quote(self.arch)}/{quote(self.encoder_name)}?{selection}"))
        if self.checkpoint_path:
            refs.append(file_model_ref(self.checkpoint_path))
        return refs

    def solidify(self) -> nn.Module:
        """Build the smp model on first call (cached in ``self.segmenter``) and return it.

        Validates the required config lazily (``arch`` / ``num_classes`` — ``0`` means unset),
        builds through :func:`smp.create_model`, and loads ``checkpoint_path`` into it when set.
        Confluid's ``flow()`` calls this after construction, and :meth:`forward` calls it too, so
        the parameters exist before an optimizer reads them. Idempotent — later calls return the
        cache.
        """
        if self.segmenter is None:
            if self.arch not in SEGMENTATION_ARCHITECTURES:
                raise ValueError(
                    f"SegmentationModelPytorch.arch must be one of {list(SEGMENTATION_ARCHITECTURES)}, "
                    f"got {self.arch!r}."
                )
            if self.num_classes < 1:
                raise ValueError(
                    f"SegmentationModelPytorch.num_classes must be >= 1 to build the model "
                    f"(got {self.num_classes}) — set it explicitly, or let the segmenter derive it "
                    "from the dataset."
                )
            self.segmenter = smp.create_model(
                arch=self.arch,
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                in_channels=self.in_channels,
                classes=self.num_classes,
                **self.model_kwargs,
            )
            logger.debug(
                f"Built {self.arch} / {self.encoder_name} "
                f"(weights={self.encoder_weights}, classes={self.num_classes})."
            )
            if self.checkpoint_path:
                load_lightning_checkpoint(self, self.checkpoint_path, strict=self.checkpoint_strict)
        return cast(nn.Module, self.segmenter)

    @property
    def backbone(self) -> nn.Module:
        """The built smp model, built + cached on first access (confluid lazy-init rule 3).

        A read-only cached ``@property`` over the same build path as :meth:`solidify`, so a
        directly constructed, **never-flowed** instance still materializes on demand.
        ``solidify`` is retained as the explicit method confluid's ``flow()`` calls — ``flow``
        never touches property getters, so the method hook is what leaves ``self.parameters()``
        ready before ``configure_optimizers``. Both share the one cache.
        """
        return self.solidify()

    def forward(self, images: Tensor) -> Tensor:
        """Run the segmentation model and emit per-pixel class logits.

        Args:
            images: ``[N, in_channels, H, W]`` float tensor.

        Returns:
            ``[N, num_classes, H, W]`` raw logits.

        The spatial-size rule is deliberately NOT checked here. smp validates it itself in
        ``SegmentationModel.check_input_shape``, against the ENCODER's actual ``output_stride``
        and with the padded size to use in the message. A hardcoded ``% 32`` here — which this
        wrapper's DeepLabV3+-only predecessor had — is wrong twice over: it rejects valid input
        for the architectures that set ``requires_divisible_input_shape = False`` (SegFormer, DPT),
        and it would not track an encoder whose stride is not 32.
        """
        if images.dim() != 4:
            raise ValueError(
                f"SegmentationModelPytorch expects 4D [N, C, H, W] images, got shape {tuple(images.shape)}."
            )
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"SegmentationModelPytorch was built for in_channels={self.in_channels}, "
                f"got images with {images.shape[1]} channels."
            )
        return cast(Tensor, self.backbone(images))


__all__ = ["SEGMENTATION_ARCHITECTURES", "SMP_URI_PREFIX", "SegmentationArchitecture", "SegmentationModelPytorch"]

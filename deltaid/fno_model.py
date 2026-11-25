"""``FourierNeuralOperator`` — a Fourier neural operator (Li et al., 2021) as a segmentation model.

A model that learns its filters in the frequency domain: each layer takes the picture's 2-D Fourier
transform, multiplies the lowest ``modes`` frequencies by learned complex weights, transforms back,
and adds a local path (a 1x1 convolution, or a wider one). Because the learned weights live on
FREQUENCIES rather than on pixels, the same weights run on a picture of any size — no stride rule,
no padding to a multiple of 32, and the output follows the input's resolution.

**What it is good and bad at.** Keeping only the lowest frequencies is a low-pass filter: wide,
smooth regions are cheap to model, a detail a few pixels across is not — only the local path sees
it. Measured on spectrogram foreground masks, it ran several times faster than a pretrained
DeepLabV3+ and scored the same when the time axis was resampled at half or double the rows, but
lost most of its accuracy on short bursts. See ``docs/architecture.md`` §18.

It wraps nothing — no zoo, no pretrained weights — and is torch-only, so it sits beside the zoo
wrapper rather than in the ``unet/`` cross-framework comparison. Like every torch model here it
serves all three torch backends (lightning, torch, fastai) unchanged.

Typical wiring (YAML) — declare it ``!lazy:`` so the segmenter injects ``num_classes``::

    model: !lazy:deltaid.FourierNeuralOperator
      width: 48
      modes: [24, 24]
      patch_size: 4
"""

from typing import Annotated, List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from annotated_types import Interval
from confluid import configurable
from loggair import get_logger
from matrainer.torch import load_lightning_checkpoint
from matrainer.uri import ModelRef, file_model_ref
from torch import Tensor, nn

logger = get_logger(__name__)

__all__ = ["FourierNeuralOperator"]

#: A layer count, channel width or kernel size: at least one, and bounded only by sanity.
Positive = Annotated[int, Interval(ge=1, le=4096)]


class _SpectralConv2d(nn.Module):
    """One Fourier layer: ``rfft2`` → the lowest ``modes`` frequencies times learned weights → ``irfft2``.

    Two weight blocks, because ``rfft2`` keeps the whole row-frequency axis (positive frequencies
    first, negative ones at the end) but only the non-negative half of the column axis. The weights
    are stored as real tensors with a trailing ``(re, im)`` axis rather than as complex parameters:
    optimizers and devices that do not handle complex parameters (MPS among them) train them as
    plain floats.

    On a picture smaller than the modes allow, the kept block shrinks to what the grid has, using
    the weights of the lowest frequencies — which is what makes the layer run at any size.
    """

    def __init__(self, channels: int, modes_rows: int, modes_columns: int) -> None:
        super().__init__()
        self.modes_rows = modes_rows
        self.modes_columns = modes_columns
        scale = 1.0 / (channels * channels)
        self.low = nn.Parameter(scale * torch.randn(channels, channels, modes_rows, modes_columns, 2))
        self.high = nn.Parameter(scale * torch.randn(channels, channels, modes_rows, modes_columns, 2))

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        # FFTs run in float32: half-precision FFTs need power-of-two sizes and lose the low modes.
        spectrum = torch.fft.rfft2(x.float())
        rows = min(self.modes_rows, height // 2)
        columns = min(self.modes_columns, width // 2 + 1)
        low = torch.view_as_complex(self.low[:, :, :rows, :columns].contiguous())
        # The negative row frequencies -rows..-1 are the LAST `rows` entries of the weight block.
        high = torch.view_as_complex(self.high[:, :, self.modes_rows - rows :, :columns].contiguous())
        out = torch.zeros(batch, channels, height, width // 2 + 1, dtype=spectrum.dtype, device=x.device)
        out[:, :, :rows, :columns] = torch.einsum("bixy,ioxy->boxy", spectrum[:, :, :rows, :columns], low)
        out[:, :, height - rows :, :columns] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[:, :, height - rows :, :columns], high
        )
        result: Tensor = torch.fft.irfft2(out, s=(height, width)).to(x.dtype)
        return result


class _FNO(nn.Module):
    """The layer graph: lift → ``layers`` x (Fourier layer + local path) → per-pixel head.

    Private: the ``@configurable`` wrapper is the public surface.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        width: int,
        modes: Tuple[int, int],
        layers: int,
        patch_size: int,
        local_kernel: int,
        projection_width: int,
        coordinates: bool,
    ) -> None:
        super().__init__()
        self.coordinates = coordinates
        # A strided lift is a patch embedding: every later layer runs on a grid `patch_size` times
        # smaller per side, which is where the speed comes from. `patch_size: 1` is the plain FNO.
        self.lift = nn.Conv2d(in_channels + (2 if coordinates else 0), width, patch_size, stride=patch_size)
        self.spectral = nn.ModuleList([_SpectralConv2d(width, modes[0], modes[1]) for _ in range(layers)])
        self.local = nn.ModuleList(
            [nn.Conv2d(width, width, local_kernel, padding=local_kernel // 2) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(width, projection_width, 1), nn.GELU(), nn.Conv2d(projection_width, num_classes, 1)
        )

    def forward(self, images: Tensor) -> Tensor:
        batch, _, height, width = images.shape
        x = images
        if self.coordinates:
            # Where a pixel IS: a Fourier layer is translation-invariant, so without these it
            # cannot tell the top of the picture from the bottom.
            rows = torch.linspace(0.0, 1.0, height, device=x.device, dtype=x.dtype)
            columns = torch.linspace(0.0, 1.0, width, device=x.device, dtype=x.dtype)
            x = torch.cat(
                [
                    x,
                    rows.view(1, 1, height, 1).expand(batch, 1, height, width),
                    columns.view(1, 1, 1, width).expand(batch, 1, height, width),
                ],
                dim=1,
            )
        x = self.lift(x)
        last = len(self.spectral) - 1
        for index, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            x = spectral(x) + local(x)
            if index < last:
                x = F.gelu(x)
        logits: Tensor = self.head(x)
        # No `typing.cast` in this forward or the layer's: TorchScript (a deployment path) refuses it.
        if logits.shape[-2] != height or logits.shape[-1] != width:
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        return logits


@configurable(task="segmentation", role="model", lazy=True, framework="torch")
class FourierNeuralOperator(nn.Module):
    """A Fourier neural operator emitting per-pixel class logits, for any picture size.

    Args:
        num_classes: Width of the per-pixel head, background included. ``0`` (the default) means
            unset: a segmenter flows the resolved count in at run time, so it is not written in YAML.
        in_channels: Input channel count. Defaults to 3.
        width: Channels in every Fourier layer.
        modes: ``(rows, columns)`` — how many of the lowest frequencies each layer learns weights
            for, per axis. Finer detail than ``size / modes`` pixels reaches the output only through
            the local path. A picture too small for them uses as many as it has.
        layers: Number of Fourier layers.
        patch_size: Stride of the lifting convolution: the layers run on a grid this many times
            smaller per side, and the logits are upsampled back. ``1`` is the plain full-resolution
            operator — accurate on small detail, and far slower.
        local_kernel: Kernel size of the convolution beside each Fourier layer, odd. ``1`` is the
            plain operator; ``3`` lets each layer see a pixel's neighbours too.
        projection_width: Channels of the hidden layer in the per-pixel head.
        coordinates: Append each pixel's normalised row and column as two extra input channels.
        checkpoint_path: Optional Lightning ``.ckpt`` / state-dict to load after the build; ``""`` = none.
        seed: Seed for torch's global RNG, set when the model is BUILT, without restoring it.

    Attributes:
        net: The built network (``None`` until :meth:`solidify`). A public submodule name, so a
            checkpoint saved through Lightning round-trips as ``net.*`` keys after
            ``load_lightning_checkpoint`` strips its ``model.`` prefix.
        checkpoint_strict: Whether the checkpoint load enforces exact keys. ``False`` so a re-headed
            ``num_classes`` does not reject the whole load.
    """

    def __init__(
        self,
        num_classes: int = 0,
        in_channels: Positive = 3,
        width: Positive = 48,
        modes: Annotated[Tuple[int, int], Interval(ge=1, le=4096)] = (24, 24),
        layers: Positive = 4,
        patch_size: Positive = 4,
        local_kernel: Annotated[int, Interval(ge=1, le=15)] = 1,
        projection_width: Positive = 128,
        coordinates: bool = True,
        checkpoint_path: str = "",
        seed: int = 0,
    ) -> None:
        super().__init__()
        # Store only — the network is allocated by `solidify()` (the workspace lazy-init rule).
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.modes = (int(modes[0]), int(modes[1]))
        self.layers = int(layers)
        self.patch_size = int(patch_size)
        self.local_kernel = int(local_kernel)
        self.projection_width = int(projection_width)
        self.coordinates = bool(coordinates)
        self.checkpoint_path = checkpoint_path
        self.seed = int(seed)
        self.checkpoint_strict: bool = False
        self.net: Optional[nn.Module] = None

    @property
    def model_refs(self) -> List[ModelRef]:
        """The weight artifacts this model loads — only a checkpoint; it has no pretrained weights."""
        return [file_model_ref(self.checkpoint_path)] if self.checkpoint_path else []

    def solidify(self) -> nn.Module:
        """Build (once) and return the network — confluid's ``flow()``-time hook.

        Required configuration is validated here rather than in ``__init__``, which keeps zero-arg
        construction working. Idempotent.
        """
        if self.net is None:
            if self.num_classes < 1:
                raise ValueError(
                    "FourierNeuralOperator.num_classes is not set — a segmenter flows the resolved "
                    "count in at run time, or set it explicitly."
                )
            if self.local_kernel % 2 == 0:
                raise ValueError(
                    f"FourierNeuralOperator.local_kernel must be odd so the local path keeps the "
                    f"grid's size, got {self.local_kernel}."
                )
            bad = {
                name: value
                for name, value in {
                    "in_channels": self.in_channels,
                    "width": self.width,
                    "modes": min(self.modes),
                    "layers": self.layers,
                    "patch_size": self.patch_size,
                    "projection_width": self.projection_width,
                }.items()
                if value < 1
            }
            if bad:
                raise ValueError(f"FourierNeuralOperator settings must be >= 1, got {bad}.")
            torch.manual_seed(self.seed)
            self.net = _FNO(
                num_classes=self.num_classes,
                in_channels=self.in_channels,
                width=self.width,
                modes=self.modes,
                layers=self.layers,
                patch_size=self.patch_size,
                local_kernel=self.local_kernel,
                projection_width=self.projection_width,
                coordinates=self.coordinates,
            )
            logger.debug(
                f"Built a Fourier neural operator ({self.num_classes} classes, width {self.width}, "
                f"modes {self.modes}, {self.layers} layers, patch {self.patch_size})."
            )
            if self.checkpoint_path:
                load_lightning_checkpoint(self, self.checkpoint_path, strict=self.checkpoint_strict)
        return self.net

    @property
    def backbone(self) -> nn.Module:
        """The network, built on first access — for a directly constructed, never-flowed instance."""
        return self.solidify()

    def forward(self, images: Tensor) -> Tensor:
        """Per-pixel class logits ``[N, num_classes, H, W]`` from an ``[N, in_channels, H, W]`` batch."""
        if images.dim() != 4:
            raise ValueError(f"FourierNeuralOperator expects 4D [N, C, H, W] images, got shape {tuple(images.shape)}.")
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"FourierNeuralOperator was built for in_channels={self.in_channels}, "
                f"got images with {images.shape[1]} channels."
            )
        return cast(Tensor, self.backbone(images))

    def __repr__(self) -> str:
        """Short and honest before the build — the default would print an empty module."""
        if self.net is None:
            return (
                f"FourierNeuralOperator(num_classes={self.num_classes}, width={self.width}, "
                f"modes={self.modes}, unbuilt)"
            )
        return super().__repr__()

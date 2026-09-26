"""Deltaid: semantic segmentation across training backends, driven by one config.

**The core is backend-free.** ``pip install deltaid`` gives you the CLI and the config surface;
a training framework arrives with the backend you select::

    pip install 'deltaid[lightning]'         # PyTorch Lightning — the reference backend

Which is why every backend name below resolves LAZILY (PEP 562): importing ``deltaid`` must not
require a framework the operator did not choose. Touching a name whose backend is missing raises
an :class:`ImportError` naming the exact extra, instead of a traceback from three libraries down.
"""

from importlib.metadata import PackageNotFoundError, version

from matrainer._exports import lazy_exports

try:
    # Single source of truth: the installed distribution's metadata, i.e. pyproject.toml's
    # `version` (the distribution is `deltaid`). Never type the number in here as well.
    __version__ = version("deltaid")
except PackageNotFoundError:  # pragma: no cover - uninstalled source checkout
    __version__ = "0.0.0.dev0"


#: Backend-owned name -> (defining module, the extra that installs it). Resolved on first attribute
#: access. Keep in step with the extras in ``pyproject.toml``: a name here whose extra does not
#: exist would advertise an install command that fails.
#
# The two kinds of model wrap different things, and only the second is the cross-framework
# comparison — so they live apart. A ZOO WRAPPER (`SegmentationModelPytorch`, reaching somebody
# else's weights) stays at the TOP LEVEL, one `<backend>_model.py` each; every implementation of
# the shared architecture spec lives in the `deltaid.unet` package (which imports nothing itself,
# so these stay lazy).
_BACKEND_NAMES = {
    "LightningSegmenter": ("deltaid.lightning_segmenter", "lightning"),
    "TorchSegmenter": ("deltaid.torch_segmenter", "torch"),
    "FastaiSegmenter": ("deltaid.fastai_segmenter", "fastai"),
    "KerasSegmenter": ("deltaid.keras_segmenter", "keras"),
    "TensorFlowSegmenter": ("deltaid.tensorflow_segmenter", "tensorflow"),
    "HuggingFaceSegmenter": ("deltaid.huggingface_segmenter", "huggingface"),
    "JaxSegmenter": ("deltaid.jax_segmenter", "jax"),
    "MlxSegmenter": ("deltaid.mlx_segmenter", "mlx"),
    "SegmentationModelPytorch": ("deltaid.smp_model", "torch"),
    "HuggingFaceSegmentationModel": ("deltaid.huggingface_model", "huggingface"),
    # The shared spec's implementations. `TorchUNet` serves EVERY torch backend (lightning, torch,
    # fastai — they differ in the loop, not in what a model is), which is why there is no separate
    # name per backend. Its extra is the SMALLEST one that supplies what it needs — plain `torch`,
    # not `lightning`, so the ImportError advises the smaller install.
    "TorchUNet": ("deltaid.unet.torch_unet", "torch"),
    "KerasUNet": ("deltaid.unet.keras_unet", "keras"),
    "TensorFlowUNet": ("deltaid.unet.tensorflow_unet", "tensorflow"),
    "FlaxUNet": ("deltaid.unet.flax_unet", "jax"),
    "MlxUNet": ("deltaid.unet.mlx_unet", "mlx"),
}

__all__ = [*_BACKEND_NAMES]

# The resolver is matrainer's `lazy_exports` (the machinery was identical in every training
# project's root); the `_BACKEND_NAMES` table above is the project DATA it runs on.
__getattr__, __dir__ = lazy_exports(__name__, _BACKEND_NAMES)

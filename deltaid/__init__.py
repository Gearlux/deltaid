"""Deltaid: semantic segmentation across training backends, driven by one config.

**The core is backend-free.** ``pip install deltaid`` gives you the CLI and the config surface;
a training framework arrives with the backend you select::

    pip install 'deltaid[lightning]'         # PyTorch Lightning — the reference backend

Which is why every backend name below resolves LAZILY (PEP 562): importing ``deltaid`` must not
require a framework the operator did not choose. Touching a name whose backend is missing raises
an :class:`ImportError` naming the exact extra, instead of a traceback from three libraries down.
"""

import importlib
from typing import Any

__version__ = "0.2.0"

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
    "SegmentationModelPytorch": ("deltaid.smp_model", "torch"),
    "HuggingFaceSegmentationModel": ("deltaid.huggingface_model", "huggingface"),
    # The shared spec's implementations. `TorchUNet` serves EVERY torch backend (lightning, torch,
    # fastai — they differ in the loop, not in what a model is), which is why there is no separate
    # name per backend. Its extra is the SMALLEST one that supplies what it needs — plain `torch`,
    # not `lightning`, so the ImportError advises the smaller install.
    "TorchUNet": ("deltaid.unet.torch_unet", "torch"),
    "KerasUNet": ("deltaid.unet.keras_unet", "keras"),
    "TensorFlowUNet": ("deltaid.unet.tensorflow_unet", "tensorflow"),
}

__all__ = [*_BACKEND_NAMES]


def __getattr__(name: str) -> Any:
    """Resolve a backend-owned name on first use, or explain which extra is missing."""
    entry = _BACKEND_NAMES.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, extra = entry
    try:
        return getattr(importlib.import_module(module_path), name)
    except ImportError as exc:
        # The failure is a MISSING BACKEND, not a broken deltaid — say so, and say what to run.
        # Chained (`from exc`) so the underlying "No module named 'torch'" stays visible.
        raise ImportError(
            f"deltaid.{name} needs the {extra!r} backend, which is not installed.\n"
            f"  pip install 'deltaid[{extra}]'\n"
            f"(underlying import error: {exc})"
        ) from exc


def __dir__() -> Any:
    """Advertise the lazy names to ``dir()`` / tab-completion."""
    return sorted({*globals(), *_BACKEND_NAMES})

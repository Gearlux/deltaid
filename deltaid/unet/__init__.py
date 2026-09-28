"""The shared U-Net spec and its per-framework implementations.

**This module imports NOTHING, and that is load-bearing.** ``deltaid/__init__.py`` resolves every
backend-owned name lazily so a machine without torch never imports it; an ``__init__`` here
re-exporting the implementations would drag in torch AND keras AND flax AND mlx the moment
anything under this package — including the dependency-free :mod:`deltaid.unet.spec` — was
touched. Import an implementation from its own module, or through the lazy names on ``deltaid``.

See :mod:`deltaid.unet.spec` for what the architecture is and why it exists.
"""

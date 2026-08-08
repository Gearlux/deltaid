"""Deltaid CLI — semantic segmentation across training backends.

Three verbs, and the training backend is a *config* dimension the CLI passes through::

    deltaid train    cfg.yaml                       # the config's default backend
    deltaid train    cfg.yaml --framework lightning # a `!scope:framework=lightning` block
    deltaid evaluate cfg.yaml
    deltaid predict  cfg.yaml
    deltaid frameworks                              # what the backends are

What stays constant across backends is the point: the same ``runnable:`` + ``task:`` config, the
same three verbs, the same tracking. Only the runnable class named in the YAML changes. (A generic
``matrainer train cfg.yaml`` or ``recordstream run cfg.yaml`` also works — the first is the same
machinery without this project's table.)

The MACHINERY is :mod:`matrainer.cli` (2026-08-07, the P4 lift) — the verb dispatch through
``materialize_runnable``, the ``--framework`` → ``--scope`` rewrite, the keras engine pin, the
registration import and their load-bearing ORDER all live on the factory's own docstrings. What
this module owns is deltaid's DATA: the FRAMEWORKS table and the verb help texts.

Adding a backend: write ``deltaid/<name>_segmenter.py``, then add a row to ``FRAMEWORKS`` below.
"""

from typing import Dict

from matrainer.cli import Framework, make_trainer_app, trainer_main

FRAMEWORKS: Dict[str, Framework] = {
    "lightning": Framework(
        "PyTorch Lightning — the reference backend (smp models, per-split torchmetrics).",
        "deltaid.torch_segmentation",
    ),
    # Shares `lightning`'s loss/metric set, and that is the point rather than an economy: a
    # `torch.nn` loss does not care whether a Trainer or a hand-written loop calls it, so the two
    # backends differ ONLY in the loop.
    "torch": Framework(
        "PyTorch — a hand-written loop, no training framework at all (the control experiment).",
        "deltaid.torch_segmentation",
    ),
    "fastai": Framework(
        "fastai — a Learner with one-cycle scheduling, over the same torch stack.",
        # The same torch losses reach it (a `BaseLoss` is a callable over torch tensors), and its
        # METRICS are names in `fastai.metrics` rather than registered classes — so there is
        # nothing extra to import for it.
        "deltaid.torch_segmentation",
    ),
    "keras": Framework(
        "Keras 3 — one API over PyTorch / TensorFlow / JAX (set KERAS_BACKEND to pick).",
        "deltaid.keras_segmentation",
    ),
    # Shares the Keras loss/metric set on purpose: in TF >= 2.16 `tf.keras` IS Keras 3, so a
    # separate `tensorflow_segmentation.py` would register the same classes under a second tag
    # and make every bare `!class:SparseCategoricalCrossentropy` ambiguous for no gain.
    "tensorflow": Framework(
        "TensorFlow — a native tf.data + tf.GradientTape loop, with SavedModel export.",
        "deltaid.keras_segmentation",
    ),
    "huggingface": Framework(
        "Transformers `Trainer` + Accelerate — fine-tuning a Hub checkpoint (SegFormer).",
        # No registrations: an `AutoModelForSemanticSegmentation` computes its own loss from
        # `labels=`, so there is nothing a config could wire.
    ),
    "jax": Framework(
        "JAX — a jitted optax loop over a Flax model, the loop written out in full.",
        # No registrations either, for a DIFFERENT reason than huggingface's: optax ships plain
        # FUNCTIONS, so a loss is resolved by NAME and there is no class for the registry to tag.
    ),
    "mlx": Framework(
        "MLX — Apple silicon, unified memory, lazy arrays (`mlx.nn` losses by NAME).",
        # `mlx.nn.losses` are functions like optax's; `mlx.optimizers` ships classes, but a
        # config names those too. Nothing to register.
    ),
}

app = make_trainer_app(
    name="deltaid",
    description="Semantic segmentation across training backends",
    frameworks=FRAMEWORKS,
    verb_docs={
        "train": "Train a semantic-segmentation model",
        "evaluate": "Evaluate a trained segmenter over the test set",
        "predict": "Stream segmentation predictions from a trained segmenter to its sink",
    },
)


def main() -> None:
    trainer_main(app, FRAMEWORKS)


if __name__ == "__main__":
    main()

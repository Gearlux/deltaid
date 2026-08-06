"""Deltaid CLI — semantic segmentation across training backends.

Three verbs, and the training backend is a *config* dimension the CLI passes through::

    deltaid train    cfg.yaml                       # the config's default backend
    deltaid train    cfg.yaml --framework lightning # a `!scope:framework=lightning` block
    deltaid evaluate cfg.yaml
    deltaid predict  cfg.yaml
    deltaid frameworks                              # what the backends are

What stays constant across backends is the point: the same ``runnable:`` + ``task:`` config, the
same three verbs, the same tracking. Only the runnable class named in the YAML changes. (A generic
``recordstream run cfg.yaml`` also works — it calls ``.run()`` with whatever ``task`` the YAML
declares, with no verb at all.)

Adding a backend: write ``deltaid/<name>_segmenter.py``, then add a row to ``FRAMEWORKS`` below.
"""

import os
import sys
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from liquifai import LiquifyApp
from loggair import get_logger

logger = get_logger(__name__)

app = LiquifyApp(name="deltaid", description="Semantic segmentation across training backends")


def _dispatch(runnable: Any, task: str) -> None:
    """Build the bound runnable, force ``task``, and run it.

    The verb wins over any ``task:`` set in YAML, so one config serves all three verbs.
    Backend-agnostic: needs only a ``task`` attribute and ``run()``.

    Building goes through ``recordstream.cli.materialize_runnable`` — NOT a bare ``flow()`` —
    because the flat config's top-level keys (``train_set`` / ``batch_size`` / ``max_epochs`` /
    ``model`` …) only broadcast into the constructor when the node is built against the document
    it came from. A bare flow drops them SILENTLY: `train_set` becomes None (surfacing much later
    as a failure to derive num_classes) and `max_epochs: 3` quietly falls back to the constructor
    default, which never surfaces at all. Same helper as `recordstream run`, so both runners share
    one answer.
    """
    if runnable is None:
        raise ValueError("No runnable was bound — provide one under 'runnable:' in your YAML.")

    from recordstream.cli import materialize_runnable

    runnable = materialize_runnable(runnable)
    if hasattr(runnable, "task"):
        runnable.task = task  # the verb wins over the YAML's task
    runnable.run()


# --------------------------------------------------------------------------- #
# The backends
# --------------------------------------------------------------------------- #
class Framework(NamedTuple):
    """One training backend: what to say about it, what to import for it."""

    description: str
    #: The module whose import registers this engine's losses/metrics under a plain NAME.
    #: ``None`` means the backend has NOTHING to register, which is a real state rather than an
    #: oversight — a model that computes its own loss leaves nothing for a config to wire.
    registrations: Optional[str] = None


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

#: Verb -> the ``task`` it forces. The verb wins over any ``task:`` in the YAML.
VERB_TASKS = {"train": "fit", "evaluate": "test", "predict": "predict"}

_VERB_DOCS = {
    "train": "Train a semantic-segmentation model",
    "evaluate": "Evaluate a trained segmenter over the test set",
    "predict": "Stream segmentation predictions from a trained segmenter to its sink",
}


def _verb_command(verb: str) -> Callable[[Any], None]:
    """One verb.

    A factory, not a default-argument closure: liquifai reads the handler's SIGNATURE to build the
    command, so smuggling the loop variable in as a default would advertise it as a ``--_verb``
    option in help and TAB completion. The signature it must expose is exactly ``(runnable)`` —
    the parameter NAME is what liquifai's DI binds to the config's ``runnable:`` key.
    """

    def command(runnable: Any) -> None:
        _dispatch(runnable, VERB_TASKS[verb])

    command.__name__ = verb
    command.__doc__ = f"{_VERB_DOCS[verb]} (task={VERB_TASKS[verb]}). Pick a backend with --framework."
    return command


for _verb in VERB_TASKS:
    app.script_command(name=_verb, flow_mode="manual")(_verb_command(_verb))


@app.command(name="frameworks")
def frameworks() -> None:
    """List the training backends `--framework` accepts."""
    width = max(len(name) for name in FRAMEWORKS)
    print("Training backends — pass one as `--framework <name>`:\n")
    for name, spec in FRAMEWORKS.items():
        print(f"  {name:<{width}}  {spec.description}")
    print(
        "\nThe name must also match a `!scope:framework=<name>` block in the config;"
        "\nconfluid lists the values a config declares when it does not."
    )


def _pin_keras_backend(framework: Optional[str]) -> None:
    """``--framework tensorflow`` must pin ``KERAS_BACKEND`` before ANYTHING imports keras.

    Keras 3 reads the variable once, at import time, and ``recordstream.keras`` defaults it to the
    first backend INSTALLED — torch, on any machine that also carries ``deltaid[lightning]``. A
    torch-backed ``keras.Model`` cannot be trained with ``tf.GradientTape``, so the default has to
    be overridden for this backend.

    It happens HERE, in ``main()``, and it must run BEFORE :func:`_register_segmentation`, which is
    what made the omission bite: ``FRAMEWORKS["tensorflow"].registrations`` names
    ``deltaid.keras_segmentation`` (there is deliberately no second loss module — ``tf.keras`` IS
    Keras 3), so registering imports ``deltaid._keras`` -> ``recordstream.keras`` and the torch
    default is already bound by the time :mod:`deltaid._tensorflow` sets its own. Every
    ``deltaid train --framework tensorflow`` then died at ``_require_tensorflow_backend()`` with
    *"needs Keras on the TensorFlow engine, but keras loaded the 'torch' one"* — the message names
    the fix, but the CLI knows the answer and can just do it (measured on the backend sweep, which
    is where a backend nobody had run since the CLI grew its registration step surfaced).

    :mod:`deltaid._tensorflow` sets the same default as a second line of defence for callers that
    never go through this CLI (``recordstream run cfg.yaml --framework tensorflow``), and
    ``TensorFlowSegmenter`` refuses to run on the wrong engine either way.

    ``setdefault``, so an explicit ``KERAS_BACKEND=…`` still wins and is then reported as a
    mismatch instead of being silently overridden.
    """
    if framework == "tensorflow":
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def _register_segmentation(framework: Optional[str]) -> None:
    """Import a backend's loss/metric registrations so a config may NAME them.

    A ``!class:`` target that is a registered name (``DiceLoss``, or a selector like
    ``CrossEntropyLoss@role=loss``) resolves only once the module that registers it has been
    imported; a dotted path resolves through importlib either way. Discovery services bootstrap
    every ``confluid.configurables`` entry point, but a CLI does not.

    It happens from ``main()``, before ``app.run()``, because the EAGER instance form
    (``!class:Foo()`` with parens) is built during config load — earlier than any handler runs.
    With no ``--framework`` this imports nothing: the flag is what names the engine, and there is
    no safe guess. A config that names no engine-specific set, or spells its losses as dotted
    paths, needs none of this.

    Best-effort: a missing backend is a normal state on a partial install, and its absent
    registrations are the correct outcome, not an error.
    """
    if framework is None:
        return
    module = FRAMEWORKS.get(framework, Framework("")).registrations
    if module is None:
        return
    import importlib

    try:
        importlib.import_module(module)
    except ImportError as exc:
        logger.debug(f"{framework}: loss/metric registrations unavailable ({module}: {exc})")


def _normalize_framework_flag(argv: List[str]) -> None:
    """Rewrite ``--framework <name>`` into the ``--scope framework=<name>`` liquifai reads.

    liquifai binds ``--KEY VAL`` to scope activation only for dimensions the config DECLARES;
    against a config with no ``!scope:framework=…`` block at all, a bare ``--framework lightning``
    would instead be read as a config override and try to set a ``framework`` attribute on
    something. Rewriting here makes the flag mean one thing regardless of what the config happens
    to declare — an undeclared dimension stays an inert no-op, which is what lets a config grow
    into a backend it does not have yet.

    Mutates ``argv`` in place because scopes are parsed from the command line, so this must land
    before ``run()``. The explicit ``--scope framework=…`` spelling is left exactly as typed.
    """
    out: List[str] = []
    i, n = 0, len(argv)
    while i < n:
        arg = argv[i]
        if arg.startswith("--framework="):
            out += ["--scope", f"framework={arg.split('=', 1)[1]}"]
        elif arg == "--framework" and i + 1 < n:
            out += ["--scope", f"framework={argv[i + 1]}"]
            i += 1
        else:
            out.append(arg)
        i += 1
    argv[:] = out


def _requested_framework(argv: List[str]) -> Optional[str]:
    """The backend named on the command line, in any spelling, or ``None``.

    Read from ``argv`` rather than from liquifai because the caller needs it BEFORE the config is
    loaded — see :func:`_register_segmentation`.
    """
    for i, arg in enumerate(argv):
        if arg.startswith("--framework="):
            return arg.split("=", 1)[1]
        if arg.startswith(("--scope=", "-s=")) and "framework=" in arg:
            return arg.split("framework=", 1)[1]
        if arg in ("--framework", "--scope", "-s") and i + 1 < len(argv):
            following = argv[i + 1]
            if arg == "--framework":
                return following
            if following.startswith("framework="):
                return following.split("=", 1)[1]
    return None


def main() -> None:
    framework = _requested_framework(sys.argv)
    _pin_keras_backend(framework)  # BEFORE the registration import — see that function
    _register_segmentation(framework)
    _normalize_framework_flag(sys.argv)
    app.run()


if __name__ == "__main__":
    main()

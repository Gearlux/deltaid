"""The CLI axis: three verbs, a ``--framework`` FLAG, and a config that stays complete under it.

The framework is a property of the CONFIG, not of the CLI, so it is a flag the CLI passes through
rather than a subcommand. The checks here are deliberately CONFIG-level — they load the example
YAML once per declared scope — so a block that drops a slot fails on the machine that introduces
the mistake, rather than only where that backend's framework happens to be installed.

The MACHINERY (the rewrite, the keras engine pin, the dispatch, the pre-step ORDER) is
`matrainer.cli`'s since 2026-08-07 and is pinned in matrainer's own `tests/test_cli.py`; what
stays here is deltaid's half of each seam — the flag against THIS project's spellings, the pin
against THIS project's tensorflow row, and the table itself.
"""

import inspect
import os
from pathlib import Path
from typing import Any, List

import confluid
import pytest
from matrainer.cli import (
    VERB_TASKS,
    normalize_framework_flag,
    pin_keras_backend,
    registration_modules,
    requested_framework,
    trainer_main,
)

from deltaid.cli import FRAMEWORKS, app, main

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "train_unet_oxford_pets.yaml"


# --------------------------------------------------------------------------- #
# The verbs
# --------------------------------------------------------------------------- #
def test_three_verbs_mapping_to_the_tasks_they_force() -> None:
    """The verb wins over the YAML's `task:`, so ONE config serves all three."""
    assert VERB_TASKS == {"train": "fit", "evaluate": "test", "predict": "predict"}


@pytest.mark.parametrize("verb", sorted(VERB_TASKS))
def test_every_verb_is_registered_as_a_command(verb: str) -> None:
    assert verb in app._commands


@pytest.mark.parametrize("verb", sorted(VERB_TASKS))
def test_a_verb_handler_exposes_exactly_the_runnable_parameter(verb: str) -> None:
    """liquifai reads the SIGNATURE to build the command, and binds by parameter NAME.

    A default-argument closure smuggling the loop variable would advertise a `--_verb` option in
    help and TAB completion — which is why the handlers come from a factory.
    """
    import inspect

    params = list(inspect.signature(app._commands[verb]).parameters)
    assert params == ["runnable"], f"{verb} exposes {params}"


def test_the_verbs_build_through_materialize_runnable_not_a_bare_flow() -> None:
    """A bare `flow()` drops EVERY top-level key silently.

    liquifai's DI only materializes a bound node against its document when the command parameter
    is annotated with a *configurable class*; these annotate `runnable: Any` because the runnable
    is polymorphic. So an auto flow would leave `train_set` as None (surfacing much later as a
    failure to derive num_classes) and `max_epochs: 5` as the constructor default (never
    surfacing at all).
    """
    import inspect

    import matrainer.cli

    for verb in VERB_TASKS:
        assert getattr(app._commands[verb], "__liquifai_flow_mode__", None) == "manual"
    assert "materialize_runnable" in inspect.getsource(matrainer.cli.default_dispatch)


# --------------------------------------------------------------------------- #
# The --framework flag
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "argv,expected",
    [
        (["deltaid", "train", "c.yaml", "--framework", "lightning"], "lightning"),
        (["deltaid", "train", "c.yaml", "--framework=lightning"], "lightning"),
        (["deltaid", "train", "c.yaml", "--scope", "framework=lightning"], "lightning"),
        (["deltaid", "train", "c.yaml", "-s", "framework=lightning"], "lightning"),
        (["deltaid", "train", "c.yaml"], None),
    ],
)
def test_the_requested_framework_is_read_from_argv_in_every_spelling(argv: List[str], expected: str) -> None:
    """Read from argv rather than liquifai because the registrations must be imported BEFORE the
    config loads — the eager `!class:Foo()` form is built during load, earlier than any handler."""
    assert requested_framework(argv) == expected


def test_the_flag_is_rewritten_to_a_scope_never_appended() -> None:
    """liquifai binds the ergonomic `--KEY VAL` flag only for dimensions the config DECLARES.

    Against a config with no `!scope:framework=…` block, a bare `--framework lightning` would be
    read as a config OVERRIDE and try to set a `framework` attribute. Rewriting makes the flag
    mean one thing either way, and an undeclared dimension stays the inert no-op that lets a
    config grow into a backend it does not have yet.
    """
    argv = ["deltaid", "train", "c.yaml", "--framework", "lightning"]
    normalize_framework_flag(argv)
    assert argv == ["deltaid", "train", "c.yaml", "--scope", "framework=lightning"]


def test_an_explicit_scope_spelling_is_left_alone() -> None:
    argv = ["deltaid", "train", "c.yaml", "--scope", "framework=lightning"]
    normalize_framework_flag(argv)
    assert argv == ["deltaid", "train", "c.yaml", "--scope", "framework=lightning"]


# --------------------------------------------------------------------------- #
# The config stays complete under every declared framework
# --------------------------------------------------------------------------- #
def test_the_example_config_declares_every_implemented_framework() -> None:
    """A backend absent from the config cannot be selected; one in the config but not in
    FRAMEWORKS is advertised by neither `deltaid frameworks` nor the registrations."""
    text = EXAMPLE.read_text()
    for name in FRAMEWORKS:
        assert f"!scope:framework={name}" in text, f"the example config declares no block for {name!r}"


@pytest.mark.parametrize("framework", sorted(FRAMEWORKS))
def test_every_declared_framework_resolves_to_a_complete_runnable(framework: str) -> None:
    """The check that catches a scope block erasing a slot it did not mean to.

    A scope block replaces a KEY and takes its whole subtree with it, so a `model:` nested inside
    `runnable:` would VANISH the moment a block swapped the runnable — a failure that surfaces as
    a missing-argument TypeError minutes into a run. Keeping every framework-specific key a
    top-level SIBLING is what prevents it, and this asserts the result rather than the layout.
    """
    doc = confluid.load(EXAMPLE, until="document", scopes=[f"framework={framework}"])
    for key in ("runnable", "model", "train_set", "val_set"):
        assert doc.get(key) is not None, f"--framework {framework} leaves {key!r} unset"


def test_the_default_backend_has_a_block_of_its_own() -> None:
    """The top level already IS lightning, but confluid raises on an activation matching no
    block — so an undeclared default would make naming it explicitly an ERROR. That raise is the
    check a backend subcommand could never do: a typo now lists the declared backends instead of
    silently training the default one."""
    with pytest.raises(confluid.ScopeError) as excinfo:
        confluid.load(EXAMPLE, until="document", scopes=["framework=lightnin"])
    assert "lightning" in str(excinfo.value), "the error must LIST the values the config declares"


def test_every_framework_names_registrations_that_import() -> None:
    """A `registrations` module that does not import leaves every bare loss name unresolvable."""
    import importlib

    for name, spec in FRAMEWORKS.items():
        for module in registration_modules(spec):
            importlib.import_module(module)  # raises if the path is wrong


# --------------------------------------------------------------------------- #
# The engine `--framework tensorflow` cannot run without
# --------------------------------------------------------------------------- #
def test_the_tensorflow_backend_pins_the_keras_engine_before_anything_imports_keras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keras 3 reads `KERAS_BACKEND` ONCE, at import, and the default here is the wrong one.

    `recordstream.keras` defaults it to the first engine INSTALLED, which is torch on any machine
    that also carries `deltaid[lightning]` — and a torch-backed `keras.Model` cannot be trained
    with `tf.GradientTape`. Without this step EVERY `deltaid train --framework tensorflow` died at
    `_require_tensorflow_backend()`, because the registration pre-step imports
    `deltaid.keras_segmentation` (there is deliberately no second loss module) and binds the torch
    default before `deltaid._tensorflow` gets to set its own.
    """
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    pin_keras_backend(requested_framework(["deltaid", "train", "c.yaml", "--framework", "tensorflow"]))

    assert os.environ.get("KERAS_BACKEND") == "tensorflow"


def test_an_explicit_keras_backend_is_never_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    """`setdefault`, so a deliberate choice survives and the segmenter reports the mismatch."""
    monkeypatch.setenv("KERAS_BACKEND", "jax")
    pin_keras_backend("tensorflow")

    assert os.environ["KERAS_BACKEND"] == "jax"


@pytest.mark.parametrize("framework", ["lightning", "torch", "keras", "fastai", "jax", "mlx", "huggingface", None])
def test_no_other_framework_touches_keras_backend(monkeypatch: pytest.MonkeyPatch, framework: Any) -> None:
    """`keras` included: it runs on whichever engine is loaded, which is the whole point of it."""
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    pin_keras_backend(framework)

    assert "KERAS_BACKEND" not in os.environ


def test_the_pin_runs_before_the_registration_import() -> None:
    """Order is the whole fix: the registration import is what binds the wrong engine.

    The order now lives in the shared `trainer_main` (matrainer pins it too); what THIS test
    keeps is the project's stake in it — `main()` must delegate there, and the shared body must
    still pin before it registers. Read from SOURCE — calling it would run the app.
    """
    assert "trainer_main" in inspect.getsource(main), "main() must run the shared pre-step order"
    body = inspect.getsource(trainer_main).split('"' * 3)[-1]

    assert body.index("pin_keras_backend") < body.index("register_frameworks")

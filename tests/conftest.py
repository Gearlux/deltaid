"""Session-wide setup for the deltaid suite.

One thing only, and it is a native-runtime ordering fix rather than a testing convenience.

**pyarrow must not be imported before TensorFlow (macOS).** If pyarrow's native thread pool
already exists when TensorFlow initialises, the first eager op inside TensorFlow's parallel-for
(``tf.vectorized_map`` -> ``pfor`` -> ``quick_execute``) NEVER RETURNS — not slow: hung, with no
error, no timeout and no output. Import TensorFlow first and the identical libraries, threads and
call all work. The classification project hit this first and documented the full elimination there
(``sonair/tests/conftest.py``); the condition is exhaustively reproducible, the mechanism is not
explained.

deltaid is exposed to it for the same reason: **pyarrow arrives without ever being asked for.**
recordstream uses it directly, so importing almost anything first-party pulls it in — and every
test module here imports recordstream.

So when the session is going to run on the TensorFlow engine, TensorFlow is imported HERE.
conftest is imported before any test module, which is the only hook early enough to win the race.
Deliberately conditional: importing TensorFlow into a torch-engine run would cost several seconds
and bind nothing useful, and on a machine without the ``tensorflow`` extra it is absent.
"""

import os


def _tensorflow_engine_selected() -> bool:
    """Is this session going to bind Keras to TensorFlow?

    Read from the ENVIRONMENT rather than from ``recordstream.keras.keras_backend()``, because
    calling that would import keras — and keras imports its engine, which is the very thing whose
    ordering is at stake.
    """
    return os.environ.get("KERAS_BACKEND") == "tensorflow"


if _tensorflow_engine_selected():  # pragma: no cover - ordering fix, see the module docstring
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        # No `deltaid[tensorflow]` here; the TF-engine tests skip themselves anyway.
        pass

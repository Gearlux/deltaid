"""Deltaid CLI — semantic-segmentation training and evaluation.

``deltaid train`` / ``deltaid evaluate`` / ``deltaid predict`` bind the single
:class:`~deltaid.segmenter.LightningSegmenter` (a standard Lightning module) and run
it with the matching ``task`` (the verb wins over the YAML's ``task:``). A generic
``recordstream run config.yaml`` also works.
"""

from typing import Any

from liquifai import LiquifyApp
from loggair import get_logger

logger = get_logger(__name__)

app = LiquifyApp(name="deltaid", description="Deltaid semantic-segmentation commands")


def _dispatch(segmenter: Any, task: str) -> None:
    if segmenter is None:
        logger.error("No segmenter was bound — provide one under 'runnable:' in your YAML.")
        return
    from confluid import flow
    from confluid.fluid import Fluid

    if isinstance(segmenter, Fluid):
        segmenter = flow(segmenter)
    if hasattr(segmenter, "task"):
        segmenter.task = task
    segmenter.run()


@app.script_command(flow_mode="auto")
def train(runnable: Any) -> None:
    """Train a semantic-segmentation model (task=fit)."""
    _dispatch(runnable, "fit")


@app.script_command(flow_mode="auto")
def evaluate(runnable: Any) -> None:
    """Evaluate a trained segmenter over the test set (task=test)."""
    _dispatch(runnable, "test")


@app.script_command(flow_mode="auto")
def predict(runnable: Any) -> None:
    """Stream segmentation predictions to the sink (task=predict)."""
    _dispatch(runnable, "predict")


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()

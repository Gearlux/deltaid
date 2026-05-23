"""Deltaid CLI — semantic-segmentation training and evaluation commands.

Provides ``deltaid train`` and ``deltaid evaluate`` as aliases to the
segmentation-specialised Lightning trainer/evaluator. These are equivalent
to ``marainer segmentation train`` / ``marainer segmentation evaluate``
(which are registered via the marainer plugin in :mod:`deltaid.plugin`).
"""

from liquifai import LiquifyApp
from logflow import get_logger

from deltaid.evaluator import LightningSegmentationEvaluator
from deltaid.trainer import LightningSegmentationTrainer

logger = get_logger(__name__)

app = LiquifyApp(name="deltaid", description="Deltaid semantic-segmentation commands")


@app.script_command(flow_mode="auto")
def train(trainer: LightningSegmentationTrainer) -> None:
    """Train a semantic-segmentation model."""
    trainer.run()


@app.script_command(flow_mode="auto")
def evaluate(evaluator: LightningSegmentationEvaluator) -> None:
    """Evaluate a trained semantic-segmentation model."""
    evaluator.run()


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()

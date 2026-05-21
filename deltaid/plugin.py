"""Deltaid marainer plugin — mounts ``marainer segmentation`` sub-app.

This module is discovered by marainer's plugin system via the
``marainer.commands`` entry-point group (declared in ``deltaid/pyproject.toml``).
It adds a ``segmentation`` command group to the running marainer app,
exposing:

* ``marainer segmentation train``    — alias to :func:`deltaid.cli.train`
* ``marainer segmentation evaluate`` — alias to :func:`deltaid.cli.evaluate`
"""

from deltaid.cli import evaluate, train
from liquifai import LiquifyApp


def register(app: LiquifyApp) -> None:
    """Mount the ``segmentation`` sub-app onto *app* (the root marainer app)."""
    segmentation = LiquifyApp(
        name="segmentation",
        description="Semantic-segmentation workflow commands (deltaid specialisation).",
    )
    segmentation.script_command(flow_mode="auto")(train)
    segmentation.script_command(flow_mode="auto")(evaluate)
    app.add_app(segmentation, name="segmentation")

"""Tests for the deltaid marainer plugin registration."""

from liquifai import LiquifyApp

from deltaid.plugin import register


def test_register_mounts_segmentation_subapp() -> None:
    app = LiquifyApp(name="marainer", description="test marainer host")
    register(app)
    # The `segmentation` sub-app should now be discoverable on the host.
    sub_app_names = {a.name for a in getattr(app, "sub_apps", []) if hasattr(a, "name")}
    if sub_app_names:
        assert "segmentation" in sub_app_names
    else:
        # Some LiquifyApp implementations store sub-apps under a private attr;
        # fall back to attribute-based checks rather than asserting a specific
        # internal layout.
        assert any(
            "segmentation" in repr(v) for v in vars(app).values()
        ), "register() must mount a `segmentation` sub-app on the host LiquifyApp."


def test_register_exposes_train_and_evaluate_commands() -> None:
    """The sub-app's command surface must include train + evaluate."""
    app = LiquifyApp(name="marainer", description="test marainer host")
    register(app)

    # Locate the segmentation sub-app (best-effort across LiquifyApp internals).
    seg_app = None
    for attr_value in vars(app).values():
        if isinstance(attr_value, list):
            for entry in attr_value:
                if getattr(entry, "name", None) == "segmentation":
                    seg_app = entry
                    break
        elif isinstance(attr_value, dict):
            seg_app = attr_value.get("segmentation", seg_app)
        elif getattr(attr_value, "name", None) == "segmentation":
            seg_app = attr_value
        if seg_app is not None:
            break

    assert seg_app is not None, "could not locate the registered `segmentation` sub-app"
    # Both commands should appear somewhere in the sub-app's command registry.
    payload = repr(vars(seg_app))
    assert "train" in payload
    assert "evaluate" in payload

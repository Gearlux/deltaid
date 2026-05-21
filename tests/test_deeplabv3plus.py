"""Tests for :class:`deltaid.deeplabv3plus.DeepLabV3PlusModel`."""

from pathlib import Path

import pytest
import torch

from deltaid.deeplabv3plus import DeepLabV3PlusModel

# DeepLabV3+ requires H, W to each be a multiple of 32 (encoder stride).
# 64 is the smallest valid square that keeps the test fast.
SIZE = 64


def _make_model(num_classes: int = 2, **kwargs: object) -> DeepLabV3PlusModel:
    """Build a tiny offline model — encoder_weights=None avoids any network IO."""
    return DeepLabV3PlusModel(
        num_classes=num_classes,
        encoder_name="resnet34",
        encoder_weights=None,
        **kwargs,  # type: ignore[arg-type]
    )


# --- Forward shape contract ------------------------------------------------


def test_forward_returns_per_pixel_logits_with_correct_shape() -> None:
    model = _make_model(num_classes=3)
    images = torch.rand(2, 3, SIZE, SIZE)
    logits = model(images)
    assert logits.shape == (2, 3, SIZE, SIZE)
    assert logits.dtype == torch.float32


def test_forward_handles_single_image_batch_in_eval_mode() -> None:
    """Single-image inference is the typical Evaluator path.

    BatchNorm in the resnet encoder cannot normalize a batch of 1 in train
    mode (variance is undefined), so the model must be flipped to ``eval()``
    for single-image forward — exactly what Lightning's ``test`` /
    ``predict`` paths do automatically.
    """
    model = _make_model(num_classes=2).eval()
    logits = model(torch.rand(1, 3, SIZE, SIZE))
    assert logits.shape == (1, 2, SIZE, SIZE)


def test_forward_supports_non_default_in_channels() -> None:
    model = DeepLabV3PlusModel(
        num_classes=2,
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
    )
    logits = model(torch.rand(2, 1, SIZE, SIZE))
    assert logits.shape == (2, 2, SIZE, SIZE)


# --- Input validation ------------------------------------------------------


def test_rejects_non_4d_input() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="4D"):
        model(torch.rand(3, SIZE, SIZE))  # missing batch dim


def test_rejects_wrong_channel_count() -> None:
    model = _make_model()  # default in_channels=3
    with pytest.raises(ValueError, match="in_channels"):
        model(torch.rand(2, 1, SIZE, SIZE))


def test_rejects_non_multiple_of_32_spatial_dims() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="multiples of 32"):
        model(torch.rand(2, 3, 50, 64))


def test_rejects_zero_num_classes() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        DeepLabV3PlusModel(num_classes=0, encoder_weights=None)


# --- Checkpoint round-trip -------------------------------------------------


def test_checkpoint_round_trip_via_init(tmp_path: Path) -> None:
    """Saving a state_dict and reloading via checkpoint_path produces identical weights."""
    src = _make_model(num_classes=2)
    ckpt_path = tmp_path / "model.pt"
    torch.save(src.state_dict(), ckpt_path)

    dst = DeepLabV3PlusModel(
        num_classes=2,
        encoder_name="resnet34",
        encoder_weights=None,
        checkpoint_path=str(ckpt_path),
    )
    for (k_src, v_src), (k_dst, v_dst) in zip(
        src.state_dict().items(), dst.state_dict().items()
    ):
        assert k_src == k_dst
        assert torch.equal(v_src, v_dst)


def test_checkpoint_strips_lightning_model_prefix(tmp_path: Path) -> None:
    """A Lightning-style ``state_dict`` with ``model.`` keys loads cleanly."""
    src = _make_model(num_classes=2)
    src_state = src.state_dict()
    lightning_state = {f"model.{k}": v for k, v in src_state.items()}
    ckpt_path = tmp_path / "lightning.ckpt"
    torch.save({"state_dict": lightning_state, "epoch": 1}, ckpt_path)

    dst = DeepLabV3PlusModel(
        num_classes=2,
        encoder_name="resnet34",
        encoder_weights=None,
        checkpoint_path=str(ckpt_path),
    )
    for k in src_state:
        assert torch.equal(src_state[k], dst.state_dict()[k])


def test_missing_checkpoint_raises() -> None:
    with pytest.raises(FileNotFoundError):
        DeepLabV3PlusModel(
            num_classes=2,
            encoder_name="resnet34",
            encoder_weights=None,
            checkpoint_path="/nonexistent/path/model.ckpt",
        )


# --- Differentiability -----------------------------------------------------


def test_logits_are_differentiable() -> None:
    """Confirm the forward path supports backprop end-to-end.

    Use batch=2 — encoder BatchNorm requires more than 1 value per channel
    in train mode (the realistic training path).
    """
    model = _make_model(num_classes=2)
    images = torch.rand(2, 3, SIZE, SIZE, requires_grad=True)
    logits = model(images)
    loss = logits.sum()
    loss.backward()
    # At least one parameter accumulated a non-zero grad.
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )

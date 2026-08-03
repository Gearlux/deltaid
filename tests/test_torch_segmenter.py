"""``TorchSegmenter`` — the control experiment: the same run with no training framework.

Two kinds of check here. The ones that could equally be asked of any backend (num_classes
resolution, the sentinels, the metric slots) exist because this backend DUPLICATES that logic
rather than sharing it, so a divergence between the two is exactly what would go unnoticed. The
ones that are this backend's alone are about what it does NOT have — no Trainer, no Lightning
import, no framework format — which is the whole claim it makes.
"""

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from confluid import LazyClass, lazy_param_names, to_pydantic
from recordstream import Mask

from deltaid.torch_segmenter import TorchSegmenter

NUM_CLASSES = 3
SIDE = 32


class _TinyNet(nn.Module):
    """A 1x1-conv 'segmenter': [N, 3, H, W] -> [N, C, H, W], trainable, instant."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(x))


def _records(n: int = 4, classes: int = NUM_CLASSES, learnable: bool = False) -> List[Dict[str, Any]]:
    """Synthetic segmentation records in the shape the config pipeline produces.

    ``learnable`` makes the mask the argmax over the input's channels, which a 1x1 conv can fit
    exactly — the only way a loss-descends assertion measures anything.
    """
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        image = torch.rand(3, SIDE, SIDE)
        mask = image.argmax(dim=0).numpy() if learnable else rng.integers(0, classes, (SIDE, SIDE))
        out.append({"image": image, "mask": Mask(mask.astype(np.int64)), "record_id": i})
    return out


def _segmenter(**kwargs: Any) -> TorchSegmenter:
    params: Dict[str, Any] = {
        "model": _TinyNet(),
        "num_classes": NUM_CLASSES,
        "max_epochs": 1,
        "batch_size": 2,
        "device": "cpu",  # explicit, so the suite is deterministic wherever it runs
    }
    params.update(kwargs)
    return TorchSegmenter(**params)


# --------------------------------------------------------------------------- #
# What this backend does NOT have — the claim it exists to make
# --------------------------------------------------------------------------- #
class TestItNeedsNoTrainingFramework:
    def test_the_module_never_imports_pytorch_lightning(self) -> None:
        """The whole point: remove the training FRAMEWORK and the config still does not change.

        AST-scanned rather than checked via `sys.modules`, because another test in the same
        process will have imported Lightning for the other backend.
        """
        source = Path(inspect.getfile(TorchSegmenter)).read_text()
        imported = {
            (node.module or "").split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "pytorch_lightning" not in imported
        # `matrainer.lightning` would pull it in transitively — that package's `__init__` runs
        # `require("pytorch_lightning")`, which is exactly how this regression happened before.
        assert "from matrainer.lightning import" not in source

    def test_it_has_no_trainer_slot(self) -> None:
        """No `lightning:` slot and no framework loggers — tracking goes through `trackers:`,
        which every backend has. Saying otherwise would be a worse Trainer."""
        params = inspect.signature(TorchSegmenter.__init__).parameters
        assert "lightning" not in params
        assert "logger" not in params

    def test_it_writes_a_plain_state_dict(self, tmp_path: Path) -> None:
        """There is no framework checkpoint format, because there is no framework."""
        seg = _segmenter(train_set=_records(4), checkpoint_dir=str(tmp_path), model=LazyClass(_TinyNet))
        seg.run()
        saved = torch.load(tmp_path / "model.pt", weights_only=True)
        assert "head.weight" in saved

    def test_the_device_knob_is_taken_at_its_word(self) -> None:
        """An explicit value never silently falls back, so a CPU-vs-GPU comparison is honest."""
        seg = _segmenter(device="cpu")
        seg.initialize()
        assert seg._device == torch.device("cpu")


# --------------------------------------------------------------------------- #
# The logic it duplicates from the Lightning backend — so it cannot drift
# --------------------------------------------------------------------------- #
class TestParityWithTheLightningBackend:
    def test_num_classes_resolution_agrees(self) -> None:
        """Model head wins; explicit next; the mask walk last."""
        assert _resolved(_segmenter(model=_TinyNet(num_classes=5), num_classes=9)) == 5
        assert _resolved(_segmenter(model=nn.Identity(), num_classes=7)) == 7
        assert _resolved(_segmenter(model=LazyClass(_TinyNet, num_classes=4), num_classes=None)) == 4
        assert _resolved(_segmenter(model=LazyClass(_TinyNet), num_classes=None, train_set=_records())) == 3

    def test_the_loss_default_matches(self) -> None:
        seg = _segmenter(ignore_index=250)
        seg.initialize()
        assert isinstance(seg.loss_fn, nn.CrossEntropyLoss)
        assert seg.loss_fn.ignore_index == 250

    def test_an_explicitly_wired_loss_is_never_replaced(self) -> None:
        chosen = nn.NLLLoss()
        seg = _segmenter()
        seg.loss_fn = chosen
        seg.initialize()
        assert seg.loss_fn is chosen

    def test_the_metric_defaults_match(self) -> None:
        seg = _segmenter()
        seg.initialize()
        assert set(seg.val_metrics.keys()) == {"val/miou", "val/dice_macro"}

    def test_a_split_slot_adds_to_the_shared_set_for_that_split_only(self) -> None:
        from torchmetrics.classification import MulticlassAccuracy

        seg = _segmenter(val_metrics={"accuracy": LazyClass(MulticlassAccuracy)})
        seg.initialize()
        assert set(seg.val_metrics.keys()) == {"val/miou", "val/dice_macro", "val/accuracy"}
        assert set(seg.train_metrics.keys()) == {"train/miou", "train/dice_macro"}

    def test_it_uses_the_shared_combination_rule(self) -> None:
        """One rule across every backend AND both tasks — never re-derived per backend."""
        assert "from matrainer.metric_slots import" in Path(inspect.getfile(TorchSegmenter)).read_text()

    def test_the_target_is_read_back_as_int64(self) -> None:
        from recordstream import collate_records

        seg = _segmenter()
        seg.initialize()
        assert seg._targets(collate_records(_records(2))).dtype == torch.int64


def _resolved(seg: TorchSegmenter) -> int:
    seg.initialize()
    assert seg.num_classes is not None
    return seg.num_classes


# --------------------------------------------------------------------------- #
# The contract every backend shares
# --------------------------------------------------------------------------- #
class TestTheSharedContract:
    def test_the_entrypoint_markers_are_the_dispatch_table(self) -> None:
        from recordstream.runnable import runnable_entrypoints

        assert set(runnable_entrypoints(TorchSegmenter)) == {"fit", "evaluate", "test", "predict"}
        assert "run_entrypoint(self, self.task)" in inspect.getsource(TorchSegmenter.run)

    def test_autograd_is_claimed_only_for_fit(self) -> None:
        assert _segmenter(task="fit").__needs_autograd__ is True
        assert _segmenter(task="predict").__needs_autograd__ is False

    def test_trackers_is_a_constructor_param_that_admits_a_list(self) -> None:
        """confluid gates LIST broadcast on the ANNOTATION — as a post-construction attribute a
        `trackers: [...]` YAML key would silently do nothing."""
        assert "trackers" in to_pydantic(TorchSegmenter).model_fields

    def test_all_four_metric_slots_are_declared(self) -> None:
        fields = to_pydantic(TorchSegmenter).model_fields
        assert {"metrics", "train_metrics", "val_metrics", "test_metrics"} <= set(fields)

    def test_the_deferred_slots_are_declared_lazy(self) -> None:
        assert {"model", "train_set", "optimizer", "train_loader"} <= lazy_param_names(TorchSegmenter)

    def test_no_local_hyperparameter_derivation_survives(self) -> None:
        """matrainer owns the ONE derivation, so two backends cannot answer differently."""
        source = Path(inspect.getfile(TorchSegmenter)).read_text()
        assert "_config_params" not in source
        assert "runnable=self" in source

    def test_the_terminal_call_is_the_one_that_ends_the_run(self) -> None:
        """`capture_log_file` finalizes every tracker — a backend calling `finalize` itself would
        be an eighth place for the tracker's status to disagree with the manifest's."""
        source = inspect.getsource(TorchSegmenter.fit)
        assert "capture_log_file" in source
        assert ".finalize(" not in source


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class TestTheLoop:
    def test_fit_trains_and_reports_both_splits(self) -> None:
        seg = _segmenter(train_set=_records(8), model=LazyClass(_TinyNet))
        seg.val_set = _records(4)
        seg.run()
        assert {"train/loss", "train/miou", "val/loss", "val/miou"} <= set(seg.computed_metrics)

    def test_the_loss_actually_descends(self) -> None:
        """A finite loss is not evidence of training — a badly wired one stays finite too.

        Learnable data on purpose: over random masks the loss would sit at ln(C) however well the
        optimizer was wired, so the test would measure nothing.
        """
        seg = _segmenter(train_set=_records(8, learnable=True), model=LazyClass(_TinyNet), max_epochs=12)
        seg.run()
        losses = seg.history["train/loss"]
        assert len(losses) == 12
        assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_the_history_output_is_per_epoch(self) -> None:
        """What a Trainer's logger would have accumulated — this loop has to keep it itself."""
        seg = _segmenter(train_set=_records(4), model=LazyClass(_TinyNet), max_epochs=3)
        seg.run()
        assert len(seg.history["train/loss"]) == 3

    def test_grad_clip_is_applied_when_set(self) -> None:
        seen: List[float] = []
        original = torch.nn.utils.clip_grad_norm_

        def _spy(params: Any, max_norm: float, *args: Any, **kwargs: Any) -> Any:
            seen.append(max_norm)
            return original(params, max_norm, *args, **kwargs)

        seg = _segmenter(train_set=_records(4), model=LazyClass(_TinyNet), grad_clip=0.5)
        torch.nn.utils.clip_grad_norm_ = _spy  # type: ignore[assignment]
        try:
            seg.run()
        finally:
            torch.nn.utils.clip_grad_norm_ = original  # type: ignore[assignment]
        assert seen and set(seen) == {0.5}

    def test_test_writes_a_metrics_sidecar(self, tmp_path: Path) -> None:
        import json

        seg = _segmenter(task="test", model=_TinyNet())
        seg.test_set = _records(4)
        seg.metrics_sidecar_path = str(tmp_path / "metrics.json")
        seg.run()
        payload = json.loads((tmp_path / "metrics.json").read_text())
        assert "test/miou" in payload["metrics"]

    def test_evaluate_scores_the_validation_split(self) -> None:
        """`evaluate` and `test` differ in the SPLIT, which is the only thing that differs."""
        seg = _segmenter(task="evaluate", model=_TinyNet())
        seg.val_set = _records(4)
        seg.run()
        assert "val/miou" in seg.computed_metrics

    def test_predict_streams_every_record_to_the_sink_and_closes_it(self) -> None:
        written: List[Any] = []
        closed: List[bool] = []

        class _Sink:
            def write(self, prediction: Any, metadata: Any) -> None:
                written.append((prediction, metadata))

            def close(self) -> None:
                closed.append(True)

        seg = _segmenter(task="predict", model=_TinyNet())
        seg.test_set = _records(6)
        seg.predictions_sink = _Sink()
        seg.run()

        assert len(written) == 6
        assert [m["record_id"] for _, m in written] == [0, 1, 2, 3, 4, 5]
        assert set(written[0][0]) == {"logits", "probs", "mask"}
        assert closed == [True]

    def test_the_outputs_raise_before_a_run(self) -> None:
        seg = _segmenter(model=LazyClass(_TinyNet))
        with pytest.raises(RuntimeError, match="only available after run"):
            _ = seg.computed_metrics

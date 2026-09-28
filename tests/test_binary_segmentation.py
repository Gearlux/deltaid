"""Binary segmentation is ``num_classes: 2`` — the evidence for that decision, pinned.

There is ONE target contract here (``[N, C, H, W]`` logits against an int64 ``[N, H, W]`` class-id
mask) and binary is ``C = 2``. smp also offers ``mode="binary"``, which expects a single-channel
``[N, 1, H, W]`` prediction, so the standing question was whether that earns a branch.

It does not, and the reason is measurable rather than stylistic: **what a binary setup actually
buys is not scoring the BACKGROUND**, and smp's own ``classes=`` knob buys that from the contract
that already exists. These tests are the numbers that settle it — they exist so a future reader
re-opening "should we add a binary mode?" finds the answer executed rather than asserted, and so
that an smp release changing either behaviour fails here.

See ``docs/architecture.md`` record 10.
"""

import segmentation_models_pytorch as smp
import torch
from torchmetrics.classification import MulticlassJaccardIndex

#: A heavily imbalanced mask — ~2% foreground. This is the regime `mode="binary"` exists for; at
#: balanced classes the dilution below is invisible, which is exactly why it catches people out.
SIDE = 32


def _imbalanced_case() -> tuple:
    """A ~2%-foreground target, and a model predicting NOTHING but background.

    The worst useful model there is: it is right about 98% of the pixels and has learned nothing.
    Every number below is about whether a given loss or metric SAYS so.
    """
    target = torch.zeros(2, SIDE, SIDE, dtype=torch.long)
    target[:, :3, :7] = 1
    background_logit = torch.full((2, 1, SIDE, SIDE), -5.0)
    logits = torch.cat([-background_logit, background_logit], dim=1)  # [N, 2, H, W]
    return logits, target


def test_the_fixture_is_actually_imbalanced() -> None:
    """Otherwise every assertion below is vacuous."""
    _, target = _imbalanced_case()
    assert float(target.float().mean()) < 0.05


def test_a_plain_multiclass_dice_DILUTES_the_failure() -> None:
    """The trap, stated as a number: the background's near-perfect score halves the loss, so a
    model that found nothing scores as though it were half right."""
    logits, target = _imbalanced_case()
    diluted = float(smp.losses.DiceLoss(mode="multiclass")(logits, target))
    assert 0.45 < diluted < 0.55, diluted


def test_smps_own_classes_knob_recovers_the_binary_signal() -> None:
    """`classes=[1]` scores the FOREGROUND only — from the same `[N, 2, H, W]` tensor. This is
    the whole reason a single-channel branch is not needed: the knob is already there."""
    logits, target = _imbalanced_case()
    foreground_only = float(smp.losses.DiceLoss(mode="multiclass", classes=[1])(logits, target))
    assert foreground_only > 0.99, foreground_only


def test_it_agrees_with_what_a_true_binary_loss_would_have_said() -> None:
    """The claim the decision rests on: `classes=[1]` over C=2 and `mode="binary"` over a
    single channel report the SAME failure, so the branch would buy no information."""
    logits, target = _imbalanced_case()
    single_channel = logits[:, 1:2]  # what a `[N, 1, H, W]` head would have emitted
    binary = float(smp.losses.DiceLoss(mode="binary")(single_channel, target))
    foreground_only = float(smp.losses.DiceLoss(mode="multiclass", classes=[1])(logits, target))
    assert binary > 0.98 and foreground_only > 0.98
    assert abs(binary - foreground_only) < 0.02, (binary, foreground_only)


def test_a_macro_averaged_metric_dilutes_it_the_same_way() -> None:
    """Not just the loss — the default metric hides it too, which is why the per-class report
    exists and why the README says to wire it for an imbalanced task."""
    logits, target = _imbalanced_case()
    macro = float(MulticlassJaccardIndex(num_classes=2)(logits, target))
    assert 0.45 < macro < 0.55, macro


def test_the_PER_CLASS_metric_names_the_failure_outright() -> None:
    """`average=None` is the answer the config recommends: it reports 0.0 for the class the model
    never found, and the per-class table renders worst-first so that 0.0 is the FIRST row."""
    logits, target = _imbalanced_case()
    per_class = MulticlassJaccardIndex(num_classes=2, average=None)(logits, target)
    assert float(per_class[0]) > 0.95, "the background is nearly perfect, which is the problem"
    assert float(per_class[1]) == 0.0, "the foreground was never found, and the vector says so"


def test_the_segmenter_grew_no_binary_mode_knob() -> None:
    """The decision, pinned where a future change would trip on it: binary is `num_classes: 2`
    and a config knob, never a second tensor shape."""
    import inspect

    from deltaid.lightning_segmenter import LightningSegmenter

    params = inspect.signature(LightningSegmenter.__init__).parameters
    assert "mode" not in params
    assert "binary" not in params

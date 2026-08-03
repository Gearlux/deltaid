# deltaid — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

  - [ ] **The remaining two backends** @feature — six ship (`lightning`, `torch`, `fastai`,
    `keras`, `tensorflow`, `huggingface`); **jax** and **mlx** drop in as
    `deltaid/<name>_segmenter.py` plus a row in `cli.FRAMEWORKS`, a `_BACKEND_NAMES` entry, an
    extra, and an entry point. Each ends with a real training run, not just a green suite.
  - [ ] **The rest of the shared U-Net spec — `flax_unet.py` / `mlx_unet.py`** @feature —
    `spec.py`, `TorchUNet`, `KerasUNet` and `TensorFlowUNet` ship and all three agree with the
    DERIVED parameter count (488,739 for 3 classes), including under custom `widths`. Note the
    three portability choices the spec encodes — parameter-free NEAREST upsampling, the
    `size_divisor` rule, and the channels-LAST `input_shape` knob — and the MLX wrinkle its
    classification twin records: MLX keeps batch-norm running stats in the SAME parameter tree, so
    `parameter_count` must exclude them by name or the match breaks.
  - [ ] **A per-split Keras metric, once Keras grows a hook for it** @question —
    `KerasSegmenter` can only UNION its four metric slots, because `keras.Model.compile` takes one
    list and reports it on both splits; the TensorFlow backend, whose loop is ours, merges them
    properly. Revisit if Keras ever exposes a phase-scoped metric set. Until then the union is
    documented and pinned, not worked around by recompiling between phases (which would discard
    the optimizer state a fit is using).
  - [ ] **Report a per-class metric instead of dropping it** @feature — `_report_epoch_metrics`
    passes only `scalar_metrics(...)` to `log_dict`, because Lightning's
    `convert_tensors_to_scalars` RAISES on anything that is not one number. So a wired
    `MulticlassJaccardIndex(average=None)` — the per-class IoU a segmentation report actually wants
    — is computed and then silently discarded. The classification task solved the same problem for
    confusion matrices (`recordstream.ops.image.confusion_matrices_payload` + marainer's
    table/figure renderers), but that path is SHAPE-specific: it matches square 2-D values, and a
    per-class IoU is a `[C]` VECTOR. This needs its own renderer (a class-name → value table, plus
    a bar chart for `log_figure`), living beside the existing ones in marainer rather than here.
    Do NOT close this by reusing the confusion-matrix path.
  - [ ] **Binary segmentation without a background class** @question — the current contract is
    multiclass logits `[N, C, H, W]` + an int64 class-id mask, which covers binary as `C=2`. A true
    single-channel `[N, 1, H, W]` + `BCEWithLogitsLoss` setup — what smp's `mode="binary"` losses
    expect — would need a different target shape and metric family. Decide whether that earns a
    branch, or whether `C=2` stays the one answer.

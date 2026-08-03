# deltaid — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

  - [ ] **A per-split Keras metric, once Keras grows a hook for it** @question —
    `KerasSegmenter` can only UNION its four metric slots, because `keras.Model.compile` takes one
    list and reports it on both splits; the TensorFlow backend, whose loop is ours, merges them
    properly. Revisit if Keras ever exposes a phase-scoped metric set. Until then the union is
    documented and pinned, not worked around by recompiling between phases (which would discard
    the optimizer state a fit is using).
  - [ ] **Binary segmentation without a background class** @question — the current contract is
    multiclass logits `[N, C, H, W]` + an int64 class-id mask, which covers binary as `C=2`. A true
    single-channel `[N, 1, H, W]` + `BCEWithLogitsLoss` setup — what smp's `mode="binary"` losses
    expect — would need a different target shape and metric family. Decide whether that earns a
    branch, or whether `C=2` stays the one answer.

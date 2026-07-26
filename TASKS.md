# deltaid — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

  - [ ] **Restore dataset-derived `num_classes` for raidar/deltaid via `initialize()`** @feature — parity with sonair: give `LightningDetectionTrainer` / `LightningSegmentationTrainer` an `initialize()` that resolves `num_classes` (detection/segmentation targets aren't scalar class-ids, so `sampleflux.num_classes` doesn't apply directly — needs a per-task derivation) and flows the model + deferred metrics with it. Until then both require explicit `num_classes`.

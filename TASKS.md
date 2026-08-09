# deltaid — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

- [ ] **Seed the zoo wrappers the way distilaide (2026-08-06) and the classification project (2026-08-09) seeded theirs** @medium @reproducibility — `SegmentationModelPytorch` and `HuggingFaceSegmentationModel` carry no `seed:` knob, so every `*_zoo` sweep row's init draws from ambient RNG — measured on the 2026-08-09 MLflow refresh (`Sun_Aug_9_2026_2` vs `Sat_Aug_8_2026_7`): all four zoo rows moved (fastai_zoo mIoU 0.7455→0.7271, huggingface_zoo 0.5619→0.6112, lightning_zoo 0.7416→0.7383, torch_zoo 0.7520→0.7405) while five of the six always-seeded unet rows were bit-exact (tensorflow_unet drifts every sweep — tf-engine nondeterminism, a separate known class). The pattern is established twice over: seed at BUILD time without restoring RNG state (`torch.manual_seed` before `smp.create_model`; for the HF wrapper the seed belongs on the SEGMENTER via `transformers.set_seed` before the model flows + joined into `_argument_floor` — the detection project's 2026-08-06 fix, mirrored in classification 2026-08-09). Knob-BINDS tests per wrapper (same seed → identical weights); the classifier knob restales navigaitor's generated segmentation examples — regenerate in the same change.

**Everything else empty as of 2026-08-05.** The eight backends ship, the shared spec has five implementations, and
everything the backend sweep (`examples/sweep/`) turned up has been closed: the Lightning training
metrics, the four missing `train_metrics` slots, the per-batch aggregation in the JAX and MLX
loops, and fastai's missing `pixel_accuracy`. See `AGENTS.md` and `docs/architecture.md` records
12–14.

The sweep's report was lifted to `matrainer.sweep` on 2026-08-06, when a third task project
needed one; `examples/sweep/` keeps the matrix and the per-engine quirks, which are this
project's half.

# deltaid — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.


**Empty as of 2026-08-09.** The eight backends ship, the shared spec has five implementations, and
everything the backend sweep (`examples/sweep/`) turned up has been closed: the Lightning training
metrics, the four missing `train_metrics` slots, the per-batch aggregation in the JAX and MLX
loops, and fastai's missing `pixel_accuracy`. See `AGENTS.md` and `docs/architecture.md` records
12–14.

The sweep's report was lifted to `matrainer.sweep` on 2026-08-06, when a third task project
needed one; `examples/sweep/` keeps the matrix and the per-engine quirks, which are this
project's half.

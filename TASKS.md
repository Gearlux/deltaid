# deltaid — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

- [ ] **Docs and comments describe the old surface** @docs @small — nine places still describe deltaid as it was before the matrainer bases and `matrainer sweep` (the `run_sweep.sh` references are on the workspace backlog; these are the ones beside them):
  - `deltaid/README.md:424` and `:438` and `deltaid/docs/architecture.md:612` — "the two blocks whose paths move under `sweep_logs/`", "Two things the script does", and record 12's account of the script; the overlay writes `./runs/exported/…` and `./runs/hf_runs` (`examples/sweep/all_backends.yaml`, pinned by `tests/test_sweep_config.py::test_the_sweep_writes_everything_under_one_tree`) and there is no script.
  - `deltaid/tests/test_sweep_config.py:40` — "The matrix the driver script walks"; the matrix is `examples/sweep/sweep.yaml`, walked by `matrainer sweep`.
  - `deltaid/pyproject.toml:196-197` — says a zoo wrapper's entry point is `deltaid-<backend>-model`; the table below it names `deltaid-smp-model`, `deltaid-torch-unet` and `deltaid-flax-unet` (`:201-226`).
  - `deltaid/deltaid/_keras.py:10-11` — "Every deltaid module writes `from deltaid._keras import keras`"; `deltaid/keras_segmenter.py:46` and `deltaid/segmentation_keras.py:42` import it from `recordstream.keras`.
  - `deltaid/deltaid/smp_model.py:241` — names "SegFormer, DPT" as the architectures with `requires_divisible_input_shape = False`; in the installed smp it is `False` on `Unet` only (`decoders/unet/model.py:103`), `True` on DPT (`decoders/dpt/model.py:77`) and inherited `True` for SegFormer (`base/model.py:20`); README `:146` and architecture §2 correctly name `Unet`.
  - `deltaid/examples/train_unet_oxford_pets.yaml:107-108` — "The nested source needs the INSTANCE form — `!class:...()` WITH parens — or `Stream.source` stays a deferred Class marker"; both spellings build now (`recordstream/recordstream/core/stream.py:56-61`).
  - `deltaid/README.md:12-13` — "no framework of our own sits in between"; every segmenter is a matrainer machinery base plus task hooks (`docs/architecture.md` §15: "matrainer is load-bearing for training").
  - `deltaid/docs/architecture.md:296,303-305,363` (record 7) — say fastai, jax and mlx score val and test only and that fastai has no `train_metrics` slot; record 13 changed that for all but huggingface, with no amendment note on record 7. `:371` says the registrations field is `Optional[str]`; `matrainer/matrainer/cli.py:94` types it `Union[None, str, Tuple[str, ...]]`.
  - `deltaid/deltaid/_tensorflow.py:19` — says `TensorFlowSegmenter.initialize()` fails on a wrong backend; the engine guard runs in `run()` before `initialize()` (`matrainer/matrainer/tensorflow/loop.py:171-174`).

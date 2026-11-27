# Deltaid Mandates

Rules only. The reasons are in [`docs/architecture.md`](docs/architecture.md) (cited as §N), and
usage is in the [README](README.md). Root `AGENTS.md` rules are not repeated here.

## Current state

Semantic segmentation on eight backends, all driven by one config: `lightning`, `torch` (a
hand-written loop), `fastai`, `keras`, `tensorflow`, `huggingface`, `jax`, `mlx`. On PyPI as
pre-release `0.1.0aN` (the version in `pyproject.toml`); the repo is public.

Each runnable `<backend>_segmenter.py` = task hooks (here) + a matrainer machinery base + its own
full flat `__init__`:

| backends | task hooks | matrainer base |
|---|---|---|
| lightning · torch · fastai | `SegmentationTorchHooks` | `LightningRunnable` · `TorchLoopRunnable` · `FastaiRunnable` |
| keras · tensorflow | `SegmentationKerasHooks` | `KerasRunnable` · `TensorFlowRunnable` |
| huggingface · jax · mlx | `SegmentationBase` | `HuggingFaceRunnable` · `JaxLoopRunnable` · `MlxLoopRunnable` |

Other modules:
- `segmentation_base.py` (torch-free task base); `segmentation_torch.py` / `segmentation_keras.py`
  (per-family hooks). Not `torch_segmentation.py` / `keras_segmentation.py`, which register the
  upstream losses and metrics.
- Zoo wrappers: `smp_model.py` (`SegmentationModelPytorch`, twelve architectures),
  `huggingface_model.py` (`HuggingFaceSegmentationModel`).
- `fno_model.py` (`FourierNeuralOperator`): a torch model that wraps nothing, any picture size. §18.
- `unet/`: `spec.py` plus five implementations of one U-Net (torch, keras, tensorflow, flax, mlx).
- `_keras.py` / `_tensorflow.py`: import shims. `cli.py`: the `FRAMEWORKS` table only, on
  `matrainer.cli`. `__init__.py`: lazy exports.

Example config: `examples/train_unet_oxford_pets.yaml`. Sweep: `examples/sweep/`.

**Who owns what:** recordstream owns records, ops (`ConvertToMask`, the joint albumentations
resize), collate, `batch_tensor`, `num_mask_classes` and the fork guards. matrainer owns the
trainer machinery, CLI factory, tracking, metric naming/slots/functions and the sweep driver.
A signal-domain project owns region rasterization and signal-domain sinks. Never add a collate, mask op, batch
reader, remap knob on `ConvertToMask`, or CLI helper here. §3.

## Backends

- Keep each backend's FULL flat `__init__`; introspection reads it. Extend through the hooks.
  Never shadow the machinery (`fit`, `initialize`, `_build_trainer`, `_run_eval`,
  `_report_epoch_metrics`, `_score`, `_epoch_metrics`, `_log_run_context`, `_callbacks`,
  `_run_epoch`, `_live_trackers`). Never write a `{task: method}` dict, a private hyperparameter
  derivation or a sidecar writer. A machinery change goes into matrainer with every consumer
  updated. (`tests/test_tracking_parity.py`, `::test_the_entrypoint_markers_are_the_dispatch_table`) §15.
- `torch` imports no training framework and has no Trainer slot, callbacks or distributed
  strategy. (`tests/test_torch_segmenter.py::TestItNeedsNoTrainingFramework`) §7.
- `fastai`: the default loss is `CrossEntropyLossFlat(axis=1, ignore_index=…)`, because fastai's
  `axis=-1` breaks on `[N, C, H, W]`. Metrics are NAMES; the one class is the `PixelAccuracy`
  accumulator behind `_metric_aliases()`. Never wrap an upstream metric. (`tests/test_fastai_segmenter.py`)
- `keras`/`tensorflow`: the default loss takes `from_logits=True` and `MeanIoU` takes
  `sparse_y_pred=False`. `keras` UNIONS its four metric slots, and that is final: never swap
  `model._compile_metrics`; narrow per split with `--framework tensorflow`. On the torch engine
  `_check_compiled_metrics` warns about `MeanIoU`; never substitute a cheaper default per engine.
  The keras module never sets `KERAS_BACKEND`. (`tests/test_keras_segmenter.py`) §7.
- `tensorflow` is a sibling of `KerasSegmenter`, never a subclass, tagged `framework="keras"`,
  with no `tensorflow_segmentation.py`. It imports keras/tf only through `deltaid._tensorflow`.
  (`tests/test_tensorflow_segmenter.py`) §7.
- `huggingface` has no loss slot and no registrations. Its `ignore_index` goes onto the model
  config from the `_resolve_task_defaults` hook, never an `initialize` override. Logits are
  upsampled before any metric or sink, and the channel axis moves last before
  `accumulate_confusion`. `TrainingArguments` defaults are matrainer's run-time floor; never pin
  them on the slot. SegFormer fails backward on MPS: keep the example's `use_cpu: true` line, and
  never add a silent CPU fallback. (`tests/test_huggingface_segmenter.py`) §7.
- `jax`/`mlx`: steps are jitted ONCE (`tests/test_jax_segmenter.py::test_the_steps_are_jitted_ONCE_and_reused`).
  `ignore_index` is masked in `_loss_fn` (mean over valid pixels, ignored label clamped first).
  Losses, metrics and optimizers are NAMES, so there is no registrations module. The eval path
  overrides `_initialize_eval` to pass the width into the flow.
- `mlx`: int32 targets, the optimizer name is INSTANTIATED (optax's is called), `mx.eval` once per
  step, train/eval mode set per phase, an Apple-silicon marker on the extra. (`tests/test_mlx_segmenter.py`)

## Config and CLI

- The backend is a flag: `--framework X` → `--scope framework=X`. No backend subcommands, no CLI
  machinery in `cli.py`. `main()` delegates to matrainer's `trainer_main`, which setdefaults
  `KERAS_BACKEND=tensorflow` for `--framework tensorflow` BEFORE registering; no other framework
  touches it. (`tests/test_cli_frameworks.py`) matrainer record 10.
- Every backend is a `!scope:framework=<name>` block holding its `runnable:`;
  `default_scopes: [framework=lightning]` names the default. `runnable`, `model`, `metrics` and
  `preprocess` are TOP-LEVEL siblings; never nest `model:` in `runnable:`.
  (`::test_every_declared_framework_resolves_to_a_complete_runnable`)
- Channels-last blocks (keras, tensorflow, jax, mlx) restate `preprocess` without the final
  `ToTensor`; huggingface keeps the torch chain. A torchmetrics mapping reaches only lightning and
  torch; the fastai block restates `metrics:` as names, and the other engines take their metrics in
  the runnable's kwargs. (`tests/test_sweep_config.py::test_a_metric_set_never_reaches_an_engine_that_cannot_consume_it`)

## Registrations and discovery

- Register upstream classes; never reimplement one. One registration module per loss API:
  `torch_segmentation.py` (lightning, torch, fastai) and `keras_segmentation.py` (keras,
  tensorflow); huggingface, jax and mlx have `registrations=None`. Every registered class goes in
  the module's literal `__all__`. (`tests/test_categories.py`)
- Every registering module gets its `confluid.configurables` entry point in the SAME change, then
  `uv pip install -e ./deltaid --no-deps`. Entries are modules, never `module:attr`.
  (`tests/test_entry_points.py`)
- Every class carries `task="segmentation"`, a `role` and a `framework` (the API it is wired to:
  the TensorFlow classes are `keras`). One class per backend; no `role="evaluator"`.
- `torch_segmentation.py` imports torch at module level ON PURPOSE, so a missing extra drops the set.
- A new constructor param makes navigaitor's generated segmentation example stale: run
  `python -m navigaitor.scripts.regen_examples` in the same change.

## Task pipeline

- Targets: an int64 `[N, H, W]` class-id mask (int32 on mlx) against `[N, C, H, W]` logits,
  `target_key="mask"`. Name int64 explicitly (albumentations returns int32). No one-hot.
  (`::test_the_target_is_read_back_as_int64`)
- Binary is `num_classes: 2`. Never add a `mode:` knob or a `[N, 1, H, W]` branch.
  (`tests/test_binary_segmentation.py`) §10.
- `num_classes`: model head → explicit → lazy kwarg → the announced mask walk; eval reads the
  model's head, never the eval data. Masks use `num_mask_classes`; never widen
  `recordstream.num_classes`. (`tests/test_lightning_segmenter.py::TestNumClassesResolution`) §4.
- `loss_fn`/`metrics` default to `None` and resolve idempotently; a wired one is never replaced.
  The default pair `mean_iou` + `dice_macro` takes the segmenter's `ignore_index`. (`::TestSentinelDefaults`)
- Every split goes through `_prepare` (`ensure_materialized`). Never answer a worker SIGSEGV with
  `num_workers: 0`. (`tests/test_tracking_parity.py::test_every_backend_warms_its_sources_before_forking`) §8.
- Predictions are `SegmentationOutput`, never raw logits.
  (`::test_predict_step_emits_the_contract_not_raw_logits`)

## Metrics, slots, outputs

- `metrics` + `train_/val_/test_metrics` are constructor params, combined only through
  `matrainer.metric_slots`. Separate per split: lightning, torch, tensorflow, jax, mlx. Union:
  keras and a fastai fit. val + test only: huggingface. Never fill a train column by an eval-mode
  rerun. §13.
- jax, mlx and huggingface accumulate NAMED metrics over a per-split confusion matrix
  (`matrainer.metric_functions`), never a mean of per-batch values; a user callable stays
  batch-averaged. (`::test_the_metric_does_not_depend_on_the_BATCH_SIZE`) §14.
- Every backend takes `metric_names` (stored verbatim) and records `<split>/<name>`. Default names
  are words every engine can produce (`mean_iou`, `pixel_accuracy`). Never translate a name a user
  wrote; `renames` is for engine-generated names. (`tests/test_tracking_parity.py`) §11.
- Per-class VECTORS go to the trackers' text channel (plus a figure with `plot_metrics`),
  worst-first, never through `log_dict`/`log_metrics`, and never via classification's
  confusion-matrix rendering. `class_names` is a config knob. (`::TestThePerClassReport`)
- `trackers` is a constructor param, `Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]]`,
  stored verbatim. There is no `logger:` slot. (`::test_no_backend_has_a_logger_slot`) §17.
- Every slot flowed at run time is `Partial[T]`; a `model` slot names its framework's type, never
  `Partial[Any]`; dataset slots use `recordstream.RecordSource`. (`tests/test_lazy_slots.py`)

## Models

- `SegmentationModelPytorch` (`tests/test_smp_model.py`) §1, §2:
  - Its `arch` Literal equals smp's set in both directions; the runtime tuple is `get_args()`.
  - `solidify()` keeps its own `arch` guard.
  - The build is cached in the public submodule `segmenter`, never `model` (it would break
    `load_lightning_checkpoint`'s prefix strip), shared by `solidify()`, `backbone` and `forward`.
  - smp kwargs pass through verbatim. `forward` checks only 4-D-ness and channels; never add a
    spatial-divisibility check.
- `model_refs` reads stored config only, returns `[]` when nothing external was loaded, and uses
  `hf_model_ref`/`file_model_ref`. A new wrapper with external weights adds it. (`tests/test_model_refs.py`)
- A model whose build draws random numbers takes `seed`, applied at BUILD time without restoring
  the RNG, on the wrapper rather than the segmenter. (`tests/test_smp_model.py::test_the_seed_actually_reproduces_the_weights`,
  `tests/test_huggingface_segmenter.py::TestTheModelWrapperSeedsItsBuild`, `TestKerasUNetMatchesTheSpec`) §16.
- `FourierNeuralOperator` (`tests/test_fno_model.py`) §18: the build is cached in the public
  submodule `net`; the forward stays TorchScript-compilable (no `typing.cast` in `_FNO` or
  `_SpectralConv2d`); a picture smaller than the modes uses the lowest ones it has.
- `unet/` (`tests/test_unet_spec.py`) §6:
  - `__init__.py` imports nothing; `spec.py` only the standard library.
  - Every implementation is checked against `spec.trainable_parameter_count` and IS its
    framework's module type. `TorchUNet` serves all three torch backends; never merge one into a
    zoo module.
  - MLX keeps the built network in the public `net` (underscore attributes are invisible to
    `parameters()`); Flax wraps the lazy build in `nnx.data`.
  - A raw-`tf.nn` module never touches `tf.keras`, `tf.initializers`, `tf.losses` or
    `tf.metrics`. §9.

## Packaging, release, tests

- `pip install deltaid` installs no framework; every backend name is lazy through
  `_BACKEND_NAMES`. Keep it in step with the extras *(unpinned beyond the release smoke test)*.
- Releases go through the hand-written `release.yml` from the public repo. Each release raises
  every workspace floor, core and extras, to the newest release and runs the suite there, and a dependent's next
  release raises its `deltaid>=` the same way (user choice 2026-11-26; aisland `tests/test_workspace_dependency_specs.py`
  checks each is satisfiable); README links are
  absolute GitHub URLs (`tests/test_docs_links.py`); the newest CHANGELOG entry is the declared
  version (`tests/test_packaging.py`).
- CI runs `mypy .` here with only `[dev,torch,lightning,keras,huggingface,mlflow,tensorboard]`:
  list every extra-gated import in `[[tool.mypy.overrides]]`, and expect `warn_unused_ignores` to
  fail a dead `# type: ignore` the root run passes. Reproduce CI in a clean Python 3.12 venv with
  CI's install line before pushing. `--cov-branch` is required (`tests/test_coverage_config.py`).
- CI does NOT run the fastai, tensorflow, jax or mlx suites; they skip without their extra. Run
  the TensorFlow ones with `KERAS_BACKEND=tensorflow pytest tests/test_tensorflow_segmenter.py`.

## Sweep

- `examples/sweep/sweep.yaml` is the 11-cell matrix and `all_backends.yaml` its overlay; the driver
  is `matrainer sweep backends|trackers|all examples/sweep/sweep.yaml`. (`tests/test_sweep_config.py`) §12.
- In the overlay: restate a framework block in full (a scope block replaces, never merges); put
  per-block metrics in the runnable's kwargs (a key only the overlay adds lands last and wins);
  non-torch blocks need no nested `unet:`; tracker blocks use `!ref:experiment_name` /
  `!ref:run_name`; run outputs go under `./runs/`.
- The keras row pins `KERAS_BACKEND=tensorflow` (the spec's `env:`). `num_workers` stays non-zero
  so the fork path runs. Never hide a ragged column by trimming the report.

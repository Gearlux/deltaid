# Deltaid

**Semantic segmentation across training backends, driven by one config.**

One `runnable:` + `task:` YAML, three verbs, and a model slot that reaches every architecture
[`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) ships.
**Eight** training backends — **PyTorch Lightning**, a **hand-written torch loop**, a
**fastai `Learner`**, **Keras 3**, a native **TensorFlow** `tf.data` + `GradientTape` loop,
**HuggingFace `transformers`**, a jitted **JAX/optax** loop and Apple's **MLX** — and swapping
between them is one flag, not one config.

Each segmenter reads like a textbook example for its framework: no framework of our own sits in
between.

```bash
pip install 'git+https://github.com/Gearlux/deltaid.git@main#egg=deltaid[lightning]'
# or: deltaid[torch] / [fastai] / [keras] / [tensorflow] / [huggingface] / [jax] / [mlx]

deltaid train    examples/train_unet_oxford_pets.yaml
deltaid evaluate examples/train_unet_oxford_pets.yaml
deltaid predict  examples/train_unet_oxford_pets.yaml
deltaid frameworks                      # what the backends are
recordstream run examples/train_unet_oxford_pets.yaml   # the generic runner, no verb
```

The core install is **backend-free**: `pip install deltaid` gives you the CLI and the config
surface, and a training framework arrives with the backend you select (`deltaid[lightning]`). Every
backend-owned name resolves lazily, so `import deltaid` pulls in no ML framework at all.

## The example: Oxford-IIIT Pet

`examples/train_unet_oxford_pets.yaml` trains end to end on the "MNIST of segmentation" — 3680
pet photos, each with a **trimap**: a greyscale PNG whose pixels are class ids, `1` = the pet,
`2` = the background, `3` = the boundary ring.

```
deltaid train examples/train_unet_oxford_pets.yaml                     # lightning
deltaid train examples/train_unet_oxford_pets.yaml --framework torch   # same config
deltaid train examples/train_unet_oxford_pets.yaml --framework fastai  # same config
```

Five epochs on a 256-row slice, changing only the flag:

| `--framework` | result |
|---|---|
| `lightning` | train/loss 0.281, val/loss 0.261 |
| `torch` | **val mIoU 0.757**, Dice 0.846 (per-epoch metrics on both splits) |
| `fastai` | **dice_multi 0.825**, jaccard 0.726 |

Two things about that dataset are silent when handled wrong, and the config encodes both:

1. **The trimap is 1-based.** A per-pixel CrossEntropy over 3 classes expects ids in `[0, 2]`, so
   `3` would index past the head.
2. **The image and the mask must be resized by the same draw.** Resizing them separately — or
   interpolating the mask — invents class ids that were never in the label scheme.

## The preprocessing chain

Segmentation needed exactly **one** new op; everything else already existed:

```yaml
preprocess:
  - !class:recordstream.ops.image.ConvertToMask   # the trimap PNG -> an int64 [H, W] Mask
    field: segmentation_mask
    output: mask                                  # albumentations' own key vocabulary
  - !class:recordstream.ops.formula.FormulaOp     # 1-based trimap -> 0-based class ids
    field: mask
    formula: a - 1
  - !class:recordstream.ops.structure.DropField   # a PIL left in the record breaks collate
    key: segmentation_mask
  - !class:albumentations.Resize                  # ONE joint draw over image AND mask
    height: 224
    width: 224
  - !class:albumentations.Normalize               # image only — the mask passes through
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  - !class:recordstream.ops.torch.ToTensor        # HWC -> CHW, for the IMAGE only
    field: image
```

`output: mask` is load-bearing rather than cosmetic: it is the key albumentations itself uses, so
recordstream's op-family dispatch hands the transform **both** `image` and `mask` in one call and a
single draw moves them together. An image-only transform like `Normalize` still touches the image
alone.

`ToTensor` only converts HWC to CHW; the pixels it receives are already float32, standardized by
`Normalize`.

## Models — a zoo wrapper, and a shared spec

Two kinds, and they answer different questions:

| | wraps | knobs are about | lives in |
|---|---|---|---|
| `SegmentationModelPytorch` | somebody else's weights (smp) | the zoo — `arch`, `encoder_name`, `encoder_weights` | the top level |
| `HuggingFaceSegmentationModel` | somebody else's weights (the Hub) | the zoo — `model_name`, `pretrained` | the top level |
| `TorchUNet` / `KerasUNet` / `TensorFlowUNet` / `FlaxUNet` / `MlxUNet` | **nothing** | the architecture — `widths`, `bottleneck_width`, `seed` | `deltaid/unet/` |

For real work prefer a zoo wrapper: a small net trained from scratch loses to a pretrained encoder
every time (in the table above, SegFormer-b0 reaches mIoU 0.659 where the from-scratch U-Nets reach
0.35–0.46 on the same slice). The `unet/` package exists because **the zoos do not line up** — smp
is torch-only and `keras.applications` ships no segmentation models, so a cross-framework comparison
built on a zoo could never include Keras, TensorFlow, Flax or MLX. `deltaid/unet/spec.py` states one
U-Net once and *derives* the trainable-parameter count from it (**488,739** for 3 classes), so every
implementation is checked against that claim rather than against whichever was written first — all
**five** agree, including under custom `widths`.

```yaml
model: !lazy:deltaid.TorchUNet     # no pretrained weights — the comparison, not the recommendation
  widths: [16, 32, 64]
```

`TorchUNet` serves all three torch backends unchanged (they differ in the loop, not in what a model
is), which is why there is no `fastai_unet.py`. `input_shape` is `(H, W, C)` in **every**
implementation, even `TorchUNet`'s, whose tensor is NCHW — one config line drives any of them.

## The zoo wrapper — one class, twelve architectures

`SegmentationModelPytorch` wraps `smp.create_model`, so the architecture is a **knob**, not a class
choice:

```yaml
model: !lazy:deltaid.SegmentationModelPytorch
  arch: unet                # or deeplabv3plus, fpn, pspnet, segformer, dpt, …
  encoder_name: resnet34    # any of smp's ~800 encoders
  encoder_weights: imagenet
```

| `arch:` | |
|---|---|
| `unet` · `unetplusplus` · `manet` · `linknet` | encoder–decoder with skip connections |
| `fpn` · `pspnet` · `pan` · `upernet` | pyramid / attention decoders |
| `deeplabv3` · `deeplabv3plus` | atrous convolution + ASPP |
| `segformer` · `dpt` | transformer decoders |

`arch` is a closed `Literal`, so a typo fails at construction listing every valid value, and a
picker or form-spec enumerates them instead of offering free text. Any further smp constructor knob
(`decoder_channels`, `encoder_depth`, …) passes through verbatim.

Wire the model `!lazy:` and `num_classes` is resolved at run time and injected — you never write it
twice. The wrapper also builds its backbone at that point rather than at config-load time, so a
config that is merely inspected downloads nothing.

**Spatial sizes are smp's rule to enforce, not ours.** Whether `H`/`W` must be divisible by the
encoder stride is per-architecture (smp's `Unet` pads internally; DeepLabV3+ does not), so the
wrapper delegates the check to smp, which names the padded size to use.

Loading a previously trained model:

```yaml
model: !class:deltaid.SegmentationModelPytorch
  arch: unet
  num_classes: 3
  checkpoint_path: lightning_logs/version_0/checkpoints/last.ckpt
```

The Lightning `model.` prefix is stripped automatically, and `checkpoint_strict` defaults to
`False` so a re-headed `num_classes` does not reject the whole load.

## `num_classes`, resolved

In order: a concrete model's head → the explicit `num_classes:` → a `!lazy:` model's pinned
kwarg → a walk of the mask column (`max + 1`). The walk is announced in the log because it decodes
every mask in the split. **Prefer stating it**: for segmentation the count is a property of the
label scheme, not of the data, and a class that appears in no training image would otherwise
silently shrink the head.

For `evaluate` / `test` / `predict` it is read off the **trained model**, never the eval dataset —
a subset whose masks never show the last class would mis-size the metrics against the head.

### Binary segmentation is `num_classes: 2`

There is one target contract — `[N, C, H, W]` logits against an int64 `[N, H, W]` class-id mask —
and binary is `C = 2`. deltaid ships no single-channel `[N, 1, H, W]` mode, and that is a decision
rather than a gap: what a binary setup actually buys is **not scoring the background**, and you get
that from a knob, not from a different tensor shape.

The trap is real, though, so it is worth stating what it looks like. On a mask with ~2%
foreground, a model that predicts **nothing but background**:

| | value |
|---|---|
| `DiceLoss(mode="multiclass")` | **0.5052** ← the background's perfect score halves it |
| `MulticlassJaccardIndex(average="macro")` | **0.4897** ← same dilution |
| `DiceLoss(mode="multiclass", classes=[1])` | **0.9999** ← foreground only |
| `DiceLoss(mode="binary")` | 0.9899 |
| `MulticlassJaccardIndex(average=None)` | `[0.9795, 0.0]` ← the failure, named |

So on an imbalanced binary task, score the foreground explicitly:

```yaml
num_classes: 2

loss_fn: !class:DiceLoss@framework=torch
  mode: multiclass
  classes: [1]              # smp's own knob — the background is not averaged in

val_metrics:
  iou_per_class: !lazy:torchmetrics.classification.MulticlassJaccardIndex
    average: null           # renders as the worst-first table, so 0.0 is the first row

class_names: [background, foreground]
```

A `[N, 1, H, W]` branch would buy a second target shape, a second loss family, a second metric
family and a second prediction contract — to reach a number `classes: [1]` already reaches from
the contract that exists.

## Losses and metrics

Off-the-shelf upstream classes, registered for discovery — deltaid reimplements none of them.

```yaml
loss_fn: !class:DiceLoss@framework=torch
  mode: multiclass          # smp's losses take this as their first argument

val_metrics:                # what VALIDATION measures, on top of the shared set
  iou_per_class: !lazy:torchmetrics.classification.MulticlassJaccardIndex
    average: null           # the per-class vector a segmentation report wants
```

- **Losses** — `torch.nn`'s `CrossEntropyLoss` (the default, with `ignore_index`),
  `BCEWithLogitsLoss`, `NLLLoss`; and smp's region-overlap family for class imbalance: `DiceLoss`,
  `JaccardLoss`, `FocalLoss`, `TverskyLoss`, `LovaszLoss`, `SoftCrossEntropyLoss`,
  `SoftBCEWithLogitsLoss`, `MCCLoss`.
- **Metrics** — the `torchmetrics.classification` multiclass family, which applies per-pixel
  unchanged (a segmentation prediction *is* a classification of every pixel). The defaults are mean
  IoU + macro Dice.

Two knobs matter here and nowhere else: **`ignore_index`** excludes a void class from the score as
well as the loss (the segmenter's own `ignore_index` is passed to both by default), and
**`average`** picks between one number (`"macro"`) and the per-class vector (`None`).

The non-torch backends cannot take a torchmetrics template, so they name their metrics instead —
`mean_iou`, `pixel_accuracy`, `iou_per_class`, plus whatever their own engine ships. **Those names
resolve to the same arithmetic the torch backends get**, over a confusion matrix accumulated across
the split and reduced once (`matrainer.metric_functions`), so a number from a hand-written loop is
the number torchmetrics reports for the same predictions. That matters more than it sounds:
averaging a per-batch mean IoU instead is wrong in a way that *moves with the batch size*, because
a class absent from a batch is skipped by that batch's mean. A metric you supply as a **callable**
keeps the per-batch protocol and is averaged, since nothing can know what your function wants
accumulated.

### Per-split metrics

`metrics:` is what every split measures; `train_metrics:` / `val_metrics:` / `test_metrics:` are
what *that* split measures on top, merged over the shared set by name. So an expensive per-class
breakdown goes on validation alone without restating the shared pair beside it.

A metric whose value is a `[C]` **vector** — what `average=None` produces — cannot go through
Lightning's `log_dict` (its `convert_tensors_to_scalars` raises on anything that is not one
number), so it is reported through the trackers' **text** channel instead: a table sorted
worst-class-first, printed to the log and logged to every wired tracker.

```yaml
val_metrics:
  iou_per_class: !lazy:torchmetrics.classification.MulticlassJaccardIndex
    average: null

class_names: [pet, background, boundary]   # rows are numbered without this
plot_metrics: true                         # ALSO a bar chart (needs `matrainer[plot]`)
```

```
val/iou_per_class (mean 0.7509):

| class      |   iou_per_class |
|------------|-----------------|
| boundary   |          0.4827 |
| pet        |          0.8549 |
| background |          0.9150 |
```

Worst-first because the scalar mean already says *how* the model is doing; what this adds is
*which class it is worst at*. In the run above, mean IoU 0.751 hides that the boundary ring scores
0.48 while both real classes are above 0.85 — and `class_names:` is what makes that sentence
readable, since a mask carries ids and there is no label map to derive names from.

**Every split is scored on every backend but one**, and the exception is declared rather than
quietly empty:

| backend | how the three slots combine |
|---|---|
| `lightning` / `torch` / `tensorflow` / `jax` / `mlx` | each split gets its **own** set — the loop is this project's |
| `keras` / `fastai` (fit) | the **union**: `compile()` and a fitting `Learner` each hold ONE list and report it on both passes |
| `huggingface` | **`val` + `test` only** |

The union is the honest failure where narrowing is impossible: a metric computed on a split that
did not ask for it is noise a reader skips, where *dropping* a wired one is a metric that silently
reports nothing. And huggingface's gap is structural — `compute_metrics` is a
`transformers.Trainer`'s *evaluation* hook, so a training set would cost a second forward pass over
the whole train split every epoch. Do not fill any of these by re-running the model in eval mode:
that costs a pass **and** measures a different network (dropout, batch norm), so an expected
train/val gap would read as a bug. The backends that own their loop score the training split from
the logits their own gradient step already produced.

## Where results go

`trackers:` is the whole of a run's output wiring — there is no `logger:` slot. Each tracker is
this run's metric sink (adapted to Lightning's logger interface) *and* receives the git
provenance, the resolved config, and the run manifest.

```yaml
trackers:
  - !class:matrainer.MlflowTracker { experiment_name: oxford_pets_segmentation }
  - !class:matrainer.TensorBoardTracker
```

`metrics_sidecar_path:` additionally writes the run's final scalars to a JSON file.

**What the metrics are called.** Every tracker records the key it is handed, and eight engines
call the same quantities eight ways — Keras writes `loss` / `val_loss`, fastai `train_loss` /
`valid_loss` plus **unprefixed validation** metrics, a `transformers.Trainer` `train_*` / `eval_*`.
So each backend translates out of its engine's vocabulary on the way to the trackers and the
metrics sidecar, and every one of them records `train/loss`, `val/loss` and `val/<metric>` — the
same prefix the per-class tables carry (`val/iou_per_class`), so a run reads the same whichever
backend produced it.

Nothing needs to be set for that. `metric_names:` is the knob for when you want otherwise, and it
broadcasts into every backend at once:

```yaml
metric_names: !class:matrainer.MetricNames
  target: keras                            # record the run in Keras spelling (`val_loss`, …)
  renames: {jaccard_coeff_multi: iou}      # align a metric the translation cannot know about
```

What an engine hands its OWN callbacks is untouched either way — `EarlyStopping(monitor="val_loss")`
still finds Keras' name, and the Keras `History` object keeps Keras' columns. The vocabularies live
in `matrainer.metric_names`.

## Predictions

`predict` streams a `recordstream.SegmentationOutput` per record — `logits`, `probs`, and the
argmax `mask` — to a wired `predictions_sink:`, paired with that record's metadata. The output is
the *contract* rather than raw logits, because a sink reads `mask` / `probs` by name and cannot
softmax on the caller's behalf.

## One config, every backend

The training backend is a property of the **config**, so the CLI passes it through as a flag:

```bash
deltaid train cfg.yaml --framework lightning
```

Everything above the framework divider in the example config is engine-agnostic; an alternative
backend is a `!scope:framework=<name>` block overriding only the keys it must. A name the config
does not declare is an **error listing the ones it does** — not a silent fall-through to the
default.

All eight backends are implemented:

| backend | the loop | the model | what it brings |
|---|---|---|---|
| `lightning` | a `pl.Trainer` | `SegmentationModelPytorch` | callbacks, loggers, distributed strategies |
| `torch` | written out by hand | `SegmentationModelPytorch` | **the control experiment** — no training framework at all |
| `fastai` | a `Learner` | `SegmentationModelPytorch` | one-cycle scheduling, a per-epoch `Recorder` |
| `keras` | `model.fit()` | `KerasUNet` | one API over torch / TensorFlow / JAX |
| `tensorflow` | `tf.data` + `GradientTape` | `TensorFlowUNet` | XLA (`jit_compile`), a SavedModel export |
| `huggingface` | `transformers.Trainer` | `HuggingFaceSegmentationModel` | fine-tuning a Hub checkpoint (SegFormer) |
| `jax` | `nnx.jit` + optax | `FlaxUNet` | the loop as a pure function, traced once |
| `mlx` | written out by hand | `MlxUNet` | Apple silicon, unified memory, lazy arrays |

The first three run on torch, so they share the model, the losses, the metrics and the batch shape;
what differs is only the loop. The last three each change something real, and every one of those
differences lives in the config where a reader can see it:

- **Keras and TensorFlow are channels-LAST**, so their `preprocess` chain is the shared one *minus*
  its final `ToTensor` step (which produces CHW). The mask needs no such change — a class-id map is
  `[H, W]` in every engine. Their losses and metrics are `keras.losses` / `keras.metrics`, not
  torchmetrics, and `from_logits: true` is mandatory: the head is a bare 1x1 conv, and Keras's
  default normalizes raw logits by the row sum instead of raising (measured: 4.6293 where the
  correct value is 1.4546).
- **HuggingFace swaps the model as well as the loop**, so its block wires its own — but it keeps
  the shared torch preprocess chain, since transformers *is* torch. It needs no `loss:` at all: an
  `AutoModelForSemanticSegmentation` computes its own from `labels=`.
- **JAX and MLX are channels-last too**, and both resolve their loss, metrics and optimizer by
  **NAME** rather than by `!class:` tag — optax and `mlx.nn.losses` ship plain *functions*, so
  there is no class for a tag to resolve. That is why neither has a registrations module.
  `mlx.optimizers` ships *classes* where optax ships *factories*, which is the one asymmetry
  between the two.

Because the zoos do not line up (smp is torch-only, `keras.applications` has no segmentation
models, `transformers` dropped Flax in v5, and `mlx-image` pins a `huggingface_hub` this workspace
cannot take), the non-torch backends train **deltaid's own shared U-Net spec** — the same
**488,739-parameter** network in five vocabularies, so a cross-framework comparison compares
frameworks rather than five different networks. See
[the shared spec](#models--a-zoo-wrapper-and-a-shared-spec).

Verified end to end on the example's 256-row Oxford-Pets slice, five epochs, changing only the flag:

| `--framework` | final train | final validation |
|---|---|---|
| `lightning` | loss 0.281 | loss 0.261 |
| `torch` | — | mIoU **0.757**, Dice 0.846 |
| `fastai` | — | dice_multi **0.825**, jaccard 0.726 |
| `keras` | loss 0.719, mIoU 0.404 | loss 0.800, mIoU 0.365 |
| `tensorflow` | loss 0.661, mIoU 0.459 | loss 1.043, mIoU 0.350 |
| `huggingface` | loss 0.569 | loss 0.359, mIoU **0.659**, pixel acc 0.888 |
| `jax` | loss 0.638 | loss 0.743, mIoU 0.452, pixel acc 0.713 |
| `mlx` | loss 0.789 | loss 0.751, mIoU 0.351, pixel acc 0.673 |

The Hub backend wins because it starts from real weights; the five U-Net backends start from
noise, and their agreement with each other is the point rather than their score.

### Running them all — the sweep

`examples/sweep/` trains every backend this machine has, into one tracker, and prints the
comparison:

```bash
examples/sweep/run_sweep.sh                     # every installed backend, into MLflow
examples/sweep/run_sweep.sh --tracker csv       # ...into the always-available CSV backend
examples/sweep/run_sweep.sh --only jax,mlx      # just those two
examples/sweep/run_sweep.sh --dry-run           # print the matrix and the commands, run nothing
```

Every run is the same `examples/sweep/all_backends.yaml` with two flags changed. That file is an
**overlay**: it `include:`s the example above and adds only what a sweep needs — a `--tracker`
dimension, one metric vocabulary, and the two blocks whose paths move under `sweep_logs/`.

Eleven cells, because a backend is run once per model it can take:

| | zoo wrapper | shared U-Net (`--model unet`) |
|---|---|---|
| `lightning` / `torch` / `fastai` | `SegmentationModelPytorch` | `TorchUNet` |
| `keras` / `tensorflow` / `jax` / `mlx` | — | `KerasUNet` / `TensorFlowUNet` / `FlaxUNet` / `MlxUNet` |
| `huggingface` | SegFormer-b0 | — |

Both gaps are structural. There is **no non-torch segmentation zoo** to wrap — which is the
observation the [shared spec](#models--a-zoo-wrapper-and-a-shared-spec) exists to answer — and
huggingface takes no U-Net, because a `transformers.Trainer` drives models that compute their own
loss from `labels=`.

Two things the script does that are worth knowing:

- **The keras row pins `KERAS_BACKEND=tensorflow`**, and that is forced rather than a preference:
  see the `MeanIoU` measurement below. It also makes the `keras` and `tensorflow` rows differ in
  exactly one thing, the loop.
- **One process per run, and failures are isolated.** Keras 3 binds its compute engine at import,
  so `keras` and `tensorflow` cannot share an interpreter; and one backend dying must not cost the
  other ten, since a missing row is itself a result.

`matrainer.sweep` names the experiment `<date>_<N>` — counting N **from the tracker**, so a wiped
tracker restarts at 1 — and renders the runs back. It lives in matrainer because naming and
reading a sweep are queries against a tracker rather than anything segmentation knows about. Only `mlflow` and `csv` can be read from here;
the rest are write surfaces locally, and the report says where to open them rather than showing an
empty table. Timings, FLOP counts and Lightning's within-epoch `train/loss_step` series are folded
away by default, named in the footer, and restored by `--all`:

```bash
python -m matrainer.sweep report csv Wed_Aug_5_2026_1 --all
```

Per-class vectors (`val/iou_per_class`) never reach the table at all — a `[C]` value cannot go
through a `{name: float}` metric channel, so each backend sends it to the trackers' **text**
channel. Read those in the tracker's own UI or in the run log, sorted worst-class first.

Verified 2026-08-06, `--tracker mlflow --epochs 1` — **11/11 ok**:

| run | val/loss | val/mean_iou | val/pixel_acc | train/loss | train/mean_iou | train/pixel_acc |
|---|---|---|---|---|---|---|
| `lightning_zoo` | 0.9045 | 0.4057 | 0.6281 | 1.1539 | 0.2462 | 0.4274 |
| `lightning_unet` | 1.0540 | 0.2958 | 0.5544 | 1.1423 | 0.1859 | 0.3665 |
| `torch_zoo` | 0.7243 | 0.4889 | 0.7491 | 0.9236 | 0.3613 | 0.6037 |
| `torch_unet` | 1.0540 | 0.2958 | 0.5544 | 1.1423 | 0.1859 | 0.3665 |
| `fastai_zoo` | 0.4784 | 0.5386 | 0.8194 | 0.6549 | 0.4478 | 0.7200 |
| `fastai_unet` | 0.9509 | 0.3340 | 0.6431 | 1.0422 | 0.2648 | 0.4994 |
| `keras_unet` | 0.9340 | 0.3039 | 0.6238 | 0.9078 | 0.2513 | 0.5872 |
| `tensorflow_unet` | 0.9241 | 0.1938 | 0.5815 | 1.0075 | 0.3096 | 0.5410 |
| `huggingface_zoo` | 0.8541 | 0.5595 | 0.8123 | 0.9624 | | |
| `jax_unet` | 0.9756 | 0.1972 | 0.5815 | 0.9402 | 0.2996 | 0.5573 |
| `mlx_unet` | 1.0468 | 0.1938 | 0.5815 | 0.9664 | 0.2321 | 0.5230 |

Two things to read in it. **`lightning_unet` and `torch_unet` agree to four decimals on all six
columns** — a `pl.Trainer` and a hand-written loop producing the same numbers from the same config
is the whole claim, executed. And the two remaining blanks are structural rather than accidental:
`compute_metrics` is a `transformers.Trainer`'s *evaluation* hook, so training metrics there would
cost a second forward pass over the whole train split at every epoch end.

Every column is one quantity computed one way — each engine's names are translated at its boundary
and the arithmetic is shared, so `val/mean_iou` compares models rather than aggregations. The
scores themselves are one epoch from noise on 256 rows and are not a leaderboard; what they are for
is the two rows that agree exactly and the five U-Nets that land near each other.

### Two engine facts that will otherwise cost you an afternoon

Both are upstream limits, both are measured, and neither is something this project can fix — so
each is stated where you meet it rather than left to be rediscovered.

**`keras.metrics.MeanIoU` on the *torch* engine makes a run look hung.** It accumulates a `[C, C]`
confusion matrix through `keras.ops.scatter`, which the torch backend implements as a Python loop
over every index — and per-pixel that index count *is* the pixel count. One 8x224x224 batch:

| engine | one `update_state` |
|---|---|
| tensorflow | 0.009 s |
| jax | 0.194 s |
| **torch** | **~160 s** |

The model step itself is 0.37 s, so on torch the metric is ~99.8% of the run. `KerasSegmenter`
warns when it sees the combination; run with `KERAS_BACKEND=tensorflow` (or `jax`), or drop the IoU
metric and keep pixel accuracy. Classification never hit this, and the asymmetry is the tell: its
metrics are per-*image*, so the same scatter loops over 8 indices rather than 401,408.

`--framework tensorflow` needs no such export — the CLI pins `KERAS_BACKEND=tensorflow` for you
before anything imports keras, since that backend cannot run on any other engine. An explicit
`KERAS_BACKEND=…` still wins, and is then reported as a mismatch rather than silently overridden.

**SegFormer's backward pass does not run on Apple's MPS accelerator**, which `transformers.Trainer`
selects by itself on Apple silicon. The run dies one step in with *"view size is not compatible with
input tensor's size and stride"* out of `torch.autograd` — identical forward+backward gives
`loss=1.1024` on CPU. The example config therefore pins `use_cpu: true` in the huggingface block;
delete that line on a CUDA box.

## What deltaid does not own

The data plumbing belongs to the dataset engine — the record model, the ops, the collate and its
read-back, the class-count walk. Rasterizing regions into a mask, and domain-specific prediction
sinks, belong to the domain package. Tracking, run artifacts and the metric-slot rule belong to the
tracking library. Deltaid is the segmenter, the model wrapper, and the registrations.

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Decision records — why each mechanism exists |
| [AGENTS.md](AGENTS.md) | The agent-facing rulebook (invariants an agent must obey) |
| [TASKS.md](TASKS.md) | Backlog |

## What the run record says about the model

Every model wrapper here names the weight artifacts it loads, so a run's manifest points at them
instead of leaving a reader to work out which registry resolves a bare architecture name. Nothing
has to be called for it — the tracking layer walks the configured graph and records what it finds:

```yaml
models:
- {kind: smp, name: unet/resnet34, uri: 'smp://unet/resnet34?weights=imagenet',
   role: pretrained, slot: model, url: null}
```

**One wrapper may report several**, each with the `role` it played: `pretrained` for weights that
came from outside the run, `checkpoint` for weights a previous run wrote. A model trained from
scratch loaded no artifact and records nothing — the architecture it used is configuration, and is
already in the manifest's `params`.

Reading the identity is free and never builds the model: the properties consult stored
configuration only. Adding one to your own wrapper is a single `model_refs` property — see
[matrainer's `uri` module](https://github.com/Gearlux/matrainer).

# Deltaid

**Semantic segmentation across training backends, driven by one config.**

One `runnable:` + `task:` YAML, three verbs, and a model slot that reaches every architecture
[`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) ships.
Six training backends today — **PyTorch Lightning**, a **hand-written torch loop**, a
**fastai `Learner`**, **Keras 3**, a native **TensorFlow** `tf.data` + `GradientTape` loop, and
**HuggingFace `transformers`** — and swapping between them is one flag, not one config.

Each segmenter reads like a textbook example for its framework: no framework of our own sits in
between.

```bash
pip install 'git+https://github.com/Gearlux/deltaid.git@main#egg=deltaid[lightning]'
# or: deltaid[torch] / deltaid[fastai] / deltaid[keras] / deltaid[tensorflow] / deltaid[huggingface]

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
    normalize: false
```

`output: mask` is load-bearing rather than cosmetic: it is the key albumentations itself uses, so
recordstream's op-family dispatch hands the transform **both** `image` and `mask` in one call and a
single draw moves them together. An image-only transform like `Normalize` still touches the image
alone.

`normalize: false` on `ToTensor` because `Normalize` has already standardized the pixels — leaving
the default `true` divides them by 255 again and squashes them to ~`[-0.01, 0.01]`.

## Models — a zoo wrapper, and a shared spec

Two kinds, and they answer different questions:

| | wraps | knobs are about | lives in |
|---|---|---|---|
| `SegmentationModelPytorch` | somebody else's weights (smp) | the zoo — `arch`, `encoder_name`, `encoder_weights` | the top level |
| `HuggingFaceSegmentationModel` | somebody else's weights (the Hub) | the zoo — `model_name`, `pretrained` | the top level |
| `TorchUNet` / `KerasUNet` / `TensorFlowUNet` | **nothing** | the architecture — `widths`, `bottleneck_width`, `seed` | `deltaid/unet/` |

For real work prefer a zoo wrapper: a small net trained from scratch loses to a pretrained encoder
every time (in the table above, SegFormer-b0 reaches mIoU 0.659 where the from-scratch U-Nets reach
0.35–0.46 on the same slice). The `unet/` package exists because **the zoos do not line up** — smp
is torch-only and `keras.applications` ships no segmentation models, so a cross-framework comparison
built on a zoo could never include Keras, TensorFlow, Flax or MLX. `deltaid/unet/spec.py` states one
U-Net once and *derives* the trainable-parameter count from it (**488,739** for 3 classes), so every
implementation is checked against that claim rather than against whichever was written first — all
three agree, including under custom `widths`.

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

### Per-split metrics

`metrics:` is what every split measures; `train_metrics:` / `val_metrics:` / `test_metrics:` are
what *that* split measures on top, merged over the shared set by name. So an expensive per-class
breakdown goes on validation alone without restating the shared pair beside it.

> A metric whose value is not a single number is currently **dropped** from the reported scalars
> rather than crashing the epoch. Rendering the per-class vector properly is on the backlog.

## Where results go

`trackers:` is the whole of a run's output wiring — there is no `logger:` slot. Each tracker is
this run's metric sink (adapted to Lightning's logger interface) *and* receives the git
provenance, the resolved config, and the run manifest.

```yaml
trackers:
  - !class:marainer.MlflowTracker { experiment_name: oxford_pets_segmentation }
  - !class:marainer.TensorBoardTracker
```

`metrics_sidecar_path:` additionally writes the run's final scalars to a JSON file.

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

Six backends are implemented:

| backend | the loop | the model | what it brings |
|---|---|---|---|
| `lightning` | a `pl.Trainer` | `SegmentationModelPytorch` | callbacks, loggers, distributed strategies |
| `torch` | written out by hand | `SegmentationModelPytorch` | **the control experiment** — no training framework at all |
| `fastai` | a `Learner` | `SegmentationModelPytorch` | one-cycle scheduling, a per-epoch `Recorder` |
| `keras` | `model.fit()` | `KerasUNet` | one API over torch / TensorFlow / JAX |
| `tensorflow` | `tf.data` + `GradientTape` | `TensorFlowUNet` | XLA (`jit_compile`), a SavedModel export |
| `huggingface` | `transformers.Trainer` | `HuggingFaceSegmentationModel` | fine-tuning a Hub checkpoint (SegFormer) |

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

Because the zoos do not line up (smp is torch-only, and `keras.applications` has no segmentation
models), the non-torch backends train **deltaid's own shared U-Net spec** — the same 488,739-parameter
network in three vocabularies, so a cross-framework comparison compares frameworks rather than
three different networks. See [the shared spec](#models--a-zoo-wrapper-and-a-shared-spec).

Verified end to end on the example's 256-row Oxford-Pets slice, five epochs, changing only the flag:

| `--framework` | final train | final validation |
|---|---|---|
| `lightning` | loss 0.281 | loss 0.261 |
| `torch` | — | mIoU **0.757**, Dice 0.846 |
| `fastai` | — | dice_multi **0.825**, jaccard 0.726 |
| `keras` | loss 0.719, mIoU 0.404 | loss 0.800, mIoU 0.365 |
| `tensorflow` | loss 0.661, mIoU 0.459 | loss 1.043, mIoU 0.350 |
| `huggingface` | loss 0.569 | loss 0.359, mIoU **0.659**, pixel acc 0.888 |

The Hub backend wins because it starts from real weights; the three U-Net backends start from
noise, and their agreement with each other is the point rather than their score.

`jax` and `mlx` drop in beside them as `deltaid/<name>_segmenter.py` plus a row in `FRAMEWORKS`.

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
[marainer's `uri` module](https://github.com/Gearlux/marainer).

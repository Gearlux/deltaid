# Deltaid

A standard PyTorch-Lightning semantic segmenter, using [Marainer](https://github.com/Gearlux/marainer) for optional experiment tracking.

Deltaid provides **`LightningSegmenter`** — ONE self-contained, standard
`pytorch_lightning.LightningModule` with its own step functions and a `run()`
that builds the loaders + `pl.Trainer` and dispatches on a `task` knob
(`fit`/`evaluate`/`test`/`predict`) — plus a thin, Confluid-configurable
wrapper around
[`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch)
(`DeepLabV3PlusModel`). The segmenter owns its segmentation defaults — an
`ignore_index` `CrossEntropyLoss` and the mean-IoU / Dice metric pair — and
uses Marainer only for optional experiment tracking and a couple of
`marainer.torch` utilities.

The first-class model is **DeepLabV3+** (`DeepLabV3PlusModel`), a strong
single-pass segmentation baseline with ASPP-style multi-scale context. Any
encoder backbone supported by smp is selectable from YAML.

## Installation

```bash
pip install git+https://github.com/Gearlux/deltaid.git@main
```

### Development

```bash
git clone https://github.com/Gearlux/deltaid.git
cd deltaid
pip install -e ".[dev]"
```

## Quick Start

### YAML wiring

The config roots at a top-level `runnable:` key bound to the segmenter, plus a
`task:` the segmenter's `run()` dispatches on:

```yaml
runnable: !class:deltaid.LightningSegmenter
  task: fit               # fit | evaluate | test | predict
  num_classes: 2          # 0 = background, 1 = foreground
  experiment_name: my_segmentation
  run_name: deeplabv3plus_resnet34

  model: !class:deltaid.deeplabv3plus.DeepLabV3PlusModel
    num_classes: 2
    encoder_name: resnet34
    encoder_weights: imagenet

  train_set: !ref:train_set   # a sampleflux Flux yielding (image tensor, mask) samples
  val_set: !ref:val_set

  # Loss (ignore_index CrossEntropyLoss), the mean-IoU / Dice metrics,
  # sampleflux.typed_collate, and AdamW are baked in as defaults. Override any of
  # them with an upstream-library class — no deltaid adapter wrapper exists:
  # loss_fn: !class:torch.nn.CrossEntropyLoss
  #   weight: [1.0, 5.0]      # bump foreground if pixels are scarce
  # val_metrics: !class:torchmetrics.classification.MulticlassJaccardIndex
  #   num_classes: 2

  # Lightning runtime knobs (epochs, devices, loggers, checkpoint callbacks):
  lightning: !lazy:pytorch_lightning.Trainer
    max_epochs: 20
    logger:
      - !class:pytorch_lightning.loggers.TensorBoardLogger
        save_dir: ./lightning_logs
```

### Full end-to-end example

`waivefront/config/train_deeplabv3plus.yaml` trains a DeepLabV3+
segmentation model on RFUAV drone-signal spectrograms. It demonstrates the
typical pipeline:

1. `RegionsJsonSource` reads per-window annotations.
2. `waivefront.spectrogram.SpectrogramOp` → `MagnitudeOp` →
   `sampleflux.ops.image.ConvertToImageOp` render the IQ → spectrogram → PIL image.
3. `waivefront.targets.RegionsToSegmentationMaskOp` rasterizes the signal-region
   annotations into a per-pixel int64 mask on `sample.target` (the input stays a
   PIL image), then `sampleflux.ops.torch.ToTensorOp` turns it into the `[3, H, W]`
   input tensor.
4. `deltaid.deeplabv3plus.DeepLabV3PlusModel` consumes the image and emits
   per-pixel class logits.
5. `LightningSegmenter`'s built-in `ignore_index` `CrossEntropyLoss` trains it.

Run (the per-project CLI, or the generic `sampleflux run`):

```bash
deltaid train waivefront/config/train_deeplabv3plus.yaml
# or: sampleflux run waivefront/config/train_deeplabv3plus.yaml
```

## Models

> `DeepLabV3PlusModel` is tagged `@configurable(category="segmentation_model")`, so navigaitor's `list_configurable_classes(category="segmentation_model")` and the visual-editor form-spec surface it as a segmentation-model option.

| Class | smp backbone | Notes |
|-------|--------------|-------|
| `DeepLabV3PlusModel` | `smp.DeepLabV3Plus` | ASPP + decoder, default `resnet34` encoder |

Any encoder name supported by smp works (`resnet50`, `efficientnet-b0`,
`mit_b0` for SegFormer-style, `mobilenet_v2`, ...). See the
[smp encoders reference](https://smp.readthedocs.io/en/latest/encoders.html).

## Adapters

| Symbol | Purpose |
|--------|---------|
| `sampleflux.typed_collate` | The generic DataLoader collate the segmenter wires in (no bespoke deltaid collate). Stacks the `[3, H, W]` `Image` input and the fully-stackable `[H, W]` `Mask` target into batched tensors and keeps aux fields as per-sample lists. It is the segmenter's default loader `collate_fn`. |

The loss is not a deltaid class — `LightningSegmenter` builds a plain
`torch.nn.CrossEntropyLoss(ignore_index=...)` internally. Pass any upstream-library
loss (`torch.nn`, `segmentation_models_pytorch.losses`, `kornia.losses`, ...) under
`loss_fn:` to override it.

Most torchmetrics segmentation metrics
(`MulticlassJaccardIndex`, `MulticlassF1Score`, ...) accept logits directly
and emit a scalar — they wire into `val_metrics` straight from YAML, no
adapter needed.

## Marainer integration

`LightningSegmenter` is a standard `pytorch_lightning.LightningModule`, so
checkpointing and loggers are wired as stock Lightning: pass
`pytorch_lightning.callbacks.ModelCheckpoint` and `pytorch_lightning.loggers.*`
classes under the `lightning:` Trainer block (see the Quick Start snippet).
Marainer is used only for optional experiment tracking (attach
`marainer.tracking.*` trackers under the segmenter's `trackers:` key) and a
couple of small `marainer.torch` utilities.

`DeepLabV3PlusModel(checkpoint_path=...)` loads weights via
`marainer.torch.load_lightning_checkpoint` (imported across the dependency
boundary, not re-implemented here) — it strips Lightning's `"model."` prefix so
a `.ckpt` saved through a Lightning wrapper resumes / evaluates cleanly.
See marainer's README → *Loading checkpoints* for the accepted formats.

## Dependencies

| Package | Purpose |
|---------|---------|
| `marainer` | Optional experiment tracking + `marainer.torch` utilities (checkpoint loading). |
| `confluid` | YAML-driven dependency injection (`@configurable`, `!class:`, `!ref:`). |
| `sampleflux` | Sample / Flux pipeline plumbing. |
| `loggair` | Structured logging. |
| `segmentation-models-pytorch` | Underlying smp segmentation model implementations. |
| `torch`, `torchvision`, `torchmetrics`, `pytorch-lightning` | Standard PyTorch stack. |

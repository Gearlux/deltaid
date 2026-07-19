# Deltaid

Semantic-segmentation models and adapters for the [Marainer](https://github.com/Gearlux/marainer) training framework.

Deltaid provides a thin, Confluid-configurable wrapper around
[`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch)
so segmentation models plug into the **unchanged** base
`marainer.trainer.LightningTrainer` — no subclassing, no overridden step functions, no
custom Lightning module. The Trainer's standard
`preds = self(batch.input); loss = self.loss_fn(preds, batch.target)` flow
fits segmentation natively.

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

```yaml
model: !class:deltaid.deeplabv3plus.DeepLabV3PlusModel
  num_classes: 2          # 0 = background, 1 = foreground
  encoder_name: resnet34
  encoder_weights: imagenet

loss_fn: !class:deltaid.segmentation.SegmentationLoss
  # class_weights: [1.0, 5.0]  # optional — bump foreground if pixels are scarce

val_metrics: !class:torchmetrics.classification.MulticlassJaccardIndex
  num_classes: 2

train_loader: !lazy:torch.utils.data.DataLoader
  batch_size: 8
  shuffle: true
  collate_fn: !ref:deltaid.segmentation.segmentation_collate_fn

trainer: !class:marainer.trainer.LightningTrainer
  experiment_name: my_segmentation
  run_name: deeplabv3plus_resnet34
  model: !ref:model
  loss_fn: !ref:loss_fn
  val_metrics: !ref:val_metrics
  ...
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
5. `deltaid.segmentation.SegmentationLoss` (CrossEntropyLoss) trains it.

Run:

```bash
marainer train waivefront/config/train_deeplabv3plus.yaml
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
| `SegmentationLoss` | `nn.CrossEntropyLoss` wrapped for Confluid configuration; supports `class_weights` and `ignore_index`. |
| `segmentation_collate_fn` | DataLoader collate that stacks per-sample image / mask tensors and preserves per-sample metadata as a list (the default collate crashes on string-valued metadata). |

Most torchmetrics segmentation metrics
(`MulticlassJaccardIndex`, `MulticlassF1Score`, ...) accept logits directly
and emit a scalar — they wire into `val_metrics` straight from YAML, no
adapter needed.

## Inherited from marainer

`LightningSegmentationTrainer` subclasses `marainer.trainer.LightningTrainer`,
so segmentation runs automatically get marainer's default `ModelCheckpoint`
(`save_last=True` + best by the first val metric) and YAML-config artifact
logging to every wired Lightning logger. Wire stock
`pytorch_lightning.loggers.*` classes under `lightning.logger:`.
See marainer's README → *Default checkpointing* for details.

`DeepLabV3PlusModel(checkpoint_path=...)` loads weights via
`marainer.checkpoint.load_lightning_checkpoint` (imported across the dependency
boundary, not re-implemented here) — it strips Lightning's `"model."` prefix so
a marainer-trained `.ckpt` resumes / evaluates cleanly.
See marainer's README → *Loading checkpoints* for the accepted formats.

## Dependencies

| Package | Purpose |
|---------|---------|
| `marainer` | Base Trainer / Evaluator. |
| `confluid` | YAML-driven dependency injection (`@configurable`, `!class:`, `!ref:`). |
| `sampleflux` | Sample / Flux pipeline plumbing. |
| `loggair` | Structured logging. |
| `segmentation-models-pytorch` | Underlying smp segmentation model implementations. |
| `torch`, `torchvision`, `torchmetrics`, `pytorch-lightning` | Standard PyTorch stack. |

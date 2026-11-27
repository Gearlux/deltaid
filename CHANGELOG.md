# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a2] — 2026-11-26

### Changed
- **deltaid requires the newest workspace releases** — confluid 0.5.0 (was `>=0.3.1`), liquifai 0.4.0 (was
  `>=0.3.0`), loggair 0.3.0 (was `>=0.2.0`), recordstream 0.1.0a4 (was `>=0.1.0a2`) and matrainer 0.1.0a3 (was
  `>=0.1.0a1`), in the core and in every extra; the suite is tested against them.
- **The licence is now the PolyForm Noncommercial License 1.0.0** (`LICENSE`, `pyproject.toml`): free for personal, academic and other noncommercial use; commercial use needs a separate licence from the author. It was MIT; the versions released under MIT keep that licence.

### Models
- `FourierNeuralOperator` — a Fourier neural operator for any picture size (`width`, `modes`,
  `layers`, `patch_size`, `local_kernel`), for the lightning, torch and fastai backends.

## [0.1.0a1] — 2026-09-28

First public pre-release. The surface it ships:

### Backends
- **Eight training engines over one config**, each a runnable class behind its own extra:
  `LightningSegmenter` (`[lightning]`, the reference), `TorchSegmenter` (`[torch]`, a
  hand-written loop), `FastaiSegmenter` (`[fastai]`), `KerasSegmenter` (`[keras]`),
  `TensorFlowSegmenter` (`[tensorflow]`), `HuggingFaceSegmenter` (`[huggingface]`),
  `JaxSegmenter` (`[jax]`) and `MlxSegmenter` (`[mlx]`, Apple silicon). Each runnable dispatches
  on `task`: `fit` / `evaluate` / `test` / `predict`.
- **Backend-free core:** `import deltaid` loads no ML framework. Touching a backend that is not
  installed raises an `ImportError` naming the extra to install.

### Models
- `SegmentationModelPytorch` — every `segmentation_models_pytorch` architecture and encoder as
  configuration knobs (`arch`, `encoder_name`, `encoder_weights`).
- `HuggingFaceSegmentationModel` — any Hub semantic-segmentation checkpoint by name.
- The `deltaid.unet` package: one U-Net spec with five native implementations (`TorchUNet`,
  `KerasUNet`, `TensorFlowUNet`, `FlaxUNet`, `MlxUNet`) whose parameter counts are checked
  against the spec, for comparing engines on one network.

### Losses, metrics and results
- Registered segmentation losses and metrics per engine, per-split metric choices, mIoU and a
  per-class confusion matrix; `num_classes` resolved from the data (binary segmentation is
  `num_classes: 2`).
- Predicted masks streamed to the configured sink.

### Command line
- `deltaid train | evaluate | predict <config.yaml> [--framework <name>]` and
  `deltaid frameworks`; the same config also runs under `recordstream run`.
- An end-to-end example on Oxford-IIIT Pet (`examples/train_unet_oxford_pets.yaml`) and a sweep
  that runs one config under every backend.

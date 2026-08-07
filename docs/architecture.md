# Deltaid — architecture decisions

Short records of *why* each mechanism exists, in the workspace format:
**Context → Decision → Consequences → Example → What you may change**.

The user-facing surface — how to write a config, wire a loss, pick an architecture — lives in
[`README.md`](../README.md). `AGENTS.md` is the agent-facing rulebook. This file answers "why does
this module/mechanism exist at all?", which neither of those does.

---

## 1. One model wrapper, not one per architecture (2026-08-02)

**Context.** The project began with `DeepLabV3PlusModel`, a `@configurable` `nn.Module` naming
`smp.DeepLabV3Plus` directly. Adding U-Net meant adding `UNetModel`, and adding SegFormer meant
adding `SegFormerModel`. Comparing the three of them meant editing the config's `!class:` tag,
which is a different kind of change from tuning a knob: a tag swap can silently drop sibling keys,
and it is not something a form-spec or a picker can offer as a choice.

Reading the two candidate wrappers side by side settled it. They differed in **one line of body**
— `DeepLabV3Plus(...)` versus `Unet(...)` — with identical constructors, identical lazy-build
caching, identical checkpoint handling, and identical `forward`. Eleven more of those would be
eleven copies of one file distinguished by a string.

**Decision.** ONE wrapper, `SegmentationModelPytorch`, over `smp.create_model(arch, ...)`. The
architecture is a **constructor knob** typed as a closed `Literal`
(`SegmentationArchitecture`), and the runtime tuple is `get_args()` of that Literal rather than a
second hand-written list.

**Consequences.**

- Twelve architectures reachable by changing one string; comparing them is a knob change, not a
  config restructure.
- Because the type is closed, confluid's pydantic pass rejects a typo **at construction**, listing
  every valid value — earlier and more informative than smp's own `KeyError` several frames down.
  Navigaitor's form-spec and StreamStudio render it as an enumerated widget rather than free text.
- A `Literal` cannot be built at runtime and still be a type, so the list is written out. That
  creates a drift risk in both directions: an smp release adding an architecture would leave it
  silently unofferable, and one removing an architecture would leave a choice that fails at build
  time. `tests/test_smp_model.py` asserts the Literal equals `smp.MODEL_ARCHITECTURES_MAPPING`
  **both ways**, which is what makes the hand-written list safe.
- `solidify()` keeps its own `arch` guard even though pydantic checks the constructor, because the
  post-construction paradigm means `configure()` and a plain setattr both bypass that check.

**Example.**

```yaml
model: !lazy:deltaid.SegmentationModelPytorch
  arch: deeplabv3plus      # the ONLY line that differs from a U-Net run
  encoder_name: resnet34
  encoder_weights: imagenet
```

**What you may change.** Adding an smp architecture: extend the Literal, and the test tells you if
you forgot. A model that is *not* an smp model (a Hub checkpoint, a hand-written net) is a
different wrapper at the top level — not an `arch` value.

---

## 2. The spatial-size rule belongs to smp (2026-08-02)

**Context.** The predecessor's `forward` raised when `H % 32 != 0`, on the reasoning that a
DeepLabV3+ encoder has stride 32 and a confusing error deep inside the decoder is worse than a
clear one at the boundary. The instinct is right; the implementation was wrong.

**Decision.** Validate only what smp does not — 4-D-ness and the channel count — and let
`SegmentationModel.check_input_shape` enforce the stride rule.

**Consequences.** Two facts make the delegation necessary rather than merely tidy, and both are
measured against smp 0.5:

1. `requires_divisible_input_shape` is **per-architecture**. `Unet` sets it `False` and pads
   internally, so 60×60 is perfectly valid input — which the hardcoded check would have *refused*.
2. The divisor is the ENCODER's `output_stride`, not the constant 32.

The bug was invisible while the wrapper supported only DeepLabV3+ (which does set it `True`) and
would have surfaced the moment `arch` became a knob — i.e. as a direct consequence of record 1.
smp's own message additionally names the padded size to use, which a reimplementation would have
to duplicate.

**Example.**

```python
SegmentationModelPytorch(arch="unet", ...)(torch.zeros(2, 3, 60, 60))          # fine — Unet pads
SegmentationModelPytorch(arch="deeplabv3plus", ...)(torch.zeros(2, 3, 60, 60)) # RuntimeError, from smp
```

**What you may change.** Add a check here for anything smp does *not* cover. Do not re-add a
divisibility check in any form.

---

## 3. Segmentation added ONE primitive to the data engine (2026-08-02)

**Context.** A segmentation target is an image whose pixels are class ids. Getting one from a
Hugging Face dataset into a loss requires: decode the PNG, turn it into an int64 array, remap the
ids if the scheme is 1-based, resize it *together with* the photo, drop the source column, stack
the batch, and read it back as a tensor of the right dtype. That reads like a task-specific
pipeline, and the tempting shape is a deltaid module that does all of it.

**Decision.** Exactly one new op in the data engine —
`recordstream.ops.image.ConvertToMask`, the segmentation counterpart of `ConvertToImage` — and
nothing in deltaid. Every other step is an op that already existed.

**Consequences.**

| step | who does it |
|---|---|
| mask-bearing field → int64 `Mask` | `ConvertToMask` — **the one addition** |
| remap ids (1-based trimap → 0-based) | `FormulaOp(field="mask", formula="a - 1")` |
| resize image AND mask together | a bare `albumentations` transform, via the op-family dispatch |
| drop the source column | `DropField` |
| stack the batch | `collate_records` |
| read it back as int64 | `batch_tensor(..., dtype=torch.int64)` |
| derive the class count | `num_mask_classes` |

The op's `output` defaults to `"mask"`, and that is load-bearing rather than a nicety: it is
albumentations' own key vocabulary, so the engine's dispatch hands a transform **both** `image` and
`mask` in ONE call and a single joint draw moves them together with the `Mask` type surviving the
re-wrap. Choosing any other default would force a rename op into every config.

The op reads through `item_value`, not `item_data`, because a source that does not know a column is
a mask ships it as a `Label` — which `item_data` hands straight back. That need is what prompted
extracting `item_value` in the engine, where the same unwrapping rule had already been written out
twice.

`ConvertToMask` deliberately has **no** remap, dtype or offset knob. Each would restate an op that
already exists, and the config reads better with the steps named.

**Example.** See the `preprocess` block in
[`examples/train_unet_oxford_pets.yaml`](../examples/train_unet_oxford_pets.yaml) — six lines, of
which one is segmentation-specific.

**What you may change.** A genuinely new *shape* of target (instance masks, panoptic) is a new
engine primitive plus a branch here. It is not a reason to grow knobs on this op.

---

## 4. `num_classes` has four sources, in a deliberate order (2026-08-02)

**Context.** The classification twin derives its class count by walking the label column — cheap,
because a label is one integer. A segmentation label is an `[H, W]` array, so the same walk decodes
every mask in the split. Meanwhile the count is usually *known*: it is a property of the label
scheme (a trimap has three classes), not of the data.

**Decision.** Resolve in order — concrete model head → explicit `num_classes:` → a `!lazy:` model's
pinned kwarg → `recordstream.num_mask_classes(train_set, target_key)`. The last step logs what it
is doing at INFO. Evaluation tasks are **model-driven only**.

**Consequences.**

- A live model's head is authoritative: metrics sized against anything else score the model against
  a head it does not have.
- The expensive walk is genuinely last, and announced, so an operator sees why the run paused.
- For eval it is not consulted at all: an eval split may be a subset whose masks never show the last
  class, which would silently shrink the metrics.
- **`num_mask_classes` is a separate function from `num_classes`, not a widening of it.** That one's
  `_to_int` rejects a non-scalar target *on purpose*, so a classification run accidentally handed
  arrays fails loudly instead of miscounting. Teaching it to take the max of whatever it gets would
  trade that guard away for both tasks.

**Example.**

```yaml
num_classes: 3    # pet / background / boundary — stated, because the scheme says so
```

Omit it and the log reads `Deriving num_classes from the 'mask' column (a full pass over the
masks)...` followed by `Derived num_classes=3`.

**What you may change.** The order. Do not make the walk cheaper by sampling — a class that appears
in one image is exactly the case a partial walk gets wrong, and silently.

---

## 5. Why deltaid looks like the classification project (2026-08-02)

**Context.** The two projects solve different tasks, and the rebuild could have taken any shape.
Copying another project's structure is usually a smell.

**Decision.** Mirror it deliberately: the same CLI (three verbs + a `--framework` flag from a
`FRAMEWORKS` table), the same lazy `_BACKEND_NAMES` map, the same per-registering-module entry
points, the same sentinel-defaults-resolved-in-`initialize()` pattern, the same `trackers:`-only
output wiring, the same per-split metric slots, the same config layout (framework-specific keys as
top-level siblings).

**Consequences.**

- Every hard-won lesson transfers without being re-learned. Concretely: the old CLI used
  `flow_mode="auto"`, which drops every top-level config key silently — `train_set` becomes `None`
  and surfaces much later as a failure to derive `num_classes`. That exact bug was already
  diagnosed and fixed in the classification project; the rebuild inherited the fix rather than the
  bug.
- An operator moving between tasks writes the same config and runs the same commands.
- The genuinely shared piece — the metric-slot combination rule — moved UP to the tracking library
  rather than being copied, because a rule that never differs should not exist twice.
- **The cost is duplication**, and it is deliberate and temporary: the label pipeline, the batch
  shape and the artifact writers now exist in both projects. An engine-neutral base is deferred
  until this project and the detection one have been through the same treatment, so it gets
  designed against three tasks rather than against one.

**Example.** The two runnables' `_build_trainer` are near-identical, and that similarity is what
makes the tracking behaviour comparable between a classification run and a segmentation one.

**What you may change.** Diverge where the TASK genuinely differs — the target contract, the metric
defaults, the `num_classes` derivation all do, and all three are documented above. Do not diverge
on the CLI, the config layout, or the discovery tags.

---

## 6. One shared U-Net, apart from the zoo wrapper (2026-08-02, five implementations as of 2026-08-03)

**Context.** Record 1 made the model a knob over `segmentation_models_pytorch`, which covers
twelve architectures over ~800 encoders. That is the right answer for real work and the wrong one
for a cross-framework comparison, because **smp is torch-only**. A Keras or Flax backend has no smp
to reach for, and the obvious substitutes — a different library per framework — would compare four
different *networks* while claiming to compare four *frameworks*.

**Decision.** A second kind of model, in its own package: `deltaid/unet/spec.py` states ONE U-Net
in prose and constants, and each framework implements it natively — five of them as of 2026-08-03
(`torch_unet.py`, `keras_unet.py`, `tensorflow_unet.py`, `flax_unet.py`, `mlx_unet.py`). The spec
**derives** `trainable_parameter_count` from the architecture, and every implementation is checked
against that derivation.

**Consequences.**

- The two kinds are told apart by what they wrap and where they live: a zoo wrapper reaches
  somebody else's weights and stays at the top level; a spec implementation wraps **nothing** and
  lives in `unet/`. Their knobs differ accordingly (`encoder_weights` vs `widths`).
- The derived count is the only thing holding the implementations together — the layer *order* is
  prose, because different APIs cannot share code. So an implementation is checked against a
  CLAIM, not against whichever one was written first. All five report **488,739** for 3 classes,
  including under custom `widths`.
- **Excluding batch-norm's running statistics takes a different mechanism in every framework**, and
  that is the one place the "same network" claim is fragile. torch holds them as BUFFERS (so
  `requires_grad` filtering does it); Keras as non-trainable WEIGHTS (`trainable_weights`); raw
  TensorFlow as `trainable=False` variables; Flax as `nnx.BatchStat`, a different variable *type*
  in the same state tree (so counting `nnx.Param` only); and **MLX keeps them in the SAME parameter
  tree as the weights**, so there the exclusion is a NAME filter. Without that last one the
  five-way match silently breaks — which is exactly the kind of drift the derived count exists to
  catch.
- **Three spec choices exist for portability, not quality**, and each would otherwise make "the
  same network" untrue by construction:
  - upsampling is parameter-free NEAREST, because a learned `ConvTranspose2d`'s parameter count
    and output size depend on padding conventions that differ between frameworks;
  - `H`/`W` must be divisible by `2 ** len(widths)` — stated as `size_divisor()` rather than a
    constant, because it follows from `widths`;
  - the `input_shape` knob is `(H, W, C)` in *every* implementation, even torch's, whose tensor is
    NCHW — so one config line drives any of them and only `C` reaches the count.
- `unet/__init__.py` **imports nothing**, and that is load-bearing: `deltaid/__init__.py` resolves
  every backend name lazily so a machine without torch never imports it, and re-exporting the
  implementations here would drag in torch (later keras, flax, mlx) the moment anything under the
  package — including the dependency-free `spec.py` — was touched.

**Example.**

```python
from deltaid.unet.spec import trainable_parameter_count
from deltaid.unet.flax_unet import FlaxUNet
from deltaid.unet.mlx_unet import MlxUNet
from deltaid.unet.torch_unet import TorchUNet

assert (
    trainable_parameter_count(3)
    == TorchUNet(num_classes=3).parameter_count   # buffers excluded by `requires_grad`
    == FlaxUNet(num_classes=3).parameter_count    # `nnx.Param` only — BatchStat is another type
    == MlxUNet(num_classes=3).parameter_count     # running stats excluded BY NAME
    == 488_739
)
```

**What you may change.** Add an implementation; the derivation tells you if it disagrees. Changing
`ENCODER_WIDTHS` or the layer order is a change to the CLAIM — every implementation moves with it,
or the comparison stops meaning anything.

---

## 7. Eight loops over one config — and what each backend does *not* have (2026-08-02, completed 2026-08-03)

**Context.** "Swap the runnable, keep the config" is easy to assert and hard to keep true. All
eight backends now run the same YAML, and the interesting question is not that they work but what
each one had to give up.

**Decision.** Implement the two torch backends that differ from Lightning in opposite directions —
one with *less* framework (`torch`, a hand-written loop) and one with a *different* framework
(`fastai`, a `Learner`) — and let each declare its absences honestly rather than emulating what it
lacks.

**Consequences.**

| | `lightning` | `torch` | `fastai` |
|---|---|---|---|
| the loop | `pl.Trainer` | written out | `Learner` |
| callbacks | Trainer slot | **none** | `callbacks:` |
| distributed | strategies | **none, deliberately** | fastai's own |
| per-split metrics | train/val/test | train/val/test | **val/test only** |
| config block size | (the default) | **the smallest in the file** | +`metrics:` |

- **`torch` has no `lightning:` slot and no distributed strategy, and that is honest rather than
  missing.** Multi-GPU is what Lightning is for; a half-`Trainer` here would be a worse one. What
  it gains is that a knob which only worked *because* Lightning implemented it shows up
  immediately — and its config block being the smallest in the file is the evidence for the claim.
- **`fastai` has no `train_metrics` slot**, because a `Learner` computes metrics on the validation
  pass only. Offering the slot would accept a wired set and silently never call it — the exact
  failure per-split slots exist to remove.
- **`fastai` must restate `metrics:`**, because the shared key is a torchmetrics template *mapping*
  and a `Learner` cannot consume one. This is the general shape of a backend override: not "this
  backend prefers different metrics" but "this engine speaks a different metric vocabulary".
- The duplication between them (`num_classes` resolution, the sentinels, the metric slots) is
  **deliberate and temporary**: an engine-neutral base is deferred until the detection project has
  been through the same treatment, so it is designed against three tasks rather than one. Until
  then the guard is the fleet tests, which read the SOURCE.

The three non-torch backends added on 2026-08-03 each give up something different, and in every
case the absence is stated rather than emulated:

| | `keras` | `tensorflow` | `huggingface` |
|---|---|---|---|
| the loop | `model.fit()` | `tf.data` + `GradientTape` | `transformers.Trainer` |
| the model | `KerasUNet` | `TensorFlowUNet` | a Hub checkpoint |
| the loss | `keras.losses` | `keras.losses` | **the model computes its own** |
| per-split metrics | **union only** | train/val/test | val/test |
| the batch layout | **channels-LAST** | **channels-LAST** | CHW (it is torch) |
| what it alone brings | any Keras engine | XLA, a SavedModel | pretrained weights |

- **`keras` can only UNION its four metric slots, and that is FINAL rather than pending.**
  Verified against Keras **3.15.1**: `Model.compile` takes `metrics` and `weighted_metrics` and
  nothing phase-scoped (`weighted_metrics` is about sample weights); the set lives in ONE
  `CompileMetrics` container; and `Model.metrics` is a read-only property. So a metric wired to
  `val_metrics` is computed on training too. Widening is the honest failure — a metric appearing
  where it was not asked for is noise a reader ignores, where DROPPING it would be a wired metric
  that silently reports nothing. `matrainer.metric_slots` exists in two functions for exactly this
  split: `merge_metric_specs` for an engine whose loop the caller owns, `union_metric_specs` for
  one that cannot be told to narrow.

  A **private** route does exist and is deliberately not taken: swapping `model._compile_metrics`
  for a fresh `CompileMetrics` in a callback's `on_test_begin` / `on_test_end` narrows correctly
  (measured — a val-only metric appears as `val_only_val` and not on training). It reaches into
  `keras.src.trainers.compile_utils` and a private attribute for a gain that is cosmetic. The
  workspace does knowingly use a private API elsewhere (the classification project's
  `_trackable_children` for SavedModel restore), but there the alternative was BROKEN; here it is
  merely noisier. **The supported way to narrow is `--framework tensorflow`** — the same Keras
  losses and metrics over a loop that IS ours, which merges per split properly.
- **`tensorflow` is a SIBLING of `keras`, not a subclass**, because it *narrows a precondition*:
  `_require_tensorflow_backend()` raises where a `KerasSegmenter` runs fine. A picker offering
  "any KerasSegmenter" would otherwise hand out something that cannot run. It earns its module by
  owning the four things Keras 3 does not expose — `tf.data`, traced `GradientTape` steps (which
  is what makes `jit_compile` a knob at all), the epoch loop, and the SavedModel export.
- **`huggingface` has no loss slot at all**, and that is the only backend of which that is true.
  `FRAMEWORKS["huggingface"].registrations is None` records it: a model that computes its own loss
  leaves nothing for a config to wire, so an empty registrations module would be a lie about the
  shape rather than an economy.

The last two, added 2026-08-03, complete the set — and both give up the same thing, for the same
reason:

| | `jax` | `mlx` |
|---|---|---|
| the loop | `nnx.jit` + optax, traced once | written out by hand |
| the model | `FlaxUNet` | `MlxUNet` |
| loss / metrics / optimizer | **names**, not `!class:` tags | **names**, not `!class:` tags |
| callbacks | a plain `(epoch, logs)` callable | a plain `(epoch, logs)` callable |
| per-split metrics | val/test only | val/test only |
| what it alone brings | the loop as a pure function | unified memory, no device transfer |

- **Neither has a registrations module**, and unlike `huggingface` — where the reason is that the
  model computes its own loss — the reason here is that **optax and `mlx.nn.losses` ship plain
  FUNCTIONS**. There is no class for a `!class:` tag to resolve or for confluid's registry to tag,
  so a loss is named (`"softmax_cross_entropy_with_integer_labels"`, `"cross_entropy"`). Three
  backends therefore have `registrations=None` for two genuinely different reasons, which is why
  the field is `Optional[str]` rather than a boolean.
- **Neither has a callback CLASS to name**, because neither framework ships a callback protocol —
  so `callbacks:` holds a plain callable invoked as `callback(epoch, logs)`. Spelling that
  signature out in the type alias is what makes the slot checkable at all.
- **`ignore_index` is applied in `_loss_fn`, not passed down**, because neither framework's loss
  takes one. The per-pixel loss is masked and the mean is over the VALID pixels; dividing by the
  full pixel count would shrink the loss in proportion to how much of the image is void — a
  silent, image-dependent learning-rate change.
- The one asymmetry between them is worth stating because it is easy to copy wrong: **optax ships
  optimizer FACTORIES (`optax.adam(lr)`) while `mlx.optimizers` ships CLASSES**, so `_as_optimizer`
  *calls* in one and *instantiates* in the other.

Two upstream limits are worth knowing before they cost an afternoon, and both are *config*
answers rather than code ones. `keras.metrics.MeanIoU` on the **torch** engine scatters once per
PIXEL through a Python loop (`keras/src/backend/torch/core.py`) — ~160 s per 8x224x224 batch
against 0.009 s on TensorFlow, measured — so `KerasSegmenter` warns and names the fix rather than
substituting a cheaper metric, because a default that changed with the engine would break the one
claim the project makes. And SegFormer's *backward* pass does not run on Apple's MPS accelerator
(`loss=1.1024` on CPU, `view size is not compatible…` on MPS, same forward+backward), so the
example config pins `use_cpu: true` in its huggingface block.

**Example.** The whole `torch` block in the example config:

```yaml
torch: !scope:framework=torch
  runnable: !class:deltaid.TorchSegmenter
    device: auto
```

**What you may change.** Add a backend; add its row to `FRAMEWORKS`, `_BACKEND_NAMES`, the extras,
the entry points and the example config, and let the fleet tests tell you what you forgot. Do not
give a backend a knob its engine cannot honour.

---

## 8. The fork hazard segmentation found (2026-08-02)

**Context.** `deltaid train --framework fastai` died with `DataLoader worker exited unexpectedly`
— no Python traceback, nothing in the log. The same config trained fine under `lightning` and
`torch`, and fine under `fastai` with `num_workers: 0`.

**Decision.** Fix both causes upstream, in the data engine, and route every wired split in every
backend through one `_prepare()` helper.

**Consequences.** There were **two** independent hazards, and the first fix was not enough:

1. `import fastai` sets the multiprocessing start method to **`fork`**, process-wide. A forked
   child that is the first to touch a lazy source runs the download there, reaching `_scproxy` /
   CoreFoundation — not fork-safe. `recordstream.ensure_materialized` warms the source in the
   parent.
2. **albumentations runs on OpenCV, whose thread pool is not fork-safe either.** With the source
   warmed and the pool live, the worker *still* SIGSEGVs (measured). recordstream now calls
   `cv2.setNumThreads(0)` the first time it invokes an albumentations op.

**Why this project found it and the classification one did not:** a classification config resizes
with `ConvertToImage` (PIL). A per-pixel task must resize the image and its mask in **one joint
draw**, which only a bare albumentations transform does — so segmentation is the first thing in
the workspace to put cv2 on the worker path. The hazard was always there; nothing had stepped on
it.

**Example.**

```python
# every backend, every split — the guard is in one helper, not at each call site
def _prepare(self, source):
    if source is None:
        return None
    return ensure_materialized(ensure_record_dataset(source))
```

**What you may change.** Nothing here, and specifically: do **not** answer a recurrence with
`num_workers: 0`. That avoids the fork rather than the bug, and turns a deterministic crash into
one that looks flaky.

---

## 9. A raw-TensorFlow module may not touch `tf.initializers` (2026-08-03)

**Context.** `TensorFlowUNet` is the shared spec written in raw `tf.nn` ops — that is its whole
reason to exist beside `KerasUNet`, which is the *Keras* entry and runs on whichever engine Keras
is bound to. It built its weights with `tf.initializers.GlorotUniform`, which looks like
TensorFlow and is not.

**Decision.** Compute Glorot-uniform weights from `tf.random.stateless_uniform` locally, and treat
every `tf.keras` / `tf.initializers` / `tf.losses` / `tf.metrics` reach-through as a Keras import
in disguise.

**Consequences.** Since TF 2.16, `tf.keras` **is** Keras 3, so `tf.initializers.GlorotUniform` is
`keras.initializers.GlorotUniform` (`is`-identical, measured). In a process where Keras is already
bound to the **torch** engine — which is this workspace's default, since `recordstream.keras`
picks the first engine installed and torch wins — it returns a **torch tensor**, and `tf.Variable`
then dies converting an MPS tensor to numpy. The failure is invisible until both engines share one
interpreter, which is why a fleet test suite found it and a single-file run never would: the same
file passes alone and fails after any Keras test has run.

`stateless_uniform` is also the better primitive on its own terms — the stateful `tf.random.uniform`
reads a GLOBAL seed another library can move, so a `seed:` knob would not actually reproduce.

**Example.**

```python
def _glorot_uniform(shape, seed):
    receptive_field = int(shape[0]) * int(shape[1])
    fan_in, fan_out = int(shape[-2]) * receptive_field, int(shape[-1]) * receptive_field
    limit = (6.0 / float(fan_in + fan_out)) ** 0.5
    return tf.random.stateless_uniform(shape, seed=(seed, 0), minval=-limit, maxval=limit)
```

**What you may change.** The initializer, if the spec's comparability is preserved — the parameter
COUNT is what `spec.trainable_parameter_count` pins, not the distribution. What you may not do is
reach back into `tf.initializers` for brevity.

---

## 10. Binary segmentation is `num_classes: 2` (2026-08-03)

**Context.** The target contract is `[N, C, H, W]` logits against an int64 `[N, H, W]` class-id
mask, which covers binary as `C = 2`. smp's losses also offer `mode="binary"`, which expects a
single-channel `[N, 1, H, W]` prediction — so the question was whether that earns a branch here.

**Decision.** It does not. There is ONE target contract; binary is `C = 2`, and the thing a binary
setup actually buys is reached with a config knob instead.

**Consequences.** The trap `mode="binary"` exists to avoid is real, and it is worth stating in
numbers. On a mask with ~2% foreground, given a model that predicts **nothing but background**:

| | value |
|---|---|
| `DiceLoss(mode="multiclass")` | **0.5052** — the background's perfect score halves it |
| `MulticlassJaccardIndex(average="macro")` | **0.4897** — the same dilution |
| `DiceLoss(mode="multiclass", classes=[1])` | **0.9999** |
| `DiceLoss(mode="binary")` | 0.9899 |
| `MulticlassJaccardIndex(average=None)` | `[0.9795, 0.0]` |

So what `mode="binary"` buys is **not scoring the background** — and `classes=[1]`, smp's own knob,
buys exactly that from the contract that already exists. The per-class vector names the failure
outright, and since record 7's per-class report it renders worst-first, so the `0.0` is the first
row a reader sees.

A `[N, 1, H, W]` branch would have cost a second target shape (so `batch_tensor` and every
backend's `_to_arrays` grow a case), a second loss family, a second metric family (`Binary*` rather
than `Multiclass*`) and a second prediction contract — eight backends wide — to reach a number the
one contract already reaches.

**Example.**

```yaml
num_classes: 2

loss_fn: !class:DiceLoss@framework=torch
  mode: multiclass
  classes: [1]              # the background is not averaged in

val_metrics:
  iou_per_class: !lazy:torchmetrics.classification.MulticlassJaccardIndex
    average: null
```

**What you may change.** Nothing about the shape. If a future task genuinely needs single-channel
outputs — a regression head, a distance transform — that is a different target TYPE and belongs in
the record model upstream, not in a `mode:` knob on the segmenter.

---

## 11. One name per quantity, on all eight backends (2026-08-04)

**Context.** A `Tracker` records the key it is handed, verbatim — every backend of matrainer's
does. So the names in a tracker are whatever the training engine called them, and the eight
engines here call the same quantities eight ways: `loss` / `val_loss` (keras, jax, mlx),
`train_loss` / `valid_loss` plus **unprefixed validation** metrics (fastai), `train_*` / `eval_*`
(a `transformers.Trainer`), `train/loss` / `val/loss` (lightning, torch), and — because Lightning
appends a suffix when one name carries both a step and an epoch series — `train/loss_epoch`.

Two runs of the same config on two backends therefore could not be read side by side, which is
the one thing this project exists to make possible. The per-class tables had the same split:
`val_iou_per_class` from the JAX loop, `val/iou_per_class` from the torch one, for the same
computation.

**Decision.** Each backend translates out of its engine's vocabulary at the boundary — on the way
to a tracker and to the metrics sidecar — through `matrainer.MetricNames`
(matrainer's `docs/architecture.md` record 6), declaring its own `source`. The canonical target is
`<split>/<name>`, `split ∈ {train, val, test}`.

The translation lives in matrainer rather than here for the `metric_slots` reason one layer up:
the classification project needs the identical table on the identical eight engines, and a copy
in each would drift. What stays here is WHEN each backend applies it and WHICH split it is
reporting — the two things only the backend can know.

**Consequences.**

- Every backend records `train/loss`, `val/loss` and `val/<metric>`; the per-class text and figure
  channels carry the same prefix, so `val/iou_per_class` means one thing across the fleet.
- `LightningSegmenter.training_step` logs the loss twice, under `train/loss` (`on_epoch`) and
  `train/loss_step` (`on_step`, `prog_bar`). The per-step series is kept — nothing else shows the
  loss moving within an epoch — and it is what the progress bar displays.
- The JAX and MLX `_report()` takes a `split` rather than a free-text `prefix`, because the same
  value reaches the per-class TEXT channel: a prefix that is not a split would rename a table
  nobody else renames.
- `HuggingFaceSegmenter.test()` passes `metric_key_prefix="test"`. Left at HF's `eval` default the
  held-out scores would be translated into the columns a validation pass fills.
- The Keras and TensorFlow backends translate `_computed_metrics` and leave `_history` alone: the
  history is Keras' own object, and a wired `EarlyStopping(monitor="val_loss")` must keep finding
  the name Keras gave it.
- Every backend gained a `metric_names:` constructor slot, so one top-level key reaches all eight —
  including `target:`, which records the whole fleet in another engine's spelling.

**Example.**

```yaml
metric_names: !class:matrainer.MetricNames     # optional; the default is <split>/<name>
  target: keras
  renames: {jaccard_coeff_multi: iou}
```

```python
# what each backend hands the trackers, after translation
{"train/loss": 0.79, "val/loss": 0.75, "val/mean_iou": 0.35}
```

**What you may change.** A ninth backend declares the slot and calls
`as_metric_names(self.metric_names)` where it reports — pinned by
`tests/test_tracking_parity.py`. Do NOT translate a name the user wrote in a config, and do not
push the translation into `Tracker.log_metrics`: a tracker cannot know which engine or which
split produced a bare key.

## 12. The sweep is an overlay, and the model is a second dimension (2026-08-05)

**Context.** Records 7 and 11 made eight backends run one config and record one vocabulary, and
record 6 made five frameworks implement one U-Net. Both claims were checked the same way: one
backend at a time, by hand, with the numbers pasted into `AGENTS.md`. Nothing ran them *together*,
so nothing could see the things only a comparison shows — a metric under two names, a column one
engine leaves blank, a block that resolves to the wrong runnable under the right name.

The obvious way to get one is a second config with every backend written out. That is the thing to
avoid: it duplicates the dataset, the trimap handling, the joint resize and every framework block,
and the copy drifts from the example the moment either is edited.

**Decision.** `examples/sweep/` is an OVERLAY plus a driver.

- `all_backends.yaml` `include:`s the example and overrides only what a sweep needs: a `tracker`
  dimension whose blocks take their names from the run through `!ref:`, one metric vocabulary, and
  the two blocks whose output paths move under `sweep_logs/`.
- The **model becomes a dimension**, `!scope:model=unet`, and that block lives in the EXAMPLE
  rather than here — making the shared spec reachable is worth having on a default checkout, not
  only under a sweep. It is what lets the three torch backends train the same 488,739 parameters
  the other five already do.
- `run_sweep.sh` walks the eleven cells one SUBPROCESS at a time and isolates failures.
- `matrainer.sweep` names the experiment `<date>_<N>` — N counted from the TRACKER — and renders
  the runs back. It is matrainer's rather than this project's: both halves are queries against a
  tracker, and three task projects held byte-identical copies before the lift.

**Consequences.**

- Three composition rules are load-bearing, and each was measured while building it. (1) A scope
  block REPLACES rather than merges, because `include:` deep-merges *mappings* and a `!scope:`
  parses to a `ScopeBlock` — so a block restating one key discards the runnable and the model with
  it. (2) A deep-merged key keeps its ORIGINAL POSITION, so a key only the overlay introduces is
  appended at the end and wins over any block the example already declared — which silently
  replaced the fastai block's metric list with a torchmetrics mapping until it moved into the
  runnable's constructor kwargs, where a broadcast cannot reach it. (3) The five non-torch blocks
  need no nested `unet:` block: each wires its implementation unconditionally and is spliced after
  the example's top-level one.
- **The keras row pins `KERAS_BACKEND=tensorflow`**, and record 9's neighbour is why: on the torch
  engine `keras.metrics.MeanIoU` scatters per PIXEL through a Python loop (~160 s per batch against
  a 0.37 s model step), so an unpinned keras row does not look slow, it looks hung. Pinning also
  leaves the `keras` and `tensorflow` rows differing in exactly one thing, the loop.
- **`num_workers` is NOT dropped to 0** for the sweep, unlike its classification counterpart:
  record 8's fork hazard lives in a DataLoader worker, and a single-process sweep would stop
  exercising the one path whose failure is a SIGSEGV with no traceback.
- What the sweep exposed immediately is what it is for: fastai names mean IoU `jaccard_coeff_multi`
  (carried into the shared column by `metric_names.renames`, since it is an engine-generated name),
  has no pixel accuracy reachable by name, and four backends compute no training metrics at all.
  The first is fixed, the other two are declared in `TASKS.md` rather than hidden by trimming the
  table.
- The tests are CONFIG-level (`tests/test_sweep_config.py`), for `test_cli_frameworks.py`'s reason:
  a machine need not carry all eight frameworks, so an import-based check would silently reduce to
  whichever extras are installed — exactly where a copy-pasted block lands.

**Example.**

```bash
examples/sweep/run_sweep.sh --tracker csv --epochs 1
# ==> lightning_zoo    ok   (56s)
# ==> lightning_unet   ok   (62s)
# ...
# Wed_Aug_5_2026_2 — 11 ok, 0 failed
#
# | run            | val/loss | val/mean_iou | val/pixel_acc | train/loss | train/mean_iou |
# | lightning_unet |   1.0540 |       0.2958 |        0.5544 |     1.1423 |         0.1859 |
# | torch_unet     |   1.0540 |       0.2958 |        0.5544 |     1.1423 |         0.1859 |  <- the claim
```

```yaml
# the whole of what the overlay adds for one tracker
csv: !scope:tracker=csv
  trackers:
    - !class:matrainer.CsvTracker
      experiment_name: !ref:experiment_name    # never a literal — it would drift from the run
      run_name: !ref:run_name
      save_dir: ./sweep_logs/csv
```

**What you may change.** A ninth backend adds a row to `MATRIX` in `tests/test_sweep_config.py`
(which asserts the matrix covers every backend the CLI advertises) and an entry to `BACKENDS` in
the driver, naming its import probes and any environment it must pin. Wire an engine-specific
metric when you want to look at THAT backend; leave it out when you want to compare backends — the
REPORT renders the union either way, and a column one engine fills is honest where a silently
dropped one is not. Do NOT resolve a ragged column by making the report hide it.

## 13. Every split is scored, and where that is not free (2026-08-05)

**Context.** Record 12's sweep put the eight backends' numbers in one table, and the training
columns were mostly empty: four backends declared no `train_metrics` slot at all, and the one that
did — Lightning — recorded nothing on a short run. Each absence had a sentence in the code
explaining it, and three of the four sentences were wrong in the same way: they described THIS
PROJECT'S LOOP and read as though they described the engine.

* *"the training pass computes the LOSS and nothing else"* (JAX, MLX) — true of the loop as
  written. `nnx.value_and_grad` takes `has_aux=True`, and MLX differentiates the FIRST element of a
  returned tuple, so the logits the gradient was already taken through can simply come back beside
  the loss.
* *"a `Learner` computes its metrics on the validation pass only"* (fastai) — fastai's DEFAULT.
  `Recorder.__init__` takes `train_metrics=False`.
* Lightning's was not a sentence but a shape: `training_step` logged the per-BATCH values with
  `on_step=True`, and `log_every_n_steps` (50) drops a series shorter than the interval.

**Decision.** All eight backends serve the per-split slots, in three groups, and the group each
belongs to is a property of its ENGINE rather than of this project:

| group | backends | why |
|---|---|---|
| each split separate | `lightning`, `torch`, `tensorflow`, `jax`, `mlx` | the loop is ours |
| the union | `keras`, `fastai` (fit) | ONE compiled metric list, reported on both passes |
| `val` + `test` only | `huggingface` | `compute_metrics` is an EVALUATION hook |

The training metrics are scored from the logits the gradient step already produced. **Never from a
second eval-mode pass**: that costs a forward pass AND measures a different network (dropout, batch
norm), so a train/val gap that is expected would read as a bug.

**Consequences.**

- `LightningSegmenter.training_step` UPDATES only; `on_train_epoch_start` resets and
  `on_train_epoch_end` reports, mirroring the eval hooks. That also makes a per-class VECTOR
  wirable on the train split — the old shape ran everything through `scalar_metrics` and dropped
  one silently.
- The union is the honest failure when narrowing is impossible. Dropping a wired metric is a bug
  nobody can see; computing one where it was not asked for is noise a reader skips.
- fastai's fit path turns `Recorder.train_metrics` on, which changes its columns to
  `train_<m>` / `valid_<m>` — the shape `matrainer.metric_names`' fastai vocabulary already read,
  so nothing downstream changed. An UNPREFIXED column now means the `validate()` path.
- What this does NOT fix is how each engine AGGREGATES: torchmetrics accumulates over the split
  while the hand-written loops average per batch, so a mean IoU is not the same number on both.
  The training columns deliberately take the same per-batch mean their own eval pass takes, so each
  row stays internally consistent; making the ROWS comparable is `TASKS.md`'s.

**Example.**

```yaml
metrics:                       # every split
  mean_iou: !lazy:torchmetrics.classification.MulticlassJaccardIndex { average: macro }
val_metrics:                   # validation only — and on keras/fastai, widened to training too
  iou_per_class: !lazy:torchmetrics.classification.MulticlassJaccardIndex { average: null }
```

```python
# jax / mlx: the logits ride out of the gradient step, so the metric costs one reduction
(loss, logits), grads = nnx.value_and_grad(forward, has_aux=True)(model)
```

**What you may change.** A ninth backend declares all four slots and picks its group by what its
engine can honour — then says so in its constructor docstring, because an undeclared group is what
this record exists to stop. Do not offer a slot the engine cannot compute: accepting a wired set
and silently never calling it is the exact failure the per-split slots exist to remove.

## 14. One aggregation across eight engines (2026-08-05)

**Context.** Record 13 filled the training columns, which made a second problem legible in the
same table: the eight backends did not agree on how a metric is AGGREGATED over a split.
`lightning` / `torch` compute theirs with torchmetrics, which accumulates across batches and
reduces once. `jax` and `mlx` computed a value per batch and averaged the batches. For mean IoU
those are not the same number, and the error is not even consistently signed — a class absent from
a batch is SKIPPED by that batch's mean rather than scored 0, so batches with fewer classes present
are over-weighted, and the result moves with the batch size (10 classes at accuracy 0.7 against an
exact 0.5922: 0.6164 at batch 4, 0.5798 at batch 32).

So the sweep's `val/mean_iou` column carried a difference that was nothing to do with the models —
the one thing a cross-framework table must not do.

**How big it actually was here is worth recording, because it is small and that is not
reassuring.** Measured on this example's 64-row validation split through the real JAX loop:
accumulated 0.104803, batch-averaged 0.104790 — a delta of 1.4e-5. Eight batches of eight
224x224 images each contain all three classes in similar proportions, so the per-batch means
average to very nearly the global value. The gap only opens when batches DIFFER in which classes
they hold, which is the ordinary case with more classes, smaller batches, or sparse ones — and
nothing about the config tells a reader which regime they are in. A difference that is invisible on
the data you happen to have and material on the data you move to is exactly the kind worth removing
structurally rather than measuring per dataset.

**Decision.** Every backend that owns its loop accumulates a `C×C` confusion matrix per split and
reduces ONCE, using `matrainer.metric_functions` — the module the classification task already
brought up for macro-F1, extended with `overall_accuracy` / `iou_per_class` / `mean_iou` and an
`ignore_index` on the accumulator. The private numpy / `jnp` / `mx` implementations are deleted.

The partition is by NAME and happens BEFORE resolution: only a named metric can be
confusion-based, because nothing here can know what a user-supplied callable would want
accumulated — one keeps the `(logits, labels, ignore_index)` protocol and its per-batch mean.

**Consequences.**

- `jax` and `mlx` change their numbers; this is a fix. **`huggingface` does not, and that
  distinction is worth keeping rather than flattening:** `batch_eval_metrics` is False by default,
  so a `Trainer` gathers the whole split and calls `compute_metrics` once (verified), and its
  hand-written numpy agreed with the shared functions to ten decimals on real arrays. What the swap
  buys there is that two implementations cannot drift, plus `iou_per_class` — the per-class vector
  its own registry never had.
- **One line is engine-specific and silent when wrong.** `accumulate_confusion` argmaxes the LAST
  axis, so the huggingface backend moves its `[N, C, H, W]` channel axis first. Getting it wrong
  scores the model against its own WIDTH: a plausible number, no error. Pinned with a non-square
  map, because `H == W` would still run.
- The property the tests pin is **batch-size independence** — the same model over the same rows at
  two batch sizes must give one number — driven through the real epoch loop rather than by
  restating the arithmetic, which would pass just as happily if code and test were wrong the same
  way. The arithmetic's agreement with torchmetrics is matrainer's to pin.
- The torch backends were always right, which is what makes them the reference the other five now
  match rather than a sixth opinion.

**Example.**

```python
confusion, batchwise = self._partition_metrics(split)      # by NAME, before resolution
matrix = self._split_confusion(confusion)                  # None when nothing named was wired
for batch in split:
    ...
    accumulate_confusion(matrix, logits, labels, self.ignore_index)
values = {name: reduce(matrix) for name, reduce in confusion.items()}   # ONCE
```

**What you may change.** A new named metric is an entry in the backend's `_METRICS` and a reduction
in `matrainer.metric_functions` — never a per-batch function averaged by the loop. If a metric
genuinely cannot come from a confusion matrix, it stays a callable and is batch-averaged, and that
is a fact about the metric rather than a shortcut. Do NOT "optimise" the accumulation by keeping a
running mean of per-batch values: that is the defect this record exists to close.

---

## 15. The Lightning backend runs on the workspace trainer bases (2026-08-06)

**Context.** The four task projects were a four-way mirror: the same merged
train/evaluate/test/predict runnable per engine, with the Lightning modules 40–61%
byte-identical ACROSS projects (tracker wiring, trainer construction, step skeletons, epoch
reporting, sidecars, outputs) and the task logic near-identical WITHIN each project across its
engines. Both projects' own comments called the duplication deliberate-and-temporary, deferred
until the fleet existed to design a base against. By 2026-08-06 it did.

**Decision.** The trainer machinery moved UP to matrainer (its `docs/architecture.md` record 9
carries the full shape): a neutral abstract API (`TrainerRunnable`), a torch family layer
pinning the types (`TorchTrainerRunnable`), and the Lightning machinery base
(`LightningRunnable`). THIS project keeps only its task half, written once for the whole engine
fleet — `segmentation_base.py` (torch-free, importable by every backend) + `segmentation_torch.py` (the
torch-typed hook implementations) — and the concrete Lightning class becomes the composition
plus its own full flat constructor (the confluid introspection surface, signature unchanged).

**Consequences.** One task behaviour = one hook implementation instead of one copy per engine;
the fleet drift tests now check "inherits the shared base and does not shadow its wiring"
instead of per-module source patterns; matrainer is load-bearing for training (the old
"removing it keeps the class training" claim is dead by design). The other engine backends
still carry their private copies until their family's migration phase lands (root workspace
backlog).

**Example.**

```python
@configurable(task="segmentation", role="trainer", framework="torch")
class LightningSegmenter(SegmentationTorchHooks, LightningRunnable):
    def __init__(self, model, train_set=None, ..., ignore_index=-100):
        super().__init__(model=model, train_set=train_set, ...)  # shared slots
        self.ignore_index = int(ignore_index)                                          # task-specific state
```

**Amendment (2026-08-07).** The bare-torch backend joined the same shape: ``TorchSegmenter`` is `(SegmentationTorchHooks, matrainer.torch.TorchLoopRunnable)` — the same task hooks, a different machinery base (the owned epoch loop instead of a `pl.Trainer`), and NOTHING else: the concrete is the flat constructor alone.

**What you may change.** Hook implementations and task state freely; the concrete signature
(it is the config surface). What you may not do: shadow the inherited machinery
(`_build_trainer` / `_live_trackers` / `_run_eval` / `_report_epoch_metrics`) — extend through
the hooks, or change the base in matrainer with every consumer updated in the same workspace
change.

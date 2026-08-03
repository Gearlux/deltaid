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
  that silently reports nothing. `marainer.metric_slots` exists in two functions for exactly this
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

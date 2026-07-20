# Dropout for Neural Network Pruning

Bachelor's thesis project investigating whether the stochastic binary masks normally used
by **Dropout** for regularization can instead be repurposed as a criterion for
**structured pruning** of convolutional neural networks — and how this compares to the
classical **L2-norm** pruning approach.

**Author:** Daryna Semenets (BP-4, *Applied Mathematics*)
**Supervisor:** Assoc. Prof. N. O. Shvai, Candidate of Physical and Mathematical Sciences

Pretrained models produced in this work are available on Kaggle:
<https://www.kaggle.com/models/semenetsdaryna/dropout-for-neural-network-pruning>

---

## Motivation

Modern neural networks are heavily over-parameterized, which makes them expensive to store
and run. Pruning is one of the standard ways to compress a trained network by removing the
parts that contribute least to its output. Dropout, on the other hand, is a regularization
technique that randomly deactivates neurons during training. Both operate on the same
object — a binary mask over units — but for opposite purposes: Dropout deactivates units
*temporarily* to improve generalization, while pruning removes them *permanently* to reduce
size.

This project explores that connection directly: can a Dropout-style mask be used to decide
which channels and neurons to prune, and does a stochastic mask perform competitively
against a deterministic importance criterion such as the L2 norm of the weights?

## Research objectives

1. Investigate whether stochastic binary Dropout masks, usually applied for regularization,
   can be used for **structured pruning** of CNNs.
2. Compare the effectiveness of **dropout-based pruning** against classical **L2-norm
   structured pruning** in terms of model sparsity and accuracy.
3. Analyze the **variability and stability** of dropout-based pruning results with respect
   to different dropout rates and random seeds.

---

## Dataset

The experiments use **Imagenette2**, a 10-class subset of ImageNet
(<https://github.com/fastai/imagenette>).

| Property | Value |
|---|---|
| Classes | 10 |
| Categories | tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute |
| Image size | 128 × 128 |
| Training images | 9 468 |
| Validation images | 3 925 |

Training images are augmented with random horizontal flips; all images are resized to
128×128 and normalized. The raw folder names (e.g. `n01440764`) are mapped to human-readable
class names and re-indexed to a contiguous `0–9` range.

## Model architecture

A compact convolutional network, `SimpleCNN`, is used throughout so that the effect of
pruning is easy to isolate:

- **Conv block 1:** `Conv2d(3 → 64, 3×3)` → SpatialDropout → ReLU → MaxPool(2)
- **Conv block 2:** `Conv2d(64 → 128, 3×3)` → SpatialDropout → ReLU → MaxPool(2)
- **Conv block 3:** `Conv2d(128 → 256, 3×3)` → SpatialDropout → ReLU → MaxPool(2)
- **Classifier:** `Linear(256·16·16 → 512)` → Dropout → `Linear(512 → 10)`

Two custom Dropout modules were implemented so that the mask can be controlled explicitly:

- **`CustomDropout`** — samples a per-neuron mask for fully-connected layers.
- **`SpatialDropout`** — samples a per-channel mask for convolutional layers (an entire
  feature map is dropped, not individual pixels).

Both support an optional `seed` (to reproduce a specific mask) and an optional dropout rate
`p`, and both apply inverted-dropout scaling (`1 / (1 − p)`) so activation magnitudes stay
consistent between training and inference.

---

## Pruning methods

### Dropout-based pruning

A binary mask is generated for each prunable layer exactly as a Dropout mask would be, at a
chosen `dropout_rate`. Channels/neurons whose mask value is zero are then **physically
removed** from the network rather than merely zeroed. The actual structural removal — and
the bookkeeping of all dependent layers — is handled with the
[`torch_pruning`](https://github.com/VainF/Torch-Pruning) dependency graph
(`DependencyGraph` + `get_pruning_group`), so the pruned model remains valid and runnable.
The remaining weights are rescaled by `1 / (1 − dropout_rate)`, and the final output layer
(`fc2`) is never pruned.

### L2-norm structured pruning (baseline)

For each convolutional filter (and each hidden linear unit) the squared L2 norm of its
weights is computed. The `ratio · num_filters` units with the **smallest** norm are removed,
again through the `torch_pruning` dependency graph. This is a standard, deterministic
"remove the least important structures" baseline, and the final output layer is likewise
skipped.

Both methods are evaluated over the full range of pruning ratios / dropout rates
`0.0, 0.1, …, 0.9`, producing accuracy-vs-sparsity curves that can be compared directly.

---

## Experiments

The work is split across two notebooks.

### `thesis_first_part.ipynb` — baseline comparison (uncontrolled masks)

- Trains the initial `SimpleCNN` model (with an extended fine-tuning phase to reach a good
  starting accuracy).
- **Experiment 1.A:** compares dropout-based vs. L2-norm pruning across all sparsity levels,
  with the mask seed left uncontrolled (each run draws a different random mask).
- Fine-tunes the pruned models (at `dropout_rate = 0.5`) for both methods and measures the
  accuracy recovered before vs. after fine-tuning.

### `thesis_second_part.ipynb` — seed control, stability & variability

- **Seed-pool training:** the model is trained while cycling through a *pool of 10 masks*
  (`seed_pool = range(10)`) instead of a fresh random mask every step.
- **Experiment 1.B:** repeats the dropout-vs-L2 comparison, this time selecting a specific
  mask by fixing the seed, and fine-tunes for `seed = 8` and `seed = 42`.
- **Experiment 2 (stability & variability):**
  - For a fixed `dropout_rate = 0.5`, pruning is repeated with each of the 10 seeds and the
    resulting accuracy spread is visualized as a boxplot.
  - Across the **full range of dropout rates × 10 seeds**, the min/max accuracy spread is
    measured in two settings — **seed-controlled training** and **uncontrolled-seed
    training** — to see how much the random mask choice affects the outcome at each sparsity
    level.

In all cases the **fine-tuning itself was performed without a fixed seed**; only the pruning
mask selection was seed-controlled where noted.

**Fine-tuning setup:** 10 epochs, Adam optimizer, learning rate `1e-4`,
`ReduceLROnPlateau` scheduler, `CrossEntropyLoss`.

---

## Key findings

- **Dropout-style masks can be used to prune CNNs.** Masks generated the same way as Dropout
  masks are a viable criterion for structured pruning of convolutional networks.
- **They are competitive but not superior to L2 pruning.** Dropout-based pruning reaches
  accuracy comparable to L2-norm pruning at a given sparsity, but generally slightly lower.
- **Controlling the mask pool helps.** Reducing the number of masks used during training
  (the seed-pool strategy) has a positive effect on the method's effectiveness.
- **Instability grows with sparsity.** The spread of accuracy across different random masks
  increases as the sparsity level increases — high-sparsity pruning is much more sensitive to
  which mask is drawn.

---

## Repository structure

```
.
├── thesis_first_part.ipynb    # Experiment 1.A + fine-tuning (uncontrolled masks)
├── thesis_second_part.ipynb   # Seed-pool training, Experiment 1.B, Experiment 2
└── README.md
```

Pretrained checkpoints referenced by the notebooks (e.g. the initial trained model and the
fine-tuned pruned models) are hosted on Kaggle at the link above.

## Requirements

The notebooks are written in Python with PyTorch. Main dependencies:

```
torch, torchvision, torch-pruning, torchsummary,
numpy, pandas, matplotlib, seaborn, tqdm, pillow
```

A CUDA-capable GPU is recommended; the code automatically falls back to CPU if none is
available.

## How to run

1. Download **Imagenette2** from <https://github.com/fastai/imagenette> and place it so that
   the paths `./imagenette2/train` and `./imagenette2/val` are valid.
2. Install the dependencies listed above.
3. Open `thesis_first_part.ipynb` and run the cells top to bottom to reproduce the baseline
   comparison and fine-tuning. Optionally load the pretrained checkpoint instead of training
   from scratch.
4. Open `thesis_second_part.ipynb` for the seed-controlled experiments and the
   stability/variability analysis.

---

*This repository documents the practical part of a bachelor's thesis in Applied Mathematics.*

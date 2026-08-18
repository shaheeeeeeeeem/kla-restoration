# KLA Image Restoration — denoise + ×2 super-resolution

Restores single-channel grayscale images degraded by **speckle noise**, **additive
Gaussian noise**, and **×2 downsampling** (applied in an undisclosed random order),
recovering the clean high-resolution image.

A 4.02 M-parameter NAFNet-style encoder–decoder with a PixelShuffle ×2 head,
trained from scratch on the organizer-provided pairs. **No pretrained weights are
used in the submitted model.**

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python inference.py --input_dir /path/to/degraded --output_dir /path/to/restored
```

That is the complete inference contract — no config edits, no notebooks, no path
edits, no other required arguments. Weights are resolved **relative to
`inference.py`**, not the working directory, so the command works from anywhere.

---

## Input / output contract

Derived in Phase 1 from the actual test inputs (`scripts/analyze_dataset.py`); no
instruction files shipped with the dataset.

| | Input | Output |
|---|---|---|
| Container | `.npy` | `.npy` |
| dtype | `float32` | `float32` |
| Shape | `(H, W)`, no channel axis | `(2H, 2W)`, no channel axis |
| Test set | 400 arrays, all `(128, 128)` | 400 arrays, all `(256, 256)` |
| Range | **unclipped**, observed −0.28 … 2.16 | **clipped to [0, 1]** |
| Naming | `000000.npy` … `000399.npy` | **identical filename to its input** |

The pipeline clips; KLA does not. Input is fed to the network **raw and unclipped** —
only a fixed dataset-level mean shift is applied (see Normalization below).

The model is **fully convolutional and resolution-agnostic**: no spatial dimension
is hardcoded anywhere. Inputs are reflect-padded to a multiple of the network
stride (8) and cropped back exactly, so odd sizes work — verified on
`(128,128)`, `(256,256)`, and `(130,97)`.

---

## Dataset facts

Full report: [`results/dataset_facts.md`](results/dataset_facts.md) (machine-readable:
`results/dataset_facts.json`). Regenerate with:

```bash
python scripts/analyze_dataset.py --json_out results/dataset_facts.json
```

- 3,200 training pairs. `GT/` is `(256,256)`, `NoisyLR/` is `(128,128)` — **×2
  confirmed on every pair inspected**, no exceptions.
- **Pairing rule: identical filename.** `GT/000123.npy` ↔ `NoisyLR/000123.npy`.
- GT is exactly in `[0,1]` (verified over all 3,200). NoisyLR exceeds it on **both**
  ends: 3,120/3,200 images contain a pixel > 1 and 1,967/3,200 contain a pixel < 0.
- Train and test NoisyLR statistics **match closely** (mean 0.4335 vs 0.4427, std
  0.2848 vs 0.2843) — the OOD half of the test set differs in *content*, not in
  degradation strength.
- Measured degradation properties: the downsampler is **anti-aliased, not
  nearest-neighbour** (nearest gives clearly worse residual RMSE), and the effective
  per-image noise level varies widely — residual std from **0.025 to 0.20**
  (median 0.081). The model must be blind to noise strength.

---

## Method

### Normalization
The raw, unclipped NoisyLR is mean-shifted by a fixed constant recorded in
`configs/final.yaml` (`norm.mean: 0.4335`, the measured pixelwise mean of train
NoisyLR; `norm.std: 1.0`). No per-image statistics, no clipping, no renormalization.

### Global residual
The network output is added to a **bicubic ×2 upsample of the input**, so it only
learns the correction. The tail convolution is **zero-initialized**, which means an
untrained model outputs exactly the bicubic baseline — training can only improve on
the floor, and convergence starts from a sensible point rather than noise.

### Architecture
NAFNet-style encoder–decoder operating at **LR resolution**, with the ×2 upsample
deferred to a PixelShuffle head at the very end. Keeping the bulk of the compute at
128×128 rather than 256×256 is ~4× cheaper for the same depth, which matters because
throughput is scored. Blocks use SimpleGate + simplified channel attention.

- width 32, encoder blocks `[2,2,4]`, 6 middle blocks, decoder blocks `[2,2,2]`
- 4.02 M parameters, stride 8

### Loss
`Charbonnier(ε=1e-3) + 0.15 · (1 − MS-SSIM)`.
**No adversarial loss and no GAN-pretrained initialization** — hallucinated texture
is explicitly penalised by the objective, so nothing in this pipeline synthesizes
detail it cannot recover.

### Validation split
200 images held out **by image** from `train/`, seeded, and committed verbatim to
[`configs/val_split.txt`](configs/val_split.txt). Never trained on.

**`Test_NoisyLR/` is never trained on, never validated on, and never used for
checkpoint selection.** It is used only to produce final submission outputs.

---

## Reproducing

```bash
python scripts/make_split.py                      # regenerate the val split (refuses to overwrite)
python train.py --config configs/final.yaml       # full training run
python scripts/run_baseline.py                    # bicubic baseline predictions
python scripts/evaluate.py --pred_dir results/preds_bicubic_val --name "bicubic x2"
```

Seed is `1337` (`configs/final.yaml`). Every checkpoint stores its **git commit hash
and seed**; `results/experiments.csv` logs the same per validation point.

### Sanity checks

```bash
python scripts/sanity_overfit.py --n 2 --iters 400   # can the net fit the mapping?
python scripts/sanity_roundtrip.py                   # is saving lossless?
```

| Check | Result |
|---|---|
| **A** — memorize 2 fixed pairs, no crop/augmentation | 26.36 → **52.13 dB** (+25.77 over bicubic) — PASS |
| **B** — save/reload round trip under the output contract | **bit-exact** float32; clip exact; channel axis stripped — PASS |

Because outputs are `.npy` float32 rather than an 8-bit image container, saving is
lossless — Sanity B asserts *bit-exactness*, not approximate equality.

---

## Results

Validation = the 200 held-out images. All metrics computed by `scripts/evaluate.py`
**on the saved `.npy` files**, not in-memory tensors.

**KLA's blend over PSNR/SSIM/LPIPS is undisclosed, so all three are reported
unweighted.**

| Model | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Params |
|---|---|---|---|---|
| Bicubic ×2 (baseline) | 22.7300 | 0.5215 | 0.4395 | — |
| **NAFNet-SR (ours)** | **27.5150** | **0.7430** | **0.2676** | 4.02 M |
| Improvement | **+4.7850 dB** | **+0.2215** | **−0.1719** | |

Submitted checkpoint: `weights/best.pt` — step 4000, EMA weights, seed 1337, git `caf4c07`.
End-to-end inference throughput including disk read and file writing:
**172.35 images/sec** on the hardware below (400 test images in 2.32 s, warm cache,
median of 5 after 2 warmup passes). A single cold-start `inference.py` invocation over
the same 400 images takes ~5.1 s wall clock including process start and model load.

Inference is **deterministic**: `cudnn.benchmark` is disabled and
`cudnn.deterministic` enabled, verified by two consecutive runs producing
**400/400 bit-identical** outputs. This costs ~2% throughput and was taken
deliberately, since determinism is part of the output contract.

PSNR and SSIM are implemented in `src/metrics/image_metrics.py` and agree with
`skimage.metrics` to **1.6e-7** (SSIM) and **9e-7** (PSNR). LPIPS uses the AlexNet
variant (evaluation only — see [`EXTERNAL_RESOURCES.md`](EXTERNAL_RESOURCES.md)).

Runtime breakdown: [`results/runtime_report.md`](results/runtime_report.md).
Figures, including a failure case: `results/figures/`.

---

## Hardware

Everything in this repo was developed, trained, and timed on:

- **NVIDIA GeForce RTX 4050 Laptop GPU**, 6.44 GB, sm_89
- torch 2.12.0+cu126, CUDA 12.6, driver 566.07, Windows 11, Python 3.14
- bf16 autocast (native on sm_89), `channels_last`, TF32 enabled

**All reported timings are measured on this laptop GPU, not on an H100.** They
should be read as relative, not as a prediction of evaluation-hardware throughput.

---

## Limitations — stated, not hidden

**The run was stopped early, on purpose.** The approved budget was 15,000
iterations; it was stopped at ~9,400 because validation had peaked at step 4,000 and
was steadily degrading while training loss kept falling — textbook overfitting, with
3,000 training images, 4.02 M parameters and `weight_decay = 0.0`. The
best-by-validation rule captured the step-4,000 peak automatically, so the submitted
checkpoint is the best this run produced. Full step-by-step table in
[`results/metrics_summary.md`](results/metrics_summary.md).

The obvious next steps, none of which were attempted: non-zero weight decay,
stronger augmentation, and a cosine schedule targeting ~4–5k steps.

Other limitations:

- **One seed, one config, one architecture.** No ablation beyond bicubic vs. final model.
- **Timings are from an RTX 4050 laptop GPU, not an H100.** Relative, not predictive.
- **The model over-smooths dense high-frequency texture.** See
  `results/figures/qualitative_failures.png` — where the ground truth is itself
  nearly noise-like, a fidelity objective correctly suppresses high frequencies and
  loses genuine detail. It still beats bicubic on every one of those cases.

---

## Assumptions

1. Output format was **derived** from the test inputs — no README or evaluator
   instructions shipped with the dataset. Restored images use the input's exact
   filename, `.npy`, `float32`, ×2 shape, clipped to `[0,1]`.
2. Blur is **not** treated as a degradation. The organizer deck shows it as
   illustration only, so no deblurring stage is built.
3. Metric weights are unknown; the model is tuned for a balanced
   Charbonnier + MS-SSIM objective rather than for any single metric.

## Repository layout

```
train.py                  training entry point
inference.py              submission entry point (the graded contract)
configs/final.yaml        the one config; seed, paths, model, loss, schedule
configs/val_split.txt     committed validation file list
src/data/                 split logic, paired dataset, augmentation
src/models/               NAFNet-style ×2 restoration network
src/losses/               Charbonnier + MS-SSIM
src/metrics/              PSNR, SSIM, LPIPS (validated against skimage)
src/engine/               normalization, EMA, LR schedule
scripts/analyze_dataset.py    Phase 1 dataset discovery
scripts/make_split.py         seeded val split generator
scripts/run_baseline.py       bicubic baseline predictions
scripts/evaluate.py           metrics on saved files
scripts/benchmark_runtime.py  per-stage timing breakdown
scripts/make_figures.py       qualitative panels incl. failure case
scripts/sanity_*.py           the two pipeline sanity checks
results/                  metrics, timing, experiment log, figures
EXTERNAL_RESOURCES.md     every external model + licence read from its LICENSE file
```

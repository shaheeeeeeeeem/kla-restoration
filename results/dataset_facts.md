# Phase 1 — Dataset facts report

Produced by `scripts/analyze_dataset.py` (every array inspected, no sampling).
Machine-readable copy: `results/dataset_facts.json`. Visual: `results/figures/phase1_dataset_peek.png`.

## Locations

| Role | Absolute path |
|---|---|
| Train pairs | `C:\Users\mdsha\Downloads\train\train\` (note the **doubled** `train\train`) |
| Test inputs | `C:\Users\mdsha\Downloads\Test_NoisyLR (1)\NoisyLR\` |

The archives extracted with a nested folder: `train/train/{GT,NoisyLR}`, and
`Test_NoisyLR (1)/NoisyLR/`. Both trees also contain a macOS `__MACOSX/` sidecar
directory, which is ignored everywhere.

`Test_NoisyLR.zip` and `Test_NoisyLR (1).zip` are **byte-identical**
(sha256 `f2904f75d6938c...b45f83`). The `(1)` is a plain re-download; only the
`(1)` copy is extracted, and it is complete.

## Completeness

`train.zip` contains 6400 `.npy` entries; 3200 GT + 3200 NoisyLR are on disk.
`Test_NoisyLR.zip` contains 400; 400 are on disk. **Both downloads are complete.**

## Container and format

Everything is `.npy`, **`float32`**, **single-channel 2-D** arrays (shape is
`(H, W)` — there is no channel axis at all, so no RGB question arises).
No PNG/TIFF anywhere.

## Counts, shapes, pairing

| Set | Count | Shape | Total |
|---|---|---|---|
| train GT | 3200 | `(256, 256)` — 100% | 800 MB |
| train NoisyLR | 3200 | `(128, 128)` — 100% | 200 MB |
| test NoisyLR | 400 | `(128, 128)` — 100% | 25 MB |

**Pairing rule: identical filename.** `GT/000000.npy` ↔ `NoisyLR/000000.npy`.
Zero-padded 6-digit index, `000000`–`003199`, contiguous, no gaps. Set difference
in both directions is empty. No suffix scheme, no offset, no index table.

**×2 scale confirmed** on 200 random pairs: every pair is GT `(256,256)` /
LR `(128,128)`, ratio exactly `(2.00, 2.00)`. No exceptions.

## Value ranges

| Set | min | max | mean | std | frac px outside [0,1] | frac px < 0 |
|---|---|---|---|---|---|---|
| train GT | 0.000000 | 1.000000 | 0.4335 | 0.2726 | **0.0000** | 0.0000 |
| train NoisyLR | −0.278563 | 2.158005 | 0.4335 | 0.2848 | 0.0339 | 0.0028 |
| test NoisyLR | −0.224881 | 2.158016 | 0.4427 | 0.2843 | 0.0374 | 0.0066 |

GT is **exactly** in `[0,1]` — verified over all 3200 arrays, not assumed.

NoisyLR exceeds `[0,1]` on **both** ends: 3120/3200 train images contain a pixel
> 1, and 1967/3200 contain a pixel < 0. The brief's Section 1 mentioned only the
upper tail (“histogram runs to ~1.5”); the real data also has a **negative tail**,
and the upper tail reaches **2.16**, not 1.5. This is consistent with additive
Gaussian noise on top of multiplicative speckle. Confirms the rule: feed raw and
unclipped, clip only at save time.

## Train vs. test distribution

| stat | train NoisyLR | test NoisyLR | delta |
|---|---|---|---|
| mean | 0.4335 | 0.4427 | +0.0092 |
| std | 0.2848 | 0.2843 | −0.0005 |
| max | 2.1580 | 2.1580 | +0.0000 |
| frac outside [0,1] | 0.0339 | 0.0374 | +0.0035 |

**No meaningful shift.** The degradation process is clearly identical; the max
agreeing to 5 decimal places suggests a shared clipping/quantization ceiling in the
generator. The OOD half of the test set differs in *content*, not in degradation
statistics — visible in the peek figure (test row includes a near-black night-sky
frame and animal fur, which are outside the train set's architecture/foliage mix).

## Degradation probe (my own measurement, not from the docs)

Comparing NoisyLR against GT downsampled ×2 with four kernels, over 120 pairs:

| kernel | residual RMSE |
|---|---|
| bicubic | 0.0853 |
| bilinear | 0.0858 |
| area | 0.0858 |
| **nearest** | **0.0979** |

Nearest is clearly worse; the other three are within noise of each other. So the
downsampler is a **proper anti-aliased kernel** (bicubic or bilinear/area — the
noise floor prevents distinguishing them). It is *not* nearest-neighbour
subsampling. Good news for the bicubic baseline and for the global-residual design.

Per-image residual std (the effective combined noise level at LR resolution):
min 0.025, p10 0.046, **median 0.081**, p90 0.125, max 0.201. That is a **wide,
per-image-varying noise level** — the model has to be blind to noise strength, and
a single fixed-σ assumption would be wrong.

## Shipped instructions

**None.** No README, txt, md, pdf, json, csv, or spreadsheet exists anywhere in
either dataset folder. Nothing overrides Section 1.

## Output contract (derived, since nothing was shipped)

Test inputs are `000000.npy` … `000399.npy`, float32, `(128,128)`.
Restored outputs will be written as:

- **same filename**, `000000.npy` … `000399.npy`
- `.npy`, **`float32`**
- shape `(256, 256)` — ×2 of the input, no channel axis, matching GT's layout
- **clipped to `[0,1]`** by our pipeline (KLA does not clip or renormalize)

## Contradictions with the brief's Section 1

1. **Test input shapes are uniform, not mixed.** Section 1 anticipated
   "~128×128 or ~256×256, possibly mixed". All 400 test inputs are `(128,128)`,
   so all outputs are `(256,256)`. The model stays fully convolutional and
   resolution-agnostic regardless, and `inference.py` will still group by shape —
   but no mixed-batch complexity is actually exercised by this test set.
2. **NoisyLR range is wider than stated**, and goes negative (−0.28 to +2.16 vs.
   the quoted "~1.5" upper bound only).
3. Format is `.npy` float32 arrays, not an image container — so there is **no
   quantization loss** at save time, and the Sanity B round-trip should be
   bit-exact rather than merely close.

---

## Blocker for Phase 3: there is no local NVIDIA GPU

The brief left the hardware placeholder unfilled (`<<<GPU + VRAM, or "CPU only">>>`).
Measured on this machine:

- `Get-CimInstance Win32_VideoController` → **Intel(R) Arc(TM) Graphics** only.
  No NVIDIA adapter present.
- `nvidia-smi` → fails (no driver).
- `torch 2.12.0+cu126`, `torch.cuda.is_available()` → **False**,
  `torch.cuda.device_count()` → **0**.

This laptop is **CPU-only for training purposes**. All training must run on Colab.
Local execution is still fine for dataset work, the bicubic baseline, sanity checks,
and CPU-timed inference.

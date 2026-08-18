# Metrics summary

Validation split: 200 images held out **by image** from `train/` (seeded, listed in
`configs/val_split.txt`). Never trained on.

All metrics are computed by `scripts/evaluate.py` **on the saved `.npy` output
files**, not on in-memory tensors, so any dtype or range loss at save time is
captured. PSNR/SSIM use `data_range=1.0`. LPIPS is the AlexNet variant, lower is better.

**KLA's scoring blend over PSNR/SSIM/LPIPS is undisclosed, so all three are
reported unweighted.**

| Model | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Params |
|---|---|---|---|---|
| Bicubic ×2 (baseline) | 22.7300 | 0.5215 | 0.4395 | — |
| **NAFNet-SR (ours)** | **27.5150** | **0.7430** | **0.2676** | 4.02 M |
| Improvement | **+4.7850 dB** | **+0.2215** | **−0.1719** | |

Submitted checkpoint: `weights/best.pt`, step 4000, EMA weights, seed 1337,
git `caf4c07`.

## Saved-file integrity

`scripts/evaluate.py` reports the maximum fraction of pixels outside `[0,1]` across
the saved predictions: **0.000000**. The saved-file PSNR (27.5150) matches the
training-time in-memory validation PSNR (27.515) exactly, confirming that writing
`.npy` float32 costs nothing.

## Metric implementation validation

`src/metrics/image_metrics.py` implements PSNR and SSIM directly in torch. To make
sure the reported numbers are not an artefact of a custom implementation, they were
checked against `skimage.metrics` over 6 noise levels:

- SSIM (Gaussian window, 11×11, σ=1.5, `use_sample_covariance=False`): max
  deviation **1.599e-07**
- PSNR (`data_range=1.0`): max deviation **8.960e-07**

## Baseline notes

The bicubic baseline is a plain ×2 bicubic upsample of the raw NoisyLR input,
clipped to [0,1] at save time. It does no denoising at all, which is why SSIM and
LPIPS are poor even though PSNR is not catastrophic — the noise survives the
upsample intact. It is the honest floor for this task.

## OOD proxy evaluation

Half the hidden test set is out-of-distribution *content*, and the released test
inputs have no ground truth, so no direct measurement is possible. The held-out
validation split was re-partitioned by content character — no new data, no new
disclosures. Method and thresholds: `results/ood_analysis.md`; figure:
`results/figures/ood_groups.png`; reproduce with `python scripts/ood_partition.py`.

| Group | n | PSNR bicubic → ours | SSIM bicubic → ours | LPIPS bicubic → ours |
|---|---|---|---|---|
| smooth / low-frequency | 67 | 23.60 → **31.36** | 0.482 → **0.842** | 0.470 → **0.205** |
| structured / edge-dominated | 67 | 23.25 → **26.73** | 0.583 → **0.781** | 0.382 → **0.229** |
| **texture-heavy (weakest)** | 66 | 21.32 → **24.40** | 0.499 → **0.603** | 0.467 → **0.370** |
| *all 200, for reference* | 200 | 22.73 → 27.52 | 0.522 → 0.743 | 0.440 → 0.268 |

**Honest OOD estimate: 24.40 dB, not the headline 27.52 dB** — a 3.11 dB gap. If the
hidden set skews toward dense texture, that is the number to expect. The model beats
bicubic in *every* group, so the ranking never inverts; only the margin changes.

Independent corroboration: all three lowest-PSNR validation images fall in the
texture-heavy group, so the content partition and the failure-case analysis agree
without being constructed to.

This is a *proxy*. It partitions in-distribution content by character; it does not
sample genuinely unseen domains, so it bounds only the variation we can observe.

## Self-ensemble — measured, then declined

×8 flip/rotate self-ensemble is implemented behind `--self_ensemble` and **defaults
OFF**. Full write-up: `results/self_ensemble.md`; figure:
`results/figures/self_ensemble_tradeoff.png`; reproduce with
`python scripts/self_ensemble_tradeoff.py`.

| Mode | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Throughput (compute only) |
|---|---|---|---|---|
| **single pass (default)** | 27.5150 | 0.7430 | 0.2676 | **204.4 img/s** |
| ×8 self-ensemble | **27.8205** | **0.7519** | **0.2555** | 25.1 img/s |
| difference | **+0.3055 dB** | +0.0089 | −0.0121 | **8.13× slower** |

The gain is real and consistent across all three metrics. It costs 8.13× the
compute — matching the theoretical ×8 exactly.

Quality and throughput are scored on separate axes with undisclosed weights, so we
take the configuration that is strong on both rather than trading a large, certain
throughput loss for a small, uncertain quality gain. Flipping the flag needs no
retraining and no new checkpoint.

A caveat on measurement: timing the whole command over only 400 images makes the
self-ensemble look cheaper (2.48×) because ~6 s of process and CUDA startup
dominates a ~2.3 s workload. The **8.13× compute-only figure is the one that
generalises**, and we report that.

**Adding the flag did not change default behaviour** — re-running the default
command afterwards reproduced the clean-room outputs 400/400 bit-identical.

## Training budget — and why the run was stopped early

The approved run was 15,000 iterations. It was **stopped at ~9,400** because
validation had clearly peaked and was degrading:

| step | train loss | val PSNR | val SSIM |
|---|---|---|---|
| 1000 | 0.0389 | 24.5440 | 0.5819 |
| 2000 | 0.0399 | 26.2598 | 0.6630 |
| 3000 | 0.0378 | 27.3359 | 0.7299 |
| **4000** | 0.0338 | **27.5150** | **0.7430** |
| 5000 | 0.0337 | 27.3137 | 0.7370 |
| 6000 | 0.0301 | 27.0902 | 0.7284 |
| 7000 | 0.0274 | 26.9022 | 0.7206 |
| 8000 | 0.0317 | 26.7345 | 0.7137 |
| 9000 | 0.0278 | 26.5777 | 0.7073 |

Training loss keeps falling while validation falls with it — **overfitting**, from
step 4000 onward. With 3,000 training images, 4.02 M parameters, and
`weight_decay = 0.0`, this is the expected failure mode. The best-by-validation
checkpoint rule captured the step-4000 peak automatically, so the submitted model is
the best one this run produced; continuing to 15,000 would only have consumed GPU
time that the evaluation and packaging phases needed.

**This is a real limitation, not a tuned result.** With more time the obvious next
steps are, in order: non-zero weight decay, stronger augmentation, and a shorter
cosine schedule targeting ~4–5k steps. None were attempted — one seed, one config,
one checkpoint, as scoped.

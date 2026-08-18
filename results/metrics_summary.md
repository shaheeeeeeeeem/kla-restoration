# Metrics summary

Validation split: 200 images held out **by image** from `train/` (seeded, listed in
`configs/val_split.txt`). Never trained on, never used for checkpoint selection
beyond the val-PSNR best-checkpoint rule stated in the README.

All metrics are computed by `scripts/evaluate.py` **on the saved `.npy` output
files**, not on in-memory tensors, so any dtype or range loss at save time is
captured. PSNR/SSIM use `data_range=1.0`; our implementations agree with
scikit-image to within 1.6e-7 (SSIM) and 9e-7 (PSNR) — see the validation note below.
LPIPS is the AlexNet variant, lower is better.

**KLA's scoring blend over PSNR/SSIM/LPIPS is undisclosed, so all three are
reported unweighted.**

| Model | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Params |
|---|---|---|---|---|
| Bicubic ×2 (baseline) | 22.7300 | 0.5215 | 0.4395 | — |
| NAFNet-SR (ours) | TBD | TBD | TBD | 4.02 M |

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

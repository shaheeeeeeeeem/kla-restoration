# Self-ensemble trade-off

×8 flip/rotate self-ensemble is implemented and available behind
`--self_ensemble`. **It is OFF by default.** This records the measurement behind
that decision.

Quality measured on the 200-image held-out validation split; runtime measured two
ways (below). Same checkpoint, no retraining.

## Quality

| Mode | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| **single pass (default)** | 27.5150 | 0.7430 | 0.2676 |
| ×8 self-ensemble | **27.8205** | **0.7519** | **0.2555** |
| gain | **+0.3055 dB** | +0.0089 | -0.0121 |

The gain is real and consistent across all three metrics.

## Runtime — two figures, and the honest one is 8×

| Measure | single pass | ×8 self-ensemble | cost |
|---|---|---|---|
| **Compute only** (batch 16 @ 128², CUDA-event timed) | **204.4 img/s** | 25.1 img/s | **8.13× slower** |
| Whole-command wall clock, 400 images incl. process start | 41.2 img/s | 16.6 img/s | 2.48× slower |

The second row flatters the self-ensemble. Our 400-image run takes only ~2.3 s of
actual compute, so roughly 6 s of Python and CUDA startup dominates the wall clock
and compresses the apparent ratio. **The 8.13× compute figure is the one that
generalises** — it matches the theoretical 8× exactly, and on a larger evaluation
set the observed cost converges to it.

## Decision: default OFF

Quality and throughput are scored on separate axes with **undisclosed weights**.
The trade is **+0.3055 dB for 8.13× the compute**.

We are unwilling to give up a large, certain throughput loss for a small, uncertain
quality gain when we cannot see the weighting. A configuration that is strong on
both axes dominates one that is excellent on one and poor on the other under an
unknown blend. If KLA's weighting were disclosed and quality-dominant, flipping the
flag is a one-word change requiring no retraining and no new checkpoint:

```bash
python inference.py --input_dir <in> --output_dir <out> --self_ensemble
```

## The default path is untouched

Adding the flag did not alter default behaviour: re-running the default command
after the change reproduced the clean-room outputs **400/400 bit-identical**.

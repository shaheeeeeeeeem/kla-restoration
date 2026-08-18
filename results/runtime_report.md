# Runtime report

**Measured on NVIDIA GeForce RTX 4050 Laptop GPU, NOT on an H100.** Treat these as relative numbers.

## Environment

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB, sm_89)
- torch 2.12.0+cu126, CUDA 12.6
- precision: bf16 autocast, channels_last, TF32 on
- checkpoint: step 4000, git `caf4c07-dirty`, seed 1337

## Method

- 400 images, batch size 16, `torch.inference_mode()`
- 2 warmup passes discarded, **median of 5** timed passes
- per-stage timing via `time.perf_counter()` with `torch.cuda.synchronize()` around device work; the end-to-end total is a single wall clock over the whole loop
- **includes disk read and file writing**, as the scoring criterion requires
- output writing is threaded (8 workers) and overlaps compute

## End-to-end

- **total wall clock: 2.336 s** for 400 images
- **throughput: 171.27 images/sec**
- per-image: 5.84 ms
- spread over 5 runs: min 2.293 s, max 2.351 s

## Per-stage breakdown

Stage times are summed across batches within one pass. They overlap with threaded writing, so they do not add up to the wall clock.

| Stage | Median (s) | % of wall clock |
|---|---|---|
| read | 0.264 | 11.3% |
| preprocess | 0.004 | 0.2% |
| h2d | 0.016 | 0.7% |
| forward | 1.981 | 84.8% |
| d2h | 0.030 | 1.3% |
| write_submit | 0.019 | 0.8% |
| write_drain | 0.011 | 0.5% |

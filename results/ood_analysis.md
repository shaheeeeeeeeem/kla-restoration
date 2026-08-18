# OOD proxy evaluation

Half the hidden test set is out-of-distribution *content*. We have no labelled
OOD data, and the released test inputs have no ground truth, so no direct
measurement is possible. This is a **re-partition of the existing held-out
validation split** by content character — no new data, no new disclosures.

## Criterion

Two features, both computed on the **ground truth only**, so the partition
cannot be biased by how the model happens to behave:

- `hf` — share of spectral power above 0.25 × Nyquist. Separates dense texture from smooth content.
- `conc` — share of total gradient magnitude carried by the strongest 10% of pixels. Edges are sparse, so concentrated; texture is dense, so spread out.

Gradient energy **alone** cannot do this: texture-heavy and edge-dominated content
are both 'high gradient'. The concentration feature is what separates them, which
is why two features are used rather than one.

Thresholds, derived from this split and stated for reproducibility:

- smooth / low-frequency: `hf <= 0.0370` (lowest tercile)
- structured / edge-dominated: `hf > 0.0370` and `conc >= 0.3248`
- texture-heavy: `hf > 0.0370` and `conc < 0.3248`

## Results per group

| Group | n | PSNR bicubic → ours | SSIM bicubic → ours | LPIPS bicubic → ours |
|---|---|---|---|---|
| smooth / low-frequency | 67 | 23.60 → **31.36** | 0.482 → **0.842** | 0.470 → **0.205** |
| structured / edge-dominated | 67 | 23.25 → **26.73** | 0.583 → **0.781** | 0.382 → **0.229** |
| texture-heavy | 66 | 21.32 → **24.40** | 0.499 → **0.603** | 0.467 → **0.370** |
| *all 200 (for reference)* | 200 | 22.73 → 27.52 | 0.522 → 0.743 | 0.439 → 0.268 |

## Honest OOD estimate

The weakest group is **texture-heavy** (66 images): PSNR **24.4047 dB**, 
SSIM **0.6035**, LPIPS **0.3703**.

We treat that as our OOD expectation rather than the headline 27.52 dB average. If the hidden set skews toward this content,
performance should be read against **24.40 dB**, not the average — a gap of 
3.11 dB.

**The model still beats bicubic in every group**, so the ranking never inverts;
what changes is the margin.

## Caveat

This is a *proxy*. It partitions in-distribution content by character; it does not
sample genuinely unseen domains. It bounds the variation we can observe, which is
strictly less than the variation the hidden OOD half may contain.

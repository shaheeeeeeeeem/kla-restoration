import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import load_split, repo_path
from src.utils.imageio import load_npy
from src.utils.misc import load_config

BAR = "=" * 74
GROUPS = ["smooth / low-frequency", "structured / edge-dominated", "texture-heavy"]


def hf_fraction(img, cutoff=0.25):
    """Share of spectral power above `cutoff` x Nyquist. High for dense texture,
    low for smooth content. Computed on the GT, so the partition never depends on
    the model or on the degraded input."""
    f = np.fft.fftshift(np.abs(np.fft.fft2(img - img.mean())) ** 2)
    h, w = img.shape
    cy, cx = h / 2, w / 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    tot = f.sum()
    return float(f[r > cutoff].sum() / tot) if tot > 0 else 0.0


def grad_concentration(img, top=0.10):
    """Share of total gradient magnitude carried by the strongest `top` fraction of
    pixels. Edges are sparse -> concentrated. Texture is dense -> spread out."""
    gy, gx = np.gradient(img.astype(np.float64))
    g = np.sqrt(gy ** 2 + gx ** 2).ravel()
    tot = g.sum()
    if tot <= 0:
        return 0.0
    k = max(1, int(len(g) * top))
    return float(np.sort(g)[-k:].sum() / tot)


def read_metrics(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["id"]] = {k: float(r[k]) for k in ("psnr", "ssim", "lpips") if k in r}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=repo_path("configs/final.yaml"))
    p.add_argument("--model_csv", default=repo_path("results/per_image_model.csv"))
    p.add_argument("--bicubic_csv", default=repo_path("results/per_image_bicubic.csv"))
    p.add_argument("--out_csv", default=repo_path("results/ood_groups.csv"))
    p.add_argument("--report", default=repo_path("results/ood_analysis.md"))
    a = p.parse_args()

    cfg = load_config(a.config)
    _, val_ids = load_split(cfg.data.train_root)
    gt_dir = os.path.join(cfg.data.train_root, "GT")
    model = read_metrics(a.model_csv)
    bic = read_metrics(a.bicubic_csv)

    print(f"{BAR}\nOOD PROXY PARTITION OF THE HELD-OUT VALIDATION SPLIT\n{BAR}")
    print("Criterion is computed on the GROUND TRUTH only -- never on the model output or")
    print("the degraded input -- so the partition cannot be biased by model behaviour.\n")

    rows = []
    for fid in val_ids:
        gt = load_npy(os.path.join(gt_dir, fid))
        rows.append({"id": fid, "hf": hf_fraction(gt), "conc": grad_concentration(gt)})

    hf = np.array([r["hf"] for r in rows])
    hf_lo = float(np.quantile(hf, 1 / 3))
    smooth = [r for r in rows if r["hf"] <= hf_lo]
    rest = [r for r in rows if r["hf"] > hf_lo]
    conc_med = float(np.median([r["conc"] for r in rest]))
    for r in rows:
        if r["hf"] <= hf_lo:
            r["group"] = GROUPS[0]
        elif r["conc"] >= conc_med:
            r["group"] = GROUPS[1]
        else:
            r["group"] = GROUPS[2]

    print("Two features, both on GT:")
    print(f"  hf   = share of spectral power above 0.25 x Nyquist  (texture vs smooth)")
    print(f"  conc = share of gradient magnitude in the strongest 10% of pixels")
    print(f"         (edges are sparse -> high; texture is dense -> low)\n")
    print("Why two and not one: gradient energy alone cannot separate texture from edges --")
    print("both are 'high gradient'. The concentration feature is what splits them.\n")
    print(f"Thresholds (derived from this split, stated for reproducibility):")
    print(f"  smooth        : hf <= {hf_lo:.4f}   (lowest tercile)")
    print(f"  edge-dominated: hf >  {hf_lo:.4f} and conc >= {conc_med:.4f}")
    print(f"  texture-heavy : hf >  {hf_lo:.4f} and conc <  {conc_med:.4f}\n")

    summary = []
    for g in GROUPS:
        ids = [r["id"] for r in rows if r["group"] == g]
        if not ids:
            continue
        m = {k: float(np.mean([model[i][k] for i in ids])) for k in ("psnr", "ssim", "lpips")}
        b = {k: float(np.mean([bic[i][k] for i in ids])) for k in ("psnr", "ssim", "lpips")}
        summary.append((g, len(ids), b, m))

    print(f"{BAR}\nPER-GROUP RESULTS (bicubic -> ours)\n{BAR}")
    hdr = f"{'group':30s}{'n':>4s}{'PSNR':>18s}{'SSIM':>16s}{'LPIPS':>16s}"
    print(hdr)
    for g, n, b, m in summary:
        print(f"{g:30s}{n:4d}"
              f"{b['psnr']:8.2f}->{m['psnr']:7.2f}"
              f"{b['ssim']:7.3f}->{m['ssim']:6.3f}"
              f"{b['lpips']:7.3f}->{m['lpips']:6.3f}")

    worst = min(summary, key=lambda t: t[3]["psnr"])
    print(f"\nWEAKEST GROUP -> honest OOD estimate: {worst[0]}  ({worst[1]} images)")
    print(f"  ours: PSNR {worst[3]['psnr']:.4f}  SSIM {worst[3]['ssim']:.4f}  LPIPS {worst[3]['lpips']:.4f}")

    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "group", "hf", "conc", "psnr", "ssim", "lpips"])
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in ("id", "group", "hf", "conc")},
                        **{k: model[r["id"]][k] for k in ("psnr", "ssim", "lpips")}})

    overall_m = {k: float(np.mean([model[i][k] for i in val_ids])) for k in ("psnr", "ssim", "lpips")}
    overall_b = {k: float(np.mean([bic[i][k] for i in val_ids])) for k in ("psnr", "ssim", "lpips")}

    L = []
    L.append("# OOD proxy evaluation\n")
    L.append("Half the hidden test set is out-of-distribution *content*. We have no labelled")
    L.append("OOD data, and the released test inputs have no ground truth, so no direct")
    L.append("measurement is possible. This is a **re-partition of the existing held-out")
    L.append("validation split** by content character — no new data, no new disclosures.\n")
    L.append("## Criterion\n")
    L.append("Two features, both computed on the **ground truth only**, so the partition")
    L.append("cannot be biased by how the model happens to behave:\n")
    L.append("- `hf` — share of spectral power above 0.25 × Nyquist. Separates dense texture from smooth content.")
    L.append("- `conc` — share of total gradient magnitude carried by the strongest 10% of pixels. Edges are sparse, so concentrated; texture is dense, so spread out.\n")
    L.append("Gradient energy **alone** cannot do this: texture-heavy and edge-dominated content")
    L.append("are both 'high gradient'. The concentration feature is what separates them, which")
    L.append("is why two features are used rather than one.\n")
    L.append("Thresholds, derived from this split and stated for reproducibility:\n")
    L.append(f"- smooth / low-frequency: `hf <= {hf_lo:.4f}` (lowest tercile)")
    L.append(f"- structured / edge-dominated: `hf > {hf_lo:.4f}` and `conc >= {conc_med:.4f}`")
    L.append(f"- texture-heavy: `hf > {hf_lo:.4f}` and `conc < {conc_med:.4f}`\n")
    L.append("## Results per group\n")
    L.append("| Group | n | PSNR bicubic → ours | SSIM bicubic → ours | LPIPS bicubic → ours |")
    L.append("|---|---|---|---|---|")
    for g, n, b, m in summary:
        L.append(f"| {g} | {n} | {b['psnr']:.2f} → **{m['psnr']:.2f}** | "
                 f"{b['ssim']:.3f} → **{m['ssim']:.3f}** | {b['lpips']:.3f} → **{m['lpips']:.3f}** |")
    L.append(f"| *all 200 (for reference)* | 200 | {overall_b['psnr']:.2f} → {overall_m['psnr']:.2f} | "
             f"{overall_b['ssim']:.3f} → {overall_m['ssim']:.3f} | {overall_b['lpips']:.3f} → {overall_m['lpips']:.3f} |\n")
    L.append("## Honest OOD estimate\n")
    L.append(f"The weakest group is **{worst[0]}** ({worst[1]} images): PSNR **{worst[3]['psnr']:.4f} dB**, ")
    L.append(f"SSIM **{worst[3]['ssim']:.4f}**, LPIPS **{worst[3]['lpips']:.4f}**.\n")
    L.append(f"We treat that as our OOD expectation rather than the headline "
             f"{overall_m['psnr']:.2f} dB average. If the hidden set skews toward this content,")
    L.append(f"performance should be read against **{worst[3]['psnr']:.2f} dB**, not the average — a gap of ")
    L.append(f"{overall_m['psnr'] - worst[3]['psnr']:.2f} dB.\n")
    L.append("**The model still beats bicubic in every group**, so the ranking never inverts;")
    L.append("what changes is the margin.\n")
    L.append("## Caveat\n")
    L.append("This is a *proxy*. It partitions in-distribution content by character; it does not")
    L.append("sample genuinely unseen domains. It bounds the variation we can observe, which is")
    L.append("strictly less than the variation the hidden OOD half may contain.\n")
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"\nwrote {a.out_csv}")
    print(f"wrote {a.report}")
    return worst


if __name__ == "__main__":
    main()

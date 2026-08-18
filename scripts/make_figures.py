import argparse
import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import load_split, repo_path
from src.metrics.image_metrics import psnr, ssim
from src.utils.imageio import load_npy
from src.utils.misc import load_config

FIGDIR = repo_path("results/figures")


def panel(rows, titles, path, suptitle):
    n, m = len(rows), len(rows[0])
    fig, ax = plt.subplots(n, m, figsize=(3.1 * m, 3.25 * n))
    ax = np.atleast_2d(ax)
    for i, row in enumerate(rows):
        for j, (img, t) in enumerate(zip(row, titles[i])):
            ax[i, j].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1,
                            interpolation="nearest")
            ax[i, j].set_title(t, fontsize=8)
            ax[i, j].axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", default=repo_path("results/preds_model_val"))
    p.add_argument("--bicubic_dir", default=repo_path("results/preds_bicubic_val"))
    p.add_argument("--per_image_csv", default=repo_path("results/per_image_model.csv"))
    p.add_argument("--n_best", type=int, default=3)
    a = p.parse_args()

    cfg = load_config(repo_path("configs/final.yaml"))
    _, val_ids = load_split(cfg.data.train_root)
    gt_dir = os.path.join(cfg.data.train_root, "GT")
    lr_dir = os.path.join(cfg.data.train_root, "NoisyLR")
    os.makedirs(FIGDIR, exist_ok=True)

    scores = []
    if os.path.exists(a.per_image_csv):
        with open(a.per_image_csv) as f:
            for r in csv.DictReader(f):
                scores.append((float(r["psnr"]), r["id"]))
    else:
        for fid in val_ids:
            pr = torch.from_numpy(load_npy(os.path.join(a.pred_dir, fid)))[None, None]
            gt = torch.from_numpy(load_npy(os.path.join(gt_dir, fid)))[None, None]
            scores.append((float(psnr(pr, gt)), fid))
    scores.sort()

    def build(fid):
        lr = load_npy(os.path.join(lr_dir, fid))
        gt = load_npy(os.path.join(gt_dir, fid))
        bic = load_npy(os.path.join(a.bicubic_dir, fid))
        pr = load_npy(os.path.join(a.pred_dir, fid))
        t = torch.from_numpy
        pp = float(psnr(t(pr)[None, None], t(gt)[None, None]))
        ps = float(ssim(t(pr)[None, None], t(gt)[None, None]))
        bp = float(psnr(t(bic)[None, None], t(gt)[None, None]))
        lr_up = F.interpolate(t(lr)[None, None], scale_factor=2, mode="nearest")[0, 0].numpy()
        return ([lr_up, bic, pr, gt],
                [f"{fid}  input (nearest-zoom)", f"bicubic  {bp:.2f} dB",
                 f"ours  {pp:.2f} dB / {ps:.3f} SSIM", "ground truth"])

    best = scores[-a.n_best:][::-1]
    rows, titles = zip(*[build(f) for _, f in best])
    panel(list(rows), list(titles), os.path.join(FIGDIR, "qualitative_best.png"),
          "Best cases -- validation split")

    worst = scores[:a.n_best]
    rows, titles = zip(*[build(f) for _, f in worst])
    panel(list(rows), list(titles), os.path.join(FIGDIR, "qualitative_failures.png"),
          "FAILURE CASES -- lowest-PSNR validation images")

    mid = scores[len(scores) // 2 - 1: len(scores) // 2 + 2]
    rows, titles = zip(*[build(f) for _, f in mid])
    panel(list(rows), list(titles), os.path.join(FIGDIR, "qualitative_typical.png"),
          "Typical cases -- median-PSNR validation images")

    fid = scores[len(scores) // 2][1]
    gt = load_npy(os.path.join(gt_dir, fid))
    pr = load_npy(os.path.join(a.pred_dir, fid))
    bic = load_npy(os.path.join(a.bicubic_dir, fid))
    h, w = gt.shape
    y, x, s = h // 3, w // 3, min(h, w) // 4
    crop = lambda im: im[y:y + s, x:x + s]
    panel([[crop(bic), crop(pr), crop(gt)]],
          [[f"{fid} bicubic (crop)", "ours (crop)", "GT (crop)"]],
          os.path.join(FIGDIR, "detail_crop.png"),
          "Detail crop -- residual speckle and fine texture")

    ps = [s for s, _ in scores]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(ps, bins=30, color="#4a7ebb", edgecolor="white")
    ax.axvline(np.mean(ps), color="crimson", ls="--",
               label=f"mean {np.mean(ps):.2f} dB")
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("val images")
    ax.set_title("Per-image PSNR distribution (validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "psnr_distribution.png"), dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.join(FIGDIR, 'psnr_distribution.png')}")

    print(f"\n  worst {a.n_best}: {[(round(s,2), f) for s, f in worst]}")
    print(f"  best  {a.n_best}: {[(round(s,2), f) for s, f in best]}")


if __name__ == "__main__":
    main()

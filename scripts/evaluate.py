import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import load_split, repo_path
from src.metrics.image_metrics import LPIPSMetric, psnr, ssim
from src.utils.imageio import load_npy
from src.utils.misc import load_config


def evaluate_dir(pred_dir, gt_dir, ids, device="cpu", lpips_metric=None):
    rows = []
    for fid in ids:
        pp = os.path.join(pred_dir, fid)
        if not os.path.exists(pp):
            raise FileNotFoundError(f"prediction missing for {fid}: {pp}")
        pred = torch.from_numpy(load_npy(pp))[None, None].to(device)
        gt = torch.from_numpy(load_npy(os.path.join(gt_dir, fid)))[None, None].to(device)
        if pred.shape != gt.shape:
            raise RuntimeError(f"{fid}: prediction {tuple(pred.shape)} != GT {tuple(gt.shape)}")

        out_of_range = float(((pred < 0) | (pred > 1)).float().mean())
        r = {
            "id": fid,
            "psnr": float(psnr(pred, gt)),
            "ssim": float(ssim(pred, gt)),
            "frac_out_of_range": out_of_range,
        }
        if lpips_metric is not None:
            r["lpips"] = float(lpips_metric(pred, gt))
        rows.append(r)
    return rows


def summarize(rows):
    keys = [k for k in ("psnr", "ssim", "lpips") if k in rows[0]]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--config", default=repo_path("configs/final.yaml"))
    p.add_argument("--name", default="model")
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--per_image_csv", default="")
    p.add_argument("--json_out", default="")
    a = p.parse_args()

    cfg = load_config(a.config)
    _, val_ids = load_split(cfg.data.train_root)
    gt_dir = os.path.join(cfg.data.train_root, "GT")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lp = None if a.no_lpips else LPIPSMetric(net="alex", device=device)
    rows = evaluate_dir(a.pred_dir, gt_dir, val_ids, device=device, lpips_metric=lp)
    s = summarize(rows)

    print(f"\n{a.name}   ({len(rows)} val images, metrics computed on SAVED .npy files)")
    print(f"  PSNR  : {s['psnr']:.4f} dB")
    print(f"  SSIM  : {s['ssim']:.4f}")
    if "lpips" in s:
        print(f"  LPIPS : {s['lpips']:.4f}   (alex, lower is better)")
    worst = max(r["frac_out_of_range"] for r in rows)
    print(f"  saved-file range check: max frac of pixels outside [0,1] = {worst:.6f}")

    if a.per_image_csv:
        import csv
        os.makedirs(os.path.dirname(a.per_image_csv) or ".", exist_ok=True)
        with open(a.per_image_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"  per-image csv -> {a.per_image_csv}")

    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        payload = {"name": a.name, "pred_dir": a.pred_dir, "n": len(rows), **s}
        with open(a.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  summary json  -> {a.json_out}")


if __name__ == "__main__":
    main()

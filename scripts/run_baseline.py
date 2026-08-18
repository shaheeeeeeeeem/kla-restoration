import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import load_split, repo_path
from src.utils.imageio import load_npy, save_npy
from src.utils.misc import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=repo_path("configs/final.yaml"))
    p.add_argument("--out_dir", default=repo_path("results/preds_bicubic_val"))
    p.add_argument("--mode", default="bicubic", choices=["bicubic", "bilinear", "nearest"])
    a = p.parse_args()

    cfg = load_config(a.config)
    _, val_ids = load_split(cfg.data.train_root)
    lr_dir = os.path.join(cfg.data.train_root, "NoisyLR")
    os.makedirs(a.out_dir, exist_ok=True)

    for fid in val_ids:
        lr = torch.from_numpy(load_npy(os.path.join(lr_dir, fid)))[None, None]
        kw = {} if a.mode == "nearest" else {"align_corners": False}
        up = F.interpolate(lr, scale_factor=cfg.scale, mode=a.mode, **kw)
        save_npy(os.path.join(a.out_dir, fid), up[0, 0].numpy())

    print(f"{a.mode} x{cfg.scale} baseline: wrote {len(val_ids)} arrays to {a.out_dir}")


if __name__ == "__main__":
    main()

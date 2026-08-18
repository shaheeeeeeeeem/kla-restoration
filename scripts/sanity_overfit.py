import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import load_split, repo_path
from src.engine.common import bicubic_up, forward_pair
from src.losses.restoration_loss import RestorationLoss
from src.metrics.image_metrics import psnr
from src.models.nafnet_sr import build_model
from src.utils.imageio import load_npy
from src.utils.misc import enable_tf32, load_config, set_seed

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=2)
p.add_argument("--iters", type=int, default=400)
p.add_argument("--lr", type=float, default=1e-3)
a = p.parse_args()

cfg = load_config(repo_path("configs/final.yaml"))
set_seed(cfg.seed)
enable_tf32()
dev = "cuda" if torch.cuda.is_available() else "cpu"

train_ids, _ = load_split(cfg.data.train_root)
ids = train_ids[: a.n]
lr = torch.stack([torch.from_numpy(load_npy(os.path.join(cfg.data.train_root, "NoisyLR", f))) for f in ids])[:, None].to(dev)
gt = torch.stack([torch.from_numpy(load_npy(os.path.join(cfg.data.train_root, "GT", f))) for f in ids])[:, None].to(dev)

print(f"SANITY A -- memorize {len(ids)} FIXED full images (no crop, no augmentation)")
print(f"  ids {ids}   LR {tuple(lr.shape)} -> GT {tuple(gt.shape)}")

base = bicubic_up(lr, cfg.scale).clamp(0, 1)
print(f"  bicubic PSNR on these images : {float(psnr(base, gt)):.3f} dB")

model = build_model(cfg).to(dev)
crit = RestorationLoss(cfg.loss.charbonnier_weight, cfg.loss.charbonnier_eps, cfg.loss.msssim_weight)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

t0 = time.time()
for i in range(a.iters):
    out = forward_pair(model, lr, cfg)
    loss, parts = crit(out, gt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if (i + 1) % max(1, a.iters // 8) == 0:
        with torch.no_grad():
            pv = float(psnr(out.detach().clamp(0, 1), gt))
        print(f"  iter {i + 1:5d}  loss {parts['total']:.6f}  PSNR {pv:.3f} dB")

with torch.no_grad():
    final = float(psnr(forward_pair(model, lr, cfg).clamp(0, 1), gt))
gain = final - float(psnr(base, gt))
print(f"\n  final PSNR {final:.3f} dB   (+{gain:.2f} dB over bicubic)   in {time.time() - t0:.0f}s")
print(f"  SANITY A: {'PASS -- network can fit the mapping' if final > 35 else 'INVESTIGATE -- did not memorize'}")

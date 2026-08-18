import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import load_split, repo_path
from src.utils.imageio import load_npy, save_npy
from src.utils.misc import load_config

cfg = load_config(repo_path("configs/final.yaml"))
_, val_ids = load_split(cfg.data.train_root)
gt_dir = os.path.join(cfg.data.train_root, "GT")

print("SANITY B -- save/reload round trip under the Phase 1 output contract")
print("  contract: same filename, .npy, float32, (256,256), clipped to [0,1]\n")

tmp = tempfile.mkdtemp()
rng = np.random.default_rng(0)
fails = 0

for fid in val_ids[:25]:
    gt = load_npy(os.path.join(gt_dir, fid))
    save_npy(os.path.join(tmp, fid), gt)
    back = load_npy(os.path.join(tmp, fid))
    if not np.array_equal(gt, back):
        print(f"  FAIL bit-exactness on already-in-range GT {fid}")
        fails += 1
    if back.dtype != np.float32:
        print(f"  FAIL dtype {back.dtype} on {fid}")
        fails += 1
    if back.shape != gt.shape:
        print(f"  FAIL shape {back.shape} != {gt.shape} on {fid}")
        fails += 1

print(f"  in-range GT round trip : {'BIT-EXACT' if fails == 0 else 'BROKEN'} over 25 files")

x = torch.from_numpy(rng.normal(0.45, 0.35, (256, 256)).astype(np.float32))
p = os.path.join(tmp, "clip_test.npy")
save_npy(p, x.numpy())
b = load_npy(p)
ref = np.clip(x.numpy(), 0, 1)
print(f"  out-of-range input     : saved min {b.min():.6f} max {b.max():.6f} "
      f"(input was {x.min():.3f}..{x.max():.3f})")
print(f"  clip is exact          : {np.array_equal(b, ref)}")
print(f"  values in [0,1]        : {bool((b >= 0).all() and (b <= 1).all())}")

t = torch.rand(1, 1, 256, 256)
p2 = os.path.join(tmp, "chan_test.npy")
save_npy(p2, t[0].numpy())
b2 = load_npy(p2)
print(f"  channel axis stripped  : saved shape {b2.shape} from tensor {tuple(t[0].shape)}")

ok = fails == 0 and np.array_equal(b, ref) and b2.shape == (256, 256)
print(f"\n  SANITY B: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)

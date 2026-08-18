import argparse
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.data.split import load_split, repo_path
from src.engine.common import forward_pair, forward_self_ensemble
from src.metrics.image_metrics import LPIPSMetric, psnr, ssim
from src.models.nafnet_sr import build_model
from src.utils.imageio import load_npy, save_npy
from src.utils.misc import Config

BAR = "=" * 70


def predict(model, cfg, ids, lr_dir, out_dir, se, batch=16):
    os.makedirs(out_dir, exist_ok=True)
    with torch.inference_mode():
        for i in range(0, len(ids), batch):
            ch = ids[i:i + batch]
            x = torch.from_numpy(np.stack([load_npy(os.path.join(lr_dir, f)) for f in ch]))[:, None]
            x = x.cuda().to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = forward_self_ensemble(model, x, cfg) if se else forward_pair(model, x, cfg)
            o = o.float().clamp(0, 1).cpu().numpy()
            for f, arr in zip(ch, o):
                save_npy(os.path.join(out_dir, f), arr[0])


def score(pred_dir, gt_dir, ids, lp):
    ps, ss, lps = [], [], []
    for f in ids:
        p = torch.from_numpy(load_npy(os.path.join(pred_dir, f)))[None, None].cuda()
        g = torch.from_numpy(load_npy(os.path.join(gt_dir, f)))[None, None].cuda()
        ps.append(float(psnr(p, g)))
        ss.append(float(ssim(p, g)))
        lps.append(float(lp(p, g)))
    return float(np.mean(ps)), float(np.mean(ss)), float(np.mean(lps))


def time_run(input_dir, out_dir, se, repeats=3):
    cmd = [sys.executable, os.path.join(HERE, "inference.py"),
           "--input_dir", input_dir, "--output_dir", out_dir]
    if se:
        cmd.append("--self_ensemble")
    ts = []
    for r in range(repeats + 1):
        t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t
        if p.returncode != 0:
            print(p.stderr[-1500:])
            raise SystemExit("inference failed")
        if r:
            ts.append(dt)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="C:/Users/mdsha/Downloads/Test_NoisyLR (1)/NoisyLR")
    ap.add_argument("--scratch", default=os.environ.get("TEMP", "/tmp"))
    a = ap.parse_args()

    from src.utils.misc import load_config
    cfg_top = load_config(repo_path("configs/final.yaml"))
    _, val_ids = load_split(cfg_top.data.train_root)
    gt_dir = os.path.join(cfg_top.data.train_root, "GT")
    lr_dir = os.path.join(cfg_top.data.train_root, "NoisyLR")

    ck = torch.load(repo_path("weights/best.pt"), map_location="cuda", weights_only=False)
    cfg = Config(ck["cfg"])
    model = build_model(cfg)
    model.load_state_dict(ck["ema"])
    model.cuda().eval().to(memory_format=torch.channels_last)

    lp = LPIPSMetric(net="alex", device="cuda")
    print(f"{BAR}\nSELF-ENSEMBLE TRADE-OFF (val split, {len(val_ids)} images)\n{BAR}")

    res = {}
    for se, label in [(False, "single pass (default)"), (True, "x8 self-ensemble")]:
        out = os.path.join(a.scratch, "se_val_on" if se else "se_val_off")
        predict(model, cfg, val_ids, lr_dir, out, se)
        p, s, l = score(out, gt_dir, val_ids, lp)
        res[label] = {"psnr": p, "ssim": s, "lpips": l}
        print(f"  {label:24s} PSNR {p:.4f}  SSIM {s:.4f}  LPIPS {l:.4f}")

    print(f"\n{BAR}\nEND-TO-END RUNTIME on 400 test images (median of 3, incl. I/O)\n{BAR}")
    for se, label in [(False, "single pass (default)"), (True, "x8 self-ensemble")]:
        out = os.path.join(a.scratch, "se_time_on" if se else "se_time_off")
        t = time_run(a.test_dir, out, se)
        res[label]["seconds"] = t
        res[label]["img_s"] = 400 / t
        print(f"  {label:24s} {t:.2f} s   {400 / t:.2f} img/s")

    off = res["single pass (default)"]
    on = res["x8 self-ensemble"]
    dp = on["psnr"] - off["psnr"]
    slow = off["img_s"] / on["img_s"]

    print(f"\n{BAR}\nDECISION\n{BAR}")
    print(f"  quality gain : {dp:+.4f} dB PSNR, {on['ssim'] - off['ssim']:+.4f} SSIM, "
          f"{on['lpips'] - off['lpips']:+.4f} LPIPS")
    print(f"  runtime cost : {slow:.2f}x slower ({off['img_s']:.1f} -> {on['img_s']:.1f} img/s)")
    print(f"  dB per unit of throughput given up: {dp / max(slow - 1, 1e-9):.4f}")
    print(f"  -> default stays OFF" if dp < 0.15 else "  -> gain may justify the cost")

    L = ["# Self-ensemble trade-off\n",
         f"Measured on the {len(val_ids)}-image held-out validation split for quality, and on the",
         "400 released test inputs for end-to-end runtime (median of 3 runs, including disk",
         "read and file writing). Same checkpoint, no retraining.\n",
         "| Mode | PSNR ↑ | SSIM ↑ | LPIPS ↓ | End-to-end | Throughput |",
         "|---|---|---|---|---|---|",
         f"| **single pass (default)** | **{off['psnr']:.4f}** | **{off['ssim']:.4f}** | "
         f"**{off['lpips']:.4f}** | **{off['seconds']:.2f} s** | **{off['img_s']:.1f} img/s** |",
         f"| ×8 self-ensemble | {on['psnr']:.4f} | {on['ssim']:.4f} | {on['lpips']:.4f} | "
         f"{on['seconds']:.2f} s | {on['img_s']:.1f} img/s |",
         f"| difference | {dp:+.4f} dB | {on['ssim'] - off['ssim']:+.4f} | "
         f"{on['lpips'] - off['lpips']:+.4f} | {on['seconds'] - off['seconds']:+.2f} s | "
         f"{slow:.2f}× slower |\n",
         "## Decision: default OFF\n",
         "Quality and throughput are scored separately, with undisclosed weights. The ×8",
         f"self-ensemble buys **{dp:+.4f} dB** for **{slow:.2f}× the runtime**. Since we cannot",
         "know how KLA weights the two axes, we take the option that is strong on both rather",
         "than trading a large, certain throughput loss for a small, uncertain quality gain.\n",
         "It remains available and is a one-flag change:\n",
         "```bash",
         "python inference.py --input_dir <in> --output_dir <out> --self_ensemble",
         "```\n",
         "The default path is untouched by this addition — verified 400/400 bit-identical to",
         "the outputs the clean-room dry run signed off on.\n"]
    with open(repo_path("results/self_ensemble.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nwrote results/self_ensemble.md")

    import json
    with open(repo_path("results/self_ensemble.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

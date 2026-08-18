import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.engine.common import forward_pair
from src.models.nafnet_sr import build_model
from src.utils.imageio import load_npy, save_npy
from src.utils.misc import Config, load_config


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def load_checkpoint(weights, device):
    ck = torch.load(weights, map_location=device, weights_only=False)
    cfg = Config(ck["cfg"]) if "cfg" in ck and ck["cfg"] else load_config(resolve("configs/final.yaml"))
    model = build_model(cfg)
    state = ck.get("ema") or ck["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg, ck


class ShapeGroupedFiles:
    def __init__(self, input_dir):
        self.dir = input_dir
        files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy"))
        if not files:
            raise RuntimeError(f"no .npy files found in {input_dir}")
        self.groups = defaultdict(list)
        for f in files:
            shape = np.load(os.path.join(input_dir, f), mmap_mode="r").shape
            self.groups[tuple(shape[-2:])].append(f)
        self.n = len(files)


def main():
    p = argparse.ArgumentParser(description="Restore degraded .npy images (denoise + x2 upscale).")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--fp32", action="store_true")
    a = p.parse_args()

    if a.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = a.device
    if device == "cpu":
        print("WARNING: no CUDA device found, running on CPU. This will be slow.")

    weights = resolve(a.weights)
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"weights not found at {weights}\n"
            f"See README.md for the download location.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model, cfg, ck = load_checkpoint(weights, device)
    model = model.to(memory_format=torch.channels_last)
    os.makedirs(a.output_dir, exist_ok=True)

    grouped = ShapeGroupedFiles(a.input_dir)
    print(f"input   : {a.input_dir}  ({grouped.n} arrays)")
    print(f"output  : {a.output_dir}")
    print(f"weights : {weights}  (step {ck.get('step', '?')}, git {ck.get('git', '?')})")
    print(f"device  : {device}   shapes: {dict((k, len(v)) for k, v in grouped.groups.items())}")

    use_amp = (device == "cuda") and not a.fp32
    t0 = time.time()
    done = 0

    with torch.inference_mode(), ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for shape, files in grouped.groups.items():
            for i in range(0, len(files), a.batch_size):
                chunk = files[i:i + a.batch_size]
                batch = np.stack([load_npy(os.path.join(a.input_dir, f)) for f in chunk])
                x = torch.from_numpy(batch)[:, None].to(device, non_blocking=True)
                x = x.to(memory_format=torch.channels_last)

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = forward_pair(model, x, cfg)
                out = out.float().clamp_(0, 1).cpu().numpy()

                for f, arr in zip(chunk, out):
                    futures.append(pool.submit(save_npy, os.path.join(a.output_dir, f), arr[0]))
                done += len(chunk)
                print(f"\r  {done}/{grouped.n}", end="", flush=True)

        for f in futures:
            f.result()

    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"\ndone: {grouped.n} images in {dt:.2f} s  ({grouped.n / dt:.2f} img/s, "
          f"includes disk read + write)")


if __name__ == "__main__":
    main()

import argparse
import os
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import repo_path
from src.engine.common import forward_pair
from src.models.nafnet_sr import build_model
from src.utils.imageio import load_npy, save_npy
from src.utils.misc import Config, load_config


class Stage:
    def __init__(self):
        self.t = defaultdict(float)

    def add(self, k, dt):
        self.t[k] += dt


def bench(input_dir, out_dir, weights, batch_size, device, repeats, warmup):
    ck = torch.load(weights, map_location=device, weights_only=False)
    cfg = Config(ck["cfg"]) if ck.get("cfg") else load_config(repo_path("configs/final.yaml"))
    model = build_model(cfg)
    model.load_state_dict(ck.get("ema") or ck["model"])
    model.to(device).eval().to(memory_format=torch.channels_last)

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy"))
    os.makedirs(out_dir, exist_ok=True)
    totals = []

    for run in range(warmup + repeats):
        s = Stage()
        torch.cuda.synchronize() if device == "cuda" else None
        t_all = time.perf_counter()

        with torch.inference_mode(), ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for i in range(0, len(files), batch_size):
                chunk = files[i:i + batch_size]

                t = time.perf_counter()
                batch = np.stack([load_npy(os.path.join(input_dir, f)) for f in chunk])
                s.add("read", time.perf_counter() - t)

                t = time.perf_counter()
                x = torch.from_numpy(batch)[:, None]
                s.add("preprocess", time.perf_counter() - t)

                t = time.perf_counter()
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                if device == "cuda":
                    torch.cuda.synchronize()
                s.add("h2d", time.perf_counter() - t)

                t = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    out = forward_pair(model, x, cfg)
                if device == "cuda":
                    torch.cuda.synchronize()
                s.add("forward", time.perf_counter() - t)

                t = time.perf_counter()
                out = out.float().clamp_(0, 1)
                arr = out.cpu().numpy()
                s.add("d2h", time.perf_counter() - t)

                t = time.perf_counter()
                for f, a in zip(chunk, arr):
                    futures.append(pool.submit(save_npy, os.path.join(out_dir, f), a[0]))
                s.add("write_submit", time.perf_counter() - t)

            t = time.perf_counter()
            for f in futures:
                f.result()
            s.add("write_drain", time.perf_counter() - t)

        total = time.perf_counter() - t_all
        if run >= warmup:
            totals.append((total, dict(s.t)))
    return totals, len(files), cfg, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", default=repo_path("results/_bench_out"))
    p.add_argument("--weights", default=repo_path("weights/best.pt"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--report", default=repo_path("results/runtime_report.md"))
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    totals, n, cfg, ck = bench(a.input_dir, a.out_dir, a.weights, a.batch_size,
                               device, a.repeats, a.warmup)

    times = [t for t, _ in totals]
    med = statistics.median(times)
    stages = defaultdict(list)
    for _, d in totals:
        for k, v in d.items():
            stages[k].append(v)

    order = ["read", "preprocess", "h2d", "forward", "d2h", "write_submit", "write_drain"]
    lines = []
    lines.append("# Runtime report\n")
    lines.append(f"**Measured on {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}, "
                 f"NOT on an H100.** Treat these as relative numbers.\n")
    lines.append("## Environment\n")
    if device == "cuda":
        pr = torch.cuda.get_device_properties(0)
        lines.append(f"- GPU: {torch.cuda.get_device_name(0)} ({pr.total_memory / 1e9:.2f} GB, sm_{pr.major}{pr.minor})")
    lines.append(f"- torch {torch.__version__}, CUDA {torch.version.cuda}")
    lines.append(f"- precision: bf16 autocast, channels_last, TF32 on")
    lines.append(f"- checkpoint: step {ck.get('step', '?')}, git `{ck.get('git', '?')}`, seed {ck.get('seed', '?')}\n")
    lines.append("## Method\n")
    lines.append(f"- {n} images, batch size {a.batch_size}, `torch.inference_mode()`")
    lines.append(f"- {a.warmup} warmup passes discarded, **median of {a.repeats}** timed passes")
    lines.append("- per-stage timing via `time.perf_counter()` with `torch.cuda.synchronize()` "
                 "around device work; the end-to-end total is a single wall clock over the whole loop")
    lines.append("- **includes disk read and file writing**, as the scoring criterion requires")
    lines.append("- output writing is threaded (8 workers) and overlaps compute\n")
    lines.append("## End-to-end\n")
    lines.append(f"- **total wall clock: {med:.3f} s** for {n} images")
    lines.append(f"- **throughput: {n / med:.2f} images/sec**")
    lines.append(f"- per-image: {med / n * 1000:.2f} ms")
    lines.append(f"- spread over {a.repeats} runs: min {min(times):.3f} s, max {max(times):.3f} s\n")
    lines.append("## Per-stage breakdown\n")
    lines.append("Stage times are summed across batches within one pass. They overlap with "
                 "threaded writing, so they do not add up to the wall clock.\n")
    lines.append("| Stage | Median (s) | % of wall clock |")
    lines.append("|---|---|---|")
    for k in order:
        if k in stages:
            m = statistics.median(stages[k])
            lines.append(f"| {k} | {m:.3f} | {100 * m / med:.1f}% |")
    lines.append("")

    with open(a.report, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {a.report}")


if __name__ == "__main__":
    main()

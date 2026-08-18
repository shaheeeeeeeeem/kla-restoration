import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.engine.common import forward_pair
from src.models.nafnet_sr import build_model
from src.utils.imageio import load_npy, save_npy
from src.utils.misc import Config

BAR = "=" * 72
STRIDE = 8


def make_inputs(src_dir, out_dir, size, n, mode="bicubic"):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".npy"))[:n]
    for f in files:
        a = torch.from_numpy(load_npy(os.path.join(src_dir, f)))[None, None]
        r = F.interpolate(a, size=(size, size), mode=mode, align_corners=False)
        # deliberately NOT clipped: the real inputs exceed [0,1] and so must these
        np.save(os.path.join(out_dir, f), r[0, 0].numpy().astype(np.float32))
    return files


def run_inference(input_dir, output_dir, extra=()):
    cmd = [sys.executable, os.path.join(HERE, "inference.py"),
           "--input_dir", input_dir, "--output_dir", output_dir, *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
    return p.returncode == 0


def check_contract(in_dir, out_dir, expect_scale=2):
    fi = sorted(f for f in os.listdir(in_dir) if f.endswith(".npy"))
    fo = sorted(f for f in os.listdir(out_dir) if f.endswith(".npy"))
    ok = True
    if fi != fo:
        print(f"    FAIL filenames differ ({len(fi)} in, {len(fo)} out)")
        ok = False
    shapes, dtypes = set(), set()
    lo, hi = 1e9, -1e9
    for f in fo:
        a = np.load(os.path.join(out_dir, f))
        b = np.load(os.path.join(in_dir, f))
        shapes.add((b.shape, a.shape))
        dtypes.add(str(a.dtype))
        lo = min(lo, float(a.min()))
        hi = max(hi, float(a.max()))
    for bs, as_ in shapes:
        if as_ != (bs[0] * expect_scale, bs[1] * expect_scale):
            print(f"    FAIL {bs} -> {as_}, expected x{expect_scale}")
            ok = False
    if dtypes != {"float32"}:
        print(f"    FAIL dtype {dtypes}")
        ok = False
    if lo < 0.0 or hi > 1.0:
        print(f"    FAIL range {lo} .. {hi}")
        ok = False
    print(f"    shapes {sorted(str(s) for s in shapes)}")
    print(f"    dtype {dtypes}  range {lo:.6f} .. {hi:.6f}  filenames match: {fi == fo}")
    return ok


def border_artifact(in_dir, out_dir, band=STRIDE):
    """A padding seam shows up as anomalous residual energy in the outermost rows/cols.
    Compare the border band against the interior, on (output - bicubic(input))."""
    worst = 0.0
    for f in sorted(os.listdir(out_dir))[:8]:
        x = torch.from_numpy(load_npy(os.path.join(in_dir, f)))[None, None]
        y = torch.from_numpy(load_npy(os.path.join(out_dir, f)))[None, None]
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        r = (y - base).abs()[0, 0]
        b = band * 2
        interior = r[b:-b, b:-b].mean().item()
        edges = torch.cat([r[:b, :].flatten(), r[-b:, :].flatten(),
                           r[:, :b].flatten(), r[:, -b:].flatten()]).mean().item()
        if interior > 1e-9:
            worst = max(worst, edges / interior)
    return worst


def vram_probe(size, batch=1):
    ck = torch.load(os.path.join(HERE, "weights", "best.pt"), map_location="cuda",
                    weights_only=False)
    cfg = Config(ck["cfg"])
    m = build_model(cfg)
    m.load_state_dict(ck["ema"])
    m.cuda().eval().to(memory_format=torch.channels_last)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, 1, size, size, device="cuda").to(memory_format=torch.channels_last)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = forward_pair(m, x, cfg)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    shape = tuple(out.shape)
    del m, x, out
    torch.cuda.empty_cache()
    return peak, shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", default="C:/Users/mdsha/Downloads/Test_NoisyLR (1)/NoisyLR")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--keep", default="")
    a = p.parse_args()

    tmp = a.keep or tempfile.mkdtemp(prefix="respaths_")
    os.makedirs(tmp, exist_ok=True)
    allok = True

    print(f"{BAR}\nRESOLUTION PATH VERIFICATION\n{BAR}")
    print("No 512x512 GT exists in the released data, so 256x256 inputs are produced by")
    print("bicubic-upsampling the released 128x128 test inputs. This is a SHAPE and")
    print("ROBUSTNESS probe, not a quality measurement -- there is no ground truth for it.")
    print(f"Network stride is {STRIDE}; sizes divisible by {STRIDE} never hit the padding path,")
    print("so non-multiple sizes are tested separately to exercise it.\n")

    cases = [
        ("256x256 (the graded-risk size)", 256, False),
        ("250x250 (non-multiple, exercises padding)", 250, True),
        ("129x129 (odd, exercises padding)", 129, True),
    ]
    for label, size, pads in cases:
        print(f"{BAR}\n{label}\n{BAR}")
        ind = os.path.join(tmp, f"in_{size}")
        outd = os.path.join(tmp, f"out_{size}")
        files = make_inputs(a.test_dir, ind, size, a.n)
        shutil.rmtree(outd, ignore_errors=True)
        ok = run_inference(ind, outd)
        print(f"    inference.py exit ok: {ok}")
        ok = check_contract(ind, outd) and ok
        ratio = border_artifact(ind, outd)
        verdict = "no seam" if ratio < 1.6 else "SEAM SUSPECTED"
        print(f"    border/interior residual ratio: {ratio:.3f}  ({verdict}, padding used: {pads})")
        if ratio >= 1.6:
            ok = False
        allok = allok and ok

    print(f"{BAR}\nMIXED-SHAPE DIRECTORY (shape-grouped batching)\n{BAR}")
    mix = os.path.join(tmp, "in_mixed")
    outmix = os.path.join(tmp, "out_mixed")
    os.makedirs(mix, exist_ok=True)
    src = sorted(f for f in os.listdir(a.test_dir) if f.endswith(".npy"))[:a.n]
    for i, f in enumerate(src):
        arr = load_npy(os.path.join(a.test_dir, f))
        if i % 3 == 1:
            arr = F.interpolate(torch.from_numpy(arr)[None, None], size=(256, 256),
                                mode="bicubic", align_corners=False)[0, 0].numpy()
        elif i % 3 == 2:
            arr = F.interpolate(torch.from_numpy(arr)[None, None], size=(250, 250),
                                mode="bicubic", align_corners=False)[0, 0].numpy()
        np.save(os.path.join(mix, f), arr.astype(np.float32))
    shutil.rmtree(outmix, ignore_errors=True)
    ok = run_inference(mix, outmix)
    print(f"    inference.py exit ok: {ok}")
    ok = check_contract(mix, outmix) and ok
    allok = allok and ok

    if torch.cuda.is_available():
        print(f"{BAR}\nPEAK VRAM (inference, batch 1)\n{BAR}")
        for size in [128, 256, 512]:
            peak, shape = vram_probe(size)
            print(f"    {size:4d}x{size:<4d} -> {str(shape):22s} peak {peak:.3f} GB")
        peak, shape = vram_probe(256, batch=16)
        print(f"    256x256 x batch16 -> {str(shape):18s} peak {peak:.3f} GB")

    print(f"\n{BAR}\nRESULT: {'ALL CHECKS PASS' if allok else 'FAILURES PRESENT'}\n{BAR}")
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())

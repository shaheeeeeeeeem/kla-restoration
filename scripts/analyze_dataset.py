import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

BAR = "=" * 70


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def list_npy(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".npy"))


def scan_dir(name, d, sample_stats=None):
    files = list_npy(d)
    print(f"\n{BAR}\n{name}\n  path: {d}\n{BAR}")
    if not files:
        print("  !! no .npy files found")
        return None

    total_bytes = sum(os.path.getsize(os.path.join(d, f)) for f in files)
    exts = Counter(os.path.splitext(f)[1] for f in os.listdir(d))
    print(f"  file count      : {len(files)}")
    print(f"  total size      : {human(total_bytes)}")
    print(f"  extensions      : {dict(exts)}")
    print(f"  first / last    : {files[0]} / {files[-1]}")

    idx = np.arange(len(files))
    if sample_stats is not None and sample_stats < len(files):
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(files), sample_stats, replace=False))

    shapes, dtypes = Counter(), Counter()
    mins, maxs, means, sqs, counts, frac_out, frac_neg = [], [], [], [], [], [], []
    rgb_equal = []

    for i in idx:
        a = np.load(os.path.join(d, files[i]))
        shapes[a.shape] += 1
        dtypes[str(a.dtype)] += 1
        if a.ndim == 3 and a.shape[-1] == 3:
            rgb_equal.append(bool(np.array_equal(a[..., 0], a[..., 1])
                                  and np.array_equal(a[..., 1], a[..., 2])))
        af = a.astype(np.float64)
        mins.append(af.min())
        maxs.append(af.max())
        means.append(af.mean())
        sqs.append((af ** 2).mean())
        counts.append(af.size)
        frac_out.append(float(((af < 0.0) | (af > 1.0)).mean()))
        frac_neg.append(float((af < 0.0).mean()))

    counts = np.asarray(counts, dtype=np.float64)
    w = counts / counts.sum()
    gmean = float(np.dot(w, means))
    gsq = float(np.dot(w, sqs))
    gstd = float(np.sqrt(max(gsq - gmean ** 2, 0.0)))

    print(f"  arrays inspected: {len(idx)}")
    print(f"  dtypes          : {dict(dtypes)}")
    print("  shape histogram :")
    for s, c in shapes.most_common():
        print(f"      {s}  x{c}")
    if rgb_equal:
        print(f"  3-channel arrays: R==G==B in {sum(rgb_equal)}/{len(rgb_equal)}")
    print(f"  min (global)    : {min(mins):.6f}")
    print(f"  max (global)    : {max(maxs):.6f}")
    print(f"  mean (pixelwise): {gmean:.6f}")
    print(f"  std  (pixelwise): {gstd:.6f}")
    print(f"  per-image min   : mean {np.mean(mins):.6f}  worst {min(mins):.6f}")
    print(f"  per-image max   : mean {np.mean(maxs):.6f}  worst {max(maxs):.6f}")
    print(f"  frac px outside [0,1] : {np.dot(w, frac_out):.6f}")
    print(f"  frac px < 0           : {np.dot(w, frac_neg):.6f}")
    print(f"  images with any px >1 : {sum(1 for m in maxs if m > 1.0)}/{len(idx)}")
    print(f"  images with any px <0 : {sum(1 for m in mins if m < 0.0)}/{len(idx)}")

    return {
        "name": name,
        "path": d,
        "count": len(files),
        "bytes": total_bytes,
        "inspected": int(len(idx)),
        "dtypes": {k: int(v) for k, v in dtypes.items()},
        "shapes": {str(k): int(v) for k, v in shapes.items()},
        "min": float(min(mins)),
        "max": float(max(maxs)),
        "mean": gmean,
        "std": gstd,
        "frac_outside_01": float(np.dot(w, frac_out)),
        "frac_negative": float(np.dot(w, frac_neg)),
        "filenames": files,
    }


def check_pairing(gt, lr):
    print(f"\n{BAR}\nPAIRING RULE  (train GT <-> train NoisyLR)\n{BAR}")
    sg, sl = set(gt["filenames"]), set(lr["filenames"])
    print(f"  GT files            : {len(sg)}")
    print(f"  NoisyLR files       : {len(sl)}")
    print(f"  exact-name matches  : {len(sg & sl)}")
    print(f"  GT without NoisyLR  : {sorted(sg - sl)[:10]}")
    print(f"  NoisyLR without GT  : {sorted(sl - sg)[:10]}")
    if sg == sl:
        print("  ==> RULE: identical filename in GT/ and NoisyLR/. No suffix, no offset.")

    print("\n  scale check (GT dim / LR dim), up to 200 random pairs:")
    common = sorted(sg & sl)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(common), min(200, len(common)), replace=False)
    ratios = Counter()
    for i in pick:
        f = common[i]
        g = np.load(os.path.join(gt["path"], f))
        n = np.load(os.path.join(lr["path"], f))
        ratios[(g.shape[0] / n.shape[0], g.shape[1] / n.shape[1], g.shape, n.shape)] += 1
    for k, c in ratios.most_common():
        print(f"      GT{k[2]} / LR{k[3]}  ->  ratio ({k[0]:.2f}, {k[1]:.2f})  x{c}")


def compare_dists(train_lr, test_lr):
    print(f"\n{BAR}\nTRAIN NoisyLR  vs  TEST NoisyLR  (distribution shift check)\n{BAR}")
    rows = [("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max"),
            ("frac_outside_01", "frac outside")]
    print(f"  {'stat':<16}{'train':>14}{'test':>14}{'delta':>14}")
    for key, label in rows:
        a, b = train_lr[key], test_lr[key]
        print(f"  {label:<16}{a:>14.6f}{b:>14.6f}{b - a:>14.6f}")
    print("\n  shape sets:")
    print(f"    train NoisyLR: {sorted(train_lr['shapes'])}")
    print(f"    test  NoisyLR: {sorted(test_lr['shapes'])}")


def find_docs(roots):
    print(f"\n{BAR}\nSHIPPED INSTRUCTIONS / README SEARCH\n{BAR}")
    pats = (".txt", ".md", ".pdf", ".json", ".csv", ".docx", ".xlsx", ".yaml", ".yml", ".rst")
    hits = []
    for r in roots:
        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = [d for d in dirnames if d != "__MACOSX"]
            for f in filenames:
                if f.lower().endswith(pats):
                    hits.append(os.path.join(dirpath, f))
    if not hits:
        print("  none found in either dataset folder.")
    for h in hits:
        print(f"  FOUND: {h}  ({human(os.path.getsize(h))})")
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", default=os.path.expanduser("~/Downloads/train/train"))
    p.add_argument("--test_root", default=os.path.expanduser("~/Downloads/Test_NoisyLR (1)"))
    p.add_argument("--sample", type=int, default=0, help="0 = inspect every file")
    p.add_argument("--json_out", default="")
    a = p.parse_args()

    sample = a.sample if a.sample > 0 else None
    print(f"numpy {np.__version__} | python {sys.version.split()[0]}")
    print(f"train_root = {a.train_root}")
    print(f"test_root  = {a.test_root}")

    gt = scan_dir("TRAIN / GT", os.path.join(a.train_root, "GT"), sample)
    lr = scan_dir("TRAIN / NoisyLR", os.path.join(a.train_root, "NoisyLR"), sample)
    te = scan_dir("TEST / NoisyLR (hidden, no GT)", os.path.join(a.test_root, "NoisyLR"), sample)

    if gt and lr:
        check_pairing(gt, lr)
    if lr and te:
        compare_dists(lr, te)
    find_docs([a.train_root, a.test_root])

    if te:
        print(f"\n{BAR}\nOUTPUT NAMING CONTRACT (derived from test filenames)\n{BAR}")
        print(f"  test inputs look like : {te['filenames'][:3]} ... {te['filenames'][-1]}")
        print(f"  container / dtype     : .npy / {list(te['dtypes'])}")
        print("  ==> write each restored image as <same filename>.npy, float32, clipped to [0,1]")

    if a.json_out:
        out = {k: v for k, v in [("train_gt", gt), ("train_lr", lr), ("test_lr", te)] if v}
        for v in out.values():
            v.pop("filenames", None)
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()

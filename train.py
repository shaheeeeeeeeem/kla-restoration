import argparse
import csv
import os
import time

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PairedRestorationDataset
from src.data.split import load_split, repo_path
from src.engine.common import EMA, forward_pair, lr_at
from src.losses.restoration_loss import RestorationLoss
from src.metrics.image_metrics import psnr, ssim
from src.models.nafnet_sr import build_model
from src.utils.misc import Logger, device_report, enable_tf32, git_hash, load_config, set_seed


def infinite(loader, ds):
    e = 0
    while True:
        ds.set_epoch(e)
        for b in loader:
            yield b
        e += 1


@torch.no_grad()
def validate(model, val_ds, cfg, device, limit=None):
    model.eval()
    ids = val_ds.ids if limit is None else val_ds.ids[:limit]
    ps, ss = [], []
    for i in range(len(ids)):
        s = val_ds[i]
        lr = s["lr"][None].to(device)
        gt = s["gt"][None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = forward_pair(model, lr, cfg)
        out = out.float().clamp(0, 1)
        ps.append(float(psnr(out, gt)))
        ss.append(float(ssim(out, gt)))
    model.train()
    return sum(ps) / len(ps), sum(ss) / len(ss)


def append_csv(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=repo_path("configs/final.yaml"))
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--overfit", type=int, default=0,
                   help="Sanity A: overfit N pairs, no val, no ckpt")
    p.add_argument("--resume", default="")
    p.add_argument("--tag", default="final")
    a = p.parse_args()

    cfg = load_config(a.config)
    if a.iters:
        cfg.train["iters"] = a.iters
    if a.lr:
        cfg.train["lr"] = a.lr
    if a.warmup is not None:
        cfg.train["warmup_iters"] = max(1, a.warmup)

    set_seed(cfg.seed)
    enable_tf32()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = Logger(repo_path(cfg.paths.log_file) if not a.overfit else None)

    log(f"device   : {device_report()}")
    log(f"git      : {git_hash()}   seed: {cfg.seed}   tag: {a.tag}")

    train_ids, val_ids = load_split(cfg.data.train_root)
    if a.overfit:
        train_ids = train_ids[: a.overfit]
        log(f"SANITY A : overfitting {len(train_ids)} pairs -> {train_ids}")

    train_ds = PairedRestorationDataset(cfg.data.train_root, train_ids, cfg.scale,
                                        cfg.data.lr_crop, train=True, seed=cfg.seed)
    val_ds = PairedRestorationDataset(cfg.data.train_root, val_ids, cfg.scale,
                                      train=False)

    workers = 0 if a.overfit else cfg.data.num_workers
    batch = min(cfg.train.batch_size, len(train_ids)) if a.overfit else cfg.train.batch_size
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                        num_workers=workers, pin_memory=cfg.data.pin_memory,
                        drop_last=not a.overfit,
                        persistent_workers=cfg.data.persistent_workers and workers > 0)

    model = build_model(cfg).to(device)
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(q.numel() for q in model.parameters())
    log(f"model    : {cfg.model.name}  {n_params / 1e6:.2f} M params  stride {model.stride}")

    crit = RestorationLoss(cfg.loss.charbonnier_weight, cfg.loss.charbonnier_eps,
                           cfg.loss.msssim_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            betas=tuple(cfg.train.betas),
                            weight_decay=cfg.train.weight_decay)
    ema = EMA(model, cfg.train.ema_decay) if not a.overfit else None

    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ema and "ema" in ck:
            ema.shadow.load_state_dict(ck["ema"])
        start = ck["step"]
        log(f"resumed  : {a.resume} at step {start}")

    ckpt_dir = repo_path(cfg.paths.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    total = cfg.train.iters
    it = infinite(loader, train_ds)
    best = -1.0
    t0 = time.time()
    log(f"training : {total} iters, batch {cfg.train.batch_size}, "
        f"crop {cfg.data.lr_crop}->{cfg.data.lr_crop * cfg.scale}, lr {cfg.train.lr}")

    for step in range(start, total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg.train.lr, cfg.train.warmup_iters, total, cfg.train.min_lr)

        b = next(it)
        lr_in = b["lr"].to(device, non_blocking=True)
        gt = b["gt"].to(device, non_blocking=True)
        if cfg.train.channels_last:
            lr_in = lr_in.to(memory_format=torch.channels_last)
            gt = gt.to(memory_format=torch.channels_last)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = forward_pair(model, lr_in, cfg)
        loss, parts = crit(out.float(), gt.float())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        if ema:
            ema.update(model)

        if (step + 1) % cfg.train.log_every == 0 or step == start:
            el = time.time() - t0
            ips = (step - start + 1) / max(el, 1e-9)
            eta = (total - step - 1) / max(ips, 1e-9)
            log(f"  step {step + 1:6d}/{total}  loss {parts['total']:.5f}  "
                f"charb {parts.get('charbonnier', 0):.5f}  "
                f"msssim {parts.get('msssim_term', 0):.5f}  "
                f"lr {opt.param_groups[0]['lr']:.2e}  "
                f"{ips:.2f} it/s  eta {eta / 60:.1f} min")

        if a.overfit:
            continue

        if (step + 1) % cfg.train.val_every == 0 or (step + 1) == total:
            vp, vs = validate(ema.shadow if ema else model, val_ds, cfg, device)
            log(f"  VAL step {step + 1}: PSNR {vp:.4f}  SSIM {vs:.4f}")
            append_csv(repo_path(cfg.paths.experiments_csv), {
                "tag": a.tag, "git": git_hash(), "seed": cfg.seed, "step": step + 1,
                "train_loss": round(parts["total"], 6),
                "val_psnr": round(vp, 4), "val_ssim": round(vs, 4),
                "params_M": round(n_params / 1e6, 3),
                "batch": cfg.train.batch_size, "crop": cfg.data.lr_crop,
                "lr": cfg.train.lr, "iters": total,
                "elapsed_min": round((time.time() - t0) / 60, 2),
            })
            if vp > best:
                best = vp
                torch.save({"model": model.state_dict(),
                            "ema": ema.shadow.state_dict() if ema else None,
                            "step": step + 1, "val_psnr": vp, "val_ssim": vs,
                            "cfg": dict(cfg), "git": git_hash(), "seed": cfg.seed},
                           os.path.join(ckpt_dir, "best.pt"))
                log(f"  saved best.pt (PSNR {vp:.4f})")

        if (step + 1) % cfg.train.ckpt_every == 0:
            torch.save({"model": model.state_dict(),
                        "ema": ema.shadow.state_dict() if ema else None,
                        "opt": opt.state_dict(), "step": step + 1,
                        "cfg": dict(cfg), "git": git_hash(), "seed": cfg.seed},
                       os.path.join(ckpt_dir, "latest.pt"))

    if a.overfit:
        vp, vs = validate(model, train_ds_eval(cfg, train_ids), cfg, device)
        log(f"SANITY A result: PSNR on the {len(train_ids)} overfit pairs = {vp:.3f} dB, SSIM {vs:.4f}")
    log(f"done in {(time.time() - t0) / 60:.1f} min")


def train_ds_eval(cfg, ids):
    return PairedRestorationDataset(cfg.data.train_root, ids, cfg.scale, train=False)


if __name__ == "__main__":
    main()

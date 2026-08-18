import os
import random
import subprocess

import numpy as np
import torch
import yaml


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class Config(dict):
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        if isinstance(v, dict) and not isinstance(v, Config):
            v = Config(v)
            self[k] = v
        return v

    def __setattr__(self, k, v):
        self[k] = v


def load_config(path):
    with open(path) as f:
        return Config(yaml.safe_load(f))


def git_hash():
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                    stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                        stderr=subprocess.DEVNULL).decode().strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def device_report():
    if not torch.cuda.is_available():
        return "cpu (no CUDA device visible)"
    p = torch.cuda.get_device_properties(0)
    return (f"{torch.cuda.get_device_name(0)} | {p.total_memory / 1e9:.2f} GB | "
            f"sm_{p.major}{p.minor} | torch {torch.__version__} | cuda {torch.version.cuda}")


class Logger:
    def __init__(self, path=None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def __call__(self, msg):
        print(msg, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(msg + "\n")

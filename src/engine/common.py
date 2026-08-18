import copy
import math

import torch
import torch.nn.functional as F


def normalize(x, mean, std):
    return (x - mean) / std


def bicubic_up(x, scale):
    return F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)


def forward_pair(model, lr_raw, cfg):
    x = normalize(lr_raw, cfg.norm.mean, cfg.norm.std)
    base = bicubic_up(lr_raw, cfg.scale)
    return model(x, base=base)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                s.copy_(m)


def lr_at(step, base_lr, warmup, total, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))


def forward_self_ensemble(model, lr_raw, cfg):
    """x8 geometric self-ensemble: average the model over the 8 flip/rotation
    symmetries, mapping each result back before averaging. Optional and OFF by
    default -- see inference.py --self_ensemble."""
    outs = []
    for k in range(4):
        for flip in (False, True):
            xt = torch.rot90(lr_raw, k, dims=(-2, -1))
            if flip:
                xt = torch.flip(xt, dims=(-1,))
            y = forward_pair(model, xt.contiguous(), cfg).float()
            if flip:
                y = torch.flip(y, dims=(-1,))
            outs.append(torch.rot90(y, -k, dims=(-2, -1)))
    return torch.stack(outs, 0).mean(0)

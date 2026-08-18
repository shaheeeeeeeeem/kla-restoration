import torch
import torch.nn as nn
import torch.nn.functional as F

MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def _gauss_win(size, sigma, device, dtype, channels):
    c = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None]
    return w.expand(channels, 1, size, size)


def _ssim_maps(x, y, win, c1, c2):
    ch = x.shape[1]
    mu1 = F.conv2d(x, win, groups=ch)
    mu2 = F.conv2d(y, win, groups=ch)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(x * x, win, groups=ch) - mu1s
    s2 = F.conv2d(y * y, win, groups=ch) - mu2s
    s12 = F.conv2d(x * y, win, groups=ch) - mu12
    cs = (2 * s12 + c2) / (s1 + s2 + c2)
    lum = (2 * mu12 + c1) / (mu1s + mu2s + c1)
    return lum, cs


def ms_ssim(x, y, data_range=1.0, size=11, sigma=1.5, k1=0.01, k2=0.03,
            weights=MS_SSIM_WEIGHTS):
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    win = _gauss_win(size, sigma, x.device, x.dtype, x.shape[1])

    smallest = min(x.shape[-2:])
    levels = min(len(weights), max(1, int(torch.log2(torch.tensor(smallest / (size - 1.0))).item()) + 1))
    w = torch.tensor(weights[:levels], device=x.device, dtype=x.dtype)
    w = w / w.sum()

    vals = []
    for i in range(levels):
        lum, cs = _ssim_maps(x, y, win, c1, c2)
        if i < levels - 1:
            vals.append(cs.flatten(1).mean(1).clamp_min(1e-6))
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        else:
            vals.append((lum * cs).flatten(1).mean(1).clamp_min(1e-6))

    out = torch.stack(vals, dim=0) ** w[:, None]
    return out.prod(dim=0).mean()


class RestorationLoss(nn.Module):
    def __init__(self, charbonnier_weight=1.0, charbonnier_eps=1e-3, msssim_weight=0.15):
        super().__init__()
        self.cw = charbonnier_weight
        self.eps = charbonnier_eps
        self.mw = msssim_weight

    def forward(self, pred, target):
        parts = {}
        loss = 0.0
        if self.cw:
            c = charbonnier(pred, target, self.eps)
            parts["charbonnier"] = float(c.detach())
            loss = loss + self.cw * c
        if self.mw:
            m = 1.0 - ms_ssim(pred.clamp(0, 1).float(), target.float())
            parts["msssim_term"] = float(m.detach())
            loss = loss + self.mw * m
        parts["total"] = float(loss.detach())
        return loss, parts

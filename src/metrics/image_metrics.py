import torch
import torch.nn.functional as F


def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)
    return (10.0 * torch.log10(data_range ** 2 / mse.clamp_min(1e-12))).mean()


def _gaussian_window(size, sigma, device, dtype):
    c = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(pred, target, data_range=1.0, size=11, sigma=1.5, k1=0.01, k2=0.03):
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    w = _gaussian_window(size, sigma, pred.device, pred.dtype)
    c = pred.shape[1]
    w = w.expand(c, 1, size, size)

    mu1 = F.conv2d(pred, w, groups=c)
    mu2 = F.conv2d(target, w, groups=c)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    s1 = F.conv2d(pred * pred, w, groups=c) - mu1s
    s2 = F.conv2d(target * target, w, groups=c) - mu2s
    s12 = F.conv2d(pred * target, w, groups=c) - mu12

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))
    return m.flatten(1).mean(1).mean()


class LPIPSMetric:
    def __init__(self, net="alex", device="cpu"):
        import lpips
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def __call__(self, pred, target):
        p = pred.to(self.device)
        t = target.to(self.device)
        if p.shape[1] == 1:
            p = p.repeat(1, 3, 1, 1)
            t = t.repeat(1, 3, 1, 1)
        return self.model(p * 2 - 1, t * 2 - 1).flatten().mean()

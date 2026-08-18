import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)

        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.conv1(self.norm1(x))
        y = self.sg(self.conv2(y))
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.sg(self.conv4(self.norm2(x)))
        y = self.conv5(y)
        return x + y * self.gamma


class NAFNetSR(nn.Module):
    def __init__(self, in_ch=1, width=48, enc_blocks=(2, 2, 4), middle_blocks=6,
                 dec_blocks=(2, 2, 2), scale=2, global_residual=True):
        super().__init__()
        if len(enc_blocks) != len(dec_blocks):
            raise ValueError("enc_blocks and dec_blocks must have the same length")
        self.scale = scale
        self.global_residual = global_residual
        self.stride = 2 ** len(enc_blocks)

        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, stride=2))
            c *= 2

        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))

        self.tail = nn.Sequential(
            nn.Conv2d(c, in_ch * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )
        nn.init.zeros_(self.tail[0].weight)
        nn.init.zeros_(self.tail[0].bias)

    def _pad(self, x):
        h, w = x.shape[-2:]
        s = self.stride
        ph, pw = (s - h % s) % s, (s - w % s) % s
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def forward(self, x, base=None):
        x_in, h, w = self._pad(x)

        y = self.intro(x_in)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            y = enc(y)
            skips.append(y)
            y = down(y)

        y = self.middle(y)

        for up, dec, skip in zip(self.ups, self.decoders, skips[::-1]):
            y = up(y)
            y = dec(y + skip)

        y = self.tail(y)
        y = y[..., : h * self.scale, : w * self.scale]

        if self.global_residual:
            if base is None:
                base = F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                                     align_corners=False)
            y = y + base
        return y


def build_model(cfg):
    m = cfg.model
    return NAFNetSR(
        in_ch=1,
        width=m.width,
        enc_blocks=tuple(m.enc_blocks),
        middle_blocks=m.middle_blocks,
        dec_blocks=tuple(m.dec_blocks),
        scale=cfg.scale,
        global_residual=m.global_residual,
    )

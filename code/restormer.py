"""
restormer.py - compact Restormer (Zamir et al., CVPR'22): transformer restoration family,
genuinely different from NAFNet (conv family). Used to show F1 (+3dB decoupled-under-prepares)
holds ACROSS architecture families, not just NAFNet capacity. Pure PyTorch, no custom CUDA.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c)); self.b = nn.Parameter(torch.zeros(c)); self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True); s = (x - u).pow(2).mean(1, keepdim=True)
        return (x - u) / torch.sqrt(s + self.eps) * self.w[None, :, None, None] + self.b[None, :, None, None]


class MDTA(nn.Module):
    """Multi-Dconv head transposed attention (channel attention)."""
    def __init__(self, c, heads):
        super().__init__()
        self.heads = heads
        self.temp = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        nh = self.heads; cph = c // nh
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, nh, cph, h * w); k = k.reshape(b, nh, cph, h * w); v = v.reshape(b, nh, cph, h * w)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temp).softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv feed-forward network."""
    def __init__(self, c, exp=2.66):
        super().__init__()
        hc = int(c * exp)
        self.project_in = nn.Conv2d(c, hc * 2, 1, bias=False)
        self.dw = nn.Conv2d(hc * 2, hc * 2, 3, padding=1, groups=hc * 2, bias=False)
        self.project_out = nn.Conv2d(hc, c, 1, bias=False)

    def forward(self, x):
        x1, x2 = self.dw(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TBlock(nn.Module):
    def __init__(self, c, heads):
        super().__init__()
        self.n1 = LayerNorm2d(c); self.attn = MDTA(c, heads)
        self.n2 = LayerNorm2d(c); self.ffn = GDFN(c)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Restormer(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=32, blocks=(2, 3, 3), heads=(1, 2, 4)):
        super().__init__()
        self.out_ch = out_ch
        self.intro = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.encs = nn.ModuleList(); self.downs = nn.ModuleList()
        c = dim
        for n, hd in zip(blocks, heads):
            self.encs.append(nn.Sequential(*[TBlock(c, hd) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, 2)); c *= 2
        self.mid = nn.Sequential(*[TBlock(c, heads[-1] * 2) for _ in range(blocks[-1])])
        self.ups = nn.ModuleList(); self.decs = nn.ModuleList()
        for n, hd in zip(reversed(blocks), reversed(heads)):
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1), nn.PixelShuffle(2))); c //= 2
            self.decs.append(nn.Sequential(*[TBlock(c, hd) for _ in range(n)]))
        self.out = nn.Conv2d(dim, out_ch, 3, padding=1)

    def forward(self, x):
        inp = x[:, :self.out_ch]
        x = self.intro(x); skips = []
        for enc, down in zip(self.encs, self.downs):
            x = enc(x); skips.append(x); x = down(x)
        x = self.mid(x)
        for i, (up, dec) in enumerate(zip(self.ups, self.decs)):
            x = up(x); x = x + skips[-(i + 1)]; x = dec(x)
        return (self.out(x) + inp).clamp(0, 1), None

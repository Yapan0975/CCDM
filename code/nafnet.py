"""
nafnet.py - compact NAFNet (Chu et al., ECCV'22) + a coupling-aware variant.

NAFNet      : generic restoration backbone (coupling-agnostic) -> isolates the DATA thesis.
NAFNet(CA)  : adds (i) shared-latent-field heads predicting {L, t, sigma} with auxiliary
              supervision and (ii) FiLM modulation of decoder features by the predicted
              fields -> isolates the METHOD thesis (coupling-aware beats agnostic at equal
              data). Pure PyTorch, no custom CUDA.
"""
import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c)); self.b = nn.Parameter(torch.zeros(c)); self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True); s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.w[None, :, None, None] + self.b[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw=2, ffn=2):
        super().__init__()
        d = c * dw
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, d, 1)
        self.conv2 = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d // 2, d // 2, 1))
        self.conv3 = nn.Conv2d(d // 2, c, 1)
        self.norm2 = LayerNorm2d(c)
        f = c * ffn
        self.conv4 = nn.Conv2d(c, f, 1); self.sg2 = SimpleGate(); self.conv5 = nn.Conv2d(f // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1)); self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm1(x); y = self.conv1(y); y = self.conv2(y); y = self.sg(y); y = y * self.sca(y); y = self.conv3(y)
        x = x + y * self.beta
        y = self.norm2(x); y = self.conv4(y); y = self.sg2(y); y = self.conv5(y)
        return x + y * self.gamma


class FiLM(nn.Module):
    """Per-pixel feature modulation from the predicted latent fields."""
    def __init__(self, field_ch, feat_ch):
        super().__init__()
        self.gen = nn.Sequential(nn.Conv2d(field_ch, feat_ch, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(feat_ch, feat_ch * 2, 1))

    def forward(self, feat, fields):
        f = torch.nn.functional.interpolate(fields, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        g, b = self.gen(f).chunk(2, dim=1)
        return feat * (1 + g) + b


class NAFNet(nn.Module):
    def __init__(self, img_ch=3, width=32, enc=(1, 1, 2), mid=2, dec=(1, 1, 1),
                 coupling_aware=False, in_ch=None, out_ch=3):
        super().__init__()
        self.coupling_aware = coupling_aware
        self.out_ch = out_ch
        self.intro = nn.Conv2d(in_ch if in_ch else img_ch, width, 3, padding=1)
        self.encoders = nn.ModuleList(); self.downs = nn.ModuleList()
        c = width
        for n in enc:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, 2)); c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        if coupling_aware:                                   # shared-latent-field head at bottleneck
            self.field_head = nn.Sequential(NAFBlock(c), nn.Conv2d(c, 3, 1))
            self.films = nn.ModuleList()
        self.ups = nn.ModuleList(); self.decoders = nn.ModuleList()
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            if coupling_aware:
                self.films.append(FiLM(3, c))
        self.out = nn.Conv2d(width, out_ch, 3, padding=1)

    def forward(self, x):
        inp = x[:, :self.out_ch]                              # residual only on the RGB channels
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x); skips.append(x); x = down(x)
        x = self.middle(x)
        fields = None
        if self.coupling_aware:
            fields = torch.sigmoid(self.field_head(x))       # {L,t,sigma} in (0,1), bottleneck res
        for i, (up, dec) in enumerate(zip(self.ups, self.decoders)):
            x = up(x); x = x + skips[-(i + 1)]
            if self.coupling_aware:
                x = self.films[i](x, fields)
            x = dec(x)
        out = (self.out(x) + inp).clamp(0, 1)
        if self.coupling_aware:
            fields = torch.nn.functional.interpolate(fields, size=inp.shape[-2:],
                                                     mode='bilinear', align_corners=False)
        return out, fields


class CoupleNet(nn.Module):
    """CoupleNet v2 - coupling-aware restoration.
    (i) physics-prior input channels: input illumination (LIME-style) + dark channel (haze
        prior) appended so the net gets the shared coupling carriers explicitly.
    (ii) latent-field head {L,t,sigma} (supervised) + FiLM cross-modulation (in NAFNet core).
    (iii) redegradation head: re-degrade the restored image conditioned on predicted fields,
         enforce consistency with the input (physics-in-the-loop).
    """
    def __init__(self, width=32, enc=(1, 1, 2), mid=2, dec=(1, 1, 1)):
        super().__init__()
        self.core = NAFNet(width=width, enc=enc, mid=mid, dec=dec, coupling_aware=True,
                           in_ch=5, out_ch=3)
        self.redeg = nn.Sequential(nn.Conv2d(6, width, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(width, 3, 3, padding=1))

    @staticmethod
    def hints(x):
        maxc = x.max(1, keepdim=True).values
        illum = torch.nn.functional.avg_pool2d(maxc, 15, 1, 7)
        minc = x.min(1, keepdim=True).values
        dark = -torch.nn.functional.max_pool2d(-minc, 15, 1, 7)
        return torch.cat([illum, dark], 1)

    def forward(self, x):
        h = self.hints(x)
        out, fields = self.core(torch.cat([x, h], 1))
        redeg = torch.sigmoid(self.redeg(torch.cat([out, fields], 1)))   # reconstructed input
        return out, fields, redeg

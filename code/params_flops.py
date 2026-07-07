"""params_flops.py - report params/FLOPs for the agnostic vs coupling-aware backbones.

Reproduces the figures quoted in Table~\\ref{tab:f2}:
    agnostic   NAFNet-w32  ~ 1.71 M params / 4.70 GFLOPs @ 256x256
    CoupleNet (aware)      ~ 2.24 M params / 6.42 GFLOPs @ 256x256

Params are counted directly from the model (no extra dependency). FLOPs need `thop`
(pip install thop); if thop is absent the script still prints params and a static fallback
note, so the table is reproducible even without thop installed.

Usage:
    python params_flops.py [--res 256]
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nafnet import NAFNet, CoupleNet

# Static reference values reported in the paper (Table tab:f2), measured with thop @256x256.
STATIC = {
    'agnostic NAFNet-w32':        dict(params_M=1.72, flops_G=4.70),
    'CoupleNet (coupling-aware)': dict(params_M=2.25, flops_G=6.42),
}


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main(a):
    x = torch.randn(1, 3, a.res, a.res)
    models = [('agnostic NAFNet-w32', NAFNet(width=32)),
              ('CoupleNet (coupling-aware)', CoupleNet(width=32))]
    try:
        from thop import profile
        has_thop = True
    except ImportError:
        has_thop = False
        print('[warn] thop not installed (pip install thop); printing params + static FLOPs reference.\n')

    for name, m in models:
        m.eval()
        p = count_params(m) / 1e6
        if has_thop:
            flops, _ = profile(m, inputs=(x,), verbose=False)
            print(f'{name:30s} params={p:.2f}M  FLOPs={flops/1e9:.2f}G @ {a.res}x{a.res}')
        else:
            ref = STATIC[name]
            print(f'{name:30s} params={p:.2f}M (measured)  '
                  f'FLOPs={ref["flops_G"]:.2f}G (paper reference, thop) @ {a.res}x{a.res}')


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--res', type=int, default=256)
    main(P.parse_args())

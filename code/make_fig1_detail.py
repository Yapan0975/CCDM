"""
make_fig1_detail.py - Figure 1 upgrade requested by Reviewer #2 (Q5): the coupled/decoupled
difference is hard to see without the amplified difference map, so this version adds
(i) local zoom-ins on a bright/dark boundary and (ii) the latent fields that drive the
cross-terms (illumination L, the two noise sigma maps, and the rain modulation).

Row 1: clean | decoupled | coupled | |coupled - decoupled| x 4
Row 2: illumination L | sigma (decoupled, flat) | sigma (coupled, signal-dependent) | rain gain
Row 3: zoom of the boxed region: clean | decoupled | coupled | difference

--pgnoise renders the noise with the Poisson-Gaussian model stated in the paper (variance a*x+b),
under which the coupled sigma map rises with the local signal; use it for every paper figure.
Run on the server (needs clean_test100 / depth / rain masks), CPU only:
  python3 make_fig1_detail.py --pgnoise --idx 4 --out fig1_detail.pdf
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ccdm
from dataset_ccd import COMBOS


def bgr2rgb(x):
    return np.clip(x[:, :, ::-1], 0, 1)


def main(a):
    params = {'noise_model': 'poisson'} if a.pgnoise else None
    cleans = sorted(glob.glob(os.path.join(a.clean_test, '*.png')))
    rains = sorted(glob.glob(os.path.join(a.rain, '*')))
    snows = sorted(glob.glob(os.path.join(a.snow, '*')))
    cf = cleans[a.idx]
    J = cv2.imread(cf).astype(np.float32) / 255.0
    H, W = J.shape[:2]; H -= H % 8; W -= W % 8; J = J[:H, :W]
    d = np.load(os.path.join(a.depth_test, os.path.basename(cf).replace('.png', '.npy')))[:H, :W]
    rm = cv2.imread(rains[a.idx % len(rains)]).astype(np.float32) / 255.0
    sm = cv2.imread(snows[a.idx % len(snows)]).astype(np.float32) / 255.0
    types = COMBOS[a.combo]

    out = {}
    for mode in ('decoupled', 'coupled'):
        rng = np.random.default_rng(10000 + a.idx)
        lq, gt = ccdm.degrade(J, d, rng, mode=mode, types=types,
                              rain_mask=rm, snow_mask=sm, params=params)
        out[mode] = (np.clip(lq, 0, 1), gt)
    I_dec, gt_dec = out['decoupled']
    I_cpl, gt_cpl = out['coupled']
    diff = np.clip(np.abs(I_cpl - I_dec) * 4.0, 0, 1)
    L = gt_cpl['L']
    rain_gain = L / (L.mean() + 1e-9)          # relative rain radiance under coupling

    # Zoom window: the mechanism is only visible across an illumination boundary, so pick the
    # patch that maximises (illumination spread) x (coupled-vs-decoupled discrepancy) -- a patch
    # containing both a bright and a dark region, where coupled noise/rain must differ most.
    k = max(H, W) // 5
    best, bxy = -1.0, (0, 0)
    step = max(k // 5, 6)
    dmap = np.abs(I_cpl - I_dec).mean(2)
    for yy in range(0, H - k, step):
        for xx in range(0, W - k, step):
            lp = L[yy:yy + k, xx:xx + k]
            v = float(lp.std() * dmap[yy:yy + k, xx:xx + k].mean())
            if v > best:
                best, bxy = v, (yy, xx)
    zy, zx = bxy

    def crop(img):
        return img[zy:zy + k, zx:zx + k]

    fig, axes = plt.subplots(3, 4, figsize=(7.4, 5.5))
    panels = [
        (bgr2rgb(J), 'clean $J$', None),
        (bgr2rgb(I_dec), 'decoupled ($\\lambda = 0$)', None),
        (bgr2rgb(I_cpl), 'coupled ($\\lambda = 1$)', None),
        (diff.mean(2), '$|$coupled $-$ decoupled$| \\times 4$', 'inferno'),
        (L, 'illumination $L$ (LIME)', 'viridis'),
        (gt_dec['sigma'], 'noise $\\sigma$, decoupled (flat)', 'magma'),
        (gt_cpl['sigma'], 'noise $\\sigma$, coupled (signal-dep.)', 'magma'),
        (rain_gain, 'rain gain under coupling ($L / \\bar{L}$)', 'cividis'),
        (bgr2rgb(crop(J)), 'zoom: clean', None),
        (bgr2rgb(crop(I_dec)), 'zoom: decoupled', None),
        (bgr2rgb(crop(I_cpl)), 'zoom: coupled', None),
        (crop(diff).mean(2), 'zoom: difference', 'inferno'),
    ]
    vlim = {}
    for i in (5, 6):
        vlim[i] = (min(gt_dec['sigma'].min(), gt_cpl['sigma'].min()),
                   max(gt_dec['sigma'].max(), gt_cpl['sigma'].max()))
    for i, (img, title, cmap) in enumerate(panels):
        ax = axes[i // 4, i % 4]
        kw = {}
        if i in vlim:
            kw = dict(vmin=vlim[i][0], vmax=vlim[i][1])
        ax.imshow(img, cmap=cmap, **kw)
        ax.set_title(title, fontsize=7.2, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color('#999')

    # the two sigma panels share a colour scale, so annotate what is actually in them:
    # matched mean, zero spatial variation on the left, signal-following structure on the right
    axes[1, 1].text(0.5, 0.06, 'mean %.3f,  std %.5f' % (gt_dec['sigma'].mean(), gt_dec['sigma'].std()),
                    transform=axes[1, 1].transAxes, ha='center', fontsize=6.4, color='white')
    axes[1, 2].text(0.5, 0.06, 'mean %.3f,  std %.3f' % (gt_cpl['sigma'].mean(), gt_cpl['sigma'].std()),
                    transform=axes[1, 2].transAxes, ha='center', fontsize=6.4, color='white')
    for i in (0, 1, 2, 3):
        axes[0, i].add_patch(Rectangle((zx, zy), k, k, fill=False, ec='#ffdd00', lw=1.1))
    for r, lab in [(0, 'rendered'), (1, 'latent fields'), (2, 'zoom of boxed region')]:
        axes[r, 0].set_ylabel(lab, fontsize=7.4, labelpad=3)
    fig.tight_layout(pad=0.45)
    fig.savefig(a.out, bbox_inches='tight', dpi=300)
    sig = gt_cpl['sigma'].ravel(); lum = cv2.blur(I_cpl.mean(2), (15, 15)).ravel()
    corr = float(np.corrcoef(sig, lum)[0, 1]) if sig.std() > 0 else float('nan')
    print('saved ' + a.out)
    print(f'  noise model: {"Poisson-Gaussian" if a.pgnoise else "legacy sigma0/L"}')
    print(f'  scene={os.path.basename(cf)} combo={a.combo} zoom=({zx},{zy},{k})')
    print(f'  sigma dec: mean={gt_dec["sigma"].mean():.4f} std={gt_dec["sigma"].std():.5f}')
    print(f'  sigma cpl: mean={gt_cpl["sigma"].mean():.4f} std={gt_cpl["sigma"].std():.5f}  '
          f'corr(sigma_cpl, local intensity)={corr:+.2f}')
    print(f'  mean |cpl-dec| = {np.abs(I_cpl - I_dec).mean():.5f}')


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--idx', type=int, default=4)
    P.add_argument('--combo', default='low_haze_rain')
    P.add_argument('--pgnoise', action='store_true')
    P.add_argument('--clean_test', default='clean_test100')
    P.add_argument('--depth_test', default='depth')
    P.add_argument('--rain', default='./OneRestore/syn_data/data/rain_mask')
    P.add_argument('--snow', default='./OneRestore/syn_data/data/snow_mask')
    P.add_argument('--out', default='fig1_detail.pdf')
    main(P.parse_args())

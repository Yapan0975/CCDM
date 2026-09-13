"""
make_qualitative_fig.py - qualitative restoration comparison requested by Reviewer #3 (Q3),
and the visual counterpart of the 2x2 protocol: the SAME scene is rendered in both coupling
structures and restored by both specialists plus the mixed model, so the reader can see the
specialisation that Table 2 reports numerically.

Layout (per scene, two blocks):
  block 1 - coupled-degraded input  : input | decoupled-trained | coupled-trained | mixed | GT
  block 2 - decoupled-degraded input: input | decoupled-trained | coupled-trained | mixed | GT
Per-panel PSNR is printed in the corner, so the cross-structure collapse is visible and
quantified in the same place.

--pgnoise renders the inputs with the Poisson-Gaussian noise model the paper's checkpoints were
trained with (defaults: the seed-0 lambda=0 / lambda=1 / mixed NAFNet-w32 models LAM_s0_*).
Run on the server, e.g. on GPU 1:
  CUDA_VISIBLE_DEVICES=1 python3 make_qualitative_fig.py --pgnoise --idx 4 --out fig_qualitative.pdf
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
import ccdm
from dataset_ccd import COMBOS
from nafnet import NAFNet

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(tag, width):
    m = NAFNet(width=width)
    m.load_state_dict(torch.load(tag + '.pth', map_location='cpu'))
    return m.to(DEV).eval()


def restore(m, lq):
    t = torch.from_numpy(lq.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
    with torch.no_grad():
        o = m(t)[0]
    return o[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)


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

    models = {'decoupled-trained': load_model(a.dec, a.width),
              'coupled-trained': load_model(a.cpl, a.width),
              'mixed-trained': load_model(a.mix, a.width)}

    fig, axes = plt.subplots(2, 5, figsize=(9.2, 4.0))
    for r, test_mode in enumerate(['coupled', 'decoupled']):
        rng = np.random.default_rng(10000 + a.idx)
        lq, _ = ccdm.degrade(J, d, rng, mode=test_mode, types=types,
                             rain_mask=rm, snow_mask=sm, params=params)
        lq = np.clip(lq, 0, 1)
        panels = [(bgr2rgb(lq), '%s-degraded input' % test_mode, psnr_fn(J, lq, data_range=1.0))]
        for name, m in models.items():
            o = restore(m, lq)
            panels.append((bgr2rgb(o), name, psnr_fn(J, o, data_range=1.0)))
        panels.append((bgr2rgb(J), 'ground truth', None))
        print(f'  test={test_mode:9s} ' + '  '.join(f'{t}: {p:.2f}' for _, t, p in panels if p is not None))
        for c, (img, title, ps) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=7.6, pad=4)
            if ps is not None:
                ax.text(0.03, 0.04, '%.2f dB' % ps, transform=ax.transAxes, fontsize=7.2,
                        color='white', va='bottom',
                        bbox=dict(fc='black', ec='none', alpha=0.55, pad=1.6))
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_color('#888')
        axes[r, 0].set_ylabel('test: %s\n($\\lambda = %d$)' % (test_mode, 1 if test_mode == 'coupled' else 0),
                              fontsize=7.8)
        # mark the in-domain specialist for this test structure
        good = 2 if test_mode == 'coupled' else 1
        for sp in axes[r, good].spines.values():
            sp.set_linewidth(1.6); sp.set_color('#111')
    fig.suptitle('Same scene, both coupling structures (bold frame = in-domain specialist)',
                 fontsize=8.4, y=0.99)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.965))
    fig.savefig(a.out, bbox_inches='tight', dpi=300)
    print('saved ' + a.out + '  scene=' + os.path.basename(cf) +
          '  noise=' + ('Poisson-Gaussian' if a.pgnoise else 'legacy sigma0/L'))


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--idx', type=int, default=4)
    P.add_argument('--combo', default='low_haze_rain')
    P.add_argument('--pgnoise', action='store_true')
    P.add_argument('--dec', default='LAM_s0_l000')
    P.add_argument('--cpl', default='LAM_s0_l100')
    P.add_argument('--mix', default='LAM_s0_mix')
    P.add_argument('--width', type=int, default=32)
    P.add_argument('--clean_test', default='clean_test100')
    P.add_argument('--depth_test', default='depth')
    P.add_argument('--rain', default='./OneRestore/syn_data/data/rain_mask')
    P.add_argument('--snow', default='./OneRestore/syn_data/data/snow_mask')
    P.add_argument('--out', default='fig_qualitative.pdf')
    main(P.parse_args())

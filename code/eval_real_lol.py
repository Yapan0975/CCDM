"""
eval_real_lol.py - real-data transfer test (F1 on REAL low-light, with ground truth).

Runs decoupled-trained and coupled-trained restorers on real low-light images and scores vs
the real normal-light reference. Reports per-image PSNR/SSIM, the paired coupled-minus-decoupled
gap, a paired t-test p-value and a sign-test win rate. If coupled-train > decoupled-train on
REAL data with significance, the F1 finding transfers beyond synthetic.
Works for LOL (low/high) and LOLv2-real (Input/GT) via --low_dir/--high_dir.
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from scipy import stats
from nafnet import NAFNet

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def load(ckpt, width):
    m = NAFNet(width=width); m.load_state_dict(torch.load(ckpt, map_location=DEV)); m.eval()
    return m.to(DEV)

def restore(m, x):
    t = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
    with torch.no_grad():
        return m(t)[0][0].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)

def main(a):
    models = {'decoupled': load(a.dec, a.width), 'coupled': load(a.cpl, a.width)}
    lows = sorted(glob.glob(os.path.join(a.root, a.low_dir, '*')))
    pi = {'decoupled': {'psnr': [], 'ssim': []}, 'coupled': {'psnr': [], 'ssim': []}, 'input': {'psnr': [], 'ssim': []}}
    for lf in lows:
        gt = cv2.imread(os.path.join(a.root, a.high_dir, os.path.basename(lf))).astype(np.float32) / 255.0
        lo = cv2.imread(lf).astype(np.float32) / 255.0
        H, W = gt.shape[:2]; H -= H % 8; W -= W % 8; gt = gt[:H, :W]; lo = lo[:H, :W]
        pi['input']['psnr'].append(psnr_fn(gt, lo, data_range=1.0))
        pi['input']['ssim'].append(ssim_fn(gt, lo, data_range=1.0, channel_axis=2))
        for k, m in models.items():
            o = restore(m, lo)
            pi[k]['psnr'].append(psnr_fn(gt, o, data_range=1.0))
            pi[k]['ssim'].append(ssim_fn(gt, o, data_range=1.0, channel_axis=2))
    n = len(lows)
    dp, cp = np.array(pi['decoupled']['psnr']), np.array(pi['coupled']['psnr'])
    ds, cs = np.array(pi['decoupled']['ssim']), np.array(pi['coupled']['ssim'])
    dpsnr = cp - dp; dssim = cs - ds
    t_p, p_p = stats.ttest_rel(cp, dp)          # paired t-test on PSNR
    t_s, p_s = stats.ttest_rel(cs, ds)          # paired t-test on SSIM
    out = {
        'n': n,
        'input':     {'PSNR': round(float(np.mean(pi['input']['psnr'])), 3), 'SSIM': round(float(np.mean(pi['input']['ssim'])), 4)},
        'decoupled': {'PSNR': round(float(dp.mean()), 3), 'SSIM': round(float(ds.mean()), 4)},
        'coupled':   {'PSNR': round(float(cp.mean()), 3), 'SSIM': round(float(cs.mean()), 4)},
        'gap_PSNR_mean': round(float(dpsnr.mean()), 3), 'gap_PSNR_p': round(float(p_p), 4),
        'gap_SSIM_mean': round(float(dssim.mean()), 4), 'gap_SSIM_p': round(float(p_s), 4),
        'win_rate_PSNR': round(float((dpsnr > 0).mean()), 3),
        'win_rate_SSIM': round(float((dssim > 0).mean()), 3),
    }
    print(json.dumps(out, indent=2)); json.dump(out, open(a.out, 'w'), indent=2); print('saved', a.out)

if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--root', default='lol_test')
    P.add_argument('--low_dir', default='low'); P.add_argument('--high_dir', default='high')
    P.add_argument('--dec', default='LL_dec.pth'); P.add_argument('--cpl', default='LL_cpl.pth')
    P.add_argument('--width', type=int, default=32)
    P.add_argument('--out', default='real_lolv2.json')
    main(P.parse_args())

"""
eval_real_lol_per.py - per-image real low-light PSNR/SSIM for a LIST of checkpoints
(scene-level bootstrap support; eval-only).

Same protocol as eval_real_lol.py (crop to /8, PSNR/SSIM vs the real reference), but
evaluates every checkpoint in --tags on every image and dumps the per-image arrays,
plus the unrestored input row. Images are looped once; all models share each load.

Run (server, composite-restore dir), Table-7 variance-matched track:
  CUDA_VISIBLE_DEVICES=0 python3 -u eval_real_lol_per.py \
      --tags LLv_dec_s0,LLv_cpl_s0,LLv_mix_s0,...  \
      --root lolv2_test --low_dir Input --high_dir GT --out per_image_LLv.json
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from nafnet import NAFNet

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load(ckpt, width):
    m = NAFNet(width=width)
    m.load_state_dict(torch.load(ckpt, map_location='cpu'))
    m.eval()
    return m.to(DEV)


def main(a):
    tags = [t for t in a.tags.split(',') if t]
    models = {t: load(t + '.pth', a.width) for t in tags}
    print(f'[eval_real_lol_per] {len(models)} models loaded', flush=True)
    lows = sorted(glob.glob(os.path.join(a.root, a.low_dir, '*')))
    out = {t: {'psnr': [], 'ssim': []} for t in tags}
    out['input'] = {'psnr': [], 'ssim': []}
    names = []
    for i, lf in enumerate(lows):
        names.append(os.path.basename(lf))
        gt = cv2.imread(os.path.join(a.root, a.high_dir, os.path.basename(lf))).astype(np.float32) / 255.0
        lo = cv2.imread(lf).astype(np.float32) / 255.0
        H, W = gt.shape[:2]; H -= H % 8; W -= W % 8
        gt = gt[:H, :W]; lo = lo[:H, :W]
        out['input']['psnr'].append(round(float(psnr_fn(gt, lo, data_range=1.0)), 4))
        out['input']['ssim'].append(round(float(ssim_fn(gt, lo, data_range=1.0, channel_axis=2)), 4))
        t_in = torch.from_numpy(lo.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
        for tag, m in models.items():
            with torch.no_grad():
                o = m(t_in)[0][0].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
            out[tag]['psnr'].append(round(float(psnr_fn(gt, o, data_range=1.0)), 4))
            out[tag]['ssim'].append(round(float(ssim_fn(gt, o, data_range=1.0, channel_axis=2)), 4))
        if (i + 1) % 20 == 0:
            print(f'[{i + 1}/{len(lows)}] images done', flush=True)
    json.dump(dict(images=names, results=out), open(a.out, 'w'))
    print('saved', a.out, flush=True)
    for tag in tags + ['input']:
        print(f"[check] {tag}: PSNR {np.mean(out[tag]['psnr']):.3f} "
              f"SSIM {np.mean(out[tag]['ssim']):.4f}", flush=True)


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--tags', required=True)
    P.add_argument('--root', default='lolv2_test')
    P.add_argument('--low_dir', default='Input')
    P.add_argument('--high_dir', default='GT')
    P.add_argument('--width', type=int, default=32)
    P.add_argument('--out', required=True)
    main(P.parse_args())

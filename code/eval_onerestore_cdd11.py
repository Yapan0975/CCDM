"""
eval_onerestore_cdd11.py - OneRestore with its released CDD-11 weights, applied off the shelf to the
CCD test set (the CDD-11 OneRestore table and the OneRestore row of the off-the-shelf table).

Replaces the pre-rendered ccd_mini manifest (80 scenes per cell, legacy sigma0/L noise) with
on-the-fly rendering under the protocol of every other CCD table: the 100 test scenes, per-index
synthesis seed 10000+idx, mask index idx % len, and the same synthesis params as training
(--pgnoise = Poisson-Gaussian noise, the model stated in the paper). Frames are cropped to a
multiple of 16 as in train_onerestore_ccd.py. Restorer, embedder, text prompts and the
word-embedding/thop stubs are those of OneRestore/eval_onerestore_ccd.py, so only the test data
changes.

Run (server, composite-restore dir):
  CUDA_VISIBLE_DEVICES=1 python3 -u eval_onerestore_cdd11.py --pgnoise --out ORcdd11_pg.json
"""
import os, sys, json, glob, argparse, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OneRestore'))
sys.path.insert(0, HERE)
import numpy as np, cv2, torch
_f = types.ModuleType('utils.utils_word_embedding')
_f.initialize_wordembedding_matrix = lambda *a, **k: (torch.zeros(12, 300), 300)
sys.modules['utils.utils_word_embedding'] = _f
_t = types.ModuleType('thop'); _t.profile = lambda *a, **k: (0, 0); _t.clever_format = lambda *a, **k: a
sys.modules['thop'] = _t
from utils.utils import load_restore_ckpt, load_embedder_ckpt
import ccdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
COMBOS = {'low_rain': ('low', 'rain', 'noise'), 'low_haze_rain': ('low', 'haze', 'rain', 'noise')}


def main(a):
    params = {}
    if a.dark: params['gamma'] = (3.0, 5.0)
    if a.pgnoise: params['noise_model'] = 'poisson'
    params = params or None
    net = load_restore_ckpt(DEV, freeze_model=True, ckpt_name=a.restore)
    emb = load_embedder_ckpt(DEV, freeze_model=True, ckpt_name=a.embedder)
    cache = {c: emb([c], 'text_encoder')[0] for c in COMBOS}
    tests = sorted(glob.glob(os.path.join(a.clean_test, '*.png')))
    if a.n:
        tests = tests[:a.n]
    rains = sorted(glob.glob(os.path.join(a.rain, '*')))
    snows = sorted(glob.glob(os.path.join(a.snow, '*')))
    print(f'[cdd11] {len(tests)} scenes, params={params}', flush=True)
    agg = {}
    for idx, cf in enumerate(tests):
        J = cv2.imread(cf).astype(np.float32) / 255.
        H, W = J.shape[:2]; H -= H % 16; W -= W % 16; J = J[:H, :W]
        d = np.load(os.path.join(a.depth_test, os.path.basename(cf).replace('.png', '.npy')))[:H, :W]
        for cname, types_ in COMBOS.items():
            for mode in ('decoupled', 'coupled'):
                rng = np.random.default_rng(a.seed0 + idx)
                rm = cv2.imread(rains[idx % len(rains)]).astype(np.float32) / 255.
                sm = cv2.imread(snows[idx % len(snows)]).astype(np.float32) / 255.
                lq, _ = ccdm.degrade(J, d, rng, mode=mode, types=types_, rain_mask=rm, snow_mask=sm,
                                     params=params)
                lq = np.clip(lq, 0, 1)
                t = torch.from_numpy(lq.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
                with torch.no_grad():
                    out = net(t, cache[cname])[0].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
                agg.setdefault((cname, mode), []).append(
                    (psnr_fn(J, out, data_range=1.0), ssim_fn(J, out, data_range=1.0, channel_axis=2),
                     psnr_fn(J, lq, data_range=1.0)))
        if (idx + 1) % 10 == 0:
            print(f'[cdd11] {idx + 1}/{len(tests)} scenes', flush=True)

    cells = {}
    print(f"{'combo':16s} {'mode':10s} {'in_PSNR':>8s} {'PSNR':>7s} {'SSIM':>7s}  n", flush=True)
    for (c, m), v in sorted(agg.items()):
        v = np.array(v)
        cells[f'{c}|{m}'] = dict(in_PSNR=round(float(v[:, 2].mean()), 3), PSNR=round(float(v[:, 0].mean()), 3),
                                 SSIM=round(float(v[:, 1].mean()), 4), n=len(v),
                                 psnr=[round(float(x), 4) for x in v[:, 0]])
        print(f'{c:16s} {m:10s} {v[:, 2].mean():8.3f} {v[:, 0].mean():7.3f} {v[:, 1].mean():7.4f}  {len(v)}', flush=True)
    cpl = float(np.mean([cells[k]['PSNR'] for k in cells if k.endswith('|coupled')]))
    dec = float(np.mean([cells[k]['PSNR'] for k in cells if k.endswith('|decoupled')]))
    res = dict(tag='ORcdd11' + ('_pg' if a.pgnoise else ''), model='onerestore_cdd11', mode='off_the_shelf',
               params=params, seed0=a.seed0, n=len(tests), coupled_test_PSNR=round(cpl, 3),
               decoupled_test_PSNR=round(dec, 3), cells=cells)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f'coupled_test_PSNR {cpl:.3f}  decoupled_test_PSNR {dec:.3f}  saved {a.out}', flush=True)


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--pgnoise', action='store_true')
    P.add_argument('--dark', action='store_true')
    P.add_argument('--n', type=int, default=0)
    P.add_argument('--seed0', type=int, default=10000)
    P.add_argument('--clean_test', default='clean_test100')
    P.add_argument('--depth_test', default='depth')
    P.add_argument('--rain', default='OneRestore/syn_data/data/rain_mask')
    P.add_argument('--snow', default='OneRestore/syn_data/data/snow_mask')
    P.add_argument('--restore', default='OneRestore/ckpts/onerestore_cdd-11.tar')
    P.add_argument('--embedder', default='OneRestore/ckpts/embedder_model.tar')
    P.add_argument('--out', required=True)
    main(P.parse_args())

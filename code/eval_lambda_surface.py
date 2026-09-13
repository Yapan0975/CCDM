"""
eval_lambda_surface.py - train-lambda x test-lambda generalisation surface on the CCD test set.

Answers the reviewers' central objection that the binary decoupled/coupled contrast could be an
artefact of two hand-picked endpoints: if coupling structure is a genuine domain axis, restoration
accuracy should degrade smoothly and monotonically with |train_lambda - test_lambda|, and the
lambda=0 / lambda=1 rows must reproduce the decoupled / coupled specialists.

Rendering is byte-identical in protocol to dataset_ccd.build_eval (same seed0 + per-index rng,
same mask indexing, same /8 cropping, same synthesis params); only the binary `mode` is replaced
by a continuous `lam`. The test renders MUST use the noise model the models were trained with:
pass --pgnoise for the Poisson-Gaussian campaign (train_probe_c.py --pgnoise). As a guard, the
lambda=0 / lambda=1 column means are printed next to the decoupled / coupled test PSNR that
train_probe_c.py stored in TAG.json; the two must agree up to AMP rounding.
Each test lambda is rendered once and shared by every model, so cost scales with renders, not
with models x renders.

Run (server, composite-restore dir):
  CUDA_VISIBLE_DEVICES=0 python3 -u eval_lambda_surface.py --pgnoise \
      --tags LAM_s0_l000,LAM_s0_l025,LAM_s0_l050,LAM_s0_l075,LAM_s0_l100,LAM_s0_rand,LAM_s0_mix \
      --test_lams 0,0.25,0.5,0.75,1.0 --combos low_rain,low_haze_rain \
      --out lam_surface_pg_s0.json
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import ccdm
from dataset_ccd import COMBOS
from nafnet import NAFNet

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def col_mean(res, tag, tl):
    return float(np.mean([v for k, c in res[tag].items() if k.endswith(f'|lam{tl:g}')
                          for v in c['psnr']]))


def main(a):
    tags = [t for t in a.tags.split(',') if t]
    test_lams = [float(v) for v in a.test_lams.split(',') if v != '']
    params = {}
    if a.dark: params['gamma'] = (3.0, 5.0)
    if a.pgnoise: params['noise_model'] = 'poisson'
    params = params or None
    models = {}
    for t in tags:
        m = NAFNet(width=a.width)
        m.load_state_dict(torch.load(t + '.pth', map_location='cpu'))
        models[t] = m.to(DEV).eval()
    print(f'[surface] {len(models)} models x {len(test_lams)} test lambdas, params={params}', flush=True)

    cleans = sorted(glob.glob(os.path.join(a.clean_test, '*.png')))
    rains = sorted(glob.glob(os.path.join(a.rain, '*')))
    snows = sorted(glob.glob(os.path.join(a.snow, '*')))
    sel = {k: COMBOS[k] for k in (a.combos.split(',') if a.combos else COMBOS)}

    out = {t: {} for t in tags}
    scenes = [os.path.basename(c) for c in cleans]
    for idx, cf in enumerate(cleans):
        J = cv2.imread(cf).astype(np.float32) / 255.0
        H, W = J.shape[:2]; H -= H % 8; W -= W % 8; J = J[:H, :W]
        d = np.load(os.path.join(a.depth_test, os.path.basename(cf).replace('.png', '.npy')))[:H, :W]
        for cname, types in sel.items():
            for tl in test_lams:
                rng = np.random.default_rng(a.seed0 + idx)      # identical to build_eval
                rm = cv2.imread(rains[idx % len(rains)]).astype(np.float32) / 255.0
                sm = cv2.imread(snows[idx % len(snows)]).astype(np.float32) / 255.0
                lq, _ = ccdm.degrade(J, d, rng, types=types, rain_mask=rm, snow_mask=sm,
                                     params=params, lam=tl)
                t_in = torch.from_numpy(lq.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
                for tag, m in models.items():
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        o = m(t_in)[0]
                    o = o[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                    key = f'{cname}|lam{tl:g}'
                    cell = out[tag].setdefault(key, {'psnr': [], 'ssim': []})
                    cell['psnr'].append(round(float(psnr_fn(J, o, data_range=1.0)), 4))
                    cell['ssim'].append(round(float(ssim_fn(J, o, data_range=1.0, channel_axis=2)), 4))
        if (idx + 1) % 10 == 0:
            print(f'[{idx + 1}/{len(cleans)}] scenes', flush=True)

    json.dump(dict(scenes=scenes, seed0=a.seed0, combos=list(sel), test_lams=test_lams,
                   width=a.width, params=params, results=out), open(a.out, 'w'))
    print('saved', a.out, flush=True)

    print('\n=== mean PSNR surface (rows = trained model, cols = test lambda) ===', flush=True)
    print('train\\test'.ljust(16) + ''.join(f'{tl:>9g}' for tl in test_lams), flush=True)
    for tag in tags:
        print(tag.ljust(16) + ''.join(f'{col_mean(out, tag, tl):9.2f}' for tl in test_lams), flush=True)

    if 0.0 in test_lams and 1.0 in test_lams:
        print('\n=== endpoint guard: lambda=0/1 columns vs training-time 2x2 eval (TAG.json) ===', flush=True)
        for tag in tags:
            if not os.path.exists(tag + '.json'):
                continue
            ref = json.load(open(tag + '.json'))
            m0, m1 = col_mean(out, tag, 0.0), col_mean(out, tag, 1.0)
            flag = ('OK' if abs(m0 - ref['decoupled_test_PSNR']) < 0.05
                    and abs(m1 - ref['coupled_test_PSNR']) < 0.05 else 'MISMATCH')
            print(f'{tag:16} lam0 {m0:.3f} vs dec-test {ref["decoupled_test_PSNR"]:.3f} | '
                  f'lam1 {m1:.3f} vs cpl-test {ref["coupled_test_PSNR"]:.3f}  {flag}', flush=True)


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--tags', required=True)
    P.add_argument('--test_lams', default='0,0.25,0.5,0.75,1.0')
    P.add_argument('--combos', default='low_rain,low_haze_rain')
    P.add_argument('--pgnoise', action='store_true')   # must match the training noise model
    P.add_argument('--dark', action='store_true')
    P.add_argument('--clean_test', default='clean_test100')
    P.add_argument('--depth_test', default='depth')
    P.add_argument('--rain', default='./OneRestore/syn_data/data/rain_mask')
    P.add_argument('--snow', default='./OneRestore/syn_data/data/snow_mask')
    P.add_argument('--width', type=int, default=32)
    P.add_argument('--seed0', type=int, default=10000)
    P.add_argument('--out', required=True)
    main(P.parse_args())

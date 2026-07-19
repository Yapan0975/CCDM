"""
eval_per_scene.py - per-scene PSNR/SSIM of released checkpoints on the deterministic CCD
test matrix (scene-level bootstrap support; eval-only, no training).

Replicates dataset_ccd.build_eval() rendering exactly (same seed0, per-index rng, mask
indexing, /8 crop), but streams one scene at a time so many models can be evaluated
without holding the full render matrix in memory.

Run (server, composite-restore dir), FW composite track:
  CUDA_VISIBLE_DEVICES=0 python3 -u eval_per_scene.py \
      --tags FW_s0_dec,FW_s0_cpl,FW_s0_mix,...  --combos low_rain,low_haze_rain \
      --out per_scene_FW.json
Validation gate: the per-cell means printed at the end must reproduce the released
FW_s*_{dec,cpl,mix}.json per_cell values (rounding aside).
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


def main(a):
    tags = [t for t in a.tags.split(',') if t]
    models = {}
    for t in tags:
        m = NAFNet(width=a.width)
        m.load_state_dict(torch.load(t + '.pth', map_location='cpu'))
        models[t] = m.to(DEV).eval()
    print(f'[eval_per_scene] {len(models)} models loaded', flush=True)
    cleans = sorted(glob.glob(os.path.join(a.clean_test, '*.png')))
    rains = sorted(glob.glob(os.path.join(a.rain, '*')))
    snows = sorted(glob.glob(os.path.join(a.snow, '*')))
    sel = {k: COMBOS[k] for k in (a.combos.split(',') if a.combos else COMBOS)}
    out = {t: {} for t in tags}
    scenes = []
    for idx, cf in enumerate(cleans):
        scenes.append(os.path.basename(cf))
        J = cv2.imread(cf).astype(np.float32) / 255.0
        H, W = J.shape[:2]; H -= H % 8; W -= W % 8; J = J[:H, :W]
        d = np.load(os.path.join(a.depth_test, os.path.basename(cf).replace('.png', '.npy')))[:H, :W]
        for cname, types in sel.items():
            for mode in ('decoupled', 'coupled'):
                rng = np.random.default_rng(a.seed0 + idx)      # identical to build_eval
                rm = cv2.imread(rains[idx % len(rains)]).astype(np.float32) / 255.0
                sm = cv2.imread(snows[idx % len(snows)]).astype(np.float32) / 255.0
                lq, _ = ccdm.degrade(J, d, rng, mode=mode, types=types, rain_mask=rm,
                                     snow_mask=sm, params=None)
                t_in = torch.from_numpy(lq.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
                for tag, m in models.items():
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        o = m(t_in)[0]
                    o = o[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                    ps = psnr_fn(J, o, data_range=1.0)
                    ss = ssim_fn(J, o, data_range=1.0, channel_axis=2)
                    key = f'{cname}|{mode}'
                    cell = out[tag].setdefault(key, {'psnr': [], 'ssim': []})
                    cell['psnr'].append(round(float(ps), 4))
                    cell['ssim'].append(round(float(ss), 4))
        if (idx + 1) % 10 == 0:
            print(f'[{idx + 1}/{len(cleans)}] scenes done', flush=True)
    json.dump(dict(scenes=scenes, seed0=a.seed0, combos=list(sel), width=a.width, results=out),
              open(a.out, 'w'))
    print('saved', a.out, flush=True)
    # validation print: per-cell means must match the released aggregate JSONs
    for tag in tags:
        cells = {k: round(float(np.mean(v['psnr'])), 3) for k, v in out[tag].items()}
        print(f'[check] {tag}: {cells}', flush=True)


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--tags', required=True)          # comma-separated checkpoint tags (tag.pth)
    P.add_argument('--combos', default='')           # e.g. low_rain,low_haze_rain
    P.add_argument('--clean_test', default='clean_test100')
    P.add_argument('--depth_test', default='depth')
    P.add_argument('--rain', default='./OneRestore/syn_data/data/rain_mask')
    P.add_argument('--snow', default='./OneRestore/syn_data/data/snow_mask')
    P.add_argument('--width', type=int, default=32)
    P.add_argument('--seed0', type=int, default=10000)
    P.add_argument('--out', required=True)
    main(P.parse_args())

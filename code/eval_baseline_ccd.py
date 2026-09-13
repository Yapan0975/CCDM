"""eval_baseline_ccd.py  — Off-the-shelf PromptIR / AdaIR on CCD (coupled + decoupled).
Produces JSON compatible with OR_{mode}_s0.json: {coupled_test_PSNR, decoupled_test_PSNR, n}.
--pgnoise renders the test set with the Poisson-Gaussian noise model stated in the paper and
writes BLpg_<model>.json (the legacy-noise output BL_<model>.json is left untouched).

Usage:
  python3 eval_baseline_ccd.py --model promptir --pgnoise \
      --ckpt ~/Documents/yping/baselines/PromptIR_ckpt/model_best.ckpt --gpu 0
  python3 eval_baseline_ccd.py --model adair --pgnoise \
      --ckpt ~/Documents/yping/baselines/AdaIR_ckpt/adair3d.ckpt --gpu 1
"""
import os, sys, json, glob, argparse
import numpy as np, cv2, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.normpath(os.path.join(HERE, '..', 'baselines'))
sys.path.insert(0, HERE)
import ccdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

COMBOS = {'low_rain':      ('low', 'rain', 'noise'),
          'low_haze_rain':  ('low', 'haze', 'rain', 'noise')}

# ── loaders ──────────────────────────────────────────────────────────────────

def load_promptir(ckpt, dev):
    sys.path.insert(0, os.path.join(BASELINES, 'PromptIR'))
    from net.model import PromptIR
    import lightning.pytorch as pl
    import torch.nn as nn

    class PromptIRModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = PromptIR(decoder=True)
            self.loss_fn = nn.L1Loss()
        def forward(self, x):
            return self.net(x)

    model = PromptIRModel.load_from_checkpoint(ckpt, map_location=dev)
    return model.eval().to(dev)

def load_adair(ckpt, dev):
    sys.path.insert(0, os.path.join(BASELINES, 'AdaIR'))
    from net.model import AdaIR
    import lightning.pytorch as pl
    import torch.nn as nn

    class AdaIRModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = AdaIR(decoder=True)
            self.loss_fn = nn.L1Loss()
        def forward(self, x):
            return self.net(x)

    model = AdaIRModel.load_from_checkpoint(ckpt, map_location=dev)
    return model.eval().to(dev)

LOADERS = {'promptir': load_promptir, 'adair': load_adair}

# ── tile-based inference (handles arbitrary resolution) ──────────────────────

def tile_eval(net, inp, tile=256, overlap=32, dev='cuda'):
    b, c, h, w = inp.shape
    tile = min(tile, h, w)
    stride = tile - overlap
    h_idx = list(range(0, h - tile, stride)) + [h - tile]
    w_idx = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h, w, device=dev)
    W = torch.zeros_like(E)
    with torch.no_grad():
        for hi in h_idx:
            for wi in w_idx:
                patch = inp[..., hi:hi+tile, wi:wi+tile]
                out   = net(patch)
                E[..., hi:hi+tile, wi:wi+tile] += out
                W[..., hi:hi+tile, wi:wi+tile] += 1
    return (E / W).clamp(0, 1)

# ── main ─────────────────────────────────────────────────────────────────────

def main(a):
    dev = f'cuda:{a.gpu}'
    params = {'noise_model': 'poisson'} if a.pgnoise else None
    print(f'Loading {a.model} from {a.ckpt}... params={params}', flush=True)
    net = LOADERS[a.model](a.ckpt, dev)
    tag = f'BLpg_{a.model}' if a.pgnoise else f'BL_{a.model}'

    tests  = sorted(glob.glob(os.path.join(a.clean_test, '*.png')))
    rains  = sorted(glob.glob(os.path.join(a.rain, '*')))
    snows  = sorted(glob.glob(os.path.join(a.snow, '*')))
    cell   = {'coupled': [], 'decoupled': []}

    for idx, cf in enumerate(tests):
        J = cv2.imread(cf).astype(np.float32) / 255.
        H, W = J.shape[:2]; H -= H % 16; W -= W % 16; J = J[:H, :W]
        d = np.load(os.path.join(a.depth_test,
                    os.path.basename(cf).replace('.png', '.npy')))[:H, :W]
        for combo, types in COMBOS.items():
            for mode in ('decoupled', 'coupled'):
                rng = np.random.default_rng(10000 + idx)
                rm  = cv2.imread(rains[idx % len(rains)]).astype(np.float32) / 255.
                sm  = cv2.imread(snows[idx % len(snows)]).astype(np.float32) / 255.
                lq, _ = ccdm.degrade(J, d, rng, mode=mode, types=types,
                                     rain_mask=rm, snow_mask=sm, params=params)
                lq = np.clip(lq, 0, 1)
                t  = torch.from_numpy(lq.transpose(2, 0, 1)).float().unsqueeze(0).to(dev)
                out = tile_eval(net, t, tile=a.tile, overlap=a.overlap, dev=dev)
                out_np = out[0].cpu().numpy().transpose(1, 2, 0)
                h2 = min(out_np.shape[0], J.shape[0])
                w2 = min(out_np.shape[1], J.shape[1])
                cell[mode].append(psnr_fn(J[:h2, :w2], out_np[:h2, :w2],
                                          data_range=1.0))
        if (idx + 1) % 10 == 0:
            print(f'[{tag}] {idx+1}/{len(tests)} '
                  f'cpl={np.mean(cell["coupled"]):.3f} '
                  f'dec={np.mean(cell["decoupled"]):.3f}', flush=True)

    res = dict(tag=tag, model=a.model, mode='off_the_shelf', params=params,
               coupled_test_PSNR=round(float(np.mean(cell['coupled'])), 3),
               decoupled_test_PSNR=round(float(np.mean(cell['decoupled'])), 3),
               n=len(tests))
    print(json.dumps(res))
    json.dump(res, open(tag + '.json', 'w'), indent=2)
    print('saved', tag + '.json')

if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--model',      choices=['promptir', 'adair'], required=True)
    P.add_argument('--ckpt',       required=True)
    P.add_argument('--gpu',        type=int, default=0)
    P.add_argument('--tile',       type=int, default=256)
    P.add_argument('--overlap',    type=int, default=32)
    P.add_argument('--pgnoise',    action='store_true')
    P.add_argument('--clean_test', default='clean_test100')
    P.add_argument('--depth_test', default='depth')
    P.add_argument('--rain',  default='OneRestore/syn_data/data/rain_mask')
    P.add_argument('--snow',  default='OneRestore/syn_data/data/snow_mask')
    main(P.parse_args())

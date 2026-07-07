"""
ablate_crossterm.py - A4 per-cross-term ablation (no retraining).

Loads a trained restorer and evaluates it on test variants that turn ON one cross term at a
time (rest decoupled): {none, rain, noise, haze, all}. Run for the decoupled-trained and the
coupled-trained model. The cross term on which the decoupled-trained model drops most is the
dominant coupling that field-standard synthesis fails to prepare for.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, glob, torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import ccdm
from nafnet import NAFNet, CoupleNet

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
VARIANTS = {'none': set(), 'rain': {'rain'}, 'noise': {'noise'}, 'haze': {'haze'},
            'all': {'rain', 'noise', 'haze'}}
TYPES = ('low', 'haze', 'rain', 'noise')   # combo containing all three cross-term carriers

def load_model(a):
    if a.model == 'couplenet':
        m = CoupleNet(width=a.width)
    elif a.model == 'restormer':
        from restormer import Restormer; m = Restormer(dim=a.width)
    else:
        m = NAFNet(width=a.width)
    m.load_state_dict(torch.load(a.ckpt, map_location=DEV)); m.eval()
    return m.to(DEV)

def main(a):
    m = load_model(a)
    cleans = sorted(glob.glob(os.path.join(a.clean, '*.png')))[:a.n]
    rains = sorted(glob.glob(os.path.join(a.rain, '*'))); snows = sorted(glob.glob(os.path.join(a.snow, '*')))
    out = {}
    for vname, cterms in VARIANTS.items():
        ps = []
        for idx, cf in enumerate(cleans):
            J = cv2.imread(cf).astype(np.float32) / 255.0
            H, W = J.shape[:2]; H -= H % 8; W -= W % 8; J = J[:H, :W]
            d = np.load(os.path.join(a.depth, os.path.basename(cf).replace('.png', '.npy')))[:H, :W]
            rng = np.random.default_rng(20000 + idx)            # fixed across variants & models
            rm = cv2.imread(rains[idx % len(rains)]).astype(np.float32) / 255.0
            sm = cv2.imread(snows[idx % len(snows)]).astype(np.float32) / 255.0
            lq, _ = ccdm.degrade(J, d, rng, types=TYPES, rain_mask=rm, snow_mask=sm, coupled_terms=cterms)
            t = torch.from_numpy(lq.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
            with torch.no_grad():
                o = m(t)[0][0].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
            ps.append(psnr_fn(J, o, data_range=1.0))
        out[vname] = round(float(np.mean(ps)), 3)
    out['drops_vs_none'] = {k: round(out['none'] - out[k], 3) for k in ['rain', 'noise', 'haze', 'all']}
    print(json.dumps(out, indent=2))
    json.dump(out, open(a.out, 'w'), indent=2); print('saved', a.out)

if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--ckpt', required=True)
    P.add_argument('--model', default='nafnet')
    P.add_argument('--width', type=int, default=32)
    P.add_argument('--clean', default='clean_test100')
    P.add_argument('--depth', default='depth')
    P.add_argument('--rain', default='OneRestore/syn_data/data/rain_mask')
    P.add_argument('--snow', default='OneRestore/syn_data/data/snow_mask')
    P.add_argument('--n', type=int, default=100)
    P.add_argument('--out', required=True)
    main(P.parse_args())

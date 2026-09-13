"""P1.2: fine-tune OneRestore on CCD (coupled vs decoupled vs mixed) and eval on the CCD 2x2.
Adds a 4th, all-in-one architecture to the specialization result (reviewer Major 8: 'OneRestore
retrained on decoupled/coupled CCD'). Reuses the frozen pretrained embedder; fine-tunes the restorer
from the CDD-11 checkpoint with Charbonnier loss on CCD composite degradation.
--pgnoise renders training and test data with the Poisson-Gaussian noise model stated in the paper
(the same flag as train_probe_c.py); the flag must be identical for training and evaluation."""
import os, sys, json, glob, time, argparse
import numpy as np, cv2, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'OneRestore'))
sys.path.insert(0, HERE)
from utils.utils import load_embedder_ckpt
from model.OneRestore import OneRestore
import ccdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

DEV = 'cuda'
COMBOS = {'low_rain': ('low', 'rain', 'noise'), 'low_haze_rain': ('low', 'haze', 'rain', 'noise')}

def charb(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()

def emb224(lq):
    r = cv2.resize(lq, (224, 224), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(r.transpose(2, 0, 1)).float().unsqueeze(0)

def embed(embedder, lq):
    return embedder(emb224(lq).to(DEV), 'image_encoder')[0]

def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    params = {}
    if a.dark: params['gamma'] = (3.0, 5.0)
    if a.pgnoise: params['noise_model'] = 'poisson'
    params = params or None
    print(f'[{a.tag}] mode={a.mode} seed={a.seed} iters={a.iters} params={params}', flush=True)
    embedder = load_embedder_ckpt(DEV, freeze_model=True, ckpt_name=a.embedder)
    restorer = OneRestore().to(DEV)
    sd = torch.load(a.restore, map_location=DEV)
    restorer.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    opt = torch.optim.AdamW(restorer.parameters(), lr=a.lr)
    cleans = sorted(glob.glob(a.clean_train + '/*.png'))[:a.train_n] if a.train_n else sorted(glob.glob(a.clean_train + '/*.png'))
    rains = sorted(glob.glob(a.rain + '/*')); snows = sorted(glob.glob(a.snow + '/*'))
    restorer.train(); t0 = time.time()
    for it in range(1, a.iters + 1):
        rng = np.random.default_rng([a.seed, it])
        cf = cleans[rng.integers(len(cleans))]
        J = cv2.imread(cf).astype(np.float32) / 255.
        d = np.load(a.depth + '/' + os.path.basename(cf).replace('.png', '.npy'))
        c = a.crop; H, W = J.shape[:2]
        if H < c or W < c:
            ph, pw = max(0, c - H), max(0, c - W)
            J = cv2.copyMakeBorder(J, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            d = cv2.copyMakeBorder(d, 0, ph, 0, pw, cv2.BORDER_REFLECT); H, W = J.shape[:2]
        y = rng.integers(0, max(1, H - c)); x0 = rng.integers(0, max(1, W - c))
        J = J[y:y + c, x0:x0 + c]; d = d[y:y + c, x0:x0 + c]
        rm = cv2.imread(rains[rng.integers(len(rains))]).astype(np.float32) / 255.
        sm = cv2.imread(snows[rng.integers(len(snows))]).astype(np.float32) / 255.
        types = COMBOS[list(COMBOS)[rng.integers(len(COMBOS))]]
        mode = a.mode if a.mode != 'mixed' else ('coupled' if rng.integers(2) == 0 else 'decoupled')
        lq, _ = ccdm.degrade(J, d, rng, mode=mode, types=types, rain_mask=rm, snow_mask=sm,
                             params=params)
        lq = np.clip(lq, 0, 1)
        lq_re = torch.from_numpy(lq.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
        gt = torch.from_numpy(J.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
        with torch.no_grad():
            emb = embed(embedder, lq)
        out = restorer(lq_re, emb)
        loss = charb(out, gt)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f'[{a.tag}] it {it}/{a.iters} loss {loss.item():.4f} {(time.time()-t0)/it*1000:.0f}ms/it', flush=True)

    restorer.eval()
    tests = sorted(glob.glob(a.clean_test + '/*.png'))
    cell = {'coupled': [], 'decoupled': []}
    with torch.no_grad():
        for idx, cf in enumerate(tests):
            J = cv2.imread(cf).astype(np.float32) / 255.; H, W = J.shape[:2]; H -= H % 16; W -= W % 16; J = J[:H, :W]
            d = np.load(a.depth_test + '/' + os.path.basename(cf).replace('.png', '.npy'))[:H, :W]
            for cname, types in COMBOS.items():
                for mode in ('decoupled', 'coupled'):
                    rng = np.random.default_rng(10000 + idx)
                    rm = cv2.imread(rains[idx % len(rains)]).astype(np.float32) / 255.
                    sm = cv2.imread(snows[idx % len(snows)]).astype(np.float32) / 255.
                    lq, _ = ccdm.degrade(J, d, rng, mode=mode, types=types, rain_mask=rm, snow_mask=sm,
                                         params=params)
                    lq = np.clip(lq, 0, 1)
                    lq_re = torch.from_numpy(lq.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
                    out = restorer(lq_re, embed(embedder, lq))[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                    cell[mode].append(psnr_fn(J, out, data_range=1.0))
    res = dict(tag=a.tag, model='onerestore', mode=a.mode, params=params,
               coupled_test_PSNR=round(float(np.mean(cell['coupled'])), 3),
               decoupled_test_PSNR=round(float(np.mean(cell['decoupled'])), 3), n=len(tests))
    print(json.dumps(res)); json.dump(res, open(a.tag + '.json', 'w'), indent=2)
    print('saved', a.tag + '.json')

if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--mode', choices=['decoupled', 'coupled', 'mixed'], required=True)
    P.add_argument('--seed', type=int, default=0)
    P.add_argument('--tag', required=True)
    P.add_argument('--iters', type=int, default=6000)
    P.add_argument('--train_n', type=int, default=0)
    P.add_argument('--crop', type=int, default=256)
    P.add_argument('--lr', type=float, default=1e-4)
    P.add_argument('--pgnoise', action='store_true')   # Poisson-Gaussian noise (paper's model)
    P.add_argument('--dark', action='store_true')
    P.add_argument('--clean_train', default='clean_train_full'); P.add_argument('--depth', default='depth_full')
    P.add_argument('--clean_test', default='clean_test100'); P.add_argument('--depth_test', default='depth')
    P.add_argument('--rain', default='OneRestore/syn_data/data/rain_mask')
    P.add_argument('--snow', default='OneRestore/syn_data/data/snow_mask')
    P.add_argument('--embedder', default='OneRestore/ckpts/embedder_model.tar')
    P.add_argument('--restore', default='OneRestore/ckpts/onerestore_cdd-11.tar')
    main(P.parse_args())

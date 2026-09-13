"""
dataset_ccd.py - on-the-fly CCDM dataset (uses precomputed MiDaS depth).
Train: random crop + random combo + random params (mode fixed per run).
Eval : deterministic (seed per index), fixed combos, both modes.
"""
import os, glob
import numpy as np, cv2, torch
from torch.utils.data import Dataset
import ccdm

COMBOS = {'low_rain': ('low', 'rain', 'noise'),
          'low_haze_rain': ('low', 'haze', 'rain', 'noise'),
          'low_noise': ('low', 'noise')}


def _stack_fields(gt):
    return np.stack([gt['L'], gt['t'], gt['sigma'] / (gt['sigma'].max() + 1e-6)], 0).astype(np.float32)


class CCDTrain(Dataset):
    def __init__(self, clean_dir, depth_dir, mode, rain, snow, crop=256, length=4000, train_n=0,
                 combos=None, params=None, seed=0, lam=None):
        self.cleans = sorted(glob.glob(os.path.join(clean_dir, '*.png')))
        if train_n:
            self.cleans = self.cleans[:train_n]
        self.depth_dir = depth_dir; self.mode = mode; self.crop = crop; self.length = length
        self.params = params; self.seed = seed
        # lam: None -> binary mode (published behaviour); float -> fixed coupling strength;
        # 'rand' -> continuous domain randomisation, lam ~ U[0,1] drawn per sample.
        self.lam = lam
        self.rains = sorted(glob.glob(os.path.join(rain, '*')))
        self.snows = sorted(glob.glob(os.path.join(snow, '*')))
        sel = combos if combos else list(COMBOS.keys())
        self.combos = [(k, COMBOS[k]) for k in sel]

    def __len__(self): return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])         # reproducible per (seed, step) -> proper multi-seed
        cf = self.cleans[rng.integers(len(self.cleans))]
        J = cv2.imread(cf).astype(np.float32) / 255.0
        d = np.load(os.path.join(self.depth_dir, os.path.basename(cf).replace('.png', '.npy')))
        c = self.crop; H, W = J.shape[:2]
        if H < c or W < c:                                   # pad small images up to crop size
            ph, pw = max(0, c - H), max(0, c - W)
            J = cv2.copyMakeBorder(J, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            d = cv2.copyMakeBorder(d, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            H, W = J.shape[:2]
        y = rng.integers(0, max(1, H - c)); x = rng.integers(0, max(1, W - c))
        J = J[y:y + c, x:x + c]; d = d[y:y + c, x:x + c]
        _, types = self.combos[rng.integers(len(self.combos))]
        rm = cv2.imread(self.rains[rng.integers(len(self.rains))]).astype(np.float32) / 255.0
        sm = cv2.imread(self.snows[rng.integers(len(self.snows))]).astype(np.float32) / 255.0
        mode = self.mode
        if mode == 'mixed':                                  # domain-randomized: 50/50 per sample
            mode = 'coupled' if rng.integers(2) == 0 else 'decoupled'
        lam = self.lam
        if lam == 'rand':                                    # continuous randomisation over lam
            lam = float(rng.uniform(0.0, 1.0))
        lq, gt = ccdm.degrade(J, d, rng, mode=mode, types=types, rain_mask=rm, snow_mask=sm,
                              params=self.params, lam=lam)
        return (torch.from_numpy(lq.transpose(2, 0, 1)),
                torch.from_numpy(J.transpose(2, 0, 1)),
                torch.from_numpy(_stack_fields(gt)))


def build_eval(clean_dir, depth_dir, rain, snow, seed0=10000, combos=None, params=None):
    """Deterministic 2x2 eval set: {combo} x {mode}. Returns list of dicts."""
    cleans = sorted(glob.glob(os.path.join(clean_dir, '*.png')))
    rains = sorted(glob.glob(os.path.join(rain, '*'))); snows = sorted(glob.glob(os.path.join(snow, '*')))
    sel = {k: COMBOS[k] for k in (combos if combos else COMBOS)}
    items = []
    for idx, cf in enumerate(cleans):
        J = cv2.imread(cf).astype(np.float32) / 255.0
        H, W = J.shape[:2]; H -= H % 8; W -= W % 8; J = J[:H, :W]
        d = np.load(os.path.join(depth_dir, os.path.basename(cf).replace('.png', '.npy')))[:H, :W]
        for cname, types in sel.items():
            for mode in ('decoupled', 'coupled'):
                rng = np.random.default_rng(seed0 + idx)        # same masks/params across modes
                rm = cv2.imread(rains[idx % len(rains)]).astype(np.float32) / 255.0
                sm = cv2.imread(snows[idx % len(snows)]).astype(np.float32) / 255.0
                lq, gt = ccdm.degrade(J, d, rng, mode=mode, types=types, rain_mask=rm, snow_mask=sm,
                                      params=params)
                items.append(dict(combo=cname, mode=mode,
                                  lq=lq.transpose(2, 0, 1), clean=J.transpose(2, 0, 1)))
    return items

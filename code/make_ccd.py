"""
make_ccd.py - generate the CCD benchmark (mini for pipeline validation, scalable to full).

For each clean image: MiDaS depth -> ccdm.degrade in BOTH modes (decoupled = field/CDD-11,
coupled = ours with cross-terms) for each combo. Saves degraded PNGs, the clean PNG, the
shared latent GT (L,t,sigma as float16 npz, for CoupleNet supervision later) and a manifest.
"""
import os, sys, json, glob, argparse
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccdm

COMBOS = {                      # name -> degradation types ; aligned to OneRestore vocabulary
    'low_rain':       ('low', 'rain', 'noise'),
    'low_haze_rain':  ('low', 'haze', 'rain', 'noise'),
}

def build_midas():
    m = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).eval()
    tf = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    def depth_of(bgr01):
        rgb = np.ascontiguousarray((bgr01[:, :, ::-1] * 255).astype(np.uint8))
        with torch.no_grad():
            pred = m(tf(rgb))
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=rgb.shape[:2],
                        mode='bicubic', align_corners=False).squeeze().cpu().numpy()
        disp = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)   # high = near
        return (1.0 - disp).astype(np.float32)                          # near=0, far=1
    return depth_of

def main(a):
    rng = np.random.default_rng(a.seed)
    depth_of = build_midas()
    rains = sorted(glob.glob(os.path.join(a.rain, '*')))
    snows = sorted(glob.glob(os.path.join(a.snow, '*')))
    cleans = sorted(glob.glob(os.path.join(a.clean, '*.png')))[:a.n]
    for sub in ['clean', 'gt'] + [f'{c}/{m}' for c in COMBOS for m in ('decoupled', 'coupled')]:
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    manifest = []
    for i, cf in enumerate(cleans):
        J = cv2.imread(cf).astype(np.float32) / 255.0
        d = depth_of(J)
        cv2.imwrite(os.path.join(a.out, 'clean', f'{i:04d}.png'), (J * 255).astype(np.uint8))
        rm = cv2.imread(rains[i % len(rains)]).astype(np.float32) / 255.0
        sm = cv2.imread(snows[i % len(snows)]).astype(np.float32) / 255.0
        for cname, types in COMBOS.items():
            for mode in ('decoupled', 'coupled'):
                r2 = np.random.default_rng(a.seed * 1000 + i)   # same params/masks across modes
                lq, gt = ccdm.degrade(J, d, r2, mode=mode, types=types,
                                      rain_mask=rm, snow_mask=sm)
                p = os.path.join(a.out, cname, mode, f'{i:04d}.png')
                cv2.imwrite(p, (lq * 255).astype(np.uint8))
                manifest.append(dict(id=i, combo=cname, mode=mode,
                                     degraded=os.path.relpath(p, a.out),
                                     clean=f'clean/{i:04d}.png'))
                if mode == 'coupled':
                    np.savez_compressed(os.path.join(a.out, 'gt', f'{i:04d}_{cname}.npz'),
                                        L=gt['L'].astype(np.float16), t=gt['t'].astype(np.float16),
                                        sigma=gt['sigma'].astype(np.float16))
        if i % 20 == 0:
            print(f'[{i}/{len(cleans)}] {cf} depth[{d.min():.2f},{d.max():.2f}]')
    with open(os.path.join(a.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f)
    print(f'done: {len(cleans)} clean x {len(COMBOS)} combos x 2 modes = {len(manifest)} degraded -> {a.out}')

if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--clean', default='./clean512')
    P.add_argument('--rain', default='./OneRestore/syn_data/data/rain_mask')
    P.add_argument('--snow', default='./OneRestore/syn_data/data/snow_mask')
    P.add_argument('--out', default='./ccd_mini')
    P.add_argument('--n', type=int, default=80)
    P.add_argument('--seed', type=int, default=7)
    main(P.parse_args())

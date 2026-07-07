"""Generate paper figures: (1) matched decoupled/coupled CCDM sample pairs + difference maps,
(2) composite specialization+cure bar chart, (3) ExDark real-detection bar chart.
Run in ~/Documents/yping/composite-restore (has ccdm.py, clean_test100, depth, masks)."""
import os, glob
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ccdm

os.makedirs('figs', exist_ok=True)

# ---------- 1. qualitative matched pairs ----------
cleans = sorted(glob.glob('clean_test100/*.png'))
rains = sorted(glob.glob('OneRestore/syn_data/data/rain_mask/*'))
snows = sorted(glob.glob('OneRestore/syn_data/data/snow_mask/*'))
TYPES = ('low', 'haze', 'rain', 'noise')
idxs = [3, 7, 12, 19, 27, 41]
for k, i in enumerate(idxs):
    cf = cleans[i]
    J = cv2.imread(cf).astype(np.float32) / 255.
    d = np.load('depth/' + os.path.basename(cf).replace('.png', '.npy'))
    H, W = J.shape[:2]; H -= H % 8; W -= W % 8; J = J[:H, :W]; d = d[:H, :W]
    rm = cv2.imread(rains[i % len(rains)]).astype(np.float32) / 255.
    sm = cv2.imread(snows[i % len(snows)]).astype(np.float32) / 255.
    o = {}
    for mode in ('decoupled', 'coupled'):
        rng = np.random.default_rng(1000 + i)          # SAME seed both modes -> matched marginals
        lq, _ = ccdm.degrade(J, d, rng, mode=mode, types=TYPES, rain_mask=rm, snow_mask=sm)
        o[mode] = np.clip(lq, 0, 1)
    diff = np.clip(np.abs(o['coupled'] - o['decoupled']) * 4.0, 0, 1)   # x4 amplified
    cv2.imwrite(f'figs/s{k}_clean.png', (J * 255).astype(np.uint8))
    cv2.imwrite(f'figs/s{k}_dec.png', (o['decoupled'] * 255).astype(np.uint8))
    cv2.imwrite(f'figs/s{k}_cpl.png', (o['coupled'] * 255).astype(np.uint8))
    cv2.imwrite(f'figs/s{k}_diff.png', (diff * 255).astype(np.uint8))
    print('scene', k, os.path.basename(cf), J.shape)

# ---------- 2. composite specialization + cure ----------
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(5.2, 3.1))
groups = ['coupled-test', 'decoupled-test']
dec = [18.48, 22.16]; cpl = [22.82, 17.63]; mix = [21.96, 21.80]
decs = [0.11, 0.04]; cpls = [0.12, 0.24]; mixs = [0.06, 0.09]
x = np.arange(2); w = 0.26
ax.bar(x - w, dec, w, yerr=decs, label='decoupled-train', capsize=3, color='#d65f5f')
ax.bar(x, cpl, w, yerr=cpls, label='coupled-train', capsize=3, color='#4c72b0')
ax.bar(x + w, mix, w, yerr=mixs, label='mixed-train (ours)', capsize=3, color='#55a868')
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel('PSNR (dB)'); ax.set_ylim(16, 24)
ax.legend(fontsize=8, ncol=1, loc='upper right', framealpha=0.9)
ax.set_title('Each specialist wins its own structure; mixed covers both', fontsize=9.5)
fig.tight_layout(); fig.savefig('figs/fig_spec.pdf'); plt.close(fig)

# ---------- 3. ExDark real detection ----------
fig, ax = plt.subplots(figsize=(4.4, 3.1))
labels = ['decoupled', 'mixed', 'coupled']
ap = [0.0439, 0.0817, 0.1059]; aps = [0.0138, 0.0042, 0.0128]  # 5-seed mean/std, agg_exdark.py
cols = ['#d65f5f', '#55a868', '#4c72b0']
x = np.arange(3)
ax.bar(x, ap, 0.6, yerr=aps, capsize=4, color=cols)
ax.axhline(0.293, ls='--', color='gray', lw=1.3, label='off-the-shelf (ref.)')
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('ExDark AP (real GT)')
ax.set_ylim(0, 0.32)
ax.legend(fontsize=8, loc='upper left')
ax.set_title('Real low-light detection: coupled $\\approx2.4\\times$ decoupled', fontsize=9.5)
fig.tight_layout(); fig.savefig('figs/fig_exdark.pdf'); plt.close(fig)
print('figures done')

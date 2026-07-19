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
# All five arms as full bars (not a dashed reference line), grouped into the two
# non-degraded references vs the three synthetic-degradation fine-tunes, so the
# absolute collapse of the degraded arms is visible at a glance and the 2.4x
# coupled/decoupled ratio reads as a controlled diagnostic, not a performance win.
fig, ax = plt.subplots(figsize=(5.2, 3.2))
labels = ['off-the-shelf\n(no fine-tune)', 'clean-COCO\nfine-tune', 'decoupled', 'mixed', 'coupled']
ap = [0.2926, 0.2623, 0.0439, 0.0817, 0.1059]   # 5-seed means (agg_exdark.py); off-the-shelf single
aps = [0.0, 0.0081, 0.0138, 0.0042, 0.0128]
cols = ['#8c8c8c', '#b8b8b8', '#d65f5f', '#55a868', '#4c72b0']
x = np.array([0.0, 1.0, 2.4, 3.4, 4.4])
ax.bar(x, ap, 0.72, yerr=aps, capsize=4, color=cols)
ax.axvline(1.7, color='k', lw=0.8, ls=':')
ax.text(0.5, 0.315, 'no degradation', ha='center', fontsize=8, color='#555555')
ax.text(3.4, 0.315, 'synthetic-degradation fine-tune', ha='center', fontsize=8, color='#555555')
ax.annotate('$\\approx2.4\\times$', xy=(4.4, 0.122), xytext=(2.9, 0.16), fontsize=9,
            arrowprops=dict(arrowstyle='->', lw=0.9, color='#333333'))
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('ExDark AP (real GT)')
ax.set_ylim(0, 0.34)
ax.set_title('Real low-light detection: a controlled coupling-structure comparison', fontsize=9.5)
fig.tight_layout(); fig.savefig('figs/fig_exdark.pdf'); plt.close(fig)
print('figures done')

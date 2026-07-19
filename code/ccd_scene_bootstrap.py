"""
ccd_scene_bootstrap.py - scene-level and seed x scene hierarchical bootstrap for the
restoration headlines (review item: quantify scene/dataset uncertainty, not only training
stochasticity).

Reads the per-scene evaluations produced by eval_per_scene.py / eval_real_lol_per.py:
  per_scene_FW.json   composite 2x2 track (100 CCD test scenes x 15 models)
  per_image_LLv.json  variance-matched real LOLv2 track (100 real pairs x 15 models)

Reported quantities:
  A. composite mixed worst-case minus best-specialist worst-case (+3.33 dB headline):
     scene-level bootstrap CI (seeds fixed) and hierarchical seed x scene bootstrap CI.
  B. specialisation gaps (own-structure minus cross-structure PSNR per specialist):
     scene-level bootstrap CIs.
  C. real LOLv2 coupled-decoupled and mixed-decoupled PSNR: image-level paired bootstrap
     CI, per-image win rate, and a Wilcoxon signed-rank test over the 100 scenes
     (seed-averaged per-image differences), which measures scene-level uncertainty at
     n=100 rather than seed-level n=5.
Validation gates: full-sample means must reproduce the released aggregate tables.

Run:  python code/ccd_scene_bootstrap.py --dir analysis --boot 5000
"""
import os, json, argparse
import numpy as np
from scipy import stats

MODES = ['dec', 'cpl', 'mix']
SEEDS = range(5)


def cell(res, tag, key):
    return np.asarray(res[tag][key]['psnr'], float)


def fw_matrix(res, combos):
    """M[mode][seed][test_structure] -> per-scene matrix (n_scenes,) per combo, stacked."""
    M = {}
    for m in MODES:
        M[m] = {}
        for s in SEEDS:
            tag = f'FW_s{s}_{m}'
            M[m][s] = {t: np.stack([cell(res, tag, f'{c}|{t}') for c in combos])   # (n_combo, n_scene)
                       for t in ('coupled', 'decoupled')}
    return M


def fw_headline(M, idx):
    """mixed worst-case minus best-specialist worst-case, per seed, on scene subset idx."""
    diffs = []
    for s in SEEDS:
        mean = {m: {t: M[m][s][t][:, idx].mean() for t in ('coupled', 'decoupled')} for m in MODES}
        wc = {m: min(mean[m]['coupled'], mean[m]['decoupled']) for m in MODES}
        diffs.append(wc['mix'] - max(wc['dec'], wc['cpl']))
    return np.asarray(diffs)


def pct(a):
    return np.percentile(a, [2.5, 97.5])


def main(a):
    rng = np.random.default_rng(0)
    fw = json.load(open(os.path.join(a.dir, 'per_scene_FW.json')))
    combos = fw['combos']; res = fw['results']
    n_scene = len(fw['scenes'])
    M = fw_matrix(res, combos)

    # ---- validation gate: reproduce released aggregates ----
    full = np.arange(n_scene)
    for m, want_c, want_d in [('dec', 18.48, 22.16), ('cpl', 22.82, 17.63), ('mix', 21.96, 21.80)]:
        c = np.mean([M[m][s]['coupled'][:, full].mean() for s in SEEDS])
        d = np.mean([M[m][s]['decoupled'][:, full].mean() for s in SEEDS])
        print(f'[gate FW {m}] coupled-test {c:.2f} (want {want_c})  decoupled-test {d:.2f} (want {want_d})')

    obs_a = fw_headline(M, full)
    print(f'\n[A] composite mixed_wc - best_specialist_wc: mean {obs_a.mean():+.3f} dB '
          f'(per-seed {np.round(obs_a, 3)})')
    # scene-level bootstrap (seeds fixed, resample scenes)
    bs = np.array([fw_headline(M, rng.integers(0, n_scene, n_scene)).mean() for _ in range(a.boot)])
    lo_a, hi_a = pct(bs)
    print(f'    scene bootstrap ({a.boot}x): 95% CI [{lo_a:+.3f}, {hi_a:+.3f}] dB')
    # hierarchical: resample seeds AND scenes
    bh = []
    for _ in range(a.boot):
        sidx = rng.integers(0, n_scene, n_scene)
        d = fw_headline(M, sidx)
        bh.append(d[rng.integers(0, 5, 5)].mean())
    lo_h, hi_h = pct(np.array(bh))
    print(f'    seed x scene hierarchical bootstrap: 95% CI [{lo_h:+.3f}, {hi_h:+.3f}] dB')

    # ---- B. specialisation gaps ----
    print('\n[B] specialisation gap (own-structure minus cross-structure PSNR):')
    out_b = {}
    for m, own, cross in [('dec', 'decoupled', 'coupled'), ('cpl', 'coupled', 'decoupled')]:
        gap_obs = np.mean([M[m][s][own][:, full].mean() - M[m][s][cross][:, full].mean() for s in SEEDS])
        bs = []
        for _ in range(a.boot):
            idx = rng.integers(0, n_scene, n_scene)
            bs.append(np.mean([M[m][s][own][:, idx].mean() - M[m][s][cross][:, idx].mean() for s in SEEDS]))
        lo, hi = pct(np.array(bs))
        print(f'    {m}-trained: {gap_obs:+.3f} dB, scene bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]')
        out_b[m] = dict(gap=gap_obs, ci=[lo, hi])

    # ---- C. real LOLv2 ----
    lv = json.load(open(os.path.join(a.dir, 'per_image_LLv.json')))
    R = lv['results']; n_img = len(lv['images'])
    P = {m: np.stack([np.asarray(R[f'LLv_{m}_s{s}']['psnr'], float) for s in SEEDS]) for m in MODES}  # (5, n_img)
    for m, want in [('dec', 17.132), ('cpl', 17.484), ('mix', 17.375)]:
        print(f'[gate LLv {m}] mean {P[m].mean():.3f} (want {want})')
    print(f'\n[C] real LOLv2 (n={n_img} scenes, 5 seeds):')
    out_c = {}
    for m in ('cpl', 'mix'):
        D = P[m] - P['dec']                       # (5, n_img) paired per seed
        per_img = D.mean(0)                       # seed-averaged per-image difference
        obs = per_img.mean()
        bs = np.array([per_img[rng.integers(0, n_img, n_img)].mean() for _ in range(a.boot)])
        lo, hi = pct(bs)
        w = stats.wilcoxon(per_img)
        win = (per_img > 0).mean()
        print(f'    {m}-dec: {obs:+.3f} dB, image bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}], '
              f'win-rate {win:.2f}, Wilcoxon (n={n_img}) p={w.pvalue:.2e}')
        out_c[m] = dict(diff=obs, ci=[lo, hi], win=win, wilcoxon_p=float(w.pvalue))

    json.dump(dict(A=dict(mean=float(obs_a.mean()), scene_ci=[float(lo_a), float(hi_a)],
                          hier_ci=[float(lo_h), float(hi_h)]),
                   B={k: dict(gap=float(v['gap']), ci=[float(v['ci'][0]), float(v['ci'][1])])
                      for k, v in out_b.items()},
                   C=out_c),
              open(os.path.join(a.dir, 'bootstrap_summary.json'), 'w'))
    print('\nsaved', os.path.join(a.dir, 'bootstrap_summary.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='analysis')
    p.add_argument('--boot', type=int, default=5000)
    main(p.parse_args())

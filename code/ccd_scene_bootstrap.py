"""
ccd_scene_bootstrap.py - scene-level and seed x scene hierarchical bootstrap for the
restoration headlines (review item: quantify scene/dataset uncertainty, not only training
stochasticity).

Reads the per-scene evaluations produced by eval_per_scene.py / eval_real_lol_per.py:
  --track pg (default)  per_scene_PG.json   composite 2x2 track with Poisson-Gaussian noise,
                        checkpoints LAM_s{s}_{l000,l100,mix} (lambda=0 == decoupled, 1 == coupled)
  --track legacy        per_scene_FW.json   earlier sigma0/L composite runs FW_s{s}_{dec,cpl,mix}
  per_image_LLv.json    variance-matched real LOLv2 track (100 real pairs x 15 models)

Reported quantities:
  A. composite mixed worst-case minus best-specialist worst-case:
     scene-level bootstrap CI (seeds fixed) and hierarchical seed x scene bootstrap CI.
  B. specialisation gaps (own-structure minus cross-structure PSNR per specialist):
     scene-level bootstrap CIs.
  C. real LOLv2 coupled-decoupled and mixed-decoupled PSNR: image-level paired bootstrap
     CI, per-image win rate, and a Wilcoxon signed-rank test over the 100 scenes
     (seed-averaged per-image differences), which measures scene-level uncertainty at
     n=100 rather than seed-level n=5.
Validation gates: full-sample means must reproduce the aggregate tables (TAG.json values found in
--dir for the pg track; the published numbers for the legacy track).

Run:  python code/ccd_scene_bootstrap.py --track pg --dir analysis/pg --llv_dir analysis --boot 5000
"""
import os, json, argparse
import numpy as np
from scipy import stats

MODES = ['dec', 'cpl', 'mix']
SEEDS = range(5)
PG_TAG = {'dec': 'l000', 'cpl': 'l100', 'mix': 'mix'}
TRACKS = {
    'pg': dict(file='per_scene_PG.json', tag=lambda m, s: f'LAM_s{s}_{PG_TAG[m]}',
               out='bootstrap_summary_pg.json', want=None),
    'legacy': dict(file='per_scene_FW.json', tag=lambda m, s: f'FW_s{s}_{m}',
                   out='bootstrap_summary.json',
                   want={'dec': (18.48, 22.16), 'cpl': (22.82, 17.63), 'mix': (21.96, 21.80)}),
}


def cell(res, tag, key):
    return np.asarray(res[tag][key]['psnr'], float)


def fw_matrix(res, combos, tag_of):
    """M[mode][seed][test_structure] -> per-scene matrix (n_combo, n_scene)."""
    M = {}
    for m in MODES:
        M[m] = {}
        for s in SEEDS:
            tag = tag_of(m, s)
            M[m][s] = {t: np.stack([cell(res, tag, f'{c}|{t}') for c in combos])
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


def wanted(a, tr):
    if tr['want'] is not None:
        return tr['want']
    want = {}
    for m in MODES:
        c, d = [], []
        for s in SEEDS:
            p = os.path.join(a.dir, tr['tag'](m, s) + '.json')
            if os.path.exists(p):
                j = json.load(open(p)); c.append(j['coupled_test_PSNR']); d.append(j['decoupled_test_PSNR'])
        want[m] = (round(float(np.mean(c)), 2), round(float(np.mean(d)), 2)) if len(c) == 5 else (None, None)
    return want


def main(a):
    tr = TRACKS[a.track]
    rng = np.random.default_rng(0)
    fw = json.load(open(os.path.join(a.dir, tr['file'])))
    combos = fw['combos']; res = fw['results']
    n_scene = len(fw['scenes'])
    M = fw_matrix(res, combos, tr['tag'])

    # ---- validation gate: reproduce the aggregate tables ----
    full = np.arange(n_scene)
    want = wanted(a, tr)
    for m in MODES:
        c = np.mean([M[m][s]['coupled'][:, full].mean() for s in SEEDS])
        d = np.mean([M[m][s]['decoupled'][:, full].mean() for s in SEEDS])
        print(f'[gate {a.track} {m}] coupled-test {c:.2f} (want {want[m][0]})  '
              f'decoupled-test {d:.2f} (want {want[m][1]})')

    obs_a = fw_headline(M, full)
    print(f'\n[A] composite mixed_wc - best_specialist_wc: mean {obs_a.mean():+.3f} dB '
          f'(per-seed {np.round(obs_a, 3)})')
    bs = np.array([fw_headline(M, rng.integers(0, n_scene, n_scene)).mean() for _ in range(a.boot)])
    lo_a, hi_a = pct(bs)
    print(f'    scene bootstrap ({a.boot}x): 95% CI [{lo_a:+.3f}, {hi_a:+.3f}] dB')
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
        bsg = []
        for _ in range(a.boot):
            idx = rng.integers(0, n_scene, n_scene)
            bsg.append(np.mean([M[m][s][own][:, idx].mean() - M[m][s][cross][:, idx].mean() for s in SEEDS]))
        lo, hi = pct(np.array(bsg))
        win = float(np.mean([(M[m][s][own].mean(0) > M[m][s][cross].mean(0)).mean() for s in SEEDS]))
        print(f'    {m}-trained: {gap_obs:+.3f} dB, scene bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}], '
              f'scenes with own > cross (seed-averaged): {win:.2f}')
        out_b[m] = dict(gap=float(gap_obs), ci=[float(lo), float(hi)], scene_win=win)

    # ---- C. real LOLv2 ----
    out_c = {}
    llv = os.path.join(a.llv_dir or a.dir, 'per_image_LLv.json')
    if os.path.exists(llv):
        lv = json.load(open(llv))
        R = lv['results']; n_img = len(lv['images'])
        P = {m: np.stack([np.asarray(R[f'LLv_{m}_s{s}']['psnr'], float) for s in SEEDS]) for m in MODES}
        for m, want_l in [('dec', 17.132), ('cpl', 17.484), ('mix', 17.375)]:
            print(f'[gate LLv {m}] mean {P[m].mean():.3f} (want {want_l})')
        print(f'\n[C] real LOLv2 (n={n_img} scenes, 5 seeds):')
        for m in ('cpl', 'mix'):
            D = P[m] - P['dec']
            per_img = D.mean(0)
            obs = per_img.mean()
            bsl = np.array([per_img[rng.integers(0, n_img, n_img)].mean() for _ in range(a.boot)])
            lo, hi = pct(bsl)
            w = stats.wilcoxon(per_img)
            win = (per_img > 0).mean()
            print(f'    {m}-dec: {obs:+.3f} dB, image bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}], '
                  f'win-rate {win:.2f}, Wilcoxon (n={n_img}) p={w.pvalue:.2e}')
            out_c[m] = dict(diff=float(obs), ci=[float(lo), float(hi)], win=float(win),
                            wilcoxon_p=float(w.pvalue))
    else:
        print(f'\n[C] skipped: {llv} not found')

    p = os.path.join(a.dir, tr['out'])
    json.dump(dict(track=a.track,
                   A=dict(mean=float(obs_a.mean()), per_seed=[float(x) for x in obs_a],
                          scene_ci=[float(lo_a), float(hi_a)], hier_ci=[float(lo_h), float(hi_h)]),
                   B=out_b, C=out_c), open(p, 'w'), indent=1)
    print('\nsaved', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', choices=list(TRACKS), default='pg')
    ap.add_argument('--dir', default='analysis')
    ap.add_argument('--llv_dir', default='')
    ap.add_argument('--boot', type=int, default=5000)
    main(ap.parse_args())

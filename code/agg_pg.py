"""
agg_pg.py - aggregate the Poisson-Gaussian composite campaign into the numbers reported in the
paper. Run in the server working dir, or locally with RESULTS_DIR pointing at the result JSONs.

Reads (missing files are skipped, so partial campaigns can be inspected):
  LAM_s{s}_{l000,l100,mix}.json          NAFNet-w32 2x2 matrix (lambda=0 == decoupled, lambda=1 == coupled)
  FW_s{s}_{dec,cpl,mix}.json             published legacy sigma0/L runs, printed for reference only
  F2cnPG_s{s}.json                       CoupleNet trained coupled (NAFNet baseline = LAM_s{s}_l100)
  AGpg_n{80,20}.json, CNpg_n{80,20}.json F2 scale ablation (single runs)
  FW64pg_{dec,cpl}.json, FRpg_{dec,cpl}.json   architecture single runs
  ORpg_{dec,cpl,mix}_s{s}.json           OneRestore retrained on CCD
  ORcdd11_pg.json, BLpg_{promptir,adair}.json  off-the-shelf restorers
  ablate_pg_{cpl,dec}.json               per-cross-term test-time ablation
  lam_surface_pg_s{s}.json               train-lambda x test-lambda surface (eval_lambda_surface.py --pgnoise)
Writes pg_summary.json.

Seed-level statistics follow stats_ci.py: seed-paired mean difference, t-based 95% CI, two-sided
paired t-test, exact sign-flip permutation p over all 2^n sign patterns, and the number of seeds
with a positive difference (direction consistency).
"""
import os, json
from itertools import product
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_RESULTS = os.path.join(_HERE, '..', 'latex', 'results')
RES = os.environ.get('RESULTS_DIR', _REPO_RESULTS if os.path.isdir(_REPO_RESULTS) else '.')
SEEDS = [0, 1, 2, 3, 4]
MODE_TAG = {'dec': 'l000', 'cpl': 'l100', 'mix': 'mix'}
FIXED = [('l000', 0.0), ('l025', 0.25), ('l050', 0.5), ('l075', 0.75), ('l100', 1.0)]


def load(tag):
    p = os.path.join(RES, tag + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def paired(d):
    d = np.asarray(d, float); n = len(d)
    if n < 2:
        return dict(n=n, mean=float(d.mean()) if n else float('nan'),
                    per_seed=[round(float(x), 3) for x in d])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    cnt = sum(1 for s in product([1, -1], repeat=n) if abs((d * np.array(s)).mean()) >= abs(m) - 1e-12)
    return dict(n=n, mean=float(m), ci=[float(m - tc * se), float(m + tc * se)],
                t_p=float(stats.ttest_1samp(d, 0.0).pvalue), perm_p=cnt / 2 ** n,
                n_pos=int((d > 0).sum()), per_seed=[round(float(x), 3) for x in d])


def fmt(r):
    if 'ci' not in r:
        return f"mean {r['mean']:+.3f} (n={r['n']})"
    return (f"{r['mean']:+.3f} dB  CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  t-p={r['t_p']:.2g}  "
            f"perm-p={r['perm_p']:.3g}  positive {r['n_pos']}/{r['n']}")


def msd(a):
    a = np.asarray(a, float)
    return [float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0]


def matrix(tag_of, label, out):
    """Decoupled/coupled/mixed x coupled-test/decoupled-test table; tag_of(mode, seed) -> tag."""
    V = {}
    for m in ('dec', 'cpl', 'mix'):
        V[m] = {}
        for s in SEEDS:
            d = load(tag_of(m, s))
            if d:
                V[m][s] = (d['coupled_test_PSNR'], d['decoupled_test_PSNR'])
    print(f'\n=== {label} ===')
    print(f'{"train":5s} {"n":>2s} {"coupled-test":>14s} {"decoupled-test":>15s} {"worst":>14s} {"cross-avg":>9s}')
    tab = {}
    for m in ('dec', 'cpl', 'mix'):
        if not V[m]:
            continue
        ss = sorted(V[m])
        c = np.array([V[m][s][0] for s in ss]); d = np.array([V[m][s][1] for s in ss])
        w = np.minimum(c, d); a = (c + d) / 2
        tab[m] = dict(seeds=ss, cpl=c.tolist(), dec=d.tolist(), cpl_msd=msd(c), dec_msd=msd(d),
                      worst_msd=msd(w), cross=float(a.mean()))
        print(f'{m:5s} {len(ss):2d} {c.mean():8.2f}+-{msd(c)[1]:.2f} {d.mean():9.2f}+-{msd(d)[1]:.2f} '
              f'{w.mean():8.2f}+-{msd(w)[1]:.2f} {a.mean():9.2f}')
    res = dict(table=tab)
    if all(V[m] for m in V):
        common = sorted(set(V['dec']) & set(V['cpl']) & set(V['mix']))
        if common:
            C = {m: np.array([V[m][s][0] for s in common]) for m in V}
            D = {m: np.array([V[m][s][1] for s in common]) for m in V}
            W = {m: np.minimum(C[m], D[m]) for m in V}
            A = {m: (C[m] + D[m]) / 2 for m in V}
            res.update(
                seeds_paired=common,
                cure_worst=paired(W['mix'] - np.maximum(W['dec'], W['cpl'])),
                cure_cross=paired(A['mix'] - np.maximum(A['dec'], A['cpl'])),
                tax_cpl_home=paired(C['mix'] - C['cpl']),
                tax_dec_home=paired(D['mix'] - D['dec']),
                gap_dec_trained=paired(D['dec'] - C['dec']),
                gap_cpl_trained=paired(C['cpl'] - D['cpl']),
                cpl_test_gap=paired(C['cpl'] - C['dec']),
                dec_test_gap=paired(D['dec'] - D['cpl']),
                cross_cpl_minus_dec=paired(A['cpl'] - A['dec']))
            print(f'  paired over seeds {common}')
            for k in ('cure_worst', 'cure_cross', 'tax_cpl_home', 'tax_dec_home', 'gap_dec_trained',
                      'gap_cpl_trained', 'cpl_test_gap', 'dec_test_gap', 'cross_cpl_minus_dec'):
                print(f'  {k:20s} {fmt(res[k])}')
    out[label] = res


def f2(out):
    print('\n=== F2: CoupleNet vs NAFNet-w32 (coupled training, coupled-test PSNR) ===')
    res = {}
    common = [s for s in SEEDS if load(f'LAM_s{s}_l100') and load(f'F2cnPG_s{s}')]
    if common:
        a = np.array([load(f'LAM_s{s}_l100')['coupled_test_PSNR'] for s in common])
        c = np.array([load(f'F2cnPG_s{s}')['coupled_test_PSNR'] for s in common])
        res['full'] = dict(seeds=common, nafnet_msd=msd(a), couplenet_msd=msd(c), diff=paired(c - a))
        print(f'  full scale, seeds {common}: NAFNet {a.mean():.2f}+-{msd(a)[1]:.2f}  '
              f'CoupleNet {c.mean():.2f}+-{msd(c)[1]:.2f}  diff {fmt(res["full"]["diff"])}')
    for n in (80, 20):
        g, k = load(f'AGpg_n{n}'), load(f'CNpg_n{n}')
        if g and k:
            res[f'n{n}'] = dict(nafnet=g['coupled_test_PSNR'], couplenet=k['coupled_test_PSNR'],
                                diff=k['coupled_test_PSNR'] - g['coupled_test_PSNR'])
            print(f'  scale {n} train: NAFNet {g["coupled_test_PSNR"]:.2f}  CoupleNet {k["coupled_test_PSNR"]:.2f}  '
                  f'diff {res[f"n{n}"]["diff"]:+.2f}')
    out['F2'] = res


def singles(out):
    print('\n=== architecture single runs (coupled-test / decoupled-test) ===')
    res = {}
    for name, pre in (('NAFNet-w64', 'FW64pg'), ('Restormer-w32', 'FRpg')):
        d, c = load(pre + '_dec'), load(pre + '_cpl')
        if d and c:
            res[name] = dict(dec=[d['coupled_test_PSNR'], d['decoupled_test_PSNR']],
                             cpl=[c['coupled_test_PSNR'], c['decoupled_test_PSNR']],
                             gap_dec_trained=d['decoupled_test_PSNR'] - d['coupled_test_PSNR'],
                             gap_cpl_trained=c['coupled_test_PSNR'] - c['decoupled_test_PSNR'])
            print(f'  {name:14s} dec-trained {d["coupled_test_PSNR"]:.2f}/{d["decoupled_test_PSNR"]:.2f}  '
                  f'cpl-trained {c["coupled_test_PSNR"]:.2f}/{c["decoupled_test_PSNR"]:.2f}  '
                  f'own-minus-cross {res[name]["gap_dec_trained"]:+.2f} / {res[name]["gap_cpl_trained"]:+.2f}')
    out['singles'] = res


def off_the_shelf(out):
    print('\n=== off-the-shelf restorers (coupled-test / decoupled-test / cross-avg / cpl-dec) ===')
    res = {}
    for tag in ('ORcdd11_pg', 'BLpg_promptir', 'BLpg_adair'):
        d = load(tag)
        if not d:
            continue
        c, e = d['coupled_test_PSNR'], d['decoupled_test_PSNR']
        res[tag] = dict(cpl=c, dec=e, cross=(c + e) / 2, gap=c - e)
        print(f'  {tag:14s} {c:6.2f} {e:6.2f} {(c + e) / 2:6.2f} {c - e:+.2f}')
        if 'cells' in d:
            res[tag]['cells'] = {k: {kk: v[kk] for kk in ('in_PSNR', 'PSNR', 'SSIM', 'n')} for k, v in d['cells'].items()}
            for k, v in sorted(d['cells'].items()):
                print(f'      {k:26s} input {v["in_PSNR"]:6.2f}  output {v["PSNR"]:6.2f}  SSIM {v["SSIM"]:.3f}  n={v["n"]}')
    out['off_the_shelf'] = res


def ablation(out):
    print('\n=== per-cross-term test-time ablation (PSNR; gain over fully decoupled render) ===')
    res = {}
    for m in ('cpl', 'dec'):
        d = load(f'ablate_pg_{m}')
        if d:
            res[m] = {k: d[k] for k in ('none', 'rain', 'noise', 'haze', 'all')}
            print(f'  {m}-trained: ' + '  '.join(f'{k} {d[k]:.2f} ({d[k] - d["none"]:+.2f})'
                                                 for k in ('none', 'rain', 'noise', 'haze', 'all')))
    out['ablation'] = res


def surface(out):
    S, tls = {}, None
    for s in SEEDS:
        p = os.path.join(RES, f'lam_surface_pg_s{s}.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); tls = d['test_lams']; R = d['results']
        rows = {}
        for r in [f for f, _ in FIXED] + ['rand', 'mix']:
            tag = f'LAM_s{s}_{r}'
            if tag in R:
                rows[r] = np.array([np.mean([v for k, c in R[tag].items() if k.endswith(f'|lam{tl:g}')
                                             for v in c['psnr']]) for tl in tls])
        S[s] = rows
    if not S:
        return
    seeds = sorted(S)
    names = [r for r in [f for f, _ in FIXED] + ['rand', 'mix'] if all(r in S[s] for s in seeds)]
    M = {r: np.mean([S[s][r] for s in seeds], axis=0) for r in names}
    print(f'\n=== lambda surface, mean over seeds {seeds} (rows = training lambda, cols = test lambda) ===')
    print('train\\test ' + ''.join(f'{tl:>8g}' for tl in tls))
    for r in names:
        print(f'{r:10s} ' + ''.join(f'{v:8.2f}' for v in M[r]))
    fixed = [(r, l) for r, l in FIXED if r in names]
    hits = {s: sum(int(tls[int(np.argmax(S[s][r]))] == l) for r, l in fixed) for s in seeds}
    argmax_mean = [tls[int(np.argmax(M[r]))] for r, _ in fixed]
    by_d = {}
    for r, l in fixed:
        for j, tl in enumerate(tls):
            by_d.setdefault(round(abs(l - tl), 2), []).append(M[r][j])
    deg = {k: float(np.mean(v)) for k, v in sorted(by_d.items())}
    print(f'  best test lambda == training lambda, per seed: {hits} (of {len(fixed)} rows); '
          f'argmax of seed-mean rows: {argmax_mean}')
    print('  mean PSNR by |train-test lambda|: ' + '  '.join(f'{k:g}: {v:.2f}' for k, v in deg.items()))
    res = dict(seeds=seeds, test_lams=tls, mean_surface={r: M[r].tolist() for r in names},
               diag_hits=hits, argmax_seed_mean=argmax_mean, by_abs_dlam=deg)
    if 'rand' in names and 'mix' in names:
        res['rand_minus_mix'] = {f'{tl:g}': paired([S[s]['rand'][j] - S[s]['mix'][j] for s in seeds])
                                 for j, tl in enumerate(tls)}
        res['rand_minus_mix_avg'] = paired([S[s]['rand'].mean() - S[s]['mix'].mean() for s in seeds])
        best_spec_worst = [max(S[s][r].min() for r, _ in fixed) for s in seeds]
        res['worst'] = dict(rand=msd([S[s]['rand'].min() for s in seeds]),
                            mix=msd([S[s]['mix'].min() for s in seeds]),
                            best_fixed=msd(best_spec_worst))
        res['worst_rand_minus_mix'] = paired([S[s]['rand'].min() - S[s]['mix'].min() for s in seeds])
        res['worst_rand_minus_best_fixed'] = paired([S[s]['rand'].min() - b for s, b in zip(seeds, best_spec_worst)])
        res['rand_cost_vs_own_specialist'] = {
            f'{l:g}': paired([S[s]['rand'][tls.index(l)] - S[s][r][tls.index(l)] for s in seeds])
            for r, l in fixed if l in tls}
        print('  rand - mix per test lambda: ' + '  '.join(
            f'{k}: {v["mean"]:+.2f}' for k, v in res['rand_minus_mix'].items()))
        print(f'  rand - mix averaged over test lambdas: {fmt(res["rand_minus_mix_avg"])}')
        print(f'  worst case over test lambdas: rand {res["worst"]["rand"][0]:.2f}  mix {res["worst"]["mix"][0]:.2f}  '
              f'best fixed-lambda specialist {res["worst"]["best_fixed"][0]:.2f}')
        print(f'  worst rand - worst mix: {fmt(res["worst_rand_minus_mix"])}')
        print(f'  worst rand - best fixed worst: {fmt(res["worst_rand_minus_best_fixed"])}')
        print('  rand minus the specialist trained at that lambda: ' + '  '.join(
            f'{k}: {v["mean"]:+.2f}' for k, v in res['rand_cost_vs_own_specialist'].items()))
    out['surface'] = res


def main():
    out = {}
    matrix(lambda m, s: f'LAM_s{s}_{MODE_TAG[m]}', 'NAFNet-w32 composite, Poisson-Gaussian', out)
    matrix(lambda m, s: f'FW_s{s}_{m}', 'NAFNet-w32 composite, legacy sigma0/L (published, reference)', out)
    matrix(lambda m, s: f'ORpg_{m}_s{s}', 'OneRestore retrained on CCD, Poisson-Gaussian', out)
    f2(out)
    singles(out)
    off_the_shelf(out)
    ablation(out)
    surface(out)
    p = os.path.join(RES, 'pg_summary.json')
    json.dump(out, open(p, 'w'), indent=1, default=float)
    print('\nsaved', p)


if __name__ == '__main__':
    main()

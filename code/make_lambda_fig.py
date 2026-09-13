"""
make_lambda_fig.py - the figure that answers the reviewers' central objection (AE points 1 and 5,
R2 Q2): if coupling structure is a genuine domain axis rather than two hand-picked endpoints,
restoration accuracy must degrade smoothly with the train/test mismatch in the continuous
coupling strength lambda.

Panel a: train-lambda x test-lambda PSNR heat map (mean over seeds); red box = best test lambda
         per row, with the number of seeds in which that cell is the row maximum.
Panel b: one curve per training distribution over test lambda (mean +- std over seeds): the five
         fixed-lambda specialists, continuous randomisation lambda ~ U[0,1], and the binary
         50/50 mix.

Reads the lam_surface_pg_s*.json files written by eval_lambda_surface.py --pgnoise (one seed per
file). Run:
  python make_lambda_fig.py --surfaces "analysis/pg/lam_surface_pg_s*.json" --outdir figs_new
"""
import os, re, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt

FIXED = [('l000', 0.0), ('l025', 0.25), ('l050', 0.5), ('l075', 0.75), ('l100', 1.0)]
RANDOM = [('rand', 'randomised $\\lambda \\sim U[0,1]$'), ('mix', 'binary 50/50 mix')]


def load_rows(path):
    d = json.load(open(path)); tls = d['test_lams']; res = d['results']
    seeds = {int(m.group(1)) for m in (re.match(r'LAM_s(\d+)_', t) for t in res) if m}
    assert len(seeds) == 1, f'{path}: expected one seed per file, got {sorted(seeds)}'
    s = seeds.pop()
    rows = {}
    for r in [f for f, _ in FIXED] + [f for f, _ in RANDOM]:
        tag = f'LAM_s{s}_{r}'
        if tag in res:
            rows[r] = np.array([np.mean([v for k, c in res[tag].items() if k.endswith('|lam%g' % tl)
                                         for v in c['psnr']]) for tl in tls])
    return s, tls, rows


def main(a):
    files = sorted(glob.glob(a.surfaces))
    assert files, 'no surface files match ' + a.surfaces
    per_seed, tls = {}, None
    for f in files:
        s, t, rows = load_rows(f)
        tls = t if tls is None else tls
        assert t == tls, 'test lambdas differ between surface files'
        per_seed[s] = rows
    seeds = sorted(per_seed); n = len(seeds)
    names = [r for r in [f for f, _ in FIXED] + [f for f, _ in RANDOM]
             if all(r in per_seed[s] for s in seeds)]
    stack = {r: np.stack([per_seed[s][r] for s in seeds]) for r in names}     # (n_seed, n_test)
    mean = {r: stack[r].mean(0) for r in names}
    sd = {r: (stack[r].std(0, ddof=1) if n > 1 else np.zeros(len(tls))) for r in names}
    fixed = [(r, l) for r, l in FIXED if r in names]
    M = np.array([mean[r] for r, _ in fixed])

    fig = plt.figure(figsize=(9.0, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.32)

    # ---------------- panel a: heat map ----------------
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(M, cmap='Greys_r', aspect='auto')
    lo, hi = np.nanmin(M), np.nanmax(M)
    wins_row = []
    for i, (r, l) in enumerate(fixed):
        best = int(np.nanargmax(M[i]))
        wins = sum(int(np.argmax(stack[r][k]) == best) for k in range(n))
        wins_row.append(wins)
        for j in range(M.shape[1]):
            shade = (M[i, j] - lo) / (hi - lo + 1e-9)
            ax.text(j, i, '%.1f' % M[i, j], ha='center', va='center', fontsize=7.4,
                    color='black' if shade > 0.5 else 'white',
                    weight='bold' if j == best else 'normal')
        ax.add_patch(plt.Rectangle((best - 0.5, i - 0.5), 1, 1, fill=False, ec='#d62728', lw=1.6))
        if n > 1:
            ax.text(best + 0.44, i + 0.44, '%d/%d' % (wins, n), ha='right', va='bottom',
                    fontsize=5.4, color='#d62728')
    ax.set_xticks(range(len(tls)), ['%g' % t for t in tls], fontsize=8)
    ax.set_yticks(range(len(fixed)), ['%g' % l for _, l in fixed], fontsize=8)
    ax.set_xlabel('test coupling strength $\\lambda$', fontsize=8.6)
    ax.set_ylabel('training coupling strength $\\lambda$', fontsize=8.6)
    scope = 'mean of %d seeds' % n if n > 1 else 'seed %d' % seeds[0]
    ax.set_title('a   generalisation surface (PSNR, dB; %s)\nred box = best test $\\lambda$ per row'
                 % scope, fontsize=8.4, pad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(tls), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(fixed), 1), minor=True)
    ax.grid(which='minor', color='white', lw=1.2)
    ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)

    # ---------------- panel b: curves ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    cols = plt.cm.viridis(np.linspace(0.08, 0.92, len(fixed)))
    eb = dict(capsize=2.0, elinewidth=0.8) if n > 1 else {}
    for (r, l), col in zip(fixed, cols):
        ax2.errorbar(tls, mean[r], yerr=sd[r] if n > 1 else None, fmt='-o', ms=4.0, lw=1.3,
                     color=col, label='train $\\lambda = %g$' % l, **eb)
    styles = {'rand': ('--', 's', '#d62728'), 'mix': (':', '^', '#333333')}
    for r, lab in RANDOM:
        if r in names:
            ls, mk, col = styles[r]
            ax2.errorbar(tls, mean[r], yerr=sd[r] if n > 1 else None, fmt=mk, ls=ls, ms=4.4, lw=1.5,
                         color=col, label=lab, **eb)
    ax2.set_xlabel('test coupling strength $\\lambda$', fontsize=8.6)
    ax2.set_ylabel('PSNR (dB)', fontsize=8.6)
    ax2.set_title('b   each fixed-$\\lambda$ specialist peaks at its own $\\lambda$;\n'
                  'randomising over $\\lambda$ flattens the curve', fontsize=8.4, pad=6)
    ax2.grid(lw=0.4, color='#dddddd')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(fontsize=6.6, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax2.set_xticks(tls, ['%g' % t for t in tls], fontsize=8)
    ax2.tick_params(labelsize=8)

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, a.name)
    fig.savefig(out, bbox_inches='tight')
    print('saved ' + out + '  seeds=%s' % seeds)

    # ---------------- printed summary ----------------
    print('train\\test ' + ''.join('%8g' % t for t in tls))
    for r in names:
        print('%-10s ' % r + ''.join('%8.2f' % v for v in mean[r]))
    print('row maximum on the diagonal, seeds agreeing per row:', dict(zip([l for _, l in fixed], wins_row)))
    for r, lab in RANDOM:
        if r in names:
            print('%-5s worst-case %.2f dB, mean %.2f dB' % (r, mean[r].min(), mean[r].mean()))
    print('best fixed-lambda specialist worst case: %.2f dB' % max(np.min(mean[r]) for r, _ in fixed))


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--surfaces', default='analysis/pg/lam_surface_pg_s*.json')
    P.add_argument('--outdir', default='figs_new')
    P.add_argument('--name', default='fig_lambda.pdf')
    main(P.parse_args())

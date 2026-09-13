"""
make_reviewer_figs.py - two figures requested by Reviewer #2 (Q6):
  (1) fig_matrix.pdf  - the train x test protocol as a heat map instead of grouped bars,
                        so the 2x2 domain relationship is readable at a glance.
  (2) fig_exdark_seeds.pdf - the ExDark bar chart with every individual seed overlaid,
                        so statistical stability is visible rather than summarised.

--track pg (default) reads the Poisson-Gaussian composite checkpoints' results
LAM_s{s}_{l000,l100,mix}.json (lambda=0 == decoupled, lambda=1 == coupled) from --pg_results;
--track legacy reads the earlier sigma0/L runs FW_s{s}_{dec,cpl,mix}.json from --results.
The ExDark JSON (exd_*) is always read from --results. Run:
  python make_reviewer_figs.py --results ../latex/results --pg_results ../analysis/pg --outdir figs_new/pg
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt

SEEDS = range(5)
PG_TAG = {'dec': 'l000', 'cpl': 'l100', 'mix': 'mix'}


def load(rdir, tag):
    p = os.path.join(rdir, tag + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def matrix_fig(rdir, out, tag_of):
    """Rows = training structure, cols = test structure; cell = mean PSNR over seeds."""
    rows = [('decoupled', 'dec'), ('coupled', 'cpl'), ('mixed', 'mix')]
    M = np.full((len(rows), 2), np.nan)
    S = np.full((len(rows), 2), np.nan)
    n_seen = []
    for i, (_, ab) in enumerate(rows):
        dec_v, cpl_v = [], []
        for s in SEEDS:
            d = load(rdir, tag_of(ab, s))
            if d:
                dec_v.append(d['decoupled_test_PSNR'])
                cpl_v.append(d['coupled_test_PSNR'])
        n_seen.append(len(dec_v))
        if dec_v:
            M[i, 0], S[i, 0] = np.mean(dec_v), (np.std(dec_v, ddof=1) if len(dec_v) > 1 else 0.0)
            M[i, 1], S[i, 1] = np.mean(cpl_v), (np.std(cpl_v, ddof=1) if len(cpl_v) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    vmin, vmax = np.nanmin(M) - 0.4, np.nanmax(M) + 0.2
    im = ax.imshow(M, cmap='Greys_r', vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            shade = (M[i, j] - vmin) / (vmax - vmin)
            ax.text(j, i, f'{M[i, j]:.2f}\n$\\pm${S[i, j]:.2f}', ha='center', va='center',
                    fontsize=9.0, color='black' if shade > 0.5 else 'white', linespacing=1.25)
    ax.set_xticks([0, 1], ['decoupled', 'coupled'], fontsize=8.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel('test coupling structure', fontsize=9)
    ax.set_ylabel('training coupling structure', fontsize=9)
    ax.set_title('PSNR (dB), mean $\\pm$ std over %d seeds' % min(n_seen), fontsize=8.8, pad=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which='minor', color='white', lw=1.4)
    ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + out + '  seeds per row: %s' % n_seen)
    return M


def exdark_seeds_fig(rdir, out):
    """All five arms with per-seed points overlaid on the seed mean."""
    arms = [('off-the-shelf\n(no fine-tune)', 'exd_offtheshelf', '#8c8c8c'),
            ('clean-COCO\nfine-tune', 'exd_cleanft_s', '#b8b8b8'),
            ('decoupled', 'exd_dec_s', '#d65f5f'),
            ('mixed', 'exd_mix_s', '#55a868'),
            ('coupled', 'exd_cpl_s', '#4c72b0')]
    means, errs, pts = [], [], []
    for _, key, _ in arms:
        if key.endswith('_s'):
            v = [load(rdir, f'{key}{s}')['ap_exdark'] for s in SEEDS
                 if load(rdir, f'{key}{s}')]
        else:
            d = load(rdir, key)
            v = [d['ap_exdark']] if d else []
        means.append(np.mean(v))
        errs.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
        pts.append(v)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.array([0.0, 1.0, 2.4, 3.4, 4.4])
    cols = [a[2] for a in arms]
    ax.bar(x, means, 0.72, yerr=errs, capsize=4, color=cols, edgecolor='#333', lw=0.6, zorder=2)
    rng = np.random.default_rng(0)
    for xi, v in zip(x, pts):
        if len(v) > 1:
            jit = rng.uniform(-0.17, 0.17, len(v))
            ax.plot(xi + jit, v, 'o', ms=3.6, mfc='white', mec='#111', mew=0.8, zorder=4,
                    linestyle='none')
    ax.axvline(1.7, color='k', lw=0.8, ls=':')
    ax.text(0.5, 0.325, 'no degradation', ha='center', fontsize=8, color='#555')
    ax.text(3.4, 0.325, 'synthetic-degradation fine-tune', ha='center', fontsize=8, color='#555')
    ax.annotate('$+0.062$ AP', xy=(4.4, means[4] + 0.016), xytext=(2.95, 0.175),
                fontsize=8.5, arrowprops=dict(arrowstyle='->', lw=0.9, color='#333'))
    ax.set_xticks(x, [a[0] for a in arms], fontsize=8)
    ax.set_ylabel('ExDark AP (real ground truth)', fontsize=9)
    ax.set_ylim(0, 0.35)
    ax.set_title('Real low-light detection: bars = seed mean, circles = individual seeds',
                 fontsize=8.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', lw=0.4, color='#ddd', zorder=0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + out)
    for (lab, _, _), m, v in zip(arms, means, pts):
        print(f'  {lab.splitlines()[0]:<16} mean={m:.4f}  seeds={[round(z, 4) for z in v]}')


def main(a):
    os.makedirs(a.outdir, exist_ok=True)
    if a.track == 'pg':
        tag_of, mdir = (lambda ab, s: f'LAM_s{s}_{PG_TAG[ab]}'), a.pg_results
    else:
        tag_of, mdir = (lambda ab, s: f'FW_s{s}_{ab}'), a.results
    M = matrix_fig(mdir, os.path.join(a.outdir, 'fig_matrix.pdf'), tag_of)
    print('matrix (rows dec/cpl/mix x cols dec-test/cpl-test):')
    print(np.round(M, 2))
    exdark_seeds_fig(a.results, os.path.join(a.outdir, 'fig_exdark_seeds.pdf'))


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--track', choices=['pg', 'legacy'], default='pg')
    P.add_argument('--results', default='../latex/results')
    P.add_argument('--pg_results', default='../analysis/pg')
    P.add_argument('--outdir', default='figs_new/pg')
    main(P.parse_args())

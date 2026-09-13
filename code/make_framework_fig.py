"""
make_framework_fig.py - overview figure requested by all three reviewers (R1 Q4, R2 Q5, R3 Q1):
CCDM synthesis -> matched-pair construction -> continuous coupling axis -> 2x2 train x test
protocol -> what the protocol measures. Outcome numbers are those of the Poisson-Gaussian campaign
(five seeds, NAFNet-w32; ExDark detection).

Greyscale-safe (hue carries no information), vector PDF, single-column friendly.
Run: python make_framework_fig.py --out figs_new/pg/fig_framework.pdf
"""
import argparse
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

DEC = '#e9e9e9'
CPL = '#bcbcbc'
MIX = '#d6d6d6'
ACC = '#333333'


def box(ax, x, y, w, h, text, fc='white', ec=ACC, fs=7.5, lw=0.9, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, weight=weight, zorder=3, linespacing=1.4)


def arrow(ax, p1, p2, lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=9,
                                 lw=lw, color=ACC, zorder=1,
                                 connectionstyle='arc3,rad=' + str(rad),
                                 shrinkA=1.5, shrinkB=1.5))


def main(out):
    fig, ax = plt.subplots(figsize=(6.6, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.5)
    ax.axis('off')

    # ---------------- a: synthesis ----------------
    ax.text(0.05, 11.25, 'a   Physics-coupled synthesis (CCDM)', fontsize=8.8, weight='bold')
    box(ax, 0.3, 10.3, 2.5, 0.68, 'clean image $J$\n(public DIV2K)')
    box(ax, 3.2, 10.3, 2.5, 0.68, 'monocular depth $d$\n(MiDaS)')
    box(ax, 6.4, 10.3, 3.3, 0.68,
        'illumination $L$ (LIME)\ntransmission $t = e^{-\\beta d}$')
    arrow(ax, (5.7, 10.64), (6.4, 10.64))
    ax.text(8.05, 10.06, 'shared latent fields', fontsize=6.8, ha='center',
            style='italic', color='#555')

    for x, fc, name, lam in [(0.3, DEC, 'DECOUPLED   (field default)', '$\\lambda = 0$'),
                             (5.2, CPL, 'COUPLED   (ours)', '$\\lambda = 1$')]:
        ax.add_patch(Rectangle((x, 6.95), 4.5, 2.85, fc=fc, ec=ACC, lw=1.0, zorder=0))
        ax.text(x + 0.18, 9.5, name, fontsize=7.9, weight='bold', ha='left')
        ax.text(x + 4.32, 9.5, lam, fontsize=7.6, ha='right', color='#444')
        ax.plot([x + 0.18, x + 4.32], [9.33, 9.33], color='#999', lw=0.6)

    rows_dec = [('rain', '$x + \\rho$', 'uniform'),
                ('airlight', '$A_0$', 'global constant'),
                ('noise', '$\\sigma = $ const', 'homoscedastic')]
    rows_cpl = [('rain', '$x + \\rho \\cdot L$', 'rain $\\times$ illumination'),
                ('airlight', '$A_0 + \\mathrm{glow}$', 'airlight $\\times$ source'),
                ('noise', '$\\sigma^2 = a\\,x + b$', 'signal-dependent')]
    for x0, rows in [(0.3, rows_dec), (5.2, rows_cpl)]:
        for i, (nm, form, note) in enumerate(rows):
            yy = 8.72 - i * 0.62
            ax.text(x0 + 0.2, yy, nm, fontsize=7.0, va='center')
            ax.text(x0 + 1.12, yy, form, fontsize=7.4, va='center')
            ax.text(x0 + 2.45, yy, note, fontsize=6.7, va='center', color='#444', style='italic')
    arrow(ax, (2.4, 10.3), (2.4, 9.82))
    arrow(ax, (7.6, 10.3), (7.6, 9.82))

    ax.annotate('', xy=(5.2, 7.4), xytext=(4.8, 7.4),
                arrowprops=dict(arrowstyle='<|-|>', lw=0.9, color=ACC))
    box(ax, 2.2, 6.2, 5.6, 0.48,
        'matched marginal severity: rain energy, mean airlight and mean noise $\\sigma$', fs=7.0)
    arrow(ax, (2.4, 6.95), (3.3, 6.7))
    arrow(ax, (7.6, 6.95), (6.7, 6.7))

    # ---------------- b: continuous axis ----------------
    ax.text(0.05, 5.74, 'b   Coupling structure as a continuous domain axis',
            fontsize=8.8, weight='bold')
    y0 = 5.18
    ax.plot([1.0, 9.0], [y0, y0], color=ACC, lw=1.0, zorder=1)
    for frac, lab in [(0.0, '0'), (0.25, '0.25'), (0.5, '0.5'), (0.75, '0.75'), (1.0, '1')]:
        xx = 1.0 + frac * 8.0
        ax.plot([xx], [y0], marker='o', ms=4.8, mfc='white', mec=ACC, mew=1.0, zorder=3)
        ax.text(xx, y0 - 0.28, lab, fontsize=7.0, ha='center')
    ax.text(1.0, y0 + 0.22, 'decoupled', fontsize=7.0, ha='center', style='italic')
    ax.text(9.0, y0 + 0.22, 'coupled', fontsize=7.0, ha='center', style='italic')
    ax.text(5.0, y0 - 0.62,
            'coupling strength $\\lambda$: every cross-term modulation field is interpolated, '
            'severity held fixed', fontsize=6.9, ha='center', color='#444')

    # ---------------- c: protocol and what it measures ----------------
    ax.text(0.05, 4.34, 'c   $2 \\times 2$ train $\\times$ test protocol, and what it measures',
            fontsize=8.8, weight='bold')
    cx, cy, cw, ch = 1.5, 2.44, 1.55, 0.58
    ax.text(cx + cw, cy + 2 * ch + 0.3, 'test structure', fontsize=7.3, ha='center', weight='bold')
    for j, tl in enumerate(['decoupled', 'coupled']):
        ax.text(cx + cw / 2 + j * cw, cy + 2 * ch + 0.08, tl, fontsize=7.1, ha='center')
    ax.text(cx - 0.1, cy + ch + ch / 2, 'train\ndecoupled', fontsize=7.1, ha='right', va='center')
    ax.text(cx - 0.1, cy + ch / 2, 'train\ncoupled', fontsize=7.1, ha='right', va='center')
    cells = [[('in-domain', DEC), ('shifted', 'white')],
             [('shifted', 'white'), ('in-domain', CPL)]]
    for i in range(2):
        for j in range(2):
            lab, fc = cells[i][j]
            yy = cy + (1 - i) * ch
            ax.add_patch(Rectangle((cx + j * cw, yy), cw, ch, fc=fc, ec=ACC, lw=0.9, zorder=2))
            ax.text(cx + j * cw + cw / 2, yy + ch / 2, lab, fontsize=7.0, ha='center',
                    va='center', zorder=3, style='italic' if lab == 'shifted' else 'normal')
    ax.text(cx + cw, cy - 0.18,
            'one column alone reads as a ranking;\nthe full matrix separates\n'
            'specialisation from generality',
            fontsize=6.9, ha='center', va='top', color='#444', linespacing=1.6)
    arrow(ax, (5.0, y0 - 0.86), (3.05, cy + 2 * ch + 0.54), rad=-0.1)

    box(ax, 5.35, 3.08, 4.35, 0.62,
        'diagnosis: each specialist wins its own\nstructure, loses 2.6-2.7 dB on the other', fs=7.1)
    box(ax, 5.35, 2.24, 4.35, 0.62,
        'remedy: randomise over coupling structure;\nworst case +2.2 dB, in-domain cost < 0.4 dB',
        fs=7.1, fc=MIX)
    box(ax, 5.35, 1.4, 4.35, 0.62,
        'real anchor: low-light detection (ExDark),\n+0.062 absolute AP over decoupled', fs=7.1)
    arrow(ax, (4.72, 2.98), (5.35, 3.39), rad=0.09)
    arrow(ax, (4.72, 2.9), (5.35, 2.55), rad=-0.04)
    arrow(ax, (4.72, 2.82), (5.35, 1.71), rad=-0.09)

    fig.savefig(out, bbox_inches='tight')
    print('saved ' + out)


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--out', default='figs_new/pg/fig_framework.pdf')
    main(P.parse_args().out)

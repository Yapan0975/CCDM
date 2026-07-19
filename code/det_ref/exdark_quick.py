"""
exdark_quick.py - per-class AP table + score-floor sensitivity from the campaign artefacts,
independent of the (slow) image bootstrap in exdark_bootstrap.py. Safe to run in parallel.

Run (server, ~/pami_pilots):  python3 -u exdark_quick.py
"""
import json, argparse, contextlib, io
import numpy as np

ARMS = ['dec', 'cpl', 'mix', 'cleanft']
SEEDS = range(5)


def load_gt(gt_json):
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(gt_json)


def eval_ap(gt, dets, floor):
    from pycocotools.cocoeval import COCOeval
    use = [dict(image_id=d['image_id'], category_id=d['category_id'], bbox=d['bbox'],
                score=d['score']) for d in dets if d['score'] >= floor and d['category_id'] != -1]
    if not use:
        return float('nan')
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(use); ev = COCOeval(gt, dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


def main(a):
    gt = load_gt(a.gt)

    print('=== per-class AP (mean over seeds, from campaign summaries) ===')
    pc_rows = {}
    ngt = {}
    for arm in ARMS:
        accum = {}
        for s in SEEDS:
            summ = json.load(open(f'rerun_exd_{arm}_s{s}.json'))
            for cname, v in summ['per_class'].items():
                accum.setdefault(cname, []).append(v['ap'])
                ngt[cname] = v['n_gt']
        pc_rows[arm] = {c: float(np.nanmean(v)) for c, v in accum.items()}
    cats = sorted(next(iter(pc_rows.values())).keys())
    print('class'.ljust(12) + 'n_gt'.rjust(6) + ''.join(f'{x:>10s}' for x in ARMS))
    for c in cats:
        print(c.ljust(12) + f'{ngt[c]:6d}' + ''.join(f'{pc_rows[a2][c]:10.4f}' for a2 in ARMS))

    print('\n=== score-floor sensitivity (AP, seed-averaged) ===')
    sens = {}
    for floor in a.floors:
        row = {}
        for arm in ARMS:
            vals = [eval_ap(gt, json.load(open(f'rerun_exd_{arm}_s{s}_dets.json'))['dets'], floor)
                    for s in SEEDS]
            row[arm] = float(np.mean(vals))
        sens[str(floor)] = row
        ratio = row['cpl'] / row['dec'] if row['dec'] else float('nan')
        print(f'floor {floor:.2f}: ' + '  '.join(f'{a2} {row[a2]:.4f}' for a2 in ARMS) +
              f'  cpl/dec {ratio:.2f}x', flush=True)

    json.dump(dict(per_class=pc_rows, n_gt=ngt, floor_sensitivity=sens),
              open('exdark_quick_summary.json', 'w'), indent=2)
    print('\nsaved exdark_quick_summary.json')


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--gt', default='/home/server/data/exdark/exdark_test.json')
    P.add_argument('--floors', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3])
    main(P.parse_args())

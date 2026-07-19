"""
exdark_bootstrap.py - image-level bootstrap CIs, rerun-consistency check, per-class table
and score-floor sensitivity for the ExDark campaign artefacts saved by pilot2_exdark_v2.py.

Reads rerun_exd_{dec,cpl,mix,cleanft}_s{0..4}_dets.json + rerun_exd_offtheshelf_dets.json
(+ their summary JSONs and the original exd_*.json for the consistency table) and the
ExDark GT COCO json. AP is recomputed with a vectorised re-implementation of
COCOeval.accumulate (area=all, maxDet=100, 101-point interpolation) that is validated
against pycocotools on the full sample before any bootstrap is trusted.

Run (server, ~/pami_pilots):  python3 -u exdark_bootstrap.py --boot 1000
"""
import os, json, argparse, contextlib, io
import numpy as np

ARMS = ['dec', 'cpl', 'mix', 'cleanft']
SEEDS = range(5)
T_IOU = np.linspace(0.5, 0.95, 10)
RECALL = np.linspace(0.0, 1.0, 101)


def load_gt(gt_json):
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(gt_json)


def eval_full(gt, dets, floor):
    """pycocotools reference AP at a score floor (sanity anchor)."""
    from pycocotools.cocoeval import COCOeval
    use = [d for d in dets if d['score'] >= floor and d['category_id'] != -1]
    if not use:
        return float('nan'), None
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes([dict(image_id=d['image_id'], category_id=d['category_id'],
                              bbox=d['bbox'], score=d['score']) for d in use])
        ev = COCOeval(gt, dt, 'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0]), ev


def collect(ev, img_ids):
    """Per (cat, img) detection arrays from ev.evalImgs for area=all, maxDet=100."""
    p = ev.params
    a_all = 0                                  # areaRng index 0 == 'all'
    n_img = len(p.imgIds); n_area = len(p.areaRng)
    maxdet = p.maxDets[-1]
    empty = (np.zeros(0), np.zeros((len(T_IOU), 0), bool), 0, np.zeros((len(T_IOU), 0), bool))
    out = {}
    for k, cid in enumerate(p.catIds):
        per_img = {}
        for i, iid in enumerate(p.imgIds):
            e = ev.evalImgs[k * n_area * n_img + a_all * n_img + i]
            if e is None:
                per_img[iid] = empty
                continue
            sc = np.asarray(e['dtScores'][:maxdet], float)
            m = np.asarray(e['dtMatches'][:, :maxdet], float) > 0
            ig = np.asarray(e['dtIgnore'][:, :maxdet], bool)
            npig = int((~np.asarray(e['gtIgnore'], bool)).sum())
            per_img[iid] = (sc, m, npig, ig)
        out[cid] = per_img
    return out


def ap_from_subset(per_cat, img_sel):
    """AP (mean over IoU thresholds and categories) on a multiset of image ids."""
    aps = []
    for cid, per_img in per_cat.items():
        scs, ms, igs, npig = [], [], [], 0
        for iid in img_sel:
            sc, m, np_i, ig = per_img[iid]
            scs.append(sc); ms.append(m); igs.append(ig); npig += np_i
        if npig == 0:
            continue
        sc = np.concatenate(scs); m = np.concatenate(ms, 1); ig = np.concatenate(igs, 1)
        order = np.argsort(-sc, kind='mergesort')
        m = m[:, order]; ig = ig[:, order]
        tps = np.logical_and(m, ~ig); fps = np.logical_and(~m, ~ig)
        tp = np.cumsum(tps, 1).astype(float); fp = np.cumsum(fps, 1).astype(float)
        ap_t = []
        for t in range(len(T_IOU)):
            rc = tp[t] / npig
            pr = tp[t] / np.maximum(tp[t] + fp[t], np.finfo(float).eps)
            # precision envelope then 101-point interpolation (as pycocotools)
            for i in range(len(pr) - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]
            inds = np.searchsorted(rc, RECALL, side='left')
            q = np.zeros(len(RECALL))
            ok = inds < len(pr)
            q[ok] = pr[inds[ok]]
            ap_t.append(q.mean())
        aps.append(np.mean(ap_t))
    return float(np.mean(aps)) if aps else float('nan')


def main(a):
    gt = load_gt(a.gt)
    img_ids = sorted(gt.getImgIds())
    n_img = len(img_ids)
    rng = np.random.default_rng(0)

    # ---- load artefacts, validate custom accumulate against pycocotools ----
    percat, ref_ap = {}, {}
    print('=== rerun consistency (AP: rerun vs original) + accumulate validation ===')
    for arm in ARMS + ['offtheshelf']:
        seeds = [None] if arm == 'offtheshelf' else list(SEEDS)
        for s in seeds:
            tag = f'rerun_exd_{arm}' if s is None else f'rerun_exd_{arm}_s{s}'
            dets = json.load(open(f'{tag}_dets.json'))['dets']
            summ = json.load(open(f'{tag}.json'))
            ap_ref, ev = eval_full(gt, dets, a.floor)
            assert ev is not None, tag
            pc = collect(ev, img_ids)
            ap_custom = ap_from_subset(pc, img_ids)
            orig = None
            canon = os.path.expanduser('~/Documents/yping/composite-restore')   # published copies
            if s is not None:
                oname = os.path.join(canon, f'exd_{arm}_s{s}.json')
                if os.path.exists(oname):
                    orig = json.load(open(oname))['ap_exdark']
            else:
                oname = os.path.join(canon, 'exd_offtheshelf.json')
                if os.path.exists(oname):
                    orig = json.load(open(oname))['ap_exdark']
            key = (arm, s)
            percat[key] = pc; ref_ap[key] = ap_ref
            do = f'{orig:.4f}' if orig is not None else 'n/a'
            print(f'{tag:28s} rerun {summ["ap_exdark"]:.4f} pycoco {ap_ref:.4f} '
                  f'custom {ap_custom:.4f} orig {do} d(custom-pycoco)={ap_custom - ap_ref:+.5f}')

    # ---- image-level bootstrap: shared resamples across all arms (paired) ----
    print(f'\n=== image bootstrap ({a.boot} resamples of {n_img} images, seeds averaged) ===')
    arm_ap = {arm: [] for arm in ARMS}
    diffs = {('cpl', 'dec'): [], ('cpl', 'mix'): [], ('mix', 'dec'): []}
    for b in range(a.boot):
        sel = [img_ids[i] for i in rng.integers(0, n_img, n_img)]
        mean_ap = {}
        for arm in ARMS:
            vals = [ap_from_subset(percat[(arm, s)], sel) for s in SEEDS]
            mean_ap[arm] = float(np.mean(vals))
            arm_ap[arm].append(mean_ap[arm])
        for (x, y) in diffs:
            diffs[(x, y)].append(mean_ap[x] - mean_ap[y])
        if (b + 1) % 100 == 0:
            print(f'  [{b + 1}/{a.boot}]', flush=True)
    out = {}
    for arm in ARMS:
        lo, hi = np.percentile(arm_ap[arm], [2.5, 97.5])
        full = float(np.mean([ref_ap[(arm, s)] for s in SEEDS]))
        out[arm] = dict(ap=full, ci=[float(lo), float(hi)])
        print(f'{arm:8s} AP {full:.4f}  95% CI [{lo:.4f}, {hi:.4f}]')
    for (x, y), v in diffs.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f'{x}-{y}'] = dict(diff=float(np.mean(v)), ci=[float(lo), float(hi)])
        print(f'{x}-{y:5s} diff {np.mean(v):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]')

    # ---- per-class table (from the v2 summary JSONs, seed-averaged) ----
    print('\n=== per-class AP (mean over seeds) ===')
    pc_rows = {}
    for arm in ARMS:
        accum = {}
        for s in SEEDS:
            summ = json.load(open(f'rerun_exd_{arm}_s{s}.json'))
            for cname, v in summ['per_class'].items():
                accum.setdefault(cname, []).append(v['ap'])
        pc_rows[arm] = {c: float(np.nanmean(v)) for c, v in accum.items()}
    cats = sorted(next(iter(pc_rows.values())).keys())
    hdr = 'class'.ljust(12) + ''.join(f'{a2:>10s}' for a2 in ARMS)
    print(hdr)
    for c in cats:
        print(c.ljust(12) + ''.join(f'{pc_rows[a2][c]:10.4f}' for a2 in ARMS))

    # ---- score-floor sensitivity (cpl-dec ratio at different floors) ----
    print('\n=== score-floor sensitivity (AP at floors, seed-averaged) ===')
    sens = {}
    for floor in a.floors:
        row = {}
        for arm in ARMS:
            vals = []
            for s in SEEDS:
                dets = json.load(open(f'rerun_exd_{arm}_s{s}_dets.json'))['dets']
                ap_ref, _ = eval_full(gt, dets, floor)
                vals.append(ap_ref)
            row[arm] = float(np.mean(vals))
        sens[floor] = row
        ratio = row['cpl'] / row['dec'] if row['dec'] else float('nan')
        print(f'floor {floor:.2f}: ' + '  '.join(f'{a2} {row[a2]:.4f}' for a2 in ARMS) +
              f'  cpl/dec {ratio:.2f}x')

    json.dump(dict(bootstrap=out, per_class=pc_rows, floor_sensitivity=sens),
              open('exdark_bootstrap_summary.json', 'w'), indent=2)
    print('\nsaved exdark_bootstrap_summary.json')


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--gt', default='/home/server/data/exdark/exdark_test.json')
    P.add_argument('--floor', type=float, default=0.05)
    P.add_argument('--floors', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3])
    P.add_argument('--boot', type=int, default=1000)
    main(P.parse_args())

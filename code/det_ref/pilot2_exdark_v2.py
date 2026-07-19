"""
pilot2_exdark_v2.py -- pilot2_exdark.py + saved artefacts for image-level bootstrap,
per-class AP/AR and sensitivity analyses (review-strengthening re-run).

Identical training and headline evaluation to pilot2_exdark.py (same defaults, same
seeds, same fixed split, same 0.05 score floor into COCOeval), plus:
  * saves the fine-tuned checkpoint            (--ckpt_out, default <out minus .json>.pth)
  * saves raw ExDark detections down to score 0.01 with the tv->ExDark class mapping
    and per-image class-agnostic AR             (--dets_out, default <out minus .json>_dets.json)
  * adds per-class AP/AR to the summary JSON

Run (one per mode x seed), in ~/pami_pilots:
  CUDA_VISIBLE_DEVICES=0 python3 -u pilot2_exdark_v2.py --mode coupled --seed 0 \
    --out rerun_exd_cpl_s0.json
"""
import os, sys, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccdm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEV = dict(gamma=(1.6, 2.3), beta=(0.7, 1.3), A0=(0.55, 0.8),
           shot=(0.008, 0.03), read=(0.0005, 0.003), noise_model="poisson")
TYPES = ("low", "haze", "noise")
SAVE_FLOOR = 0.01          # detections saved for offline sensitivity analyses
EVAL_FLOOR = 0.05          # detections fed to COCOeval (identical to pilot2_exdark.py)


def proxy_depth(h, w):
    return np.repeat(np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None], w, axis=1)


def iou_xyxy(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None); inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / (aa[:, None] + ab[None, :] - inter + 1e-9)).astype(np.float32)


def avg_recall(pred, gt):
    if len(gt) == 0:
        return np.nan
    if len(pred) == 0:
        return 0.0
    M = iou_xyxy(pred, gt)
    return float(np.mean([(M.max(0) >= t).mean() for t in np.arange(0.5, 1.0, 0.05)]))


class CocoDeg(torch.utils.data.Dataset):
    """COCO val split; __getitem__ returns a CCDM-degraded image + torchvision detection target."""
    def __init__(self, coco, root, ids, cocoid2tv, mode, seed=0):
        self.coco, self.root, self.ids = coco, root, ids
        self.c2tv, self.mode, self.seed = cocoid2tv, mode, seed
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        import cv2
        iid = int(self.ids[i]); info = self.coco.loadImgs(iid)[0]
        bgr = cv2.imread(os.path.join(self.root, "val2017", info["file_name"]))
        rgb = (bgr[:, :, ::-1].astype(np.float32) / 255.).copy()
        if self.mode in ("decoupled", "coupled", "mixed"):
            rng = np.random.default_rng(self.seed * 100000 + iid)
            mode = self.mode
            if mode == "mixed":                                # domain-randomized over coupling structure
                mode = "coupled" if rng.integers(2) == 0 else "decoupled"
            rgb, _ = ccdm.degrade(rgb, proxy_depth(*rgb.shape[:2]), rng, mode=mode,
                                  types=TYPES, params=SEV)
            rgb = np.clip(rgb, 0, 1)
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid, iscrowd=False))
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w > 1 and h > 1 and a["category_id"] in self.c2tv:
                boxes.append([x, y, x + w, y + h]); labels.append(self.c2tv[a["category_id"]])
        if not boxes:
            boxes = [[0, 0, 1, 1]]; labels = [1]
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        target = dict(boxes=torch.tensor(boxes, dtype=torch.float32),
                      labels=torch.tensor(labels, dtype=torch.int64))
        return t, target


def build_model():
    from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
    w = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    m = retinanet_resnet50_fpn_v2(weights=w)
    return m, w.meta["categories"]


@torch.no_grad()
def eval_coupled(model, coco, root, ids, seed=123):
    """AR on COUPLED-degraded held-out val (class-agnostic) -- synthetic reference, as in F3."""
    import cv2
    model.eval(); ars = []
    for iid in ids:
        iid = int(iid); info = coco.loadImgs(iid)[0]
        bgr = cv2.imread(os.path.join(root, "val2017", info["file_name"]))
        rgb = (bgr[:, :, ::-1].astype(np.float32) / 255.).copy()
        rng = np.random.default_rng(seed * 100000 + iid)
        lq, _ = ccdm.degrade(rgb, proxy_depth(*rgb.shape[:2]), rng, mode="coupled",
                             types=TYPES, params=SEV)
        t = torch.from_numpy(np.clip(lq, 0, 1)).permute(2, 0, 1).float().to(DEV)
        o = model([t])[0]
        pb = o["boxes"].cpu().numpy()[o["scores"].cpu().numpy() > 0.3]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False))
        gt = np.array([[x, y, x + w, y + h] for (x, y, w, h) in [a["bbox"] for a in anns]], np.float32) \
            if anns else np.zeros((0, 4), np.float32)
        ars.append(avg_recall(pb, gt))
    return float(np.nanmean(ars))


@torch.no_grad()
def eval_exdark(model, exdark_root, exdark_json, tv_cats, dets_out=""):
    """REAL low-light detection on ExDark with GT boxes: COCO AP/AP50 + class-agnostic AR,
    plus per-class AP/AR and (optionally) a dump of raw detections for offline analyses."""
    import cv2, contextlib, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(exdark_json)
    exname2id = {c["name"].lower(): c["id"] for c in gt.loadCats(gt.getCatIds())}
    alias = {"motorcycle": "motorbike", "person": "people", "dining table": "table"}  # COCO name -> ExDark name
    tv2ex = {}
    for tv_idx, n in enumerate(tv_cats):
        key = alias.get(n.lower(), n.lower())
        if key in exname2id:
            tv2ex[tv_idx] = exname2id[key]
    model.eval(); dets = []; saved = []; ars = []; img_ids = []; missing = 0
    for iid in gt.getImgIds():
        info = gt.loadImgs(iid)[0]
        bgr = cv2.imread(os.path.join(exdark_root, info["file_name"]))
        if bgr is None:                                        # GT json (HF) vs images (zip): ext/case may differ
            base = os.path.splitext(info["file_name"])[0]
            for ext in (".jpg", ".png", ".JPG", ".PNG", ".jpeg", ".JPEG"):
                bgr = cv2.imread(os.path.join(exdark_root, base + ext))
                if bgr is not None:
                    break
        if bgr is None:
            missing += 1; continue
        rgb = (bgr[:, :, ::-1].astype(np.float32) / 255.).copy()
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(DEV)
        o = model([t])[0]
        boxes = o["boxes"].cpu().numpy(); scores = o["scores"].cpu().numpy(); labels = o["labels"].cpu().numpy()
        anns = gt.loadAnns(gt.getAnnIds(imgIds=iid))
        gtb = np.array([[x, y, x + w, y + h] for (x, y, w, h) in [a["bbox"] for a in anns]], np.float32) \
            if anns else np.zeros((0, 4), np.float32)
        img_ids.append(int(iid))
        ars.append(avg_recall(boxes[scores > 0.3], gtb))
        for b, s, l in zip(boxes, scores, labels):
            if s < SAVE_FLOOR:
                continue
            x1, y1, x2, y2 = b
            rec = dict(image_id=int(iid), tv_label=int(l),
                       category_id=tv2ex.get(int(l), -1),
                       bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)], score=float(s))
            saved.append(rec)
            if s >= EVAL_FLOOR and rec["category_id"] != -1:
                dets.append(dict(image_id=rec["image_id"], category_id=rec["category_id"],
                                 bbox=rec["bbox"], score=rec["score"]))
    ar = float(np.nanmean(ars))
    ap = ap50 = 0.0; per_class = {}
    if dets:
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(dets); ev = COCOeval(gt, dt, "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
        ap, ap50 = float(ev.stats[0]), float(ev.stats[1])
        prec = ev.eval["precision"]            # [T, R, K, A, M]
        rec = ev.eval["recall"]                # [T, K, A, M]
        for k, cid in enumerate(ev.params.catIds):
            name = gt.loadCats([cid])[0]["name"]
            p = prec[:, :, k, 0, -1]; p = p[p > -1]
            r = rec[:, k, 0, -1]; r = r[r > -1]
            per_class[name] = dict(ap=float(p.mean()) if p.size else float("nan"),
                                   ar=float(r.mean()) if r.size else float("nan"),
                                   n_gt=len(gt.getAnnIds(catIds=[cid])))
    if dets_out:
        json.dump(dict(img_ids=img_ids, per_image_ar=[None if np.isnan(v) else round(v, 4) for v in ars],
                       tv2ex={str(k): v for k, v in tv2ex.items()},
                       save_floor=SAVE_FLOOR, eval_floor=EVAL_FLOOR, dets=saved),
                  open(dets_out, "w"))
        print(f"saved {dets_out} ({len(saved)} dets over {len(img_ids)} imgs)", flush=True)
    return dict(ar_exdark=ar, ap_exdark=ap, ap50_exdark=ap50, missing_imgs=missing,
                per_class=per_class)


def build_loss(model, imgs, tgts):
    return model(imgs, tgts)


def main(a):
    from pycocotools.coco import COCO
    global SEV, TYPES
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    TYPES = tuple(a.types.split(","))
    SEV = dict(gamma=(a.gamma, a.gamma + 0.01), shot=(0.004, 0.012), read=(0.0003, 0.0015), noise_model="poisson")
    coco = COCO(os.path.join(a.coco, "annotations", "instances_val2017.json"))
    cocoid2tv = {}
    model, tv_cats = build_model()
    coco_name2id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    for tv_idx, n in enumerate(tv_cats):
        if n in coco_name2id:
            cocoid2tv[coco_name2id[n]] = tv_idx

    ids = np.random.default_rng(0).permutation(sorted(coco.getImgIds()))   # FIXED split across seeds
    eval_ids = ids[:a.eval_n].tolist()
    train_ids = ids[a.eval_n:a.eval_n + a.train_n].tolist()
    print(f"[split] train={len(train_ids)} eval={len(eval_ids)} | mode={a.mode} seed={a.seed} epochs={a.epochs}", flush=True)

    model = model.to(DEV)
    if a.epochs > 0 and a.mode != "clean":
        ds = CocoDeg(coco, a.coco, train_ids, cocoid2tv, a.mode, seed=a.seed)
        g = torch.Generator(); g.manual_seed(a.seed)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=4,
                                         collate_fn=lambda b: tuple(zip(*b)), generator=g)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                              lr=a.lr, momentum=0.9, weight_decay=1e-4)
        for ep in range(a.epochs):
            model.train(); tot = 0.0
            for k, (imgs, tgts) in enumerate(dl):
                imgs = [im.to(DEV) for im in imgs]; tgts = [{kk: vv.to(DEV) for kk, vv in t.items()} for t in tgts]
                loss = sum(build_loss(model, imgs, tgts).values())
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
                if k % 200 == 0:
                    print(f"  [ep{ep} {k}/{len(dl)}] loss={float(loss):.3f}", flush=True)
            print(f"[ep{ep}] mean_loss={tot/max(1,len(dl)):.3f}", flush=True)

    stem = a.out[:-5] if a.out.endswith(".json") else a.out
    ckpt_out = a.ckpt_out or (stem + ".pth")
    dets_out = a.dets_out or (stem + "_dets.json")
    if a.mode != "clean":                       # off-the-shelf weights are torchvision's own
        torch.save(model.state_dict(), ckpt_out)
        print(f"saved {ckpt_out}", flush=True)

    ar_syn = eval_coupled(model, coco, a.coco, eval_ids)
    print(f"[RESULT] mode={a.mode} seed={a.seed}  synthetic AR_coupled = {ar_syn:.4f}", flush=True)
    exd = eval_exdark(model, a.exdark_root, a.exdark_json, tv_cats, dets_out=dets_out)
    print(f"[RESULT] REAL ExDark: AP={exd['ap_exdark']:.4f} AP50={exd['ap50_exdark']:.4f} "
          f"AR={exd['ar_exdark']:.4f} (missing {exd['missing_imgs']})", flush=True)
    out = dict(mode=a.mode, seed=a.seed, epochs=a.epochs, train_n=len(train_ids), eval_n=len(eval_ids),
               ar_coupled_syn=ar_syn, **exd)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--coco", default=os.path.expanduser("~/data/coco"))
    # clean   = off-the-shelf, no fine-tuning (reference row).
    # cleanft = SAME-budget clean-COCO fine-tune (no degradation), the same 4000 imgs / epochs /
    #           optimizer / seed as decoupled|coupled|mixed -- the control asked for by review.
    P.add_argument("--mode", default="coupled", choices=["clean", "cleanft", "decoupled", "coupled", "mixed"])
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--train_n", type=int, default=4000)
    P.add_argument("--eval_n", type=int, default=1000)
    P.add_argument("--epochs", type=int, default=4)
    P.add_argument("--bs", type=int, default=4)
    P.add_argument("--lr", type=float, default=0.005)
    P.add_argument("--out", default="exd_result.json")
    P.add_argument("--ckpt_out", default="")
    P.add_argument("--dets_out", default="")
    P.add_argument("--gamma", type=float, default=2.0)
    P.add_argument("--types", default="low,haze,noise")
    P.add_argument("--exdark_root", default="/home/server/data/exdark/ExDark")
    P.add_argument("--exdark_json", default="/home/server/data/exdark/exdark_test.json")
    main(P.parse_args())

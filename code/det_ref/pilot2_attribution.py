"""
pilot2_attribution.py — "WHAT RESTORES ROBUSTNESS" leg of the FM-coupling audit (gap 1).
Tests whether COUPLED-synthesis fine-tuning recovers a detector's robustness to coupled
degradation better than DECOUPLED-synthesis fine-tuning or a clean baseline / enhancement
front-end. If coupled-FT wins, that is IJCV Finding 2 ("training distribution is the lever,
not the architecture") extended to detection robustness -> the reviewer-proof positive finding.

Self-contained on COCO val2017 (no 18GB train download): split into disjoint FT/eval sets.
On-the-fly CCDM degradation (proxy depth). Eval metric = class-agnostic AR on COUPLED-degraded
held-out val (same metric as pilot2_fm_coupling).

Variants (run separately, --mode):
  clean      : off-the-shelf detector, NO fine-tune (baseline)            (--epochs 0)
  decoupled  : fine-tune on DECOUPLED-degraded train split
  coupled    : fine-tune on COUPLED-degraded train split
All evaluated on the SAME coupled-degraded held-out val split.

Run:
  CUDA_VISIBLE_DEVICES=0 python3 pilot2_attribution.py --mode coupled   --out ft_coupled.json
  CUDA_VISIBLE_DEVICES=1 python3 pilot2_attribution.py --mode decoupled --out ft_decoupled.json
  CUDA_VISIBLE_DEVICES=2 python3 pilot2_attribution.py --mode clean --epochs 0 --out ft_clean.json
"""
import os, sys, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccdm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEV = dict(gamma=(1.6, 2.3), beta=(0.7, 1.3), A0=(0.55, 0.8),
           shot=(0.008, 0.03), read=(0.0005, 0.003), noise_model="poisson")  # default 'medium'
TYPES = ("low", "haze", "noise")


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
        if self.mode in ("decoupled", "coupled"):
            rng = np.random.default_rng(self.seed * 100000 + iid)
            rgb, _ = ccdm.degrade(rgb, proxy_depth(*rgb.shape[:2]), rng, mode=self.mode,
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
    """AR on COUPLED-degraded held-out val (class-agnostic)."""
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


def main(a):
    from pycocotools.coco import COCO
    global SEV, TYPES
    TYPES = tuple(a.types.split(","))
    SEV = dict(gamma=(a.gamma, a.gamma+0.01), shot=(0.004,0.012), read=(0.0003,0.0015), noise_model="poisson")
    coco = COCO(os.path.join(a.coco, "annotations", "instances_val2017.json"))
    cocoid2tv = {}
    model, tv_cats = build_model()
    coco_name2id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    for tv_idx, n in enumerate(tv_cats):
        if n in coco_name2id:
            cocoid2tv[coco_name2id[n]] = tv_idx

    ids = np.random.default_rng(0).permutation(sorted(coco.getImgIds()))
    eval_ids = ids[:a.eval_n].tolist()
    train_ids = ids[a.eval_n:a.eval_n + a.train_n].tolist()
    print(f"[split] train={len(train_ids)} eval={len(eval_ids)} | mode={a.mode} epochs={a.epochs}", flush=True)

    model = model.to(DEV)
    if a.epochs > 0 and a.mode != "clean":
        ds = CocoDeg(coco, a.coco, train_ids, cocoid2tv, a.mode)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=4,
                                         collate_fn=lambda b: tuple(zip(*b)))
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                              lr=a.lr, momentum=0.9, weight_decay=1e-4)
        for ep in range(a.epochs):
            model.train(); tot = 0.0
            for k, (imgs, tgts) in enumerate(dl):
                imgs = [im.to(DEV) for im in imgs]; tgts = [{kk: vv.to(DEV) for kk, vv in t.items()} for t in tgts]
                loss = sum(build_loss(model, imgs, tgts).values())
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
                if k % 100 == 0:
                    print(f"  [ep{ep} {k}/{len(dl)}] loss={float(loss):.3f}", flush=True)
            print(f"[ep{ep}] mean_loss={tot/max(1,len(dl)):.3f}", flush=True)

    if a.save_ckpt:
        torch.save(model.state_dict(), a.save_ckpt)
        print(f"[ckpt] saved {a.save_ckpt}", flush=True)
    ar = eval_coupled(model, coco, a.coco, eval_ids)
    print(f"\n[RESULT] mode={a.mode}  AR_on_coupled_degraded = {ar:.4f}", flush=True)
    json.dump(dict(mode=a.mode, epochs=a.epochs, train_n=len(train_ids), eval_n=len(eval_ids),
                   ar_coupled=ar), open(a.out, "w"), indent=2)
    print(f"saved {a.out}")


def build_loss(model, imgs, tgts):
    return model(imgs, tgts)


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--coco", default=os.path.expanduser("~/data/coco"))
    P.add_argument("--mode", default="coupled", choices=["clean", "decoupled", "coupled"])
    P.add_argument("--train_n", type=int, default=4000)
    P.add_argument("--eval_n", type=int, default=1000)
    P.add_argument("--epochs", type=int, default=4)
    P.add_argument("--bs", type=int, default=4)
    P.add_argument("--lr", type=float, default=0.005)
    P.add_argument("--out", default="ft_result.json")
    P.add_argument("--save_ckpt", default="")
    P.add_argument("--gamma", type=float, default=2.0)
    P.add_argument("--types", default="low,haze,noise")
    main(P.parse_args())

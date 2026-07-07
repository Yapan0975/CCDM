"""
pilot3_mix.py -- R1 experiment: can coupled synthesis + a little real low-light data close the
absolute-AP gap to the off-the-shelf detector?

Three arms, all evaluated on the SAME real ExDark test split (1473 imgs) as pilot2, so numbers are
directly comparable to off-the-shelf AP=0.293 and coupled-only AP=0.106:
  --mode real   : fine-tune only on real ExDark train (5890 imgs, GT boxes)
  --mode mixed  : fine-tune on COCO coupled-synthesis (4000) + real ExDark train (5890), concatenated
  (coupled-only is pilot2 --mode coupled, already run: 0.106)

Reuses pilot2_exdark's model builder, CCDM CocoDeg synthesis, and eval_exdark.
Run:
  CUDA_VISIBLE_DEVICES=2 python3 pilot3_mix.py --mode real  --seed 0 --out p3_real_s0.json
  CUDA_VISIBLE_DEVICES=3 python3 pilot3_mix.py --mode mixed --seed 0 --out p3_mixed_s0.json
"""
import os, sys, json, argparse, contextlib, io
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot2_exdark as p2                      # reuse build_model / CocoDeg / eval_exdark / proxy_depth
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class ExDarkTrain(Dataset):
    """Real ExDark training images with GT boxes, categories mapped to torchvision COCO indices."""
    def __init__(self, root, json_path, tv_cats):
        from pycocotools.coco import COCO
        with contextlib.redirect_stdout(io.StringIO()):
            self.gt = COCO(json_path)
        self.root = root
        self.ids = self.gt.getImgIds()
        exname2id = {c['name'].lower(): c['id'] for c in self.gt.loadCats(self.gt.getCatIds())}
        alias = {'motorcycle': 'motorbike', 'person': 'people', 'dining table': 'table'}
        self.ex2tv = {}                          # ExDark category id -> torchvision class index
        for tv_idx, n in enumerate(tv_cats):
            key = alias.get(n.lower(), n.lower())
            if key in exname2id:
                self.ex2tv[exname2id[key]] = tv_idx

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import cv2
        iid = int(self.ids[i]); info = self.gt.loadImgs(iid)[0]
        bgr = cv2.imread(os.path.join(self.root, info['file_name']))
        if bgr is None:
            base = os.path.splitext(info['file_name'])[0]
            for ext in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg', '.JPEG'):
                bgr = cv2.imread(os.path.join(self.root, base + ext))
                if bgr is not None:
                    break
        if bgr is None:
            bgr = np.zeros((64, 64, 3), np.uint8)
        rgb = (bgr[:, :, ::-1].astype(np.float32) / 255.).copy()
        anns = self.gt.loadAnns(self.gt.getAnnIds(imgIds=iid))
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a['bbox']
            if w > 1 and h > 1 and a['category_id'] in self.ex2tv:
                boxes.append([x, y, x + w, y + h]); labels.append(self.ex2tv[a['category_id']])
        if not boxes:
            boxes = [[0, 0, 1, 1]]; labels = [1]
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        target = dict(boxes=torch.tensor(boxes, dtype=torch.float32),
                      labels=torch.tensor(labels, dtype=torch.int64))
        return t, target


def main(a):
    from pycocotools.coco import COCO
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    # match pilot2 synthesis severity for the coupled COCO arm
    p2.SEV = dict(gamma=(a.gamma, a.gamma + 0.01), shot=(0.004, 0.012),
                  read=(0.0003, 0.0015), noise_model='poisson')
    p2.TYPES = ('low', 'haze', 'noise')

    model, tv_cats = p2.build_model()
    model = model.to(p2.DEV)

    datasets = []
    if a.mode == 'mixed':
        coco = COCO(os.path.join(a.coco, 'annotations', 'instances_val2017.json'))
        coco_name2id = {c['name']: c['id'] for c in coco.loadCats(coco.getCatIds())}
        cocoid2tv = {coco_name2id[n]: i for i, n in enumerate(tv_cats) if n in coco_name2id}
        ids = np.random.default_rng(0).permutation(sorted(coco.getImgIds()))   # same fixed split as pilot2
        train_ids = ids[a.eval_n:a.eval_n + a.train_n].tolist()
        datasets.append(p2.CocoDeg(coco, a.coco, train_ids, cocoid2tv, 'coupled', seed=a.seed))
    datasets.append(ExDarkTrain(a.exdark_root, a.exdark_train_json, tv_cats))   # real, in both arms

    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f'[split] mode={a.mode} seed={a.seed} epochs={a.epochs} '
          f'train_size={len(ds)} (' + ' + '.join(str(len(d)) for d in datasets) + ')', flush=True)

    g = torch.Generator(); g.manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=4,
                    collate_fn=lambda b: tuple(zip(*b)), generator=g)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=a.lr, momentum=0.9, weight_decay=1e-4)
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for k, (imgs, tgts) in enumerate(dl):
            imgs = [im.to(p2.DEV) for im in imgs]
            tgts = [{kk: vv.to(p2.DEV) for kk, vv in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            if k % 300 == 0:
                print(f'  [ep{ep} {k}/{len(dl)}] loss={float(loss):.3f}', flush=True)
        print(f'[ep{ep}] mean_loss={tot / max(1, len(dl)):.3f}', flush=True)

    exd = p2.eval_exdark(model, a.exdark_root, a.exdark_test_json, tv_cats)
    print(f"[RESULT] mode={a.mode} seed={a.seed}  ExDark AP={exd['ap_exdark']:.4f} "
          f"AP50={exd['ap50_exdark']:.4f} AR={exd['ar_exdark']:.4f} (missing {exd['missing_imgs']})", flush=True)
    out = dict(mode=a.mode, seed=a.seed, epochs=a.epochs, train_size=len(ds), **exd)
    json.dump(out, open(a.out, 'w'), indent=2)
    print('saved', a.out)


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--mode', required=True, choices=['real', 'mixed'])
    P.add_argument('--seed', type=int, default=0)
    P.add_argument('--coco', default=os.path.expanduser('~/data/coco'))
    P.add_argument('--train_n', type=int, default=4000)
    P.add_argument('--eval_n', type=int, default=1000)
    P.add_argument('--epochs', type=int, default=4)
    P.add_argument('--bs', type=int, default=4)
    P.add_argument('--lr', type=float, default=0.005)
    P.add_argument('--gamma', type=float, default=2.0)
    P.add_argument('--exdark_root', default='/home/server/data/exdark/ExDark')
    P.add_argument('--exdark_train_json', default='/home/server/data/exdark/exdark_train.json')
    P.add_argument('--exdark_test_json', default='/home/server/data/exdark/exdark_test.json')
    P.add_argument('--out', required=True)
    main(P.parse_args())

"""
pilot2_realdata.py — gap 3: REAL-DATA validation on LOLv2 (paired real low-light / clean).
Two questions, no manual GT needed (clean-image detections = pseudo-GT reference):

  FAITHFULNESS (off-the-shelf detector, --ckpt ""):
    AR_real  = detector on REAL low-light vs reference
    AR_coup  = detector on COUPLED-synth low-light (CCDM, applied to clean) vs reference
    AR_dec   = detector on DECOUPLED-synth vs reference
    -> if AR_real ≈ AR_coup < AR_dec, coupled synthesis is faithful to real, decoupled over-optimistic.

  TRANSFER (--ckpt ckpt_{coupled,decoupled}.pth):
    AR_real for a coupled-FT vs decoupled-FT detector -> does coupled-synth training transfer to
    REAL low-light better? (= IJCV real-transfer logic, for detection)

LOLv2: --root .../lolv2_test  with Input/ (low) and GT/ (clean), matched filenames.
Run: CUDA_VISIBLE_DEVICES=0 python3 pilot2_realdata.py --root ~/Documents/yping/composite-restore/lolv2_test --ckpt "" --out real_offshelf.json
"""
import os, sys, json, glob, argparse
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccdm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEV = dict(gamma=(1.6, 2.3), shot=(0.008, 0.03), read=(0.0005, 0.003), noise_model="poisson")


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


def boot(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 3:
        return float("nan"), float("nan")
    r = np.random.default_rng(0); bs = x[r.integers(0, len(x), (5000, len(x)))].mean(1)
    return float(x.mean()), float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))


def main(a):
    from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
    W = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    # FIXED reference detector (always off-the-shelf) -> same pseudo-GT for every model under test,
    # so coupled-FT / decoupled-FT / offshelf are compared on the SAME images against the SAME ref.
    ref_model = retinanet_resnet50_fpn_v2(weights=W).to(DEV).eval()
    model = retinanet_resnet50_fpn_v2(weights=W)
    tag = "offshelf"
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu")); tag = os.path.basename(a.ckpt)
    model = model.to(DEV).eval()

    @torch.no_grad()
    def _run(m, rgb01, thr):
        t = torch.from_numpy(np.clip(rgb01, 0, 1).astype(np.float32)).permute(2, 0, 1).to(DEV)
        o = m([t])[0]
        return o["boxes"].cpu().numpy()[o["scores"].cpu().numpy() > thr]

    def ref_det(rgb01, thr):
        return _run(ref_model, rgb01, thr)

    def det(rgb01, thr):
        return _run(model, rgb01, thr)

    lows = sorted(glob.glob(os.path.join(a.root, "Input", "*")))[:a.n]
    real, coup, dec = [], [], []
    for i, lf in enumerate(lows):
        name = os.path.basename(lf)
        gtf = os.path.join(a.root, "GT", name)
        if not os.path.exists(gtf):
            continue
        clean = cv2.imread(gtf)[:, :, ::-1].astype(np.float32) / 255.
        lowreal = cv2.imread(lf)[:, :, ::-1].astype(np.float32) / 255.
        ref = ref_det(clean, 0.5)                   # pseudo-GT = FIXED off-the-shelf detector on clean
        if len(ref) == 0:
            continue
        real.append(avg_recall(det(lowreal, 0.3), ref))
        if not a.ckpt:                              # faithfulness: also synth coupled/decoupled
            h, w = clean.shape[:2]; depth = np.repeat(np.linspace(1, 0, h, dtype=np.float32)[:, None], w, 1)
            rc = np.random.default_rng(7 * 1000 + i)
            lc, _ = ccdm.degrade(clean, depth, rc, mode="coupled", types=("low", "noise"), params=SEV)
            rd = np.random.default_rng(7 * 1000 + i)
            ld, _ = ccdm.degrade(clean, depth, rd, mode="decoupled", types=("low", "noise"), params=SEV)
            coup.append(avg_recall(det(lc, 0.3), ref)); dec.append(avg_recall(det(ld, 0.3), ref))
        if i % 25 == 0:
            print(f"  [{i}/{len(lows)}]", flush=True)

    ar_real, _ = boot(real)
    print(f"\n==================  REAL-DATA (LOLv2)  model={tag}  ==================")
    print(f"AR on REAL low-light = {ar_real:.4f}   (n={len([x for x in real if np.isfinite(x)])})")
    out = dict(model=tag, n=len(real), ar_real=ar_real)
    if not a.ckpt:
        ac, _ = boot(coup); ad, _ = boot(dec)
        gap_rc = np.array(dec, float) - np.array(coup, float)         # decoupled higher (over-optimistic)?
        gap_real_c = np.array(real, float) - np.array(coup, float)    # real vs coupled (closer to 0 = faithful)
        gap_real_d = np.array(real, float) - np.array(dec, float)
        print(f"AR coupled-synth     = {ac:.4f}")
        print(f"AR decoupled-synth   = {ad:.4f}")
        print(f"|real-coupled|={abs(np.nanmean(gap_real_c)):.4f}  |real-decoupled|={abs(np.nanmean(gap_real_d)):.4f}")
        print(f"-> coupled more faithful to real if |real-coupled| < |real-decoupled|")
        out.update(ar_coupled_synth=ac, ar_decoupled_synth=ad,
                   abs_real_minus_coupled=float(abs(np.nanmean(gap_real_c))),
                   abs_real_minus_decoupled=float(abs(np.nanmean(gap_real_d))))
    print("=====================================================================")
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print(f"saved {a.out}")


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--root", default=os.path.expanduser("~/Documents/yping/composite-restore/lolv2_test"))
    P.add_argument("--ckpt", default="")
    P.add_argument("--n", type=int, default=200)
    P.add_argument("--out", default="real_lolv2.json")
    main(P.parse_args())

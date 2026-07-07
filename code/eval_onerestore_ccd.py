"""
eval_onerestore_ccd.py - run OneRestore (CDD-11 weights) on the CCD benchmark manifest
and report PSNR/SSIM per (combo x mode). Closes the data->baseline->metric loop and gives
the first proper decoupled-vs-coupled baseline numbers (with real MiDaS depth + full ccdm).
"""
import os, sys, json, argparse, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
_f = types.ModuleType('utils.utils_word_embedding')
_f.initialize_wordembedding_matrix = lambda *a, **k: (torch.zeros(12, 300), 300)
sys.modules['utils.utils_word_embedding'] = _f
_t = types.ModuleType('thop'); _t.profile = lambda *a, **k: (0, 0); _t.clever_format = lambda *a, **k: a
sys.modules['thop'] = _t
from utils.utils import load_restore_ckpt, load_embedder_ckpt
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = {'low_rain': 'low_rain', 'low_haze_rain': 'low_haze_rain'}

def restore(net, emb, x):
    t = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
    with torch.no_grad():
        return net(t, emb)[0].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)

def main(a):
    net = load_restore_ckpt(DEV, freeze_model=True, ckpt_name=a.restore)
    emb = load_embedder_ckpt(DEV, freeze_model=True, ckpt_name=a.embedder)
    cache = {p: emb([p], 'text_encoder')[0] for p in set(PROMPT.values())}
    man = json.load(open(os.path.join(a.root, 'manifest.json')))
    agg = {}
    for e in man:
        J = cv2.imread(os.path.join(a.root, e['clean'].replace('\\', '/'))).astype(np.float32) / 255.0
        x = cv2.imread(os.path.join(a.root, e['degraded'].replace('\\', '/'))).astype(np.float32) / 255.0
        out = restore(net, cache[PROMPT[e['combo']]], x)
        ps = psnr_fn(J, out, data_range=1.0); ss = ssim_fn(J, out, data_range=1.0, channel_axis=2)
        # also score the raw degraded input (reference point)
        ps_in = psnr_fn(J, x, data_range=1.0)
        k = (e['combo'], e['mode'])
        agg.setdefault(k, []).append((ps, ss, ps_in))
    out = {}
    print(f"{'combo':16s} {'mode':10s} {'in_PSNR':>8s} {'PSNR':>7s} {'SSIM':>7s}  n")
    for (c, m), v in sorted(agg.items()):
        v = np.array(v)
        out[f'{c}|{m}'] = dict(in_PSNR=round(float(v[:,2].mean()),3),
                               PSNR=round(float(v[:,0].mean()),3),
                               SSIM=round(float(v[:,1].mean()),4), n=len(v))
        print(f"{c:16s} {m:10s} {v[:,2].mean():8.3f} {v[:,0].mean():7.3f} {v[:,1].mean():7.4f}  {len(v)}")
    # decoupled-vs-coupled gap per combo
    for c in set(k[0] for k in agg):
        d = out.get(f'{c}|decoupled'); cp = out.get(f'{c}|coupled')
        if d and cp:
            print(f"  -> {c}: dPSNR(dec-cpl) = {d['PSNR']-cp['PSNR']:+.3f}  dSSIM = {d['SSIM']-cp['SSIM']:+.4f}")
    json.dump(out, open(a.out, 'w'), indent=2)
    print('saved', a.out)

if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--root', default='./ccd_mini')
    P.add_argument('--restore', default='./ckpts/onerestore_cdd-11.tar')
    P.add_argument('--embedder', default='./ckpts/embedder_model.tar')
    P.add_argument('--out', default='./ccd_mini_baseline.json')
    main(P.parse_args())

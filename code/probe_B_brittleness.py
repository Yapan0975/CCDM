"""
Probe B - Severity-matched brittleness test.

Question: trained on the field-standard DECOUPLED composite model (CDD-11 /
OneRestore, ECCV'24), is OneRestore brittle to the physical CROSS-MODULATION
it never saw, *at equal degradation severity*?

We compare, per clean image, two composites that have the SAME total degradation
energy but differ only in spatial COUPLING:
  G1 rain :  decoupled = uniform additive rain   vs
             coupled   = rain radiance scaled by local illumination
                         (re-normalised to identical total rain energy)
  G2 haze :  decoupled = spatially-uniform airlight A  vs
             coupled   = airlight = non-uniform glow around light sources
                         (re-normalised to identical mean airlight)
Transmission t (haze) and the low-light map are identical across dec/cpl, so the
ONLY difference is the cross-term spatial structure -> a confound-free test.

If PSNR(restore decoupled) >> PSNR(restore coupled) the model cannot handle the
cross-term -> real restoration headroom -> thesis has legs.
"""
import os, sys, json, argparse, glob, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
# Pre-register a stub for utils.utils_word_embedding so importing Embedder does NOT
# pull in fasttext/gensim/glove (unavailable on the offline server). The 12x300
# word matrix is overwritten by the checkpoint anyway.
_fake = types.ModuleType('utils.utils_word_embedding')
_fake.initialize_wordembedding_matrix = lambda *a, **k: (torch.zeros(12, 300), 300)
sys.modules['utils.utils_word_embedding'] = _fake
_thop = types.ModuleType('thop')          # FLOPs counter, unused at inference
_thop.profile = lambda *a, **k: (0, 0)
_thop.clever_format = lambda *a, **k: a
sys.modules['thop'] = _thop
from utils.utils import load_restore_ckpt, load_embedder_ckpt
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
rng = np.random.default_rng(0)

def guideFilter(I, p, win, eps):
    mI = cv2.blur(I, win); mp = cv2.blur(p, win)
    mII = cv2.blur(I*I, win); mIp = cv2.blur(I*p, win)
    a = (mIp - mI*mp)/(mII - mI*mI + eps); b = mp - a*mI
    return cv2.blur(a, win)*I + cv2.blur(b, win)

def illum(img):                      # LIME-style illumination in (0,1]
    mx = img.max(2).astype(np.float32)
    return np.clip(guideFilter(mx, mx, (15, 15), 1e-2), 1e-3, 1.0)

def syn_low(img, L, gamma):          # faithful to OneRestore syn_low (no noise)
    Lf = L[:, :, None]
    return np.clip((img/(Lf+1e-7))*(Lf+1e-7)**gamma, 0, 1)

def restore(restorer, emb, x):
    t = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
    with torch.no_grad():
        out = restorer(t, emb)
    return out[0].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)

def met(ref, test):
    return (psnr_fn(ref, test, data_range=1.0),
            ssim_fn(ref, test, data_range=1.0, channel_axis=2))

def main(a):
    restorer = load_restore_ckpt(DEV, freeze_model=True, ckpt_name=a.restore)
    embedder = load_embedder_ckpt(DEV, freeze_model=True, ckpt_name=a.embedder)
    emb_rain, _, t1 = embedder(['low_rain'], 'text_encoder')
    emb_haze, _, t2 = embedder(['low_haze'], 'text_encoder')
    print('embeddings ok:', t1, t2)

    rain_masks = sorted(glob.glob(os.path.join(a.rain, '*')))
    cleans = sorted(glob.glob(os.path.join(a.clean, '*.png')))[:a.n]
    R = []
    for idx, cf in enumerate(cleans):
        J = cv2.imread(cf).astype(np.float32)/255.0
        H, W = J.shape[:2]
        L = illum(J); gamma = float(rng.uniform(2.0, 3.0))
        low = syn_low(J, L, gamma)
        darkf = np.clip(low.sum(2)/(J.sum(2)+1e-6), 0, 1)          # per-pixel darkening (<=1)

        # ---- G1 rain: matched total energy ----
        rm = cv2.imread(rain_masks[idx % len(rain_masks)]).astype(np.float32)/255.0
        rm = cv2.resize(rm, (W, H))
        dec_rain = rm
        cpl_rain = rm*darkf[:, :, None]
        cpl_rain *= (dec_rain.sum()/(cpl_rain.sum()+1e-9))          # equal total rain energy
        dec_r = np.clip(low + dec_rain, 0, 1)
        cpl_r = np.clip(low + cpl_rain, 0, 1)
        pr_d = met(J, restore(restorer, emb_rain, dec_r))
        pr_c = met(J, restore(restorer, emb_rain, cpl_r))

        # ---- G2 haze: matched mean airlight ----
        d = guideFilter(1.0 - L, 1.0 - L, (25, 25), 1e-2)          # smooth pseudo-depth (same for both)
        beta = 1.6
        t = np.exp(-np.minimum(d, 0.9)*beta)[:, :, None]
        A0 = 0.8
        src = (low.mean(2) > np.quantile(low.mean(2), 0.97)).astype(np.float32)  # light sources
        glow = cv2.GaussianBlur(src, (0, 0), sigmaX=W*0.03)
        glow = glow/(glow.max()+1e-9)
        A_dec = np.full((H, W, 1), A0, np.float32)
        A_cpl = (A0*(0.3 + 1.6*glow))[:, :, None]
        A_cpl *= (A_dec.mean()/(A_cpl.mean()+1e-9))                 # equal mean airlight
        dec_h = np.clip(low*t + A_dec*(1-t), 0, 1)
        cpl_h = np.clip(low*t + A_cpl*(1-t), 0, 1)
        ph_d = met(J, restore(restorer, emb_haze, dec_h))
        ph_c = met(J, restore(restorer, emb_haze, cpl_h))

        R.append(dict(rain_dec=pr_d, rain_cpl=pr_c, haze_dec=ph_d, haze_cpl=ph_c))
        if idx < 3 or idx % 20 == 0:
            print(f'[{idx}] rain dec/cpl PSNR {pr_d[0]:.2f}/{pr_c[0]:.2f}  haze {ph_d[0]:.2f}/{ph_c[0]:.2f}')

    def agg(key, i): return float(np.mean([r[key][i] for r in R]))
    out = {
        'n': len(R),
        'rain': {'dec_PSNR': round(agg('rain_dec',0),3), 'cpl_PSNR': round(agg('rain_cpl',0),3),
                 'dPSNR_dec_minus_cpl': round(agg('rain_dec',0)-agg('rain_cpl',0),3),
                 'dec_SSIM': round(agg('rain_dec',1),4), 'cpl_SSIM': round(agg('rain_cpl',1),4)},
        'haze': {'dec_PSNR': round(agg('haze_dec',0),3), 'cpl_PSNR': round(agg('haze_cpl',0),3),
                 'dPSNR_dec_minus_cpl': round(agg('haze_dec',0)-agg('haze_cpl',0),3),
                 'dec_SSIM': round(agg('haze_dec',1),4), 'cpl_SSIM': round(agg('haze_cpl',1),4)},
        'note': 'severity-matched (equal total rain energy / equal mean airlight); dec=field CDD-11 model, cpl=physical cross-term',
    }
    print(json.dumps(out, indent=2))
    with open(a.out, 'w') as f: json.dump(out, f, indent=2)
    print('saved', a.out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--restore', default='./ckpts/onerestore_cdd-11.tar')
    p.add_argument('--embedder', default='./ckpts/embedder_model.tar')
    p.add_argument('--clean', default='./clean512')
    p.add_argument('--rain', default='./syn_data/data/rain_mask')
    p.add_argument('--n', type=int, default=80)
    p.add_argument('--out', default='./probe_B.json')
    main(p.parse_args())

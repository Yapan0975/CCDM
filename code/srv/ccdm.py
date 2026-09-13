"""
ccdm.py - Coupled Composite Degradation Model (CCDM)

Physical core (contribution C1). Synthesises composite degradations in two modes:
  mode='decoupled'  - field-standard CDD-11 / OneRestore behaviour: each degradation's
                      parameters are independent (additive rain, uniform-variance noise,
                      global-constant airlight). Used as the control.
  mode='coupled'    - explicit cross-terms (this paper):
                        rain   radiance  scales with local illumination L   (rain x L)
                        noise  variance  follows the Poisson-Gaussian signal model var=a*x+b
                                          (signal-dependent shot + read floor; see step 6)
                        airlight          glows around light sources         (airlight x source)
                      plus motion blur (shared capture kernel, absent from CDD-11).
  NOTE: the legacy sigma0/L "gain" noise (var ~ 1/L) is retained only as a reference branch
        (noise_model!='poisson'); all results in the paper use the Poisson-Gaussian model.

The ONLY difference between the two modes is the cross-term structure (rain scaling,
noise variance map, airlight map). Low-light, transmission t and the motion kernel are
shared, so decoupled-vs-coupled is a confound-free comparison.

degrade() also returns the shared latent fields {L, t, sigma, clean} as ground truth,
which supervise CoupleNet's latent-field heads (contribution C2).

Pure numpy + OpenCV; no torch, no nvcc.
"""
import numpy as np
import cv2

DEGR_TYPES = ('low', 'noise', 'rain', 'haze', 'snow', 'blur')


def guide_filter(I, p, win, eps):
    mI = cv2.blur(I, win); mp = cv2.blur(p, win)
    mII = cv2.blur(I * I, win); mIp = cv2.blur(I * p, win)
    a = (mIp - mI * mp) / (mII - mI * mI + eps); b = mp - a * mI
    return cv2.blur(a, win) * I + cv2.blur(b, win)


def lime_illumination(img):
    """LIME-style illumination map L in (0,1] (max channel, edge-aware smoothed)."""
    mx = img.max(2).astype(np.float32)
    return np.clip(guide_filter(mx, mx, (15, 15), 1e-2), 1e-3, 1.0)


def transmission(depth, beta):
    """Atmospheric-scattering transmission t = exp(-beta*d); depth in [0,1] (0 near,1 far)."""
    return np.exp(-np.minimum(depth.astype(np.float32), 0.97) * beta)[:, :, None]


def motion_psf(length, angle_deg):
    """Linear motion-blur kernel."""
    length = max(int(length), 1)
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    return k / s if s > 0 else k


def degrade(J, depth, rng, *, mode='coupled', types=('low', 'rain', 'haze'),
            rain_mask=None, snow_mask=None, params=None, coupled_terms=None, lam=None):
    """
    J     : HxWx3 float32 in [0,1]
    depth : HxW   float32 in [0,1] (near=0, far=1)
    rng   : np.random.Generator
    coupled_terms : optional iterable subset of {'rain','noise','haze'}. If given, each term uses
                    its coupled (cross-modulated) form iff it is in the set, else the decoupled
                    form -- enables per-cross-term ablation (A4). If None, `mode` decides all.
    lam   : optional float in [0,1] -- CONTINUOUS coupling strength. Each cross-term's
            modulation FIELD is linearly interpolated between its decoupled (lam=0) and
            coupled (lam=1) form, while the marginal severity is held fixed at every lam
            (mean airlight, total rain energy, mean noise sigma are re-normalised exactly as
            in the binary case). lam=0 and lam=1 reproduce mode='decoupled' / 'coupled'
            pixel-for-pixel under the same rng, because no branch consumes extra randomness.
            If lam is None the binary `mode` / `coupled_terms` path is used (published
            behaviour is untouched).
    returns (lq, gt) ; gt = {'clean','L','t','sigma','types','mode','lam'}
    """
    assert mode in ('coupled', 'decoupled')
    def is_coupled(term):
        return (term in coupled_terms) if coupled_terms is not None else (mode == 'coupled')
    H, W, _ = J.shape
    p = dict(gamma=(2.0, 3.0), beta=(1.0, 2.0), A0=(0.6, 0.9),
             sigma0=(0.03, 0.08), rain_c=(0.7, 1.0), blur_len=(7, 21))
    if params:
        p.update(params)

    L = lime_illumination(J)                       # shared illumination (latent GT)
    t = transmission(depth, rng.uniform(*p['beta']))  # shared transmission (latent GT)
    x = J.astype(np.float32).copy()
    sigma_map = np.zeros((H, W), np.float32)

    # 1) low-light (shared form; faithful to OneRestore syn_low: x = J * L^(gamma-1))
    if 'low' in types:
        gamma = rng.uniform(*p['gamma'])
        x = np.clip(x * (L[:, :, None] ** (gamma - 1.0)), 0, 1)

    # 2) motion blur (absent from CDD-11; applied at capture, shared kernel)
    if 'blur' in types:
        k = motion_psf(int(rng.integers(p['blur_len'][0], p['blur_len'][1] + 1)),
                       float(rng.uniform(0, 180)))
        x = cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)

    # 3) haze (cross-term: airlight x light-source in coupled mode)
    if 'haze' in types:
        A0 = rng.uniform(*p['A0'])
        if lam is not None:                        # continuous coupling strength
            src = (x.mean(2) > np.quantile(x.mean(2), 0.97)).astype(np.float32)
            glow = cv2.GaussianBlur(src, (0, 0), sigmaX=W * 0.03)
            glow = glow / (glow.max() + 1e-9)
            m = (1.0 - lam) + lam * (0.3 + 1.6 * glow)   # modulation field, lam-interpolated
            A = (A0 * m)[:, :, None]
            A *= A0 / (A.mean() + 1e-9)            # equal mean airlight at EVERY lam
        elif is_coupled('haze'):
            src = (x.mean(2) > np.quantile(x.mean(2), 0.97)).astype(np.float32)
            glow = cv2.GaussianBlur(src, (0, 0), sigmaX=W * 0.03)
            glow = glow / (glow.max() + 1e-9)
            A = (A0 * (0.3 + 1.6 * glow))[:, :, None]
            A *= A0 / (A.mean() + 1e-9)             # equal mean airlight as decoupled
        else:
            A = np.full((H, W, 1), A0, np.float32)
        x = np.clip(x * t + A * (1 - t), 0, 1)

    # 4) snow (alpha-blend; shared form)
    if 'snow' in types and snow_mask is not None:
        sm = cv2.resize(snow_mask, (W, H))
        x = np.clip(x * (1 - sm) + sm, 0, 1)

    # 5) rain (cross-term: rain radiance x illumination in coupled mode)
    if 'rain' in types and rain_mask is not None:
        rm = cv2.resize(rain_mask, (W, H)) * rng.uniform(*p['rain_c'])
        if lam is not None:                        # continuous coupling strength
            rm_l = rm * ((1.0 - lam) + lam * L[:, :, None])
            rm_l *= rm.sum() / (rm_l.sum() + 1e-9)  # equal total rain energy at EVERY lam
            x = np.clip(x + rm_l, 0, 1)
        elif is_coupled('rain'):
            rm_c = rm * L[:, :, None]
            rm_c *= rm.sum() / (rm_c.sum() + 1e-9)  # equal total rain energy as decoupled
            x = np.clip(x + rm_c, 0, 1)
        else:
            x = np.clip(x + rm, 0, 1)

    # 6) sensor noise. noise_model='poisson' = physically-faithful heteroscedastic
    #    Poisson-Gaussian (var = a*signal + b: shot noise prop to signal + read-noise floor).
    #    coupled keeps the signal-dependent structure; decoupled flattens to the same per-image
    #    mean (homoscedastic). 'gain' is the earlier ad-hoc sigma0/L model (kept for reference).
    if 'noise' in types:
        if p.get('noise_model') == 'poisson':
            a = rng.uniform(*p.get('shot', (0.01, 0.06)))
            b = rng.uniform(*p.get('read', (0.0005, 0.004)))
            sig = np.sqrt(np.maximum(a * x.mean(2) + b, 1e-8)).astype(np.float32)
            if lam is not None:                    # continuous coupling strength
                flat = (float(np.sqrt((sig ** 2).mean())) if p.get('noise_match') == 'var'
                        else float(sig.mean()))
                sigma_map = ((1.0 - lam) * flat + lam * sig).astype(np.float32)
            elif is_coupled('noise'):
                sigma_map = sig
            elif p.get('noise_match') == 'var':                # variance-matched: equal total noise energy
                sigma_map = np.full((H, W), float(np.sqrt((sig ** 2).mean())), np.float32)
            else:                                              # std-matched (default; mean sigma)
                sigma_map = np.full((H, W), float(sig.mean()), np.float32)
        else:                                                  # legacy gain model (sigma0 / L)
            s0 = rng.uniform(*p['sigma0'])
            if lam is not None:                    # continuous coupling strength
                sc = (s0 / (L + 0.15)).astype(np.float32)
                sc = sc * (s0 / (sc.mean() + 1e-9))            # equal mean sigma as decoupled
                sigma_map = ((1.0 - lam) * s0 + lam * sc).astype(np.float32)
            elif is_coupled('noise'):
                sigma_map = (s0 / (L + 0.15)).astype(np.float32); sigma_map *= s0 / (sigma_map.mean() + 1e-9)
            else:
                sigma_map = np.full((H, W), s0, np.float32)
        x = np.clip(x + rng.normal(0, 1, (H, W, 1)) * sigma_map[:, :, None], 0, 1)

    gt = dict(clean=J.astype(np.float32), L=L.astype(np.float32),
              t=t[:, :, 0].astype(np.float32), sigma=sigma_map,
              types=tuple(types), mode=mode, lam=lam)
    return x.astype(np.float32), gt

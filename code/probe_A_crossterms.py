"""
Probe A — Quantify the missing physical cross-modulation terms in the
field-standard composite-degradation model (OneRestore / CDD-11, ECCV'24).

We faithfully reproduce OneRestore's syn_data.py operators and contrast the
"decoupled" field model (each degradation's parameters sampled independently)
against the physically-coupled model. Two falsifiable measurements:

  G1  rain<->illumination: additive rain `lq + rain_mask` keeps a CONSTANT
      absolute amplitude regardless of scene illumination. Physically, a rain
      streak's radiance scales with the ambient illumination it scatters.
      Consequence: under low-light the rain Weber contrast is inflated in
      shadow regions -> an unphysical cue a trained net overfits to.
      Metric: median rain Weber contrast in the darkest vs brightest quartile.

  G2  airlight<->light-source: haze airlight A is a GLOBAL random scalar
      (spatially uniform). Physically (nighttime), airlight is non-uniform glow
      emanating from light sources. Metric: spatial coefficient of variation
      (CoV) of the airlight map: field model = 0, source-coupled model >> 0.

  G3  motion blur entirely absent + fixed composition order (reported).

No network weights needed; this characterises the DATA model the whole
sub-field trains on.
"""
import os, json
import numpy as np
import cv2

REPO = "D:/_7_sci/LSD/paper2_restore/OneRestore/syn_data/data"
OUT  = "D:/_7_sci/LSD/paper2_restore/probe_out"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

# ---------- faithful OneRestore operators (from syn_data.py) ----------
def guideFilter(I, p, winSize, eps):
    mean_I = cv2.blur(I, winSize); mean_p = cv2.blur(p, winSize)
    mean_II = cv2.blur(I*I, winSize); mean_Ip = cv2.blur(I*p, winSize)
    var_I = mean_II - mean_I*mean_I; cov_Ip = mean_Ip - mean_I*mean_p
    a = cov_Ip/(var_I+eps); b = mean_p - a*mean_I
    return cv2.blur(a, winSize)*I + cv2.blur(b, winSize)

def syn_low_components(img, light, img_gray, light_max=3, light_min=2):
    """Return (low_light_img, scene_illumination_map L). Faithful to syn_low
    but exposes the per-pixel illumination factor L used to darken the scene."""
    light_f = guideFilter(light, img_gray, (3,3), 0.01)[:, :, np.newaxis]
    R = img/(light_f+1e-7)
    L = (light_f+1e-7)**rng.uniform(light_min, light_max)   # multiplicative illumination
    low = np.clip(R*L, 0, 1)                                # (drop noise term for clean comparison)
    return low, L[:, :, 0]

# ---------- load the shipped sample ----------
clear = cv2.imread(os.path.join(REPO, "clear", "1.jpg"))/255.0
light = cv2.cvtColor(cv2.imread(os.path.join(REPO, "light_map", "1.jpg")), cv2.COLOR_BGR2GRAY)/255.0
depth = cv2.imread(os.path.join(REPO, "depth_map", "1.jpg"))/255.0
img_gray = cv2.cvtColor((clear*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)/255.0
H, W, _ = clear.shape
rain_files = sorted(os.listdir(os.path.join(REPO, "rain_mask")))
rain = cv2.imread(os.path.join(REPO, "rain_mask", rain_files[0]))/255.0
rain = cv2.resize(rain, (W, H))

low, L = syn_low_components(clear, light, img_gray)
lum = low.mean(axis=2)                     # observed scene luminance after low-light
rain_amp = rain.mean(axis=2)              # additive rain layer amplitude (field model)

# ================= G1: rain Weber contrast inflation in shadows =================
# Field model rain layer == rain_mask (constant amplitude, illumination-independent).
# Weber contrast C = rain_amplitude / local_background_luminance.
streak = rain_amp > 0.05                  # actual rain-streak pixels
bg = cv2.blur(lum.astype(np.float32), (15,15)) + 1e-3
weber = rain_amp/bg                       # field model Weber contrast per pixel

q = np.quantile(lum[streak], [0.25, 0.75])
dark = streak & (lum <= q[0]); bright = streak & (lum >= q[1])
weber_dark = float(np.median(weber[dark])); weber_bright = float(np.median(weber[bright]))
# physically-coupled rain (radiance scales with local illumination, normalised)
Ln = L/ (L.max()+1e-9)
weber_cpl = (rain_amp*Ln)/bg
weber_cpl_dark = float(np.median(weber_cpl[dark])); weber_cpl_bright = float(np.median(weber_cpl[bright]))

# correlation of rain-layer amplitude with illumination over streak pixels
corr_field = float(np.corrcoef(rain_amp[streak], L[streak])[0,1])      # ~0 by construction
corr_cpl   = float(np.corrcoef((rain_amp*Ln)[streak], L[streak])[0,1]) # strong

# ================= G2: airlight uniformity vs source-coupled glow =================
# Field haze airlight: global scalar A in [0.6,0.9] -> uniform map.
A_scalar = 0.75
A_field = np.full((H, W), A_scalar, np.float32)
# Source-coupled nighttime airlight: glow from bright light sources.
src = (lum > np.quantile(lum, 0.97)).astype(np.float32)        # light sources = brightest 3%
glow = cv2.GaussianBlur(src, (0,0), sigmaX=W*0.03)
glow = glow/ (glow.max()+1e-9)
A_coupled = A_scalar*(0.3 + 1.4*glow)                          # non-uniform airlight
cov_field   = float(A_field.std()/ (A_field.mean()+1e-9))
cov_coupled = float(A_coupled.std()/ (A_coupled.mean()+1e-9))

# ================= G3: composition inventory =================
CDD11 = ["low","haze","rain","snow","low_haze","low_rain","low_snow",
         "haze_rain","haze_snow","low_haze_rain","low_haze_snow"]
has_blur = any("blur" in c for c in CDD11)

res = {
  "G1_rain_illumination": {
     "field_weber_shadow": round(weber_dark,3), "field_weber_highlight": round(weber_bright,3),
     "field_shadow_to_highlight_ratio": round(weber_dark/ (weber_bright+1e-9),2),
     "coupled_weber_shadow": round(weber_cpl_dark,3), "coupled_weber_highlight": round(weber_cpl_bright,3),
     "coupled_shadow_to_highlight_ratio": round(weber_cpl_dark/(weber_cpl_bright+1e-9),2),
     "corr_rain_vs_illum_FIELD": round(corr_field,3),
     "corr_rain_vs_illum_COUPLED": round(corr_cpl,3),
  },
  "G2_airlight_source": {
     "field_airlight_CoV": round(cov_field,4),
     "coupled_airlight_CoV": round(cov_coupled,4),
  },
  "G3_composition": {
     "num_CDD11_categories": len(CDD11),
     "categories_with_motion_blur": int(has_blur),
     "fixed_order": "low -> rain -> snow -> haze (syn_data.py:56-63)",
  },
}
print(json.dumps(res, indent=2))
with open(os.path.join(OUT, "probe_A.json"), "w") as f: json.dump(res, f, indent=2)

# ---- figure ----
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,3, figsize=(13,4))
    ax[0].imshow(cv2.cvtColor((low*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax[0].set_title("low-light scene"); ax[0].axis("off")
    im1 = ax[1].imshow(np.clip(weber,0,np.quantile(weber,0.99)), cmap="inferno")
    ax[1].set_title(f"G1 field rain Weber contrast\nshadow:highlight = {weber_dark/ (weber_bright+1e-9):.1f}x")
    ax[1].axis("off"); fig.colorbar(im1, ax=ax[1], fraction=0.046)
    im2 = ax[2].imshow(A_coupled, cmap="viridis")
    ax[2].set_title(f"G2 source-coupled airlight\nCoV {cov_coupled:.2f} vs field {cov_field:.2f}")
    ax[2].axis("off"); fig.colorbar(im2, ax=ax[2], fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"probe_A.png"), dpi=130)
    print("saved", os.path.join(OUT,"probe_A.png"))
except Exception as e:
    print("fig skipped:", e)

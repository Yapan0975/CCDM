import json
def truec(t):
    pc=json.load(open(t+'.json'))['per_cell']
    return (pc['low_rain|coupled']['PSNR']+pc['low_haze_rain|coupled']['PSNR'])/2
d=truec('FR_dec'); c=truec('FR_cpl')
print('=== F1 CROSS-FAMILY (Restormer / transformer, 600 train / 100 test) ===')
print(f'Restormer: decoupled-train={d:.3f}  coupled-train={c:.3f}  GAP(F1)={c-d:+.3f} dB')
print('(NAFNet conv family full-scale: w32 +4.24, w64 +4.84)')

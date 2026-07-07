import json
def truec(t):
    pc=json.load(open(t+'.json'))['per_cell']
    return (pc['low_rain|coupled']['PSNR']+pc['low_haze_rain|coupled']['PSNR'])/2
d=truec('F1w64_dec'); c=truec('F1w64_cpl')
print('=== F1 CROSS-ARCH CONFIRM (NAFNet width64) ===')
print(f'decoupled-train coupled_test = {d:.3f}')
print(f'coupled-train   coupled_test = {c:.3f}')
print(f'GAP (cpl-dec) = {c-d:+.3f} dB   [width32 ref: +3.01 dB]')

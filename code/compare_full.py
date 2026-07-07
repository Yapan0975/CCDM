import json
def truec(t):
    pc=json.load(open(t+'.json'))['per_cell']
    return (pc['low_rain|coupled']['PSNR']+pc['low_haze_rain|coupled']['PSNR'])/2,(pc['low_rain|decoupled']['PSNR']+pc['low_haze_rain|decoupled']['PSNR'])/2
print('=== FULL-SCALE F1 (600 train / 100 test) : coupled_test PSNR ===')
for w in ['32','64']:
    dc,_=truec(f'FW{w}_dec'); cc,_=truec(f'FW{w}_cpl')
    print(f'NAFNet-w{w}: decoupled-train={dc:.3f}  coupled-train={cc:.3f}  GAP(F1)={cc-dc:+.3f} dB')
print('(mini-scale 80/20 ref: w32 +3.01, w64 +3.18)')

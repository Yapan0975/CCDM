import json
def cell(t,k): return json.load(open(t+'.json'))['per_cell'][k]['PSNR']
dec=cell('LL_dec','low_noise|coupled'); cpl=cell('LL_cpl','low_noise|coupled')
print('=== LL (low-light specialist) SYNTHETIC low_noise coupled-test ===')
print(f'decoupled-train={dec:.3f}  coupled-train={cpl:.3f}  gap={cpl-dec:+.3f} dB')

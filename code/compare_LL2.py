import json
def cell(t,k): return json.load(open(t+'.json'))['per_cell'][k]['PSNR']
dec=cell('LL2_dec','low_noise|coupled'); cpl=cell('LL2_cpl','low_noise|coupled')
print('=== LL2 (Poisson-Gaussian) SYNTHETIC low_noise coupled-test ===')
print(f'decoupled(homosced)={dec:.3f}  coupled(heterosced)={cpl:.3f}  gap={cpl-dec:+.3f} dB')

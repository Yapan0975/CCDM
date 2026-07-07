"""
train_probe_c.py - trained-comparison de-risk of the method thesis.

Trains a small NAFNet on CCDM data (one mode per run) and evaluates on a deterministic
2x2 test matrix {combo} x {decoupled,coupled}. Three runs answer:
  M1 mode=decoupled, agnostic  vs  M2 mode=coupled, agnostic   -> DATA thesis
  M2 vs  M3 mode=coupled, coupling-aware (--ca)                -> METHOD thesis
The decisive cell is PSNR on the coupled test split.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from nafnet import NAFNet, CoupleNet
from dataset_ccd import CCDTrain, build_eval

def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()

def main(a):
    dev = 'cuda'
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    combos = a.combos.split(',') if a.combos else None
    params = {}
    if a.dark: params['gamma'] = (3.0, 5.0)                   # darker low-light (closer to real LOL)
    if a.pgnoise: params['noise_model'] = 'poisson'          # physically-faithful heteroscedastic noise
    params = params or None
    ds = CCDTrain(a.clean_train, a.depth, a.mode, a.rain, a.snow, crop=a.crop,
                  length=a.bs * a.iters, train_n=a.train_n, combos=combos, params=params, seed=a.seed)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, shuffle=False, drop_last=True, pin_memory=True)
    is_cn = (a.model == 'couplenet')
    if a.model == 'couplenet':
        model = CoupleNet(width=a.width)
    elif a.model == 'restormer':
        from restormer import Restormer
        model = Restormer(dim=a.width)
    else:
        model = NAFNet(width=a.width, coupling_aware=a.ca)
    model = model.to(dev)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=a.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler()
    print(f'[{a.tag}] model={a.model} mode={a.mode} ca={a.ca} train_n={a.train_n or "all"} params={nparam:.2f}M iters={a.iters}')

    model.train(); t0 = time.time(); it = 0
    for lq, gt, fld in dl:
        lq, gt, fld = lq.to(dev), gt.to(dev), fld.to(dev)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            if is_cn:
                out, pf, redeg = model(lq)
                loss = charbonnier(out, gt) + 0.1 * torch.nn.functional.l1_loss(pf, fld) \
                       + 0.1 * charbonnier(redeg, lq)
            else:
                out, pf = model(lq)
                loss = charbonnier(out, gt)
                if a.ca and pf is not None:
                    loss = loss + 0.1 * torch.nn.functional.l1_loss(pf, fld)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        it += 1
        if it % 1000 == 0:
            print(f'  [{a.tag}] it {it}/{a.iters} loss {loss.item():.4f} {(time.time()-t0)/it*1000:.0f}ms/it')

    # ---- eval on deterministic 2x2 matrix ----
    model.eval()
    items = build_eval(a.clean_test, a.depth_test or a.depth, a.rain, a.snow, combos=combos, params=params)
    agg = {}
    with torch.no_grad():
        for e in items:
            lq = torch.from_numpy(e['lq']).unsqueeze(0).to(dev)
            with torch.cuda.amp.autocast():
                out = model(lq)[0]
            out = out[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            J = e['clean'].transpose(1, 2, 0)
            ps = psnr_fn(J, out, data_range=1.0); ss = ssim_fn(J, out, data_range=1.0, channel_axis=2)
            agg.setdefault((e['combo'], e['mode']), []).append((ps, ss))
    res = {}
    for (c, m), v in sorted(agg.items()):
        v = np.array(v); res[f'{c}|{m}'] = dict(PSNR=round(float(v[:,0].mean()),3), SSIM=round(float(v[:,1].mean()),4), n=len(v))
    coupled = np.mean([res[k]['PSNR'] for k in res if k.endswith('|coupled')])
    decoup = np.mean([res[k]['PSNR'] for k in res if k.endswith('|decoupled')])
    out = dict(tag=a.tag, model=a.model, mode=a.mode, ca=bool(a.ca), train_n=a.train_n, params_M=round(nparam,2),
               coupled_test_PSNR=round(float(coupled),3), decoupled_test_PSNR=round(float(decoup),3),
               per_cell=res)
    print(json.dumps(out, indent=2))
    json.dump(out, open(f'{a.tag}.json', 'w'), indent=2)
    torch.save(model.state_dict(), f'{a.tag}.pth')
    print('saved', f'{a.tag}.json /', f'{a.tag}.pth')

if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--mode', choices=['decoupled', 'coupled', 'mixed'], required=True)
    P.add_argument('--seed', type=int, default=0)
    P.add_argument('--model', choices=['nafnet', 'couplenet', 'restormer'], default='nafnet')
    P.add_argument('--ca', action='store_true')
    P.add_argument('--train_n', type=int, default=0)
    P.add_argument('--combos', default='')        # comma-sep subset of COMBOS (e.g. low_noise)
    P.add_argument('--dark', action='store_true') # darker low-light (gamma 3-5) to match real LOL
    P.add_argument('--pgnoise', action='store_true') # physically-faithful Poisson-Gaussian noise
    P.add_argument('--tag', required=True)
    P.add_argument('--clean_train', default='./clean_train')
    P.add_argument('--clean_test', default='./clean_test')
    P.add_argument('--depth', default='./depth')
    P.add_argument('--depth_test', default='')
    P.add_argument('--rain', default='./OneRestore/syn_data/data/rain_mask')
    P.add_argument('--snow', default='./OneRestore/syn_data/data/snow_mask')
    P.add_argument('--iters', type=int, default=12000)
    P.add_argument('--bs', type=int, default=8)
    P.add_argument('--crop', type=int, default=256)
    P.add_argument('--width', type=int, default=32)
    P.add_argument('--lr', type=float, default=1e-3)
    P.add_argument('--workers', type=int, default=6)
    main(P.parse_args())

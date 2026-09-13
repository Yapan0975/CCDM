#!/bin/bash
# run_surfaces_pg.sh - as soon as all seven lambda-sweep checkpoints of a seed exist, evaluate that
# seed's train-lambda x test-lambda surface with Poisson-Gaussian test renders (GPU 1 only).
# Seeds are handled in parallel; each one waits for its own checkpoints and gives up if one of
# its training jobs failed.
cd ~/Documents/yping/composite-restore
SEEDS="${SEEDS:-1 2 3 4}"
for s in $SEEDS; do
  (
    tags="LAM_s${s}_l000,LAM_s${s}_l025,LAM_s${s}_l050,LAM_s${s}_l075,LAM_s${s}_l100,LAM_s${s}_rand,LAM_s${s}_mix"
    while true; do
      ready=1
      for t in ${tags//,/ }; do
        if [ -f "$t.failed" ]; then echo "[$(date +%H:%M:%S)] seed $s: $t failed, surface skipped"; exit 1; fi
        [ -f "$t.pth" ] && [ ! -f "$t.running" ] || ready=0
      done
      [ "$ready" = 1 ] && break
      sleep 60
    done
    echo "[$(date +%H:%M:%S)] seed $s: checkpoints ready, evaluating surface"
    CUDA_VISIBLE_DEVICES=1 python3 -u eval_lambda_surface.py --pgnoise --tags "$tags" \
      --test_lams 0,0.25,0.5,0.75,1.0 --combos low_rain,low_haze_rain \
      --out "lam_surface_pg_s${s}.json" > "lam_surface_pg_s${s}.log" 2>&1
    echo "[$(date +%H:%M:%S)] seed $s: surface finished (rc=$?)"
  ) &
done
wait
echo SURFACES_PG_DONE > surfaces_pg_DONE.flag
echo "[$(date +%H:%M:%S)] all surfaces finished"

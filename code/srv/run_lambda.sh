#!/bin/bash
# run_lambda.sh - continuous coupling-strength sweep on the composite CCD track.
# 7 configs per seed: lam in {0,.25,.5,.75,1} + continuous randomisation (lam~U[0,1])
# + binary 50/50 mixed control. All with Poisson-Gaussian noise (--pgnoise) so the
# experiment matches the noise model stated in the paper.
# Uses ONLY GPU 0 and 1 (two other cards left free for other users).
cd ~/Documents/yping/composite-restore
GPUS=(0 1)
SEEDS="${SEEDS:-0}"
declare -A PIDS
JOBS=()
for s in $SEEDS; do
  JOBS+=("LAM_s${s}_l000|--lam 0.0|$s")
  JOBS+=("LAM_s${s}_l025|--lam 0.25|$s")
  JOBS+=("LAM_s${s}_l050|--lam 0.5|$s")
  JOBS+=("LAM_s${s}_l075|--lam 0.75|$s")
  JOBS+=("LAM_s${s}_l100|--lam 1.0|$s")
  JOBS+=("LAM_s${s}_rand|--lam rand|$s")
  JOBS+=("LAM_s${s}_mix|--mode mixed|$s")
done
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag extra seed <<< "$job"
  [ -f "${tag}.pth" ] && { echo "[skip] $tag already done"; continue; }
  placed=
  while [ -z "$placed" ]; do
    for g in "${GPUS[@]}"; do
      p=${PIDS[$g]}
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
        MODE=coupled
        case "$extra" in *"--mode mixed"*) MODE=mixed; extra="";; esac
        CUDA_VISIBLE_DEVICES=$g nohup python3 -u train_probe_c.py \
          --mode $MODE --seed "$seed" --tag "$tag" $extra \
          --clean_train clean_train_full --depth depth_full \
          --clean_test clean_test100 --depth_test depth \
          --combos low_rain,low_haze_rain --pgnoise \
          --width 32 --iters 16000 > "${tag}.log" 2>&1 &
        PIDS[$g]=$!
        echo "[$(date +%H:%M:%S)] launch $tag on GPU$g (pid ${PIDS[$g]})"
        placed=1; sleep 8; break
      fi
    done
    [ -z "$placed" ] && sleep 20
  done
done
wait
echo ALL_LAMBDA_DONE > lambda_DONE.flag
echo "[$(date +%H:%M:%S)] all lambda training finished"

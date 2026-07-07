#!/bin/bash
# Experiment A launcher: 15 detection FT jobs (5 seeds x {decoupled,coupled,mixed}),
# one job per GPU (17.5GB each), GPU slots = 3,0,2. Auto-fills a GPU when its job finishes.
cd ~/pami_pilots
GPUS=(3 0 2)
declare -A PIDS
JOBS=()
for s in 0 1 2 3 4; do
  for m in decoupled coupled mixed; do
    ab=dec; [ "$m" = coupled ] && ab=cpl; [ "$m" = mixed ] && ab=mix
    JOBS+=("exd_${ab}_s${s}|$m|$s")
  done
done
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag m s <<< "$job"
  placed=
  while [ -z "$placed" ]; do
    for g in "${GPUS[@]}"; do
      p=${PIDS[$g]}
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
        CUDA_VISIBLE_DEVICES=$g nohup python3 -u pilot2_exdark.py --mode "$m" --seed "$s" \
          --out "${tag}.json" > "${tag}.log" 2>&1 &
        PIDS[$g]=$!
        echo "[$(date +%H:%M:%S)] launch $tag on GPU$g pid ${PIDS[$g]}"
        placed=1; sleep 12; break
      fi
    done
    [ -z "$placed" ] && sleep 20
  done
done
wait
echo ALL_EXDARK_DONE > exdark_DONE.flag

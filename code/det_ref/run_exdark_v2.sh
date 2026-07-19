#!/bin/bash
# Re-run the ExDark fine-tuning campaign with artefact saving (pilot2_exdark_v2.py):
# 4 modes x 5 seeds + one off-the-shelf eval = 21 jobs, SINGLE GPU (sequential).
# Outputs rerun_exd_*.json / .pth / _dets.json in ~/pami_pilots (originals untouched).
cd ~/pami_pilots
GPUS=(0)
declare -A PIDS
JOBS=()
for s in 0 1 2 3 4; do
  for m in decoupled coupled mixed cleanft; do
    ab=dec; [ "$m" = coupled ] && ab=cpl; [ "$m" = mixed ] && ab=mix; [ "$m" = cleanft ] && ab=cleanft
    JOBS+=("rerun_exd_${ab}_s${s}|$m|$s")
  done
done
JOBS+=("rerun_exd_offtheshelf|clean|0")
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag m s <<< "$job"
  placed=
  while [ -z "$placed" ]; do
    for g in "${GPUS[@]}"; do
      p=${PIDS[$g]}
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
        CUDA_VISIBLE_DEVICES=$g nohup python3 -u pilot2_exdark_v2.py --mode "$m" --seed "$s" \
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
echo ALL_EXDARK_V2_DONE > exdark_v2_DONE.flag

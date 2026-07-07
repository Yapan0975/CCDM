#!/bin/bash
# P1.1 (variance-matched LL, 15 runs) + P1.3 (F2 multi-seed, 10 runs) on GPUs 0/1/2 (2-up each).
# Set CCD_DIR to your CCD working directory (holds clean_train_full/, clean_test100/, depth*/, masks).
cd "${CCD_DIR:-$HOME/Documents/yping/composite-restore}"
LANES=(0 0 1 1 2 2)
MAX=${#LANES[@]}
JOBS=()
for s in 0 1 2 3 4; do
  for m in decoupled coupled mixed; do
    ab=dec; [ "$m" = coupled ] && ab=cpl; [ "$m" = mixed ] && ab=mix
    JOBS+=("LLv_${ab}_s${s}|--mode $m --seed $s --combos low_noise --dark --pgnoise --varmatch --iters 12000")
  done
done
for s in 0 1 2 3 4; do
  JOBS+=("F2ag_s${s}|--mode coupled --model nafnet --seed $s --combos low_rain,low_haze_rain --iters 16000")
  JOBS+=("F2cn_s${s}|--mode coupled --model couplenet --seed $s --combos low_rain,low_haze_rain --iters 16000")
done
i=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag args <<< "$job"
  while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do sleep 5; done
  g=${LANES[$((i % MAX))]}
  echo "[$(date +%H:%M:%S)] launch $tag on GPU$g"
  CUDA_VISIBLE_DEVICES=$g python3 -u train_probe_c.py $args --tag "$tag" \
    --clean_train clean_train_full --depth depth_full --clean_test clean_test100 --depth_test depth \
    --width 32 > "${tag}.log" 2>&1 &
  i=$((i+1)); sleep 3
done
wait
echo "[train done] running P1.1 real evals"
for s in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 python3 eval_real_lol.py --dec LLv_dec_s${s}.pth --cpl LLv_cpl_s${s}.pth \
    --root lolv2_test --low_dir Input --high_dir GT --width 32 --out realv_s${s}_cpl.json > /dev/null 2>&1
  CUDA_VISIBLE_DEVICES=0 python3 eval_real_lol.py --dec LLv_dec_s${s}.pth --cpl LLv_mix_s${s}.pth \
    --root lolv2_test --low_dir Input --high_dir GT --width 32 --out realv_s${s}_mix.json > /dev/null 2>&1
done
python3 agg_p1.py > p1_RESULTS.txt 2>&1
echo P1_DONE > p1_DONE.flag

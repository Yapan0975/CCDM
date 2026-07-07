#!/bin/bash
# Backfill the one missing LL decoupled seed (s1) + its real evals, then re-aggregate.
cd ~/Documents/yping/composite-restore
CUDA_VISIBLE_DEVICES=0 python3 -u train_probe_c.py --mode decoupled --seed 1 --tag LL_s1_dec \
  --clean_train clean_train_full --depth depth_full --clean_test clean_test100 --depth_test depth \
  --combos low_noise --dark --pgnoise --width 32 --iters 12000
CUDA_VISIBLE_DEVICES=0 python3 eval_real_lol.py --dec LL_s1_dec.pth --cpl LL_s1_cpl.pth \
  --root lolv2_test --low_dir Input --high_dir GT --width 32 --out real_s1_cpl.json
CUDA_VISIBLE_DEVICES=0 python3 eval_real_lol.py --dec LL_s1_dec.pth --cpl LL_s1_mix.pth \
  --root lolv2_test --low_dir Input --high_dir GT --width 32 --out real_s1_mix.json
python3 agg_multiseed.py > multiseed_RESULTS.txt 2>&1
echo BACKFILL_DONE > backfill_DONE.flag

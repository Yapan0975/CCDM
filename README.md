# CCDM / CCD - Coupled Composite Degradation Model and Benchmark

Code, benchmark generator, and raw results for the paper
**"Coupling Structure as a Training Domain in Composite Restoration and Low-light Detection."**

The central object is the **coupling structure** of composite-degradation synthesis: whether
co-occurring degradations are composed independently (*decoupled*, the field default) or with explicit
physical cross-terms (*coupled*). CCDM renders both as matched pairs and interpolates between them with a
continuous **coupling strength** `lambda` (0 = decoupled, 1 = coupled) at matched marginal severity. CCD
evaluates every trained model across coupling structures: a 2x2 train-structure x test-structure
protocol and a train-lambda x test-lambda surface.

> **Update (2026-09-13).** All composite-track experiments were re-run with the Poisson-Gaussian noise
> model stated in the paper (`--pgnoise`), and the coupling-strength sweep was added. The earlier
> composite runs used the legacy `sigma0/L` noise branch; they are kept for reference in
> `latex/results/legacy_sigma0L/` and are not reported in the current paper. The low-light (LOLv2),
> real-noise and ExDark detection results already used the Poisson-Gaussian model and are unchanged.

---

## 1. Directory layout

```
code/
  ccdm.py                  canonical CCDM synthesis (Poisson-Gaussian noise, continuous lambda, --varmatch control)
  make_ccd.py              generate the CCD benchmark (depth -> degrade in both modes)
  dataset_ccd.py           on-the-fly CCDM dataset (train: random combo/params, fixed or random lambda; eval: deterministic)
  nafnet.py                NAFNet (agnostic) and CoupleNet (coupling-aware) backbones
  restormer.py             Restormer backbone (cross-architecture check)
  train_probe_c.py         train a restorer (--mode decoupled/coupled/mixed, --lam coupling strength, --pgnoise)
  verify_lambda.py         check that lambda=0/1 reproduce the decoupled/coupled renders at fixed severity
  eval_lambda_surface.py   train-lambda x test-lambda PSNR surface for one seed
  eval_per_scene.py        per-scene PSNR/SSIM of checkpoints on the CCD test matrix
  ablate_crossterm.py      per-cross-term test-time ablation
  eval_onerestore_cdd11.py OneRestore with its released CDD-11 weights on the CCD test
  eval_baseline_ccd.py     PromptIR / AdaIR with their released weights on the CCD test
  eval_real_lol.py         evaluate a trained low-light model on real LOLv2
  eval_real_lol_per.py     per-image LOLv2 evaluation (scene-level statistics)
  real_coupling_stats.py   signal dependence of real LOLv2 noise vs CCDM renders
  agg_pg.py                composite, OneRestore, off-the-shelf and architecture tables; ablation; lambda surface
  stats_ci.py              seed-paired 95% CIs + exact sign-flip permutation p (TRACK=pg default, TRACK=legacy)
  ccd_scene_bootstrap.py   scene-level and hierarchical seed x scene bootstrap (--track pg default, --track legacy)
  params_flops.py          params/FLOPs for NAFNet-w32 vs CoupleNet
  gen_pairs.py, make_fig1_detail.py, make_qualitative_fig.py, make_lambda_fig.py,
  make_framework_fig.py, make_reviewer_figs.py      paper figures
  agg_multiseed.py, gen_figs.py, eval_onerestore_ccd.py, compare_*.py, probe_*.py   earlier scripts
  srv/
    run_pg_campaign.sh, pg_queue.txt   job launcher and the exact command of every Poisson-Gaussian composite run
    run_lambda.sh, run_surfaces_pg.sh  seed-0 lambda sweep; per-seed surface evaluation
    train_onerestore_ccd.py   fine-tune OneRestore on CCD (all-in-one cross-architecture check, --pgnoise)
    agg_p1.py                 aggregate the variance-matched LOLv2 table
    run_p1.sh                 earlier LOLv2 + architecture sweep
    ccdm.py, train_probe_c.py identical copies of the code/ versions
  det_ref/
    pilot2_exdark.py          fine-tune RetinaNet on degraded (or clean, --mode cleanft) COCO, eval real ExDark
    pilot3_mix.py             real/mixed fine-tune: closing the synthetic-to-real gap
    agg_exdark.py, agg_p3.py  aggregate the detection tables
latex/results/             raw result files (193) and legacy_sigma0L/ (66); SHA-256 of all 259 in MANIFEST.sha256
```

This repository is **code + raw results only**; the LaTeX manuscript, supplement, and bibliography
are uploaded separately to the journal. All aggregation scripts read from `latex/results/` by default;
override with the environment variable `RESULTS_DIR`.

**Dependencies** (`requirements.txt`): aggregation only needs `numpy`, `scipy`; `params_flops.py`
additionally needs `torch` (`thop` optional, for measured FLOPs); full training/evaluation needs
`torch`, `torchvision`, `opencv-python`, `scikit-image`, `pycocotools`, `tqdm`. The figure scripts need
`matplotlib`. The PromptIR/AdaIR evaluation expects their official repositories and weights in a sibling
`baselines/` directory.

---

## 2. Data sources and licences

| Asset | Source | Use |
|-------|--------|-----|
| Clean images | DIV2K (NTIRE 2017 challenge) | research; redistributed by the original NTIRE terms |
| Rain / snow masks | OneRestore / CDD-11 `syn_data` | research; cite OneRestore |
| Monocular depth | MiDaS (`intel-isl/MiDaS`, small) | computed locally, not redistributed |
| Real low-light restoration | LOLv2 (100 paired scenes) | research |
| Real low-light detection | ExDark (1473 images, 4795 boxes) | research |
| Detection fine-tuning images | COCO val2017 (4000/1000 split) | research |

Snow masks are loaded by the generator but **no reported result uses snow**; the main composite
protocol is `low+rain+noise` and `low+haze+rain+noise`, plus `low+noise` for the LOLv2 cross-term only.

---

## 3. Generate the CCD benchmark

```bash
python code/make_ccd.py \
    --clean  <DIV2K_clean_dir> \
    --rain   <rain_mask_dir> \
    --snow   <snow_mask_dir> \
    --out    ccd \
    --n      <num_scenes>
```
Produces matched `decoupled/` and `coupled/` renders per combo, the shared latent GT (L, t, sigma), and
`manifest.json`. Test scenes come from `clean_test100` (disjoint DIV2K split). Check the coupling-strength
construction with `python code/verify_lambda.py --pgnoise --n 100` (output released as
`latex/results/verify_lambda_pg_n100.txt`).

---

## 4. Training (five seeds, `s = 0..4`)

The scripts assume a working directory containing the generated CCD (`clean_train_full`,
`clean_test100`, `depth_full`, `depth`, masks). Every composite command below also takes
`--clean_train clean_train_full --depth depth_full --clean_test clean_test100 --depth_test depth
--combos low_rain,low_haze_rain --pgnoise`; the exact command line of every run is listed in
`code/srv/pg_queue.txt`.

| Experiment | Command (per seed/mode) |
|------------|--------------------------|
| Composite, fixed coupling strength (NAFNet-w32; `lambda=0/1` are the decoupled/coupled specialists) | `python code/train_probe_c.py --mode coupled --lam {0.0,0.25,0.5,0.75,1.0} --seed s --width 32 --iters 16000 --tag LAM_s${s}_l{000,025,050,075,100}` |
| Composite, randomised | binary 50/50: `--mode mixed --tag LAM_s${s}_mix`; continuous `lambda ~ U[0,1]`: `--mode coupled --lam rand --tag LAM_s${s}_rand` |
| Architecture (CoupleNet vs NAFNet) | `python code/train_probe_c.py --model couplenet --mode coupled --seed s --width 32 --iters 16000 --tag F2cnPG_s${s}` (NAFNet baseline = `LAM_s${s}_l100`) |
| Scale ablation (single runs) | `--model {nafnet,couplenet} --mode coupled --train_n {80,20} --width 32 --iters 12000 --tag {AGpg,CNpg}_n{80,20}` |
| Architecture checks (single runs) | `--mode {decoupled,coupled} --width 64 --iters 16000 --tag FW64pg_{dec,cpl}`; `--model restormer --mode {decoupled,coupled} --width 32 --iters 12000 --tag FRpg_{dec,cpl}` |
| OneRestore re-training | `python code/srv/train_onerestore_ccd.py --mode {decoupled,coupled,mixed} --seed s --pgnoise --iters 6000 --tag ORpg_{dec,cpl,mix}_s${s}` |
| Variance-matched LOLv2 specialist | `python code/train_probe_c.py --mode {decoupled,coupled,mixed} --seed s --combos low_noise --dark --pgnoise --varmatch --iters 12000 --tag LLv_{ab}_s${s}` |
| ExDark detection | `python code/det_ref/pilot2_exdark.py --mode {clean,cleanft,decoupled,coupled,mixed} --seed s --coco <coco> --exdark_root <exdark> --exdark_json <exdark_test.json>` (`cleanft` = same-budget clean-COCO fine-tune control, no degradation) |
| Closing the synthetic-to-real gap | `python code/det_ref/pilot3_mix.py --mode {real,mixed} --seed s` (real ExDark train, optionally mixed with COCO coupled synthesis) |

`code/srv/run_pg_campaign.sh` runs `pg_queue.txt` on two GPUs, three jobs per card; the `cd` at the
top of each sweep script should point to your working directory. The launcher does not account for GPU
memory: Restormer-w32 at batch 8 needs about 12 GB, so do not put two Restormer jobs on one 32 GB card.

**Evaluation-only runs** (need the trained checkpoints):

```bash
# train-lambda x test-lambda surface, one seed
python code/eval_lambda_surface.py --pgnoise --test_lams 0,0.25,0.5,0.75,1.0 --combos low_rain,low_haze_rain \
    --tags LAM_s0_l000,LAM_s0_l025,LAM_s0_l050,LAM_s0_l075,LAM_s0_l100,LAM_s0_rand,LAM_s0_mix --out lam_surface_pg_s0.json
# per-scene evaluation of the 15 decoupled/coupled/mixed checkpoints
python code/eval_per_scene.py --pgnoise --combos low_rain,low_haze_rain \
    --tags LAM_s0_l000,LAM_s0_l100,LAM_s0_mix,...,LAM_s4_mix --out per_scene_PG.json
# per-cross-term ablation
python code/ablate_crossterm.py --pgnoise --ckpt LAM_s0_l100.pth --out ablate_pg_cpl.json
python code/ablate_crossterm.py --pgnoise --ckpt LAM_s0_l000.pth --out ablate_pg_dec.json
# off-the-shelf restorers
python code/eval_onerestore_cdd11.py --pgnoise --out ORcdd11_pg.json
python code/eval_baseline_ccd.py --model promptir --pgnoise --gpu 0 --ckpt <PromptIR weights>
python code/eval_baseline_ccd.py --model adair --pgnoise --gpu 0 --ckpt <AdaIR weights>
```

---

## 5. Reproduce the tables

The aggregation scripts locate `latex/results/` relative to their own file, so they run from any
working directory with no extra setup. Table numbers follow the current manuscript.

```bash
python code/agg_pg.py               # Tables 3, 4, 5, 7; ablation; lambda surface statistics (writes pg_summary.json)
python code/srv/agg_p1.py           # Table 6  (variance-matched LOLv2)
python code/det_ref/agg_exdark.py   # Table 8  (ExDark AP/AP50/AR + seed-paired tests)
python code/det_ref/agg_p3.py       # Table 9  (closing the synthetic-to-real gap: real vs mixed)
python code/params_flops.py         # Table 7  params/FLOPs
python code/stats_ci.py             # 95% CIs + exact sign-flip permutation p (Table 2 family)
python code/make_lambda_fig.py --surfaces "latex/results/lam_surface_pg_s*.json" --outdir figs   # Figure 4
```
To read results from elsewhere, set `RESULTS_DIR` --- Bash: `RESULTS_DIR=/path python code/agg_pg.py`;
PowerShell: `$env:RESULTS_DIR='C:\path'; python code\agg_pg.py`. `agg_pg.py` and `ccd_scene_bootstrap.py`
write their summary JSON into the results directory; point `RESULTS_DIR` / `--dir` at a copy if the
released files should stay byte-identical. The Bonferroni threshold (0.007 across seven tests) is applied
where stated, and per-seed values are tabulated in supplementary Section S4.

The earlier `sigma0/L` composite runs can be inspected with, e.g.,
`RESULTS_DIR=latex/results/legacy_sigma0L python code/agg_multiseed.py`.

### 5b. Scene-level robustness analyses (supplement S5/S6)

Seed-paired tests quantify training stochasticity; these analyses quantify scene-sampling uncertainty
and measure the coupling of real data directly.

```bash
# scene-level + hierarchical seed x scene bootstrap (reads per_scene_PG.json and per_image_LLv.json)
python code/ccd_scene_bootstrap.py --track pg --dir latex/results --llv_dir latex/results --boot 5000
# real-noise coupling measurement (LOLv2 pairs + CCDM renders; CPU-only)
python code/real_coupling_stats.py --synthetic --out real_coupling_stats_v2.json
```

The corresponding outputs are released in `latex/results/`: `per_scene_PG.json`, `per_image_LLv.json`,
`bootstrap_summary_pg.json`, `real_coupling_stats_v2.json`.

### 5c. ExDark campaign re-run with saved artefacts

`code/det_ref/pilot2_exdark_v2.py` repeats the ExDark fine-tuning campaign identically to
`pilot2_exdark.py` but saves the fine-tuned checkpoint, the raw detections (score >= 0.01), and
per-class AP/AR; `code/det_ref/run_exdark_v2.sh` runs all 21 jobs (4 modes x 5 seeds +
off-the-shelf) on a single GPU. `code/det_ref/exdark_bootstrap.py` then computes image-level
bootstrap CIs (with a custom COCO `accumulate` validated against pycocotools on the full sample),
a rerun-vs-published consistency table, the per-class AP table, and score-floor sensitivity.
The 21 re-run summary JSONs are released as `latex/results/rerun_exd_*.json`; per-seed APs drift
by up to ~0.02-0.03 from the published values (training nondeterminism across environments) while
the ordering (coupled > mixed > decoupled, both far below the non-degraded references) is preserved.
Full detection dumps are regenerable with the script.

---

## 6. Raw results manifest

`latex/results/` holds one file per (experiment, mode, seed), plus the earlier `sigma0/L` composite runs
in `legacy_sigma0L/`. Integrity is recorded in `latex/results/MANIFEST.sha256` (259 entries). Verify with:
```bash
cd latex/results && sha256sum -c MANIFEST.sha256     # Bash / Git Bash
```
On Windows PowerShell, compare hashes with
`Get-FileHash latex\results\*.json -Algorithm SHA256` against the values in `MANIFEST.sha256`.

---

## 7. Notes on reproducibility

- The **canonical** synthesis is `code/ccdm.py` (identical to `code/srv/ccdm.py`). `lambda=0` and
  `lambda=1` reproduce the decoupled and coupled renders to within 5e-7 per pixel; the mean noise
  standard deviation is 3.1% lower at `lambda=1` than at `lambda=0` because the rain and airlight cross
  terms change the signal on which the Poisson-Gaussian variance depends.
- The legacy `sigma0/L` "gain" noise is retained only as a labelled reference branch. It was used by the
  earlier composite runs in `legacy_sigma0L/` and by the negative pilot in supplement S2, and by no other
  reported result.
- Seeds fix both the model initialisation and the on-the-fly synthesis stream, so seed-to-seed
  variation reflects genuine training stochasticity.
- Unless stated otherwise, p-values are raw seed-paired t-tests over five seeds; multiple-comparison
  handling is stated explicitly in the paper (Bonferroni threshold 0.007).
- **ExDark detection split.** `pilot2_exdark.py` / `pilot3_mix.py` use the official ExDark train/test
  partition converted to COCO format (`exdark_train.json`, 5890 imgs / `exdark_test.json`, 1473 imgs,
  disjoint). The twelve ExDark categories map to torchvision-COCO classes by motorbike->motorcycle,
  people->person, table->dining table (identity otherwise). Configure dataset paths via the
  `EXDARK_ROOT` / `EXDARK_TRAIN` / `EXDARK_TEST` / `COCO_ROOT` env vars or the `--exdark_*` / `--coco`
  flags.

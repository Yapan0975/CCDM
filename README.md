# CCDM / CCD - Coupled Composite Degradation Model and Benchmark

Code, benchmark generator, and raw results for the paper
**"Degradations Do Not Add Up: Coupling Structure as a Training Domain in Composite Restoration and
Low-light Detection."**

The central object is the **coupling structure** of composite-degradation synthesis: whether
co-occurring degradations are composed independently (*decoupled*, the field default) or with explicit
physical cross-terms (*coupled*). CCDM renders both as matched pairs; CCD is the resulting 2x2
train-structure x test-structure benchmark.

---

## 1. Directory layout

```
code/
  ccdm.py                 canonical CCDM synthesis (Poisson-Gaussian noise; --varmatch control)
  make_ccd.py             generate the CCD benchmark (depth -> degrade in both modes)
  dataset_ccd.py          on-the-fly CCDM dataset (train: random combo/params; eval: deterministic)
  nafnet.py               NAFNet (agnostic) and CoupleNet (coupling-aware) backbones
  restormer.py            Restormer backbone (cross-architecture check)
  train_probe_c.py        train a restorer on CCD (mode = decoupled/coupled/mixed)
  eval_real_lol.py        evaluate a trained low-light model on real LOLv2
  params_flops.py         params/FLOPs for NAFNet-w32 vs CoupleNet (Table 8)
  gen_figs.py             paper figures
  agg_multiseed.py        aggregate composite 2x2 + mixed cure (Tables 2, 6)
  srv/
    train_onerestore_ccd.py   fine-tune OneRestore on CCD (all-in-one cross-arch check)
    agg_p1.py                 aggregate var-matched LOLv2, architecture, OneRestore (Tables 5, 7, 8)
  det_ref/
    pilot2_exdark.py          fine-tune RetinaNet on degraded COCO, evaluate on real ExDark (Table 9)
    pilot3_mix.py             real/mixed fine-tune: closing the synthetic-to-real gap (Table 10)
    agg_exdark.py             aggregate ExDark AP/AP50/AR + seed-paired tests (Table 9)
    agg_p3.py                 aggregate closing-the-gap real/mixed runs (Table 10)
latex/
  main_pr.tex             manuscript (Elsevier elsarticle, Pattern Recognition)
  supplement.tex          supplementary material (ad-hoc-noise pilot, per-seed tables, repro)
  refs.bib                bibliography
  results/                raw per-seed JSON for every reported cell (150 files, SHA-256 in MANIFEST)
```

All aggregation scripts read JSON from `latex/results/` by default; override with the environment
variable `RESULTS_DIR`.

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
`manifest.json`. Test scenes come from `clean_test100` (disjoint DIV2K split).

---

## 4. Training (five seeds, `s = 0..4`)

The scripts assume a working directory containing the generated CCD (`clean_train_full`,
`clean_test100`, `depth_full`, `depth`, masks). Set it once:
```bash
export CCD_DIR=/path/to/your/ccd_workdir   # used by the run_*.sh sweeps
```

| Experiment | Command (per seed/mode) |
|------------|--------------------------|
| Composite specialization (NAFNet-w32) | `python code/train_probe_c.py --model nafnet --mode {decoupled,coupled,mixed} --seed s --combos low_rain,low_haze_rain --width 32 --iters 16000 --tag FW_s${s}_{dec,cpl,mix}` |
| Variance-matched LOLv2 specialist | `python code/train_probe_c.py --mode {decoupled,coupled,mixed} --seed s --combos low_noise --dark --pgnoise --varmatch --iters 12000 --tag LLv_{ab}_s${s}` |
| Architecture (agnostic vs aware) | `python code/train_probe_c.py --model {nafnet,couplenet} --mode coupled --seed s --combos low_rain,low_haze_rain --iters 16000` |
| OneRestore re-training | `python code/srv/train_onerestore_ccd.py --mode {decoupled,coupled,mixed} --seed s` |
| ExDark detection | `python code/det_ref/pilot2_exdark.py --mode {clean,decoupled,coupled,mixed} --seed s --coco <coco> --exdark_root <exdark> --exdark_json <exdark_test.json>` |
| Deployment: closing the gap (Table 10) | `python code/det_ref/pilot3_mix.py --mode {real,mixed} --seed s` (real ExDark train, optionally mixed with COCO coupled synthesis) |

Convenience sweeps: `code/srv/run_p1.sh` (LOLv2 + architecture), `code/det_ref/run_exdark.sh`
(detection). These were run on a 4xGPU host; the `cd` at the top of each sweep should point to your
`$CCD_DIR`.

---

## 5. Reproduce the tables

The aggregation scripts locate `latex/results/` relative to their own file, so they run from any
working directory with no extra setup:

```bash
python code/agg_multiseed.py        # Tables 2, 6  (composite 2x2 + mixed cure)
python code/srv/agg_p1.py           # Tables 5, 7, 8  (var-matched LOLv2, architecture, OneRestore)
python code/det_ref/agg_exdark.py   # Table 9      (ExDark AP/AP50/AR + seed-paired tests)
python code/det_ref/agg_p3.py       # Table 10     (closing the synthetic-to-real gap: real vs mixed)
python code/params_flops.py         # Table 8      params/FLOPs
python code/stats_ci.py             # 95% CIs + exact sign-flip permutation p (Statistics para)
```
To read results from elsewhere, set `RESULTS_DIR` --- Bash: `RESULTS_DIR=/path python code/agg_multiseed.py`;
PowerShell: `$env:RESULTS_DIR='C:\path'; python code\agg_multiseed.py`. Each script prints means +/- std
and the seed-paired t-tests; the Bonferroni threshold (0.007 across seven tests) is applied where
stated. Per-seed values are also tabulated in `supplement.tex` (S2).

---

## 6. Raw results manifest

`latex/results/` holds one JSON per (experiment, mode, seed). Integrity is recorded in
`latex/results/MANIFEST.sha256`. Verify with:
```bash
cd latex/results && sha256sum -c MANIFEST.sha256     # Bash / Git Bash
```
On Windows PowerShell, compare hashes with
`Get-FileHash latex\results\*.json -Algorithm SHA256` against the values in `MANIFEST.sha256`.

---

## 7. Notes on reproducibility

- The **canonical** synthesis is `code/ccdm.py` (identical to `code/srv/ccdm.py`); the legacy
  `sigma0/L` "gain" noise is retained only as a labelled reference branch and is used in no reported
  result (see supplement S1).
- Seeds fix both the model initialisation and the on-the-fly synthesis stream, so seed-to-seed
  variation reflects genuine training stochasticity.
- All p-values are raw seed-paired t-tests over five seeds; multiple-comparison handling is stated
  explicitly in the paper (Bonferroni threshold 0.007).

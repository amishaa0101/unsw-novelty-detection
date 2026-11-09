# PROJECT_SPEC.md: Detecting Unseen Cyberattacks with Statistical Novelty Detection (UNSW-NB15)

You are Claude Code. Build this project from scratch, following this spec phase by phase.
This is an M.Tech course project (CS 328: Data Science, IIT Gandhinagar) by a team of two students.
We must understand and defend every line in a viva, so correctness, simplicity and honesty matter more than speed or fancy code.

---

## 0. How you must work (read first)

1. **Plan first.** Before writing code, read this whole file, then show me a short plan for Phase 0 and Phase 1.
2. **Work in phases.** Finish one phase, run it, show me the real output, then move to the next one. Do not jump ahead.
3. **Stop at checkpoints** marked `CHECKPOINT` and wait for my confirmation before continuing.
4. **Never invent numbers.** Every number in any README, report table or figure must come from a file the code produced. If something failed or was not run, say so.
5. **Verify, don't assume.** This spec contains facts from memory (column names, class counts, file names). Check each against the real data and tell me when the data disagrees with this spec. Adapt the code to the real data.
6. **Keep code simple and readable.** Plain functions and small classes. Short comments in simple English explaining *why*, not just *what*. No unnecessary abstraction, no frameworks beyond the listed libraries.
7. **Reproducibility.** A single global seed (default 42) controls every random step. Log the seed, library versions and config with every run.
8. **Git.** Initialise a git repo. Commit at the end of each phase with a clear message. Add `data/` and large outputs to `.gitignore`.
9. **Ask when unsure.** If a design choice in this spec turns out to be impossible or clearly wrong on the real data, stop and explain the problem with evidence, then propose a fix.

---

## 1. Project summary

**Problem.** Normal attack classifiers assume the attacks seen in training are the only ones that exist (closed world). A new (zero-day) attack gets forced into a known label, often "Normal", often with high confidence.

**Main question.** Can we detect attack types the system has never seen, while keeping false alarms on normal traffic below a fixed level, and does this still work after removing the dataset's known shortcut features (TTL related)?

**Approach.**
- Simulate new attacks with **leave-one-attack-out (LOAO)**: for each of the 9 attack types, hide it completely from training and calibration, and use it only in testing.
- Learn what known traffic looks like with **three novelty detectors**, each fitted **per known class**:
  1. GMM (fitted with EM): score = negative log likelihood
  2. PCA: score = reconstruction error
  3. k nearest neighbour with FAISS: score = mean distance to the k nearest training flows
- Normalise each class's scores on one calibration part (`calib_norm`), combine them into one statistic, and turn it into a **pooled conformal p-value** using a second calibration part (`calib_final`), so the false alarm rate on known traffic is controlled (default level alpha = 0.01). Also try **Benjamini-Hochberg (BH)**. (Two-stage design, see Phase 5.)
- Compare against a **baseline**: a standard classifier with a confidence threshold.
- Run everything **with and without shortcut features**, and report **bootstrap 95% confidence intervals**.

**Outcomes for a hidden attack flow** (report all three separately):
1. Flagged **unknown** (best case)
2. Assigned a **wrong known attack** label (an alarm, but novelty not recognised)
3. **Missed**, assigned "Normal" (worst case)

---

## 2. Environment

- Python 3.11 (3.10+ is fine). Use a virtual environment.
- Target machine: a normal laptop, CPU only, 8 to 16 GB RAM. No GPU.
- `requirements.txt` (pin versions after installing):
  - numpy, pandas, scipy, scikit-learn, faiss-cpu, matplotlib, seaborn, pyyaml, tqdm, joblib, pytest
- If `faiss-cpu` fails to install, tell me. Do not silently replace FAISS with something else. (A fallback using `sklearn.neighbors.NearestNeighbors` is acceptable only after I agree.)

---

## 3. Data

### 3.1 Source
Use the **official partitioned files** of UNSW-NB15:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Official page: https://research.unsw.edu.au/projects/unsw-nb15-dataset (the download may require a manual step).

**Do not scrape or auto-download.** Expect the two CSV files in `data/raw/`. If they are missing, print clear instructions telling me where to put them, then exit.

### 3.2 Expected facts to verify (from memory, may be slightly off)
- About 175,341 rows (training file) and 82,332 rows (testing file).
- About 45 columns, including: `id, dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports, attack_cat, label`.
- No IP addresses, ports or timestamps in these files (good: those would be shortcuts).
- `attack_cat` values: Normal, Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms.
- Text columns: `proto` (100+ distinct values), `service` (`-` means unknown), `state`.

### 3.3 Data rules
- **Combine** both files into one table. We make our own splits (the official split is not designed for LOAO).
- **Drop `id`** (a row index, not a feature; it can leak ordering).
- **Normalise labels**: strip whitespace in `attack_cat`, and unify spelling variants if any (for example "Backdoors" to "Backdoor"). Log what you changed.
- `label` (0/1) must equal `attack_cat == "Normal"` negated. Check this and report mismatches.
- **Duplicates**, handled in this order and reported with counts:
  1. Exact duplicate rows (all columns equal): keep one.
  2. **Conflicting duplicates**: identical feature vectors with different `attack_cat`. Report how many groups and which label pairs. Default: drop all rows in conflicting groups (config option `conflict_policy: drop | majority`).
- **Never use `label` or `attack_cat` as features.**
- Loading adds a `source_file` column (which official CSV a row came from). It is for the audit only and **must never be used as a feature** (the Preprocessor asserts this).
- Read the CSVs with `encoding="utf-8-sig"` (the headers start with a BOM).
- `label` vs `attack_cat` mismatches: trust `attack_cat` and report the count. If there are more than 10, stop and show examples.
- Duplicate checks (exact and conflicting) are computed on **feature columns only** (excluding `id`, `label`, `attack_cat`, `source_file`).

---

## 4. Repository structure

Repo root is `D:\IIT\Data_Science` (decided at the start; no `unsw-novelty/` subfolder).

```
D:\IIT\Data_Science/
  PROJECT_SPEC.md          # this file
  README.md                # how to run + results summary (generated numbers only)
  requirements.txt
  configs/
    default.yaml
    quick.yaml             # small subsample for fast smoke tests
  data/
    raw/                   # the two official CSVs (git-ignored)
    processed/             # cleaned parquet/CSV (git-ignored)
  src/
    __init__.py
    config.py              # load YAML config, set seeds
    data.py                # load, combine, clean, deduplicate, audit
    shortcut_audit.py      # single-feature leakage audit
    preprocess.py          # Preprocessor: fit on train only, transform others
    detectors.py           # GMMDetector, PCADetector, KNNDetector, per-class wrapper
    baseline.py            # classifier + confidence-threshold baseline
    calibration.py         # conformal p-values, decision rule, BH
    split.py               # LOAO split: hide one class, stratified 60/10/10/20 (train/calib_norm/calib_final/test)
    metrics.py             # outcome rates, AUROC, bootstrap CIs
    experiment.py          # main LOAO loop
    plots.py               # all figures
    report.py              # build results tables + results.md from saved outputs
  scripts/
    run_audit.py
    run_experiments.py
    make_figures.py
  tests/
    test_preprocess.py
    test_calibration.py
    test_split.py
    test_detectors.py
  notebooks/
    01_eda.ipynb           # exploration only; final numbers come from scripts
  outputs/
    audit/                 # data audit + shortcut audit results
    runs/                  # one JSON per run
    tables/
    figures/
    logs/
```

---

## 5. Config (`configs/default.yaml`)

All tunable choices live here. Suggested defaults:

```yaml
seed: 42
data_dir: data/raw
conflict_policy: drop
split: {train: 0.6, calib_norm: 0.1, calib_final: 0.1, test: 0.2}
shortcut_features: [sttl, dttl, ct_state_ttl]   # final list confirmed after the shortcut audit
feature_settings: [with_shortcut, without_shortcut]
attack_types: [Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms]
preprocess:
  log1p: true
  proto_top_n: 10           # keep top N proto values (by training frequency), rest -> "other"
detectors: [gmm, pca, knn, baseline]
gmm:
  k_grid: [1, 2, 4, 8, 16]
  covariance_type: diag
  n_init: 3
  reg_covar: 1.0e-4
  max_fit_rows: 20000       # random subsample per class for speed; log when used
pca:
  explained_variance: 0.95
knn:
  k: 5
  index: flat               # flat = exact search
baseline:
  model: hist_gradient_boosting
calibration:
  alphas: [0.005, 0.01, 0.05, 0.1]
  main_alpha: 0.01
  bh_q: 0.01
  log_eps: 1.0e-12          # PCA and kNN scores use log(score + log_eps)
  min_scale: 1.0e-6         # IQR below this falls back to std, then to 1.0
bootstrap:
  n_resamples: 1000
  ci: 0.95
```

`quick.yaml`: same, but subsample to about 10% of rows, `n_resamples: 100`, and only 2 hidden attack types. Used for smoke tests.

---

## 6. Phases

### Phase 0: Setup
- Create the repo structure, virtual environment, `requirements.txt`, `.gitignore`, config loading and seed setting.
- Add simple `python ...` commands in the README (no Makefile, this is Windows): `audit`, `experiments`, `figures`, `test`, `quick`.
- Verify imports work (including FAISS). Commit.

### Phase 1: Load, clean, audit
In `src/data.py` and `scripts/run_audit.py`:
- Load and combine the two files, drop `id`, normalise labels.
- Verify every fact in section 3.2 and report differences.
- Deduplicate as in section 3.3.
- Save the cleaned data to `data/processed/`.
- Write `outputs/audit/data_audit.md` containing: row counts before and after each cleaning step, class counts before and after dedup, number of conflicting groups, column list with dtypes, missing values per column, number of distinct values for each text column, basic numeric summaries (min, median, max) showing how skewed the byte and packet features are, and per-class row counts for train, calib_norm, calib_final and test after dedup (with no class hidden, as a preview of split sizes).

**CHECKPOINT 1.** Show me the audit summary and wait.

### Phase 2: Shortcut audit
In `src/shortcut_audit.py`:
- For **each feature on its own**, train a depth-2 decision tree (normal vs attack) and measure **5-fold stratified cross-validated AUROC**. For text columns, use the one-hot columns of that feature together.
- Also run the same audit per attack type (that attack vs Normal), because a feature may give away one attack type only.
- Save a ranked table to `outputs/audit/shortcut_audit.csv` and a bar chart of the top 20 features.
- Flag features with AUROC at or above 0.95 as "suspect".
- In `outputs/audit/shortcut_audit.md`, explain in simple words which features are suspect and whether each looks like a **data collection artifact** (like TTL, which depends on the sending machine and network distance, not on attack behaviour) or like **real attack behaviour** (like packet counts). Be careful and honest; do not over-claim.

**CHECKPOINT 2.** Show me the ranked list and your reasoning. I will decide the final `shortcut_features` list. Default: `sttl, dttl, ct_state_ttl`.

### Phase 3: Preprocessing (`src/preprocess.py`)
A `Preprocessor` class with `fit(train_df)` and `transform(df)`:
- Select feature columns (all columns except labels, and minus shortcut features when the setting is `without_shortcut`).
- Numeric features: apply `log1p` to non-negative numeric columns (check for negatives; if a column has negative values, skip the log and report it). Then standardise with mean and standard deviation **computed on training data only**. Guard against zero standard deviation.
- Text features:
  - `proto`: keep the top N values by training frequency, map everything else to `other`.
  - `service`, `state`: one-hot, with unseen values at transform time mapped to `other` (never crash on an unseen value; a new attack may use one).
- Output float32 numpy arrays and a list of feature names.
- **Leakage rule**: `fit` sees training rows only. `transform` never refits.
- Assert that none of `label`, `attack_cat`, `source_file`, `id` is in the selected feature columns.

### Phase 4: Detectors (`src/detectors.py`)
Common interface for one class's detector: `fit(X)` and `score(X)`, where **a higher score means more unusual**.

1. **GMMDetector**
   - Choose K from `k_grid` by **lowest BIC** on that class's training data.
   - Cap K so each component has enough data: K at most `n_rows / 20` (and at least 1). Document this rule.
   - `covariance_type: diag` by default, `n_init: 3`, `reg_covar` from config, `random_state` from the seed.
   - If the class has more than `max_fit_rows` rows, fit on a random subsample (log this).
   - Score = `-score_samples(X)` (negative log likelihood).
   - Never use `warm_start=True`.
2. **PCADetector**
   - Fit PCA on that class's training data; choose the smallest k reaching `explained_variance`, capped at `min(n_rows - 1, n_features - 1)`.
   - Score = squared reconstruction error `||x - x_hat||^2`, with `x_hat = mean + V_k V_k^T (x - mean)`.
   - Save the explained-variance curve for plotting.
3. **KNNDetector**
   - Build a **new** `faiss.IndexFlatL2` for each fit call. Never reuse an index across classes or rounds.
   - `k` from config, capped at `n_rows - 1` for tiny classes.
   - FAISS returns **squared** L2 distances: take the square root, then the mean over the k neighbours.
   - Inputs must be contiguous float32 arrays.
   - Optional extra (only after the main results work): an approximate `IndexIVFFlat` variant, comparing speed and agreement with exact search.

**PerClassDetector wrapper**: given a detector type, fits one detector per known class (Normal + the 8 known attacks), and returns a score matrix of shape `(n_samples, n_known_classes)`.

### Phase 5: Calibration and decision rule (`src/calibration.py`): two-stage design

**Why not per-class p-values.** A per-class conformal p-value has a floor: `p_c(x) >= 1/(n_c + 1)`. If a small class such as Worms has about 30 calibration rows, `p_Worms(x) >= 0.032` for every flow, so `max_c p_c(x) > 0.01` always, and nothing could ever be flagged unknown at alpha = 0.01 in any round where that class is known. So we use one pooled p-value instead.

**Calibration parts.** The known-class calibration share (20%) is split, stratified by `attack_cat`, into `calib_norm` (10%) and `calib_final` (10%). The two parts are disjoint and are used for different things.

**Step 1: per-class scores.** For each known class c, the detector fitted on train rows of class c gives a score `s_c(x)` (higher = more unusual).
- PCA and kNN: use `log(score + log_eps)` (their raw scores are positive and very skewed).
- GMM: negative log likelihood as is.

**Step 2: per-class normalisation (uses `calib_norm` only).** On `calib_norm` rows of class c, compute the median `m_c` and the IQR `d_c` of the class-c scores. If `d_c < min_scale`, fall back to the standard deviation, then to 1.0, and log which fallback was used. Normalised score:

```
z_c(x) = (s_c(x) - m_c) / d_c
```

This puts every class on a comparable scale (a varied class like Normal has larger typical raw scores than a repetitive class like Generic).

**Step 3: combined statistic.**

```
T(x) = min_c z_c(x)            (how unusual x is compared with the class it fits best)
best_class(x) = argmin_c z_c(x)
```

**Step 4: pooled conformal p-value (uses `calib_final` only).** Compute `T_i` for every `calib_final` row of all known classes (n_final rows in total):

```
P(x) = (1 + #{ i in calib_final : T_i >= T(x) }) / (n_final + 1)
```

Implement with sorting and `np.searchsorted` (not a Python loop over rows).

**Decision rule** at level alpha: output **unknown** if `P(x) <= alpha`, else `best_class(x)`.

**Guarantee and its limits.** `calib_final` rows and known test rows are exchangeable draws from the same known-traffic mix, and nothing in T was fitted on `calib_final`, so for a known test flow `Prob(P(x) <= alpha) <= alpha`. This guarantee is **marginal over the known-traffic mix**, not per class: a small class (for example Worms) can have a false unknown rate above alpha while big classes stay below it. So we report the false unknown rate **per known class** as well, and list this in the limitations.

**BH variant**: on each test set, apply Benjamini-Hochberg at level `bh_q` to the pooled `P(x)` values and flag the rejected flows as unknown (the others get `best_class`). Implement BH yourself (sort, compare `p_(i) <= i q / m`, take the largest such i) and test it against `scipy.stats.false_discovery_control`. These are standard pooled conformal p-values, so the BH theory for conformal p-values applies; BH controls a different quantity (the share of false alarms among all alarms). Still report the empirical false discovery proportion.

### Phase 6: Baseline (`src/baseline.py`)
- Train `HistGradientBoostingClassifier` (seeded) on the training rows of the known classes, using the same preprocessed features.
- Novelty score: `s(x) = 1 - max predicted probability`.
- Calibrate with one pooled conformal p-value using the **same `calib_final` rows** as the detectors: `p(x) = (1 + #{calib_final scores >= s(x)}) / (n_final + 1)`. (`calib_norm` is not used by the baseline.)
- Decision: unknown if `p(x) <= alpha`, else the predicted class.
- Purpose: tests the claim that "classifier confidence is not a reliable unknown detector". Compare at the **same** alpha on the **same** rows.

### Phase 7: LOAO experiment loop (`src/split.py`, `src/experiment.py`, `scripts/run_experiments.py`)
For each `hidden` in `attack_types`, for each `feature_setting`:
1. Split: the hidden class rows go 100% to test. The known rows are split stratified by `attack_cat` into 60/20/20 train/calib/test (seeded), then the calib part is split again, stratified by `attack_cat`, into `calib_norm` and `calib_final` (10% + 10% of known rows).
2. **Assertions** (fail loudly if violated): the hidden class has zero rows in train, calib_norm and calib_final; no row index appears in two splits; preprocessing was fitted only on train; normalisation used only calib_norm rows.
3. Fit a **fresh** Preprocessor on train. Transform calib_norm, calib_final and test.
4. For each detector in `[gmm, pca, knn, baseline]`: fit **fresh** detectors on train, compute normalisation constants on calib_norm (not for the baseline), compute the pooled calibration statistics on calib_final, compute test p-values, apply the decision rule at every alpha in `alphas`, and also BH.
5. Record runtime for fit and score.
6. Save one JSON per (hidden, feature_setting, detector) in `outputs/runs/` containing the config, seed, sizes of each split, runtimes, and all metrics below. Also save the raw test p-values and labels (compressed `.npz`) so metrics can be recomputed without re-running.

Everything that learns from data (preprocessor, GMM, PCA, FAISS index, classifier) must be **created inside the loop**.

### Metrics (`src/metrics.py`), at `main_alpha` and at all alphas
On **hidden attack** test rows:
- `frac_unknown`, `frac_wrong_known`, `frac_missed` (these sum to 1)
- `detection_rate = frac_unknown + frac_wrong_known`

On **Normal** test rows:
- `false_alarm_rate`: share assigned anything other than Normal (unknown or any attack)
- `false_unknown_rate_normal`: share flagged unknown (should be at most about alpha)

On **known attack** test rows:
- `known_attack_correct`: share assigned their true class
- `known_attack_flagged_unknown`: share flagged unknown

Threshold-free:
- `auroc_hidden_vs_normal`: AUROC using `T(x)` (for the baseline, `s(x)`) as the novelty score, hidden attack rows (positive) vs Normal test rows (negative).

For BH: the same outcome fractions, plus the empirical false discovery proportion (among flows flagged unknown, the share that are actually known classes).

**Bootstrap 95% CIs** (percentile method, `n_resamples` from config, seeded): resample test rows with replacement **within each group** (hidden rows, Normal rows, known attack rows) and recompute each metric. Report the point estimate and the CI.

**Calibration check:** for each run, compare the pooled false unknown rate on all known test rows against alpha across all alphas (this is what the guarantee covers). Also report the false unknown rate **for each known class** (including `false_unknown_rate_normal`), because the guarantee is marginal and small classes may exceed alpha.

### Phase 8: Figures and tables (`src/plots.py`, `src/report.py`)
Save to `outputs/figures/` (PNG, 200 dpi, readable labels) and `outputs/tables/` (CSV + markdown):
1. Class counts before and after dedup (bar, log scale).
2. Shortcut audit: top 20 single-feature AUROCs.
3. Heatmap: detection rate (hidden attack x detector), one panel with shortcut features and one without.
4. Grouped bar chart: detection rate with vs without shortcut features per detector, with CI error bars.
5. Stacked bars: unknown / wrong known / missed for each hidden attack (main detector), with vs without shortcuts.
6. Calibration plot: empirical false unknown rate vs alpha, with the line y = alpha.
7. False alarm rate per detector at main_alpha, with CIs.
8. PCA explained variance curve for the Normal class.
9. Runtime table per detector.
10. Main results table: per hidden attack and detector, detection rate [95% CI], false alarm rate [95% CI], and AUROC, for both feature settings.

Then generate `outputs/results.md` automatically from the saved run files: tables plus short factual sentences (for example "Without shortcut features, the mean detection rate of X fell from A to B"). Every number must come from the saved outputs. No interpretation beyond what the numbers show; put open questions in a "Notes for the authors" section.

---

## 7. Tests (`tests/`, run with `pytest`)

- `test_preprocess.py`: fitting on train and transforming test does not change the train-fitted mean and std; an unseen `service` value maps to `other` without crashing; shortcut columns are absent in the `without_shortcut` setting; passing `source_file`, `label` or `attack_cat` as a feature raises an assertion error.
- `test_split.py`: the hidden class is absent from train, calib_norm and calib_final; all four splits are disjoint; the class proportions in each split are close to the overall proportions.
- `test_calibration.py`: the p-value formula on a tiny hand-made example (for example calib statistics `[1,2,3,4]`, test value `3.5` gives `P = (1+1)/5 = 0.4`); `P(x)` is never below `1/(n_final + 1)`; on synthetic data with several classes of very different sizes and scales, the pooled false unknown rate on known test data is close to alpha (within sampling error); normalisation constants depend only on calib_norm rows (changing calib_final or test rows does not change them); the IQR fallback works for a constant-score class; BH matches `scipy.stats.false_discovery_control`.
- `test_detectors.py`: on synthetic 2D data, a far-away point gets a higher score than a central point for all three detectors; PCA gives near-zero error for a point exactly on the main direction; the KNN detector builds a new index on every fit (two fits on different data give different results).

All tests must pass before Phase 7 starts.

---

## 8. Quick mode and performance

- `python scripts/run_experiments.py --config configs/quick.yaml` must finish in a few minutes. Run it first, check the outputs look sensible, then run the full config.
- Show progress with `tqdm`. Log to `outputs/logs/`.
- If the full run looks like it will take more than about 1 hour, tell me the estimate and suggest options (larger subsampling for GMM, fewer bootstrap resamples, parallel runs with joblib) before starting.

---

## 9. README.md (generated at the end)

Include, in simple English:
1. The problem and main question (from section 1).
2. How to set up and run (exact commands).
3. The pipeline, step by step.
4. Which course topics are used where: GMM and EM; SVD, PCA, low rank; nearest neighbours and FAISS; hypothesis testing, conformal p-values, multiple testing (BH); bootstrap; data cleaning and shortcut feature analysis.
5. Key results, copied from `outputs/results.md` (generated numbers only).
6. Limitations: lab data from 2015; only 9 attack types; tiny classes (for example Worms) give wide CIs; the conformal guarantee is marginal over the known-traffic mix, so a small known class can have a false unknown rate above alpha (report the per-class rates).

---

## 10. Things you must NOT do

- Do not use `label` or `attack_cat` as features.
- Do not fit any preprocessing, detector or threshold on calibration or test rows (except, with already-fitted detectors, computing normalisation constants on calib_norm and the pooled calibration statistics on calib_final).
- Do not reuse a fitted model, scaler or FAISS index across rounds.
- Do not set thresholds from training-set scores.
- Do not report results from a single lucky seed as final; if time allows, repeat the full LOAO with 3 seeds and report the spread.
- Do not write claims about results that the code did not produce.
- Do not add deep learning models unless I ask.

---

## 11. Start

Begin with: read this file, list anything that is unclear or that you think is wrong, then show the plan for Phase 0 and Phase 1.

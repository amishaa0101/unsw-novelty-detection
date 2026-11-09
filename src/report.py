"""Build outputs/results.md and tables from saved run files only. Facts, no interpretation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import UNKNOWN, benjamini_hochberg
from src.metrics import outcome_metrics

BH_LEVELS = [0.01, 0.05, 0.1]
AUROC_TEXT = {
    "novelty_auroc_vs_known": "Novelty AUROC (vs known): how well the novelty statistic T(x) ranks hidden-attack "
                              "flows above all known-class test flows (Normal and known attacks).",
    "alarm_auroc": "Alarm AUROC: how well the normalised Normal-class score z_Normal(x) ranks hidden-attack flows "
                   "above Normal test flows, i.e. whether the flow can be told apart from Normal traffic at all, "
                   "novel or not (baseline: 1 - classifier probability of Normal).",
}


def run_label(run_dir):
    """Every table and figure says which run it came from (read from the run's own saved config)."""
    first = next(Path(run_dir).glob("*.json"))
    with open(first, encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    name = Path(run_dir).name
    if cfg["subsample_frac"] < 1.0:
        return f"QUICK smoke-test run (configs/{name}.yaml): not results"
    return f"FULL run (configs/{name}.yaml, seed {cfg['seed']})"


def normal_test_overlap(runs):
    """For every pair of rounds (one detector, so one file per round): share of Normal test flows in common."""
    det = runs[0]["detector"]
    sets = []
    for r in runs:
        if r["detector"] != det:
            continue
        z = np.load(r["_npz"])
        sets.append(set(z["row"][z["y"] == r["class_codes"].index("Normal")].tolist()))
    return np.array([len(a & b) / len(a) for i, a in enumerate(sets) for b in sets[i + 1:]])


def seed_summary(run_dirs):
    """Mean and SD across seeds of (a) per-seed means over hidden attacks and (b) pooled false unknown rates.

    Each seed contributes one number per (setting, detector): the mean over its 9 hidden-attack rounds.
    """
    per_seed, calib = [], []
    for d in run_dirs:
        runs = load_runs(d)
        seed = runs[0]["config"]["seed"]
        t = main_table(runs)
        m = t.groupby(["setting", "detector"])[["detection_rate", "frac_unknown", "false_alarm_rate",
                                                 "novelty_auroc_vs_known", "alarm_auroc"]].mean().reset_index()
        m["seed"] = seed
        per_seed.append(m)
        for r in runs:
            for a in r["config"]["calibration"]["alphas"]:
                calib.append({"seed": seed, "detector": r["detector"], "alpha": a,
                              "rate": r["metrics"][f"hybrid|alpha={a}"]["false_unknown_rate_known"]})
    ps = pd.concat(per_seed)
    metrics = ps.groupby(["setting", "detector"]).agg(["mean", "std"]).drop(columns="seed")
    metrics.columns = [f"{a} {b}" for a, b in metrics.columns]
    c = pd.DataFrame(calib).groupby(["seed", "detector", "alpha"])["rate"].mean().reset_index()
    cal = c.groupby(["detector", "alpha"])["rate"].agg(["mean", "std", "min", "max"]).reset_index()
    cal = cal.rename(columns={"min": "lowest seed", "max": "highest seed"})
    return metrics.reset_index(), cal, sorted(ps.seed.unique())


def load_runs(run_dir):
    runs = []
    for p in sorted(Path(run_dir).glob("*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        r["_npz"] = p.with_suffix(".npz")
        runs.append(r)
    return runs


def ci_text(v, ci):
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def main_table(runs, mode="hybrid"):
    rows = []
    for r in runs:
        key = f"{mode}|alpha={r['config']['calibration']['main_alpha']}"
        m, ci = r["metrics"][key], r["metrics_ci"][key]
        rows.append({
            "hidden": r["hidden"], "setting": r["setting"], "detector": r["detector"],
            "n_hidden_test": r["split_sizes"]["test"].get(r["hidden"], 0),
            "detection_rate": m["detection_rate"], "detection_rate_ci": ci["detection_rate"],
            "frac_unknown": m["frac_unknown"], "frac_unknown_ci": ci["frac_unknown"],
            "frac_missed": m["frac_missed"],
            "false_alarm_rate": m["false_alarm_rate"], "false_alarm_rate_ci": ci["false_alarm_rate"],
            "false_unknown_rate_known": m["false_unknown_rate_known"],
            "novelty_auroc_vs_known": r["auroc"]["novelty_auroc_vs_known"],
            "novelty_auroc_vs_known_ci": r["auroc_ci"]["novelty_auroc_vs_known"],
            "novelty_auroc_vs_normal": r["auroc"]["novelty_auroc_vs_normal"],
            "alarm_auroc": r["auroc"]["alarm_auroc"], "alarm_auroc_ci": r["auroc_ci"]["alarm_auroc"],
            "fit_s": r["runtime_s"]["fit"], "score_s": r["runtime_s"]["score"],
        })
    return pd.DataFrame(rows)


def bh_levels(runs, mode="hybrid"):
    """BH at several q from the saved p-values (no re-run), plus the resolution numbers per round."""
    rows = []
    for r in runs:
        z = np.load(r["_npz"])
        P, y = z["P"], z["y"]
        labels = z["clf_pred"] if mode == "hybrid" else z["best"]
        names = r["class_codes"]
        H, N = names.index(r["hidden"]), names.index("Normal")
        n_final, m = r["n_calib_final"], len(P)
        p_min = 1.0 / (n_final + 1)
        row = {"hidden": r["hidden"], "setting": r["setting"], "detector": r["detector"],
               "n_final": n_final, "m": m, "p_min": p_min, "n_at_floor": int(np.isclose(P, p_min).sum())}
        for q in BH_LEVELS:
            pred = np.where(benjamini_hochberg(P, q), UNKNOWN, labels)
            o = outcome_metrics(pred, y, H, N)
            row[f"needed_q{q}"] = p_min * m / q
            row[f"flagged_q{q}"] = o["n_flagged"]
            row[f"frac_unknown_q{q}"] = o["frac_unknown"]
            row[f"fdp_q{q}"] = o["false_discovery_proportion"]
        rows.append(row)
    return pd.DataFrame(rows)


def combo_section(run_dir, tables_dir):
    """Markdown lines for the combo_<D> detectors (built by scripts/run_combo.py from saved p-values)."""
    combo = load_runs(Path(run_dir) / "combo")
    if not combo:
        return []
    main = {(r["hidden"], r["setting"], r["detector"]): r for r in load_runs(run_dir)}
    cfg = combo[0]["config"]
    alpha = cfg["calibration"]["main_alpha"]
    key = f"combo|alpha={alpha}"
    rows, cal_rows, cls_rows = [], [], []
    for r in combo:
        m, ci = r["metrics"][key], r["metrics_ci"][key]
        d = r["detector"].replace("combo_", "")
        alone = main[(r["hidden"], r["setting"], d)]["metrics"][f"hybrid|alpha={alpha}"]["frac_unknown"]
        base = main[(r["hidden"], r["setting"], "baseline")]["metrics"][f"hybrid|alpha={alpha}"]["frac_unknown"]
        rows.append({"hidden": r["hidden"], "setting": r["setting"], "detector": r["detector"],
                     "detection": ci_text(m["detection_rate"], ci["detection_rate"]),
                     "unknown": ci_text(m["frac_unknown"], ci["frac_unknown"]),
                     "unknown (D alone)": f"{alone:.3f}", "unknown (baseline alone)": f"{base:.3f}",
                     "false alarm": ci_text(m["false_alarm_rate"], ci["false_alarm_rate"]),
                     "novelty AUROC (vs known)": ci_text(r["auroc"]["novelty_auroc_vs_known"],
                                                         r["auroc_ci"]["novelty_auroc_vs_known"]),
                     "_unk": m["frac_unknown"], "_det": m["detection_rate"], "_fa": m["false_alarm_rate"],
                     "_nov": r["auroc"]["novelty_auroc_vs_known"], "_alone": alone, "_base": base})
        for a in cfg["calibration"]["alphas"]:
            cal_rows.append({"detector": r["detector"], "alpha": a,
                             "rate": r["metrics"][f"combo|alpha={a}"]["false_unknown_rate_known"]})
        for c, v in r["false_unknown_by_class"][key].items():
            cls_rows.append({"class": c, "rate": v})
    t = pd.DataFrame(rows)
    t.drop(columns=[c for c in t.columns if c.startswith("_")]).to_csv(tables_dir / "combo_results.csv", index=False)
    means = t.groupby(["setting", "detector"])[["_det", "_unk", "_alone", "_base", "_fa", "_nov"]].mean().reset_index()
    means.columns = ["setting", "detector", "detection", "unknown", "unknown (detector alone)",
                     "unknown (baseline alone)", "false alarm", "novelty AUROC (vs known)"]
    cal = pd.DataFrame(cal_rows)
    over = cal.assign(over=cal.rate > cal.alpha).groupby(["detector", "alpha"])["over"].sum()
    cal = cal.groupby(["detector", "alpha"])["rate"].agg(["mean", "min", "max"]).join(over).reset_index()
    cal = cal.rename(columns={"over": "rounds above alpha"})
    cls = pd.DataFrame(cls_rows).groupby("class")["rate"].agg(["mean", "max"]).reset_index().sort_values("max", ascending=False)

    lines = ["## Combined detectors (combo_<D>), from saved p-values, no re-fitting", "",
             f"Rule: a flow is unknown if min(P_D(x), P_baseline(x)) <= alpha / 2 (Bonferroni), otherwise it gets "
             f"the classifier's label. By Bonferroni the false unknown rate on known flows is at most alpha, with no "
             f"assumption about how the two p-values depend on each other. Novelty AUROC uses -min(P_D, P_baseline) "
             f"as the score. Main alpha = {alpha}; 95% bootstrap CIs.", ""]
    for setting in cfg["feature_settings"]:
        s = t[t.setting == setting]
        lines += [f"### {setting}", "", md(s.drop(columns=[c for c in s.columns if c.startswith("_")] + ["setting"])), ""]
    lines += ["Means over hidden attack types:", "", md(means), "",
              "Calibration check: pooled false unknown rate on known test rows over all rounds, and the number of "
              "rounds where it is above alpha:", "", md(cal, ".4f"), "",
              f"Per known class at alpha = {alpha}, over all rounds and combos:", "", md(cls, ".4f"), ""]
    return lines


def runtime_section(runs, tables_dir):
    """Runtime per detector over all rounds, from the saved run files."""
    rows = [{"detector": r["detector"], **{k: v for k, v in r["runtime_s"].items()}} for r in runs]
    t = pd.DataFrame(rows).groupby("detector").agg(["mean", "max"])
    t.columns = [f"{a} s ({b})" for a, b in t.columns]
    t = t.reset_index()
    t.to_csv(tables_dir / "runtime.csv", index=False, float_format="%.2f")
    return ["## Runtime per detector (seconds per round)", "",
            "`fit`: fitting on train (GMM includes fitting every reg_covar value and choosing on calib_norm; the "
            "baseline's fit is the classifier, which hybrid mode reuses for all detectors). `score`: scoring "
            "calib_norm, calib_final and test, and normalising. `bootstrap`: all CIs for both modes.", "",
            md(t, ".1f"), ""]


def md(df, floatfmt=".3f"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(format(v, floatfmt) if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(out)


def build(run_dir, out_path, tables_dir, figures, seed_dirs=(), extra_sections=()):
    """Write results.md. `figures` is a list of (relative path, caption).

    seed_dirs: complete runs with other seeds (same config otherwise) for the seed-repeat section.
    extra_sections: lists of markdown lines appended before the figures.
    """
    runs = load_runs(run_dir)
    assert runs, f"no run files in {run_dir}"
    cfg = runs[0]["config"]
    alpha = cfg["calibration"]["main_alpha"]
    t = main_table(runs)
    tables_dir.mkdir(parents=True, exist_ok=True)
    flat = t.copy()
    for c in [c for c in flat.columns if c.endswith("_ci")]:
        flat[c] = flat[c].apply(lambda v: f"[{v[0]:.4f}, {v[1]:.4f}]")
    flat.to_csv(tables_dir / "main_results_hybrid.csv", index=False, float_format="%.4f")

    out = [f"# Results (generated by `scripts/make_figures.py` from `{Path(run_dir).as_posix()}`)", "",
           f"**Source: {run_label(run_dir)}.** {len(runs)} run files; seed {cfg['seed']}; "
           f"{cfg['bootstrap']['n_resamples']} bootstrap resamples; main alpha {alpha}. "
           "All numbers below are read from the saved run files.", "",
           "## Threshold-free metrics", ""]
    out += [f"- {s}" for s in AUROC_TEXT.values()] + [""]

    # ---- Main table, one per feature setting
    out += [f"## Main table (hybrid mode, alpha = {alpha}, 95% bootstrap CIs)", "",
            "Hybrid mode: a flow is *unknown* if its pooled conformal p-value is <= alpha; otherwise it gets "
            "the classifier's label. Detection = unknown + wrong known attack (not labelled Normal).", ""]
    for setting in cfg["feature_settings"]:
        s = t[t.setting == setting]
        view = pd.DataFrame({
            "hidden": s.hidden, "detector": s.detector, "n hidden": s.n_hidden_test,
            "detection": [ci_text(v, c) for v, c in zip(s.detection_rate, s.detection_rate_ci)],
            "unknown": [ci_text(v, c) for v, c in zip(s.frac_unknown, s.frac_unknown_ci)],
            "false alarm": [ci_text(v, c) for v, c in zip(s.false_alarm_rate, s.false_alarm_rate_ci)],
            "novelty AUROC (vs known)": [ci_text(v, c) for v, c in zip(s.novelty_auroc_vs_known, s.novelty_auroc_vs_known_ci)],
            "alarm AUROC": [ci_text(v, c) for v, c in zip(s.alarm_auroc, s.alarm_auroc_ci)],
        })
        out += [f"### {setting}", "", md(view), ""]

    # ---- Means and with vs without sentences
    means = t.groupby(["setting", "detector"])[["detection_rate", "frac_unknown", "false_alarm_rate",
                                                 "novelty_auroc_vs_known", "alarm_auroc"]].mean()
    out += ["## Mean over hidden attack types (point estimates)", "", md(means.reset_index()), ""]
    if {"with_shortcut", "without_shortcut"} <= set(t.setting):
        out += ["Without vs with shortcut features (means over hidden attack types):", ""]
        for det in cfg["detectors"]:
            w, wo = means.loc[("with_shortcut", det)], means.loc[("without_shortcut", det)]
            out.append(f"- {det}: detection rate {w.detection_rate:.3f} -> {wo.detection_rate:.3f}; "
                       f"unknown {w.frac_unknown:.3f} -> {wo.frac_unknown:.3f}; "
                       f"novelty AUROC {w.novelty_auroc_vs_known:.3f} -> {wo.novelty_auroc_vs_known:.3f}; "
                       f"alarm AUROC {w.alarm_auroc:.3f} -> {wo.alarm_auroc:.3f}.")
        out.append("")

    # ---- Detector-only mode, for comparison
    d = main_table(runs, "detector_only").groupby(["setting", "detector"])[["detection_rate", "frac_unknown",
                                                                           "false_alarm_rate"]].mean()
    out += [f"## Detector-only mode (alpha = {alpha}), means over hidden attack types", "",
            "Same unknown decision; a flow that is not unknown gets the best-fitting detector's class instead "
            "of the classifier's label.", "", md(d.reset_index()), ""]

    # ---- Calibration check
    cal_rows, cls_rows = [], []
    for r in runs:
        for a in cfg["calibration"]["alphas"]:
            k = f"hybrid|alpha={a}"
            cal_rows.append({"detector": r["detector"], "alpha": a,
                             "rate": r["metrics"][k]["false_unknown_rate_known"]})
            for c, v in r["false_unknown_by_class"][k].items():
                cls_rows.append({"alpha": a, "class": c, "rate": v})
    cal = pd.DataFrame(cal_rows).groupby(["detector", "alpha"])["rate"].agg(["mean", "min", "max"]).reset_index()
    cls = pd.DataFrame(cls_rows)
    cls = cls[cls.alpha == alpha].groupby("class")["rate"].agg(["mean", "max"]).reset_index().sort_values("max", ascending=False)
    out += ["## Calibration check", "",
            f"Pooled false unknown rate on all known test rows (what the conformal guarantee covers), over "
            f"{t[['hidden', 'setting']].drop_duplicates().shape[0]} rounds:", "", md(cal, ".4f"), "",
            f"Per known class at alpha = {alpha} (the guarantee is marginal, not per class), over all rounds and detectors:",
            "", md(cls, ".4f"), ""]

    # ---- BH
    bh = bh_levels(runs)
    bh.to_csv(tables_dir / "bh_levels.csv", index=False, float_format="%.6g")
    summ = bh.groupby("detector")[[f"flagged_q{q}" for q in BH_LEVELS] + [f"fdp_q{q}" for q in BH_LEVELS]].agg(["mean", "max"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    out += ["## Benjamini-Hochberg (hybrid labels), recomputed from the saved p-values", "",
            "Rows flagged unknown and empirical false discovery proportion (share of flagged rows that are known "
            "classes), mean and max over rounds:", "", md(summ.reset_index(), ".4f"), "",
            "**Resolution limit.** A conformal p-value can never be smaller than P_min = 1 / (n_final + 1). "
            "BH rejects the k smallest p-values only if p_(k) <= k q / m, so with the smallest p-values sitting at "
            "P_min, BH can reject only when at least about P_min * m / q test flows share that floor "
            "(m = number of test flows). The table gives, for each round, n_final, m, P_min, how many test flows "
            "are at the floor, and how many would be needed at each q. The floor count is a guide, not an exact "
            "condition: BH can also reject when the smallest p-values are just above the floor, as long as "
            "p_(k) <= k q / m holds (compare the 'flagged' and 'at floor' columns).", ""]
    res = bh[bh.detector == bh.detector.iloc[0]][["hidden", "setting", "n_final", "m", "p_min"]].copy()
    floor = bh.pivot_table(index=["hidden", "setting"], columns="detector", values="n_at_floor").reset_index()
    for c in floor.columns[2:]:
        floor[c] = floor[c].astype(int)
    floor.columns = [c if c in ("hidden", "setting") else f"at floor ({c})" for c in floor.columns]
    res = res.merge(floor, on=["hidden", "setting"])
    for q in BH_LEVELS:
        res[f"needed q={q}"] = (res.p_min * res.m / q).round(0).astype(int)
    res["p_min"] = res.p_min.map(lambda v: f"{v:.2e}")
    out += [md(res), ""]

    # ---- Seed repeat
    if seed_dirs:
        metrics, scal, seeds = seed_summary([run_dir, *seed_dirs])
        metrics.to_csv(tables_dir / "seed_repeat_metrics.csv", index=False, float_format="%.4f")
        scal.to_csv(tables_dir / "seed_repeat_calibration.csv", index=False, float_format="%.5f")
        ov = normal_test_overlap(runs)
        out += [f"## Seed repeat (seeds {', '.join(str(s) for s in seeds)}; hybrid mode)", "",
                "The rounds within one seed are not independent: they share the seed and all use the same Normal "
                "flows (Normal is never hidden). With the same seed, pairs of rounds often put the same Normal "
                f"flows into the test split: over all pairs of rounds in seed {cfg['seed']}, the share of Normal "
                f"test flows in common ranges from {ov.min():.2f} to {ov.max():.2f} (median {np.median(ov):.2f}; "
                f"about {cfg['split']['test']:.2f} would be expected by chance). The spread across rounds therefore "
                "understates run-to-run variation, and the seed repeat is the right check for the small excess "
                "of the pooled false unknown rate above alpha.", "",
                f"Mean and SD across seeds of each seed's mean over the hidden attack types (alpha = {alpha}):", "",
                md(metrics, ".4f"), "",
                "Pooled false unknown rate on known test rows: each seed's mean over its rounds, then mean, SD, "
                "lowest and highest across seeds:", "", md(scal, ".5f"), ""]
    for section in extra_sections:
        out += list(section)

    # ---- Figures
    out += ["## Figures", ""]
    for path, caption in figures:
        out += [f"![{caption}]({path})", "", f"*{caption}*", ""]

    # ---- Notes for the authors: facts that may need a look
    notes = []
    low = t[t.novelty_auroc_vs_known < 0.5]
    if len(low):
        notes.append("Rounds with novelty AUROC (vs known) below 0.5: "
                     + "; ".join(f"{r.hidden}/{r.setting}/{r.detector} {r.novelty_auroc_vs_known:.3f}" for r in low.itertuples()))
    over = cal[(cal.alpha == alpha) & (cal["mean"] > alpha)]
    if len(over):
        notes.append(f"Detectors whose mean pooled false unknown rate exceeds alpha = {alpha}: "
                     + ", ".join(f"{r.detector} ({r.mean:.4f})" for r in over.itertuples()))
    out += ["## Notes for the authors", ""] + [f"- {n}" for n in notes or ["None generated."]] + [""]

    Path(out_path).write_text("\n".join(out), encoding="utf-8")
    return t, cal, bh

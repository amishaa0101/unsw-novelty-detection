"""Phase 8 (so far): figures and outputs/results.md from saved run files.

Usage: python scripts/make_figures.py [--config configs/default.yaml]
Quick-config outputs go to outputs/figures/quick/ and outputs/results_quick.md, so they can never be
mistaken for full results.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import report  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.plots import (HYBRID_CAPTION, calibration_plot, confusion_figure, detection_heatmaps,  # noqa: E402
                       frac_unknown_vs_alpha, shortcut_bars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    name = Path(cfg["config_path"]).stem
    full = name == "default"
    run_dir = ROOT / "outputs" / "runs" / name
    fig_dir = ROOT / "outputs" / "figures" / ("" if full else name)
    fig_dir.mkdir(parents=True, exist_ok=True)
    label = report.run_label(run_dir)
    short = "FULL run" if full else "QUICK smoke test"

    figures = []  # (path relative to outputs/, caption)
    runs = report.load_runs(run_dir)
    dets, sets = cfg["detectors"], cfg["feature_settings"]

    def add(out, caption):
        figures.append((out.relative_to(ROOT / "outputs").as_posix(), f"[{short}] {caption}"))

    out = fig_dir / "detection_heatmaps.png"
    detection_heatmaps(runs, dets, sets, short, out)
    add(out, "Detection rate (top) and share flagged unknown (bottom) per hidden attack and detector, hybrid mode, "
             "main alpha. In hybrid mode detection rates are close across detectors because they share the "
             "classifier's labels; the unknown share is what differs.")
    for metric, ylabel in [("detection_rate", "detection rate"), ("frac_unknown", "share flagged unknown")]:
        out = fig_dir / f"shortcut_bars_{metric}.png"
        shortcut_bars(runs, dets, metric, ylabel, short, out)
        add(out, f"{ylabel.capitalize()} with vs without shortcut features, per hidden attack and detector, "
                 "hybrid mode, main alpha, 95% bootstrap CIs.")
    out = fig_dir / "calibration.png"
    calibration_plot(runs, dets, short, out)
    add(out, "Empirical false unknown rate vs alpha. Left: pooled over known test rows (what the conformal "
             "guarantee covers), mean over rounds with min to max bars. Right: per known class, mean over rounds "
             "and detectors (the guarantee is marginal, not per class).")
    out = fig_dir / "frac_unknown_vs_alpha.png"
    frac_unknown_vs_alpha(runs, cfg["detectors"], cfg["feature_settings"], short, out)
    figures.append((out.relative_to(ROOT / "outputs").as_posix(),
                    f"[{short}] Share of hidden-attack flows flagged unknown at each alpha (hybrid mode; the unknown "
                    "decision is the same in both modes). Solid: with shortcut features; dashed: without."))
    for hidden in ["DoS", "Worms"]:
        if hidden not in cfg["attack_types"]:
            continue
        for setting in cfg["feature_settings"]:
            out = fig_dir / f"confusion_{hidden}_{setting}_hybrid.png"
            confusion_figure(run_dir, hidden, setting, cfg["detectors"], "hybrid", out, short)
            figures.append((out.relative_to(ROOT / "outputs").as_posix(),
                            f"[{short}] Confusion matrix, hidden = {hidden}, {setting}, hybrid mode, row shares. "
                            + HYBRID_CAPTION))
    for f in figures:
        print("Saved", f[0])

    results = ROOT / "outputs" / ("results.md" if full else f"results_{name}.md")
    tables = ROOT / "outputs" / "tables" / ("" if full else name)
    # Seed-repeat runs (configs/seed*.yaml) are included only when complete (same number of run files).
    seed_dirs = []
    if full:
        for d in sorted((ROOT / "outputs" / "runs").glob("seed*")):
            if len(list(d.glob("*.json"))) == len(runs):
                seed_dirs.append(d)
            else:
                print(f"Skipping incomplete seed run {d.name}")
    tables.mkdir(parents=True, exist_ok=True)
    extra = [report.low_auroc_section(runs), report.runtime_section(runs, tables)]
    if (run_dir / "combo").exists():
        extra.append(report.combo_section(run_dir, tables))
    dup_csv = ROOT / "outputs" / "tables" / "dup_inflation.csv"
    if full and dup_csv.exists():
        extra.append(report.dup_inflation_section(dup_csv))
    report.build(run_dir, results, tables, figures, seed_dirs=seed_dirs, extra_sections=extra)
    print("Saved", results, f"({label})")


if __name__ == "__main__":
    main()

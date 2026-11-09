"""Figures, drawn only from saved run files."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


HYBRID_CAPTION = ("In hybrid mode all detectors share the classifier's known-class labels, so the panels differ "
                  "only in the 'unknown' column. To compare detectors, use frac_unknown, the novelty AUROCs and "
                  "the alarm AUROC.")


def frac_unknown_vs_alpha(runs, detectors, settings, label, path):
    """One panel per hidden attack: frac_unknown vs alpha, a line per detector, solid = with, dashed = without shortcuts."""
    hidden = sorted({r["hidden"] for r in runs})
    ncols = 3
    nrows = int(np.ceil(len(hidden) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False, sharey=False)
    colors = dict(zip(detectors, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    styles = {"with_shortcut": "-", "without_shortcut": "--", "strict": ":"}
    index = {(r["hidden"], r["setting"], r["detector"]): r for r in runs}
    for ax in axes.flat[len(hidden):]:
        ax.axis("off")
    for ax, h in zip(axes.flat, hidden):
        for det in detectors:
            for s in settings:
                r = index.get((h, s, det))
                if r is None:
                    continue
                alphas = r["config"]["calibration"]["alphas"]
                ys = [r["metrics"][f"hybrid|alpha={a}"]["frac_unknown"] for a in alphas]
                ax.plot(alphas, ys, styles[s], marker="o", ms=3, color=colors[det],
                        label=f"{det} ({s.replace('_shortcut', '')})")
        ax.plot(alphas, alphas, color="grey", lw=0.8, label="y = alpha")
        ax.set_xscale("log")
        ax.set_xticks(alphas, [str(a) for a in alphas])
        ax.set_title(f"hidden = {h}", fontsize=10)
        ax.set_xlabel("alpha")
        ax.set_ylabel("share of hidden flows flagged unknown")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"frac_unknown vs alpha ({label}). Grey line: y = alpha, the expected false unknown rate "
                 "on known traffic.", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _main_key(r):
    return f"hybrid|alpha={r['config']['calibration']['main_alpha']}"


def detection_heatmaps(runs, detectors, settings, label, path):
    """Hidden attack x detector heatmaps: detection rate (top) and frac_unknown (bottom), one column per setting."""
    hidden = sorted({r["hidden"] for r in runs})
    index = {(r["hidden"], r["setting"], r["detector"]): r for r in runs}
    metrics = [("detection_rate", "detection rate"), ("frac_unknown", "share flagged unknown")]
    fig, axes = plt.subplots(len(metrics), len(settings), figsize=(5.5 * len(settings), 5.5 * len(metrics)), squeeze=False)
    for i, (key, title) in enumerate(metrics):
        for j, s in enumerate(settings):
            M = np.array([[index[(h, s, d)]["metrics"][_main_key(index[(h, s, d)])][key] for d in detectors] for h in hidden])
            ax = axes[i][j]
            ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(detectors)), detectors)
            ax.set_yticks(range(len(hidden)), hidden)
            for a in range(M.shape[0]):
                for b in range(M.shape[1]):
                    ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if M[a, b] > 0.6 else "black")
            ax.set_title(f"{title}, {s}", fontsize=10)
    alpha = runs[0]["config"]["calibration"]["main_alpha"]
    fig.suptitle(f"Hidden-attack outcomes, hybrid mode, alpha = {alpha} ({label})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def shortcut_bars(runs, detectors, metric, ylabel, label, path):
    """One panel per detector: metric with vs without shortcut features per hidden attack, with 95% CIs."""
    hidden = sorted({r["hidden"] for r in runs})
    index = {(r["hidden"], r["setting"], r["detector"]): r for r in runs}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False, sharey=True)
    x = np.arange(len(hidden))
    for ax, det in zip(axes.flat, detectors):
        for k, (s, color) in enumerate([("with_shortcut", "#4C72B0"), ("without_shortcut", "#DD8452")]):
            vals, lo, hi = [], [], []
            for h in hidden:
                r = index[(h, s, det)]
                v = r["metrics"][_main_key(r)][metric]
                c = r["metrics_ci"][_main_key(r)][metric]
                vals.append(v); lo.append(v - c[0]); hi.append(c[1] - v)
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, yerr=[lo, hi], capsize=2, color=color,
                   label=s.replace("_", " "))
        ax.set_xticks(x, hidden, rotation=40, ha="right", fontsize=8)
        ax.set_title(det, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
    axes.flat[0].legend(fontsize=8)
    alpha = runs[0]["config"]["calibration"]["main_alpha"]
    fig.suptitle(f"{ylabel} with vs without shortcut features, hybrid mode, alpha = {alpha}, 95% CIs ({label})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def calibration_plot(runs, detectors, label, path):
    """Empirical false unknown rate vs alpha. Left: pooled over known rows (mean over rounds, bars = min to max).
    Right: per known class (mean over rounds and detectors)."""
    alphas = runs[0]["config"]["calibration"]["alphas"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for k, det in enumerate(detectors):
        rates = np.array([[r["metrics"][f"hybrid|alpha={a}"]["false_unknown_rate_known"] for a in alphas]
                          for r in runs if r["detector"] == det])
        mean = rates.mean(axis=0)
        offset = 1 + 0.04 * (k - (len(detectors) - 1) / 2)  # small horizontal shift so markers do not overlap
        a1.errorbar(np.array(alphas) * offset, mean, yerr=[mean - rates.min(axis=0), rates.max(axis=0) - mean],
                    marker="o", ms=4, capsize=3, lw=1, label=det)
    classes = sorted({c for r in runs for c in r["false_unknown_by_class"][f"hybrid|alpha={alphas[0]}"]})
    for c in classes:
        vals = [np.mean([r["false_unknown_by_class"][f"hybrid|alpha={a}"][c] for r in runs
                         if c in r["false_unknown_by_class"][f"hybrid|alpha={a}"]]) for a in alphas]
        a2.plot(alphas, vals, marker="o", ms=3, lw=1, label=c)
    for ax, title in [(a1, "pooled over all known test rows (bars: min to max over rounds)"),
                      (a2, "per known class (mean over rounds and detectors)")]:
        ax.plot(alphas, alphas, color="black", ls="--", lw=1, label="y = alpha")
        ax.set_xscale("log")  # y stays linear so per-class rates of exactly 0 are still shown
        ax.set_xticks(alphas, [str(a) for a in alphas])
        ax.set_xlabel("alpha (log scale)")
        ax.set_ylabel("empirical false unknown rate")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"Calibration check, hybrid mode ({label})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion_figure(run_dir, hidden, setting, detectors, mode, path, label=""):
    """One heatmap per detector: true class x assigned label, row-normalised (share of each true class)."""
    ncols = 2
    nrows = int(np.ceil(len(detectors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6.2 * nrows), squeeze=False)
    for ax in axes.flat[len(detectors):]:
        ax.axis("off")
    for ax, det in zip(axes.flat, detectors):
        with open(run_dir / f"{hidden}__{setting}__{det}.json", encoding="utf-8") as f:
            r = json.load(f)
        cm = r["confusion_main_alpha"][mode]
        counts = np.array(cm["counts"], dtype=float)
        share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cm["cols"])), cm["cols"], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(cm["rows"])), [f"{n} (hidden)" if n == hidden else n for n in cm["rows"]], fontsize=8)
        for i in range(share.shape[0]):
            for j in range(share.shape[1]):
                if counts[i, j] > 0:
                    ax.text(j, i, f"{share[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if share[i, j] > 0.6 else "black")
        alpha = r["config"]["calibration"]["main_alpha"]
        ax.set_title(f"{det}", fontsize=10)
        ax.set_xlabel("assigned label")
        ax.set_ylabel("true class")
    fig.suptitle(f"Hidden = {hidden}, {setting}, {mode} mode, alpha = {alpha} (row shares). {label}", fontsize=11)
    if mode == "hybrid":
        fig.text(0.5, 0.005, HYBRID_CAPTION, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)

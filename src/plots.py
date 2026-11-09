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

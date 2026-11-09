"""Single-feature leakage audit.

For each feature on its own, a depth-2 decision tree tries to separate Normal from attack.
If one feature alone gets AUROC >= 0.95, it is "suspect": the label may be leaking through it.
We run this for Normal vs all attacks, and for each attack type vs Normal, because a feature
may give away one attack type only.

Trees only care about the order of values, so no scaling or log transform is needed here.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.data import TEXT_COLS, feature_columns

SUSPECT_AUROC = 0.95


def feature_matrix(df, feature):
    """One numeric column, or all one-hot columns of a text feature together."""
    if feature in TEXT_COLS:
        # One-hot on the whole table only fixes the list of categories; no statistic is learned.
        return pd.get_dummies(df[feature]).to_numpy(dtype=np.float32)
    return df[[feature]].to_numpy(dtype=np.float64)


def cv_auroc(X, y, seed, max_depth=2):
    """Mean and std of 5-fold stratified cross-validated AUROC for a small decision tree."""
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(tree, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean(), scores.std()


def run_audit(df, attack_types, seed):
    """Return a table with one row per feature, ranked by overall AUROC.

    Columns: overall AUROC (Normal vs all attacks) and its std, AUROC for each
    attack type vs Normal, the highest per-attack AUROC and which attack it was.
    """
    is_attack = (df["attack_cat"] != "Normal").to_numpy().astype(int)
    rows = []
    for feat in feature_columns(df):
        X = feature_matrix(df, feat)
        mean, std = cv_auroc(X, is_attack, seed)
        row = {"feature": feat, "kind": "text" if feat in TEXT_COLS else "numeric",
               "auroc_overall": mean, "auroc_overall_std": std}
        for att in attack_types:
            mask = df["attack_cat"].isin(["Normal", att]).to_numpy()
            row[f"auroc_{att}"] = cv_auroc(X[mask], is_attack[mask], seed)[0]
        rows.append(row)

    table = pd.DataFrame(rows)
    per_attack = table[[f"auroc_{a}" for a in attack_types]]
    table["auroc_max_attack"] = per_attack.max(axis=1)
    table["max_attack"] = per_attack.idxmax(axis=1).str.replace("auroc_", "", regex=False)
    table["n_attacks_ge_095"] = (per_attack >= SUSPECT_AUROC).sum(axis=1)
    table["suspect_overall"] = table["auroc_overall"] >= SUSPECT_AUROC
    table["suspect_any_attack"] = table["auroc_max_attack"] >= SUSPECT_AUROC
    return table.sort_values("auroc_overall", ascending=False).reset_index(drop=True)


# Our judgement of each reviewed feature, from what the feature measures (see NUSW-NB15_features.csv).
# Type is one of: identity artifact | real behaviour | noise | unclear. No numbers here: they come from the data.
NOTES = {
    "sttl": ("identity artifact",
             "Time-to-live set by the sender's operating system, minus one per router hop. It tells which "
             "machine sent the flow and how far away it is, not what the flow did."),
    "dttl": ("identity artifact", "Same as sttl, for the reply direction (identifies the responding machine)."),
    "ct_state_ttl": ("identity artifact",
                     "Computed directly from state together with ranges of sttl/dttl, so it carries the same TTL information."),
    "is_sm_ips_ports": ("unclear (leans artifact)",
                        "Directly encodes whether source and destination IP and port are equal. Equal addresses can be "
                        "real behaviour (for example a LAND attack), but in this data the value 1 occurs only in Normal rows, "
                        "mostly ARP, so here it marks a property of the lab's normal traffic. Its AUROC is low only because it is rare."),
    "stcpb": ("noise", "Random initial TCP sequence number chosen by the sender. Its only signal is whether it is zero "
                       "(no TCP handshake), which swin already carries."),
    "dtcpb": ("noise", "Same as stcpb for the reply direction."),
    "sbytes": ("real behaviour", "Bytes sent: what the flow carried."),
    "dbytes": ("real behaviour", "Bytes received: what the flow carried."),
    "smean": ("real behaviour", "Mean packet size sent: what the flow carried."),
    "dmean": ("real behaviour", "Mean packet size received: what the flow carried."),
    "dloss": ("real behaviour", "Packets retransmitted or dropped by the destination: how the exchange went."),
}
# Always reviewed, even below the threshold.
ALWAYS_REVIEW = ["sttl", "dttl", "ct_state_ttl", "is_sm_ips_ports", "stcpb", "dtcpb"]


def value_breakdown(df, feature, min_rows=50):
    """For each value with at least min_rows rows: row count and share of Normal rows."""
    g = df.groupby(feature)["attack_cat"].agg(rows="size", normal_share=lambda s: (s == "Normal").mean())
    g["share_of_all_normal"] = df[df["attack_cat"] == "Normal"][feature].value_counts() / (df["attack_cat"] == "Normal").sum()
    return g[g["rows"] >= min_rows].fillna(0)


def report(table, attack_types, df, seed):
    """Markdown with the ranked table, the suspect list and our classification.

    Every number comes from `table` or is computed here from `df`.
    """
    def fmt(v):
        return f"{v:.3f}"

    cols = ["feature", "auroc_overall"] + [f"auroc_{a}" for a in attack_types] + ["n_attacks_ge_095"]
    head = "| " + " | ".join(c.replace("auroc_", "") for c in cols) + " |"
    lines = [head, "|" + "---|" * len(cols)]
    for _, r in table.iterrows():
        cells = [r["feature"]] + [fmt(r[c]) for c in cols[1:-1]] + [str(r["n_attacks_ge_095"])]
        mark = " **" if r["suspect_overall"] else ""
        lines.append("| " + " | ".join(cells) + " |" + mark)

    sus = table[table["suspect_any_attack"]]
    out = [
        "# Shortcut audit (generated by `scripts/run_audit.py`)", "",
        "Each feature on its own is given to a depth-2 decision tree. The score is the mean AUROC over "
        "5-fold stratified cross-validation. `overall` = Normal vs all attacks; each attack column = that "
        f"attack vs Normal. A feature is **suspect** if its AUROC is >= {SUSPECT_AUROC} overall or for at "
        "least one attack type. Text features (proto, service, state) use all their one-hot columns together.", "",
        f"- Suspect overall: {int(table['suspect_overall'].sum())} features: "
        + ", ".join(f"`{f}`" for f in table.loc[table['suspect_overall'], 'feature']),
        f"- Suspect for at least one attack type: {len(sus)} features: "
        + ", ".join(f"`{f}`" for f in sus["feature"]), "",
    ]

    # ---- Classification of suspect and always-reviewed features
    t = table.set_index("feature")
    review = list(dict.fromkeys(list(sus["feature"]) + ALWAYS_REVIEW))
    out += ["## Classification", "",
            "- **Identity artifact**: reveals which machine or lab setup produced the flow. Candidate for removal.",
            "- **Real behaviour**: describes what the flow or host did. Kept, even if its AUROC is high.",
            "- Being computed from IP addresses does not by itself make a feature an identity artifact: a count "
            "does not reveal which IP it was.", "",
            "| feature | type | overall AUROC | highest single-attack AUROC | why |", "|---|---|---|---|---|"]
    for f in review:
        kind, why = NOTES.get(f, ("not yet classified", ""))
        r = t.loc[f]
        out.append(f"| `{f}` | {kind} | {fmt(r['auroc_overall'])} | {fmt(r['auroc_max_attack'])} ({r['max_attack']}) | {why} |")
    out.append("")

    # ---- Evidence for the TTL features: AUROC hides a one-sided pattern
    is_attack = (df["attack_cat"] != "Normal").astype(int).to_numpy()
    deep = {f: cv_auroc(df[[f]].to_numpy(dtype=float), is_attack, seed, max_depth=None)[0]
            for f in ["sttl", "dttl", "ct_state_ttl"]}
    out += ["## TTL features: what the AUROC does not show", "",
            "No TTL feature reaches the threshold overall, but they rank near the top. An unlimited-depth tree "
            "gives overall AUROC " + ", ".join(f"`{f}` {fmt(a)} (depth 2: {fmt(t.loc[f, 'auroc_overall'])})" for f, a in deep.items())
            + ", so the limit is in the values themselves, not the tree: some TTL values occur only in Normal "
            "traffic, while the values most attacks use are shared with part of Normal. Values with at least 50 rows:", ""]
    for f in ["sttl", "dttl", "ct_state_ttl"]:
        vb = value_breakdown(df, f)
        out += [f"`{f}`:", "", "| value | rows | Normal share of these rows | share of all Normal rows |", "|---|---|---|---|"]
        out += [f"| {v} | {int(r.rows):,} | {r.normal_share:.3f} | {r.share_of_all_normal:.3f} |" for v, r in vb.iterrows()]
        out.append("")

    # ---- Evidence for is_sm_ips_ports, stcpb, dtcpb
    one = df[df["is_sm_ips_ports"] == 1]
    out += ["## Other reviewed features", "",
            f"- `is_sm_ips_ports` = 1 in {len(one):,} rows; classes: {one['attack_cat'].value_counts().to_dict()}; "
            f"proto: {one['proto'].value_counts().to_dict()}."]
    for f in ["stcpb", "dtcpb"]:
        nz = (df[f] > 0).to_numpy()
        auc_zero, _ = cv_auroc(nz.reshape(-1, 1).astype(float), is_attack, seed)
        auc_nz, _ = cv_auroc(df.loc[nz, [f]].to_numpy(dtype=float), is_attack[nz], seed)
        same_as_swin = ((df[f] > 0) == (df["swin"] > 0)).mean()
        out.append(f"- `{f}`: zero in {100 * (~nz).mean():.1f}% of rows. AUROC of the value: "
                   f"{fmt(t.loc[f, 'auroc_overall'])}; of only \"is zero\": {fmt(auc_zero)}; of the value among "
                   f"non-zero rows: {fmt(auc_nz)}. \"Non-zero\" agrees with `swin` > 0 on {100 * same_as_swin:.1f}% of rows.")
    out.append("")

    out += ["## Ranked table (rows ending in ** are suspect overall)", ""] + lines + [""]
    return "\n".join(out)


def plot_top(table, path, n=20):
    """Horizontal bar chart of the n features with the highest overall AUROC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = table.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#c0392b" if s else "#7f8c8d" for s in top["suspect_overall"]]
    ax.barh(top["feature"], top["auroc_overall"], xerr=top["auroc_overall_std"], color=colors)
    ax.scatter(top["auroc_max_attack"], top["feature"], marker="|", s=200, color="black",
               label="highest AUROC for a single attack type vs Normal", zorder=3)
    ax.axvline(SUSPECT_AUROC, ls="--", color="black", lw=1)
    ax.text(SUSPECT_AUROC, len(top) - 0.3, f" {SUSPECT_AUROC}", va="bottom", fontsize=9)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("5-fold CV AUROC, depth-2 tree on one feature (bars: Normal vs all attacks)")
    ax.set_title(f"Single-feature shortcut audit: top {n} features\n(red bars = overall AUROC >= {SUSPECT_AUROC})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

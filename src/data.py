"""Load, combine, clean and deduplicate the UNSW-NB15 partitioned files."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILES = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]

# Columns that are never features.
LABEL_COLS = ["label", "attack_cat"]
META_COLS = ["id", "source_file"]  # source_file is added by us, for the audit only
TEXT_COLS = ["proto", "service", "state"]

# Known spelling variants of attack names (variant -> name we use).
LABEL_FIXES = {"Backdoors": "Backdoor"}


def check_raw_files(data_dir):
    """Exit with clear instructions if the two official CSVs are missing."""
    data_dir = Path(data_dir)
    missing = [f for f in RAW_FILES if not (data_dir / f).exists()]
    if missing:
        print("Missing data files:", ", ".join(missing))
        print(f"Please download the UNSW-NB15 partitioned files from")
        print("  https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        print(f"and put both CSVs in: {data_dir.resolve()}")
        sys.exit(1)


def load_raw(data_dir):
    """Read both files and stack them. Adds source_file so the audit can count per file."""
    data_dir = Path(data_dir)
    parts = []
    for name in RAW_FILES:
        # utf-8-sig removes the BOM at the start of the header (else the first column is '﻿id').
        df = pd.read_csv(data_dir / name, encoding="utf-8-sig")
        df["source_file"] = name
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def feature_columns(df):
    """Every column except labels and our metadata columns."""
    return [c for c in df.columns if c not in LABEL_COLS + META_COLS]


def normalise_labels(df):
    """Strip spaces and unify spelling in attack_cat. Returns the new df and a dict of changes."""
    df = df.copy()
    before = df["attack_cat"].astype(str)
    after = before.str.strip().replace(LABEL_FIXES)
    changed = before != after
    changes = (
        pd.DataFrame({"from": before[changed], "to": after[changed]})
        .value_counts()
        .to_dict()
    )
    df["attack_cat"] = after
    return df, changes


def label_mismatches(df):
    """Rows where label (0/1) disagrees with attack_cat (Normal = 0, any attack = 1)."""
    expected = (df["attack_cat"] != "Normal").astype(int)
    return df[df["label"] != expected]


def deduplicate(df, policy="drop"):
    """Remove exact duplicates, then handle conflicting duplicates.

    Duplicates are judged on feature columns only (id and source_file are ignored).
    1. Exact duplicates: same features and same attack_cat -> keep one row.
    2. Conflicting duplicates: same features but different attack_cat.
       policy 'drop'     -> drop every row of every conflicting group.
       policy 'majority' -> keep the attack_cat that had most rows before step 1
                            (ties are dropped, since there is no majority).
    Returns the clean df and a dict of counts for the audit.
    """
    feats = feature_columns(df)
    info = {"rows_before": len(df)}

    # How many original rows each (features, attack_cat) group had; needed for 'majority'.
    df = df.copy()
    df["_n_orig"] = df.groupby(feats + ["attack_cat"], dropna=False)["attack_cat"].transform("size")

    # Step 1: exact duplicates.
    df = df.drop_duplicates(subset=feats + ["attack_cat"], keep="first")
    info["rows_after_exact"] = len(df)
    info["exact_removed"] = info["rows_before"] - len(df)
    info["class_after_exact"] = df["attack_cat"].value_counts().to_dict()

    # Step 2: conflicting duplicates (same features, different attack_cat).
    df["_group"] = df.groupby(feats, dropna=False).ngroup()
    in_conflict = df.duplicated(subset=feats, keep=False)
    conflict = df[in_conflict]
    info["conflict_groups"] = int(conflict["_group"].nunique())
    info["conflict_rows"] = int(len(conflict))
    # Which label combinations clash, e.g. "Exploits | Reconnaissance".
    pairs = conflict.groupby("_group")["attack_cat"].apply(lambda s: " | ".join(sorted(s)))
    info["conflict_label_sets"] = pairs.value_counts().to_dict()
    # Original row counts (before step 1) involved in conflicts, per class.
    info["conflict_orig_rows_by_class"] = conflict.groupby("attack_cat")["_n_orig"].sum().to_dict()

    if policy == "drop":
        keep = ~in_conflict
    elif policy == "majority":
        # Within each conflicting group keep the row whose label had the most original rows.
        top = conflict.groupby("_group")["_n_orig"].transform("max")
        n_top = conflict.assign(_is_top=conflict["_n_orig"] == top).groupby("_group")["_is_top"].transform("sum")
        winners = conflict.index[(conflict["_n_orig"] == top) & (n_top == 1)]
        keep = ~in_conflict | df.index.isin(winners)
    else:
        raise ValueError(f"unknown conflict_policy: {policy}")

    df = df[keep].drop(columns=["_n_orig", "_group"]).reset_index(drop=True)
    info["policy"] = policy
    info["rows_after_conflicts"] = len(df)
    info["conflict_removed"] = info["rows_after_exact"] - len(df)
    return df, info


def numeric_summary(df, cols):
    """min / median / max / share of zeros / negatives per numeric column."""
    rows = []
    for c in cols:
        x = df[c]
        rows.append({
            "column": c,
            "min": x.min(),
            "median": x.median(),
            "max": x.max(),
            "mean": x.mean(),
            "zeros_%": 100 * (x == 0).mean(),
            "negatives": int((x < 0).sum()),
        })
    return pd.DataFrame(rows)


def class_counts(df):
    return df["attack_cat"].value_counts()


def numeric_columns(df):
    return [c for c in feature_columns(df) if c not in TEXT_COLS and np.issubdtype(df[c].dtype, np.number)]

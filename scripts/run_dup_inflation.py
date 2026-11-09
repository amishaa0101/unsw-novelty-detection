"""Optional experiment: how much do duplicates inflate a normal classifier's test scores?

The baseline classifier is trained and tested on
  (a) the raw data: both files combined, labels cleaned, id dropped, duplicates and conflicts KEPT
  (b) the deduplicated data (data/processed/clean.parquet)
each with a random stratified 80/20 split (no LOAO, all 10 classes known), same features and preprocessing.

Usage: python scripts/run_dup_inflation.py [--config configs/default.yaml]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.baseline import ConfidenceBaseline  # noqa: E402
from src.config import ROOT, load_config, save_run_info, set_seed  # noqa: E402
from src.preprocess import Preprocessor  # noqa: E402


def load_raw_clean_labels(cfg):
    raw = data.load_raw(ROOT / cfg["data_dir"])
    raw, _ = data.normalise_labels(raw)
    raw["label"] = (raw["attack_cat"] != "Normal").astype(int)
    return raw.drop(columns=["id"]).reset_index(drop=True)


def evaluate(df, setting, cfg):
    seed = cfg["seed"]
    tr, te = train_test_split(df.index, test_size=0.2, stratify=df["attack_cat"], random_state=seed)
    prep = Preprocessor(setting, cfg["shortcut_features"], cfg["noise_features"], cfg["strict_extra_features"],
                        cfg["preprocess"]["proto_top_n"], cfg["preprocess"]["log1p"]).fit(df.loc[tr])
    names = sorted(df["attack_cat"].unique())
    code = {n: i for i, n in enumerate(names)}
    y = df["attack_cat"].map(code).to_numpy()
    clf = ConfidenceBaseline(seed).fit(prep.transform(df.loc[tr]), y[tr])
    pred = clf.model.predict(prep.transform(df.loc[te]))

    # Share of test rows whose exact feature vector also appears in the training split, and the share
    # whose feature vector appears in training with a different attack_cat (a conflicting copy).
    feats = data.feature_columns(df)
    train_labels = {}
    for key, lab in zip(df.loc[tr, feats].itertuples(index=False, name=None), df.loc[tr, "attack_cat"]):
        train_labels.setdefault(key, set()).add(lab)
    keys = list(df.loc[te, feats].itertuples(index=False, name=None))
    labs = df.loc[te, "attack_cat"].tolist()
    in_train = sum(k in train_labels for k in keys) / len(te)
    other_label = sum(bool(train_labels.get(k, set()) - {lab}) for k, lab in zip(keys, labs)) / len(te)
    return {"setting": setting, "n_rows": len(df), "n_test": len(te),
            "accuracy": accuracy_score(y[te], pred), "macro_f1": f1_score(y[te], pred, average="macro"),
            "test_rows_with_exact_copy_in_train": in_train,
            "test_rows_with_copy_in_train_under_another_label": other_label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    save_run_info(cfg, "dup_inflation")
    datasets = {"raw (duplicates and conflicts kept)": load_raw_clean_labels(cfg),
                "deduplicated": pd.read_parquet(ROOT / cfg["processed_dir"] / "clean.parquet").reset_index(drop=True)}
    rows = []
    for name, df in datasets.items():
        for setting in cfg["feature_settings"]:
            r = {"data": name, **evaluate(df, setting, cfg)}
            print(r)
            rows.append(r)
    out = ROOT / "outputs" / "tables" / "dup_inflation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    print("Saved", out)


if __name__ == "__main__":
    main()

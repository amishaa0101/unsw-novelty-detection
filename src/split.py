"""Leave-one-attack-out split: hide one class, split the known rows into 4 parts."""
from sklearn.model_selection import train_test_split


def make_splits(df, hidden, split_cfg, seed):
    """Return a dict of row-index arrays: train, calib_norm, calib_final, test.

    - Rows of the `hidden` class go 100% to test (hidden=None hides nothing).
    - Known rows are split stratified by attack_cat into train / calib / test
      (default 60 / 20 / 20), then calib is split again, stratified, into
      calib_norm and calib_final (default 10 / 10).
    """
    is_hidden = df["attack_cat"] == hidden  # all False when hidden is None
    known = df[~is_hidden]

    test_frac = split_cfg["test"]
    calib_frac = split_cfg["calib_norm"] + split_cfg["calib_final"]

    rest_idx, test_idx = train_test_split(
        known.index, test_size=test_frac, stratify=known["attack_cat"], random_state=seed
    )
    rest = known.loc[rest_idx]
    # calib share of what is left after removing test, e.g. 0.2 / 0.8 = 0.25
    train_idx, calib_idx = train_test_split(
        rest.index, test_size=calib_frac / (1 - test_frac), stratify=rest["attack_cat"], random_state=seed
    )
    calib = known.loc[calib_idx]
    norm_idx, final_idx = train_test_split(
        calib.index,
        test_size=split_cfg["calib_final"] / calib_frac,
        stratify=calib["attack_cat"],
        random_state=seed,
    )

    hidden_idx = df.index[is_hidden]
    return {
        "train": train_idx.to_numpy(),
        "calib_norm": norm_idx.to_numpy(),
        "calib_final": final_idx.to_numpy(),
        "test": test_idx.append(hidden_idx).to_numpy(),
    }

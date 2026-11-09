"""Turn the clean table into float32 matrices. Fit on training rows only."""
import numpy as np

from src.data import LABEL_COLS, META_COLS, TEXT_COLS

SETTINGS = ("with_shortcut", "without_shortcut", "strict")
OTHER = "other"  # bucket for rare or unseen category values


def select_features(columns, setting, shortcut_features, noise_features, strict_extra_features):
    """Which columns are features in this setting.

    - noise features are dropped in every setting
    - shortcut features are dropped in without_shortcut and strict
    - round-trip time features are dropped in strict only
    """
    if setting not in SETTINGS:
        raise ValueError(f"unknown feature setting: {setting}")
    drop = set(LABEL_COLS) | set(META_COLS) | set(noise_features)
    if setting in ("without_shortcut", "strict"):
        drop |= set(shortcut_features)
    if setting == "strict":
        drop |= set(strict_extra_features)
    return [c for c in columns if c not in drop]


class Preprocessor:
    """log1p + standardise numeric columns, one-hot encode text columns.

    fit() learns everything (which columns get log1p, means, stds, category lists) from
    training rows only. transform() only applies what fit() learned and never refits.
    """

    def __init__(self, setting, shortcut_features, noise_features, strict_extra_features,
                 proto_top_n=10, log1p=True):
        self.setting = setting
        self.shortcut_features = list(shortcut_features)
        self.noise_features = list(noise_features)
        self.strict_extra_features = list(strict_extra_features)
        self.proto_top_n = proto_top_n
        self.log1p = log1p

    def fit(self, train_df):
        cols = select_features(train_df.columns, self.setting, self.shortcut_features,
                               self.noise_features, self.strict_extra_features)
        # Labels and metadata (including our audit-only source_file column) must never be features.
        assert not set(cols) & (set(LABEL_COLS) | set(META_COLS)), "label or metadata column selected as feature"
        self.text_cols = [c for c in cols if c in TEXT_COLS]
        self.num_cols = [c for c in cols if c not in TEXT_COLS]

        # log1p only where training values are all >= 0 (log1p of a value below -1 is undefined).
        num = train_df[self.num_cols].to_numpy(dtype=np.float64)
        has_neg = (num < 0).any(axis=0)
        self.log_cols = [c for c, neg in zip(self.num_cols, has_neg) if self.log1p and not neg]
        self.skipped_log = [c for c, neg in zip(self.num_cols, has_neg) if neg]
        self._log_mask = np.array([c in self.log_cols for c in self.num_cols])

        num = self._log(num)
        self.mean_ = num.mean(axis=0)
        std = num.std(axis=0)
        self.zero_std_cols = [c for c, s in zip(self.num_cols, std) if s < 1e-12]
        self.std_ = np.where(std < 1e-12, 1.0, std)  # constant column: avoid dividing by zero

        # Category lists come from training rows only; everything else maps to "other".
        self.categories_ = {}
        for c in self.text_cols:
            counts = train_df[c].value_counts()
            keep = counts.index[: self.proto_top_n] if c == "proto" else counts.index
            self.categories_[c] = sorted(set(keep)) + [OTHER]

        self.feature_names = list(self.num_cols) + [
            f"{c}={v}" for c in self.text_cols for v in self.categories_[c]]
        self.onehot_mask = np.array([False] * len(self.num_cols) + [True] * (len(self.feature_names) - len(self.num_cols)))
        self.fit_index_ = train_df.index.copy()  # lets the experiment assert we fitted on train only
        return self

    def _log(self, num):
        num = num.copy()
        num[:, self._log_mask] = np.log1p(num[:, self._log_mask])
        return num

    def transform(self, df):
        num = (self._log(df[self.num_cols].to_numpy(dtype=np.float64)) - self.mean_) / self.std_
        parts = [num]
        for c in self.text_cols:
            cats = self.categories_[c]
            values = df[c].where(df[c].isin(cats), OTHER).to_numpy()  # unseen value -> "other"
            parts.append((values[:, None] == np.array(cats)[None, :]).astype(np.float64))
        X = np.hstack(parts).astype(np.float32)
        assert np.isfinite(X).all(), "non-finite value after preprocessing"
        return np.ascontiguousarray(X)

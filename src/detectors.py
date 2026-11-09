"""Novelty detectors. Each has fit(X) and score(X); a higher score means more unusual."""
import faiss
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture


class GMMDetector:
    """Gaussian mixture fitted with EM. Score = negative log likelihood."""

    def __init__(self, k_grid, covariance_type="diag", n_init=3, reg_covar=1e-4, max_fit_rows=20000, seed=42):
        self.k_grid = list(k_grid)
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.reg_covar = reg_covar
        self.max_fit_rows = max_fit_rows
        self.seed = seed

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)  # float32 makes EM fail on tight one-hot columns
        self.n_rows = len(X)
        self.subsampled = len(X) > self.max_fit_rows
        if self.subsampled:  # speed: fit on a random subsample of a big class
            rng = np.random.default_rng(self.seed)
            X = X[rng.choice(len(X), self.max_fit_rows, replace=False)]
        # Rule: at most one component per 20 rows, so every component has enough data.
        k_max = max(1, len(X) // 20)
        grid = [k for k in self.k_grid if k <= k_max] or [1]
        self.bic = {}
        best = None
        for k in grid:
            gm = GaussianMixture(n_components=k, covariance_type=self.covariance_type, n_init=self.n_init,
                                 reg_covar=self.reg_covar, random_state=self.seed)  # never warm_start
            gm.fit(X)
            self.bic[k] = float(gm.bic(X))
            if best is None or self.bic[k] < self.bic[best[0]]:
                best = (k, gm)
        self.k, self.model = best
        return self

    def score(self, X):
        return -self.model.score_samples(np.asarray(X, dtype=np.float64))

    def info(self):
        return {"k": self.k, "bic": self.bic, "n_rows": self.n_rows, "subsampled": self.subsampled,
                "reg_covar": self.reg_covar}

    def nll_breakdown(self, X, onehot_mask):
        """Split the NLL under each row's most responsible component into per-column parts.

        For a diag Gaussian: -log N(x) = sum over columns of
            0.5 * log(2 pi var_d)          (log-det part, same for every row)
          + (x_d - mu_d)^2 / (2 var_d)     (squared-distance part)
        Returns a dict of per-row sums for one-hot and numeric columns.
        """
        assert self.covariance_type == "diag"
        j = self.model.predict_proba(X).argmax(axis=1)
        mu, var = self.model.means_[j], self.model.covariances_[j]
        quad = (X - mu) ** 2 / (2 * var)
        logdet = 0.5 * np.log(2 * np.pi * var)
        return {
            "quad_onehot": quad[:, onehot_mask].sum(axis=1),
            "quad_numeric": quad[:, ~onehot_mask].sum(axis=1),
            "logdet_onehot": logdet[:, onehot_mask].sum(axis=1),
            "logdet_numeric": logdet[:, ~onehot_mask].sum(axis=1),
        }


class PCADetector:
    """Low-rank model from the SVD. Score = squared reconstruction error."""

    def __init__(self, explained_variance=0.95):
        self.explained_variance = explained_variance

    def fit(self, X):
        X = X.astype(np.float64)
        self.mean_ = X.mean(axis=0)
        _, s, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        var = s ** 2
        self.evr_ = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
        cum = np.cumsum(self.evr_)
        k = int(np.searchsorted(cum, self.explained_variance) + 1)  # smallest k reaching the target
        cap = min(len(X) - 1, X.shape[1] - 1)
        self.k = max(1, min(k, cap))
        self.V_ = Vt[: self.k]  # (k, d), rows are orthonormal directions
        self.n_rows = len(X)
        return self

    def score(self, X):
        Xc = X.astype(np.float64) - self.mean_
        x_hat = Xc @ self.V_.T @ self.V_  # projection onto the top-k directions
        return ((Xc - x_hat) ** 2).sum(axis=1)

    def info(self):
        return {"k": self.k, "n_rows": self.n_rows, "evr": self.evr_.tolist()}


class KNNDetector:
    """Exact nearest neighbours with FAISS. Score = mean distance to the k nearest training rows."""

    def __init__(self, k=5):
        self.k_requested = k

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.index = faiss.IndexFlatL2(X.shape[1])  # a new index on every fit, never reused
        self.index.add(X)
        self.k = max(1, min(self.k_requested, len(X) - 1))
        self.n_rows = len(X)
        return self

    def score(self, X):
        D, _ = self.index.search(np.ascontiguousarray(X, dtype=np.float32), self.k)
        # FAISS returns squared L2 distances; tiny negatives can appear from rounding.
        return np.sqrt(np.maximum(D, 0)).mean(axis=1)

    def info(self):
        return {"k": self.k, "n_rows": self.n_rows}


class PerClassDetector:
    """One detector per known class. score() returns a matrix (n_samples, n_classes)."""

    def __init__(self, make_detector, classes):
        self.make_detector = make_detector
        self.classes = list(classes)

    def fit(self, X, y):
        self.detectors = {}
        for c in self.classes:
            rows = X[y == c]
            assert len(rows) > 0, f"no training rows for class {c}"
            self.detectors[c] = self.make_detector().fit(rows)
        return self

    def score(self, X):
        return np.column_stack([self.detectors[c].score(X) for c in self.classes])

    def info(self):
        return {c: d.info() for c, d in self.detectors.items()}


def class_separation_auroc(S, y, classes):
    """For each class c: AUROC of score column c, other classes' rows (positive) vs class c rows (negative)."""
    out = {}
    for i, c in enumerate(classes):
        neg = y == c
        if neg.sum() == 0 or (~neg).sum() == 0:
            continue
        out[c] = float(roc_auc_score((~neg).astype(int), S[:, i]))
    return out


def select_gmm_reg_covar(gmm_cfg, seed, classes, X_train, y_train, X_norm, y_norm):
    """Fit per-class GMMs for each reg_covar and keep the one that best separates known classes on calib_norm.

    Only train rows (fitting) and calib_norm rows (choosing) are used; calib_final and test are never seen.
    Returns the fitted PerClassDetector and a table of mean AUROC per reg_covar.
    """
    table, best = {}, None
    for reg in gmm_cfg["reg_covar_grid"]:
        det = PerClassDetector(lambda reg=reg: GMMDetector(
            gmm_cfg["k_grid"], gmm_cfg["covariance_type"], gmm_cfg["n_init"], reg, gmm_cfg["max_fit_rows"], seed),
            classes).fit(X_train, y_train)
        per_class = class_separation_auroc(det.score(X_norm), y_norm, classes)
        table[reg] = {"mean_auroc": float(np.mean(list(per_class.values()))), "per_class": per_class}
        if best is None or table[reg]["mean_auroc"] > table[best[0]]["mean_auroc"]:
            best = (reg, det)
    return best[1], best[0], table

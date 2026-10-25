"""
lcmi_c — local (conditional) mutual information for categorical data.

Covers the cases the regression-based ``geocmi.lcmi.CMI`` cannot: a NOMINAL outcome
(classes) with nominal and/or continuous predictors. (Categorical predictors
with a continuous outcome are handled by ``CMI(..., categorical=[...])``.)

All estimators are invariant to how categories are labelled.

  nominal x  → nominal y   debiased local plug-in MI: Miller–Madow corrected
                           contingency tables minus a permutation null;
                           optional conditioning on nominal / binned covariates
                           (local CMI  Σ_z p(z) I(x; y | z))
  continuous x → nominal y local discrete–continuous MI (Ross 2014) between y and
                           x adjusted for the other predictors (one-hot for
                           nominal ones)

Quick start
-----------
    from geocmi import lcmi_c

    df["edu_q5"] = lcmi_c.quantile_codes(df["edu"], 5)
    m = lcmi_c.CMIc(df, u="lon", v="lat", y="vote_class",
             xlist=["edu_q5", "income"], categorical=["edu_q5"]).fit(condition="auto")
    m.summary()                 # mean local MI / CMI, ranking, reliability check
    m.plot_maps(kind="cmi")
    m.df_out                    # data + {x}_mi, {x}_cmi and smoothed versions

Notes on conditioning
---------------------
Local CMI needs every stratum of z to be populated inside a neighbourhood.
Keep categories coarse (e.g. quintiles for x, terciles for z) and windows large
enough. ``fit`` reports a reliability check: the CMI obtained with a shuffled
(irrelevant) z should retain ≈ 100 % of the pairwise MI; values well below
~80 % mean the strata are too sparse for the chosen window.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from .lcmi import _mi_ross

__all__ = ["CMIc", "quantile_codes", "cramers_v", "onehot", "conditioning_matrix",
           "local_mi_mixed", "local_mi_nominal"]


# ── estimators (label-invariant) ──────────────────────────────────────

def onehot(codes) -> np.ndarray:
    """One-hot dummies for a nominal variable (first level dropped)."""
    codes = np.asarray(codes)
    levels = np.unique(codes)
    return np.column_stack([(codes == lv).astype(float) for lv in levels[1:]]) \
        if levels.size > 1 else np.zeros((codes.size, 0))


def conditioning_matrix(X, categorical_mask) -> np.ndarray:
    """Stack numeric columns as is and categorical columns as one-hot dummies."""
    X = np.asarray(X)
    cols = []
    for j, is_cat in enumerate(categorical_mask):
        cols.append(onehot(X[:, j]) if is_cat else X[:, [j]].astype(float))
    return np.column_stack(cols) if cols else np.zeros((X.shape[0], 0))


def local_mi_mixed(codes, x, nn_idx, k: int = 3) -> np.ndarray:
    """Local MI between a nominal and a continuous variable in each
    neighbourhood (rows of ``nn_idx``), Ross (2014) estimator. Works for
    nominal x / continuous y and continuous x / nominal y alike."""
    codes = np.asarray(codes)
    x = np.asarray(x, dtype=float)
    out = np.empty(nn_idx.shape[0])
    for i, nb in enumerate(nn_idx):
        xi = x[nb]
        out[i] = _mi_ross(codes[nb], (xi - xi.mean()) / (xi.std() + 1e-12), k=k)
    return out


def _codes(a):
    _, inv = np.unique(np.asarray(a), return_inverse=True)
    return inv


def _local_cmi_counts(x, y, z, nn_idx, cx, cy, cz):
    n, bw = nn_idx.shape
    J = ((z * cx + x) * cy + y)[nn_idx]
    cnt = np.zeros((n, cz * cx * cy))
    np.add.at(cnt, (np.repeat(np.arange(n), bw), J.ravel()), 1)
    N = cnt.reshape(n, cz, cx, cy)
    out = np.zeros(n)
    for zz in range(cz):
        Nz = N[:, zz]
        m = Nz.sum((1, 2))
        P = Nz / np.maximum(m, 1)[:, None, None]
        px, py = P.sum(2, keepdims=True), P.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(P > 0, P * np.log(P / (px * py)), 0.0)
        mi = t.sum((1, 2))
        kx = (px[:, :, 0] > 0).sum(1)
        ky = (py[:, 0, :] > 0).sum(1)
        kxy = (P > 0).sum((1, 2))
        mm = np.where(m > 1, mi + (kxy - kx - ky + 1) / (2 * np.maximum(m, 1)), 0.0)  # Miller–Madow
        out += (m / bw) * mm
    return out


def local_mi_nominal(x, y, nn_idx, z=None, n_perm: int = 5, seed: int = 0) -> np.ndarray:
    """Debiased local mutual information between nominal x and nominal y.

    Plug-in estimate from local contingency tables (Miller–Madow corrected),
    minus the mean of a permutation null in which x is shuffled (within the
    strata of z, if given) — this removes the small-sample bias of sparse
    local tables. With ``z`` (a nominal conditioning variable; combine several
    with ``np.unique(np.c_[...], axis=0, return_inverse=True)[1]``) the result
    is the local conditional MI  Σ_z p(z) I(x; y | z). Keep z coarse: each
    stratum needs enough points inside a neighbourhood.
    """
    rng = np.random.default_rng(seed)
    x, y = _codes(x), _codes(y)
    z = np.zeros_like(x) if z is None else _codes(z)
    cx, cy, cz = x.max() + 1, y.max() + 1, z.max() + 1
    obs = _local_cmi_counts(x, y, z, nn_idx, cx, cy, cz)
    null = np.zeros_like(obs)
    for _ in range(int(n_perm)):
        xs = x.copy()
        for zz in range(cz):
            idx = np.where(z == zz)[0]
            xs[idx] = x[rng.permutation(idx)]
        null += _local_cmi_counts(xs, y, z, nn_idx, cx, cy, cz)
    return obs - null / max(int(n_perm), 1)


# ── helpers ───────────────────────────────────────────────────────────

def quantile_codes(x, q: int) -> np.ndarray:
    """Integer codes 0..q-1 of the q quantile classes of a numeric variable."""
    return pd.qcut(pd.Series(x).rank(method="first"), q=q, labels=False).to_numpy().astype(int)


def cramers_v(a, b) -> float:
    """Cramér's V between two nominal variables."""
    tab = pd.crosstab(np.asarray(a), np.asarray(b)).to_numpy().astype(float)
    n = tab.sum()
    e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / n
    chi2 = ((tab - e) ** 2 / e).sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1)))) if min(tab.shape) > 1 else 0.0


def _knn_smooth(surf, coords, k):
    k = min(k, len(surf))
    d, idx = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(coords)
    h = d[:, -1:] / 2 + 1e-12
    w = np.exp(-d ** 2 / (2 * h ** 2))
    return (w * surf[idx]).sum(1) / w.sum(1)


# ── model ─────────────────────────────────────────────────────────────

class CMIc:
    """Local (conditional) mutual information maps for a nominal outcome.

    Parameters
    ----------
    df, u, v     : data and coordinate columns
    y            : nominal outcome column (any labels)
    xlist        : predictor columns
    categorical  : predictors that are nominal (default: non-numeric columns and
                   integer columns with at most 20 distinct values)
    bw           : neighbourhood size (number of nearest neighbours); default
                   min(max(200, n // 50), 1000)
    n_perm       : permutations for the debiasing null
    smooth_k     : neighbours of the Gaussian smoothing used for *_smooth maps
    """

    def __init__(self, df: pd.DataFrame, u: str, v: str, y: str, xlist: Sequence[str],
                 categorical: Optional[Sequence[str]] = None, bw: Optional[int] = None,
                 n_perm: int = 5, smooth_k: int = 60, seed: int = 42, verbose: bool = True):
        self.df_raw = df.copy()
        self.u_col, self.v_col, self.y_col = u, v, y
        self.xlist = list(xlist)
        self.n = len(df)
        self.verbose = verbose
        self.n_perm, self.smooth_k, self.seed = int(n_perm), int(smooth_k), seed
        if categorical is None:
            categorical = [c for c in self.xlist
                           if not pd.api.types.is_numeric_dtype(df[c])
                           or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= 20)]
        unknown = set(categorical) - set(self.xlist)
        if unknown:
            raise ValueError(f"categorical variables not in xlist: {sorted(unknown)}")
        self.categorical = list(categorical)
        self._is_cat = {x: x in self.categorical for x in self.xlist}
        self.Y = _codes(df[y].to_numpy())
        raw = df[[u, v]].to_numpy(float)
        rng_ = np.where(np.ptp(raw, 0) > 0, np.ptp(raw, 0), 1.0)
        self._coords = (raw - raw.min(0)) / rng_
        self.bw = int(min(max(200, self.n // 50), 1000, self.n - 1) if bw is None else min(bw, self.n - 1))
        self._nn = NearestNeighbors(n_neighbors=self.bw).fit(self._coords).kneighbors(self._coords)[1]
        self.df_out: Optional[pd.DataFrame] = None
        self.meta: dict = {}

    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _code_of(self, col, levels_z):
        """Nominal codes of a conditioning column (numeric ones are binned)."""
        x = self.df_raw[col].to_numpy()
        if self._is_cat.get(col, not pd.api.types.is_numeric_dtype(self.df_raw[col])):
            return _codes(x)
        return quantile_codes(x, levels_z)

    def _partner(self, x, levels_z):
        others = [c for c in self.xlist if c != x]
        xc = self._code_of(x, levels_z) if not self._is_cat[x] else _codes(self.df_raw[x])
        scores = {c: cramers_v(xc, self._code_of(c, levels_z)) for c in others}
        return max(scores, key=scores.get) if scores else None

    def fit(self, condition: Union[None, str, dict] = None, levels_z: int = 3,
            check: bool = True) -> "CMIc":
        """Compute local MI maps (and local CMI maps when ``condition`` is given).

        condition : None   — pairwise maps only
                    "auto" — condition each nominal x on its most associated other
                             predictor (Cramér's V)
                    dict   — {x: [conditioning columns]}; any column of df can be
                             used (numeric ones are binned into ``levels_z`` quantiles)
        levels_z  : quantile bins for numeric conditioning columns
        check     : reliability check of the conditioning (shuffled-z retention)
        """
        out = self.df_raw.copy()
        cat_mask = [self._is_cat[c] for c in self.xlist]
        X_all = self.df_raw[self.xlist].to_numpy()
        rows = []
        rng = np.random.default_rng(self.seed)
        for j, x in enumerate(self.xlist):
            row = dict(variable=x, type="nominal" if self._is_cat[x] else "continuous")
            if self._is_cat[x]:
                mi = local_mi_nominal(self.df_raw[x], self.Y, self._nn, n_perm=self.n_perm, seed=self.seed)
            else:
                others = [i for i in range(len(self.xlist)) if i != j]
                Z = conditioning_matrix(X_all[:, others], [cat_mask[i] for i in others])
                xv = self.df_raw[x].to_numpy(float)
                if Z.shape[1]:
                    A = np.c_[np.ones(self.n), Z]
                    xv = xv - A @ np.linalg.lstsq(A, xv, rcond=None)[0]
                mi = local_mi_mixed(self.Y, xv, self._nn)
            out[f"{x}_mi"] = mi
            out[f"{x}_mi_smooth"] = _knn_smooth(mi, self._coords, self.smooth_k)
            row["mean_mi"] = float(mi.mean())

            zcols = None
            if condition == "auto" and self._is_cat[x]:
                p = self._partner(x, levels_z)
                zcols = [p] if p else None
            elif isinstance(condition, dict):
                zcols = condition.get(x)
            if zcols:
                if not self._is_cat[x]:
                    warnings.warn(f"{x} is continuous: it is already adjusted for the other "
                                  "predictors; explicit conditioning applies to nominal x only.")
                else:
                    Zc = np.column_stack([self._code_of(c, levels_z) for c in zcols])
                    z = np.unique(Zc, axis=0, return_inverse=True)[1].ravel()
                    cmi = local_mi_nominal(self.df_raw[x], self.Y, self._nn, z=z,
                                           n_perm=self.n_perm, seed=self.seed)
                    out[f"{x}_cmi"] = cmi
                    out[f"{x}_cmi_smooth"] = _knn_smooth(cmi, self._coords, self.smooth_k)
                    row.update(given=", ".join(zcols), mean_cmi=float(cmi.mean()))
                    if check:
                        zs = z[rng.permutation(self.n)]
                        null_c = local_mi_nominal(self.df_raw[x], self.Y, self._nn, z=zs,
                                                  n_perm=self.n_perm, seed=self.seed)
                        ret = float(null_c.mean() / mi.mean()) if mi.mean() > 1e-9 else np.nan
                        row["retention_check"] = ret
                        row["cmi_over_baseline"] = (float(cmi.mean() / null_c.mean())
                                                    if null_c.mean() > 1e-9 else np.nan)
                        if np.isfinite(ret) and ret < 0.8:
                            warnings.warn(f"{x}: conditioning on {zcols} is under-powered "
                                          f"(retention {ret:.0%}); use coarser categories or a "
                                          "larger bw.")
            rows.append(row)
            self._log(f"  [{x}] mean MI = {row['mean_mi']:.4f}"
                      + (f"  mean CMI | {row['given']} = {row['mean_cmi']:.4f}" if "mean_cmi" in row else ""))
        self.df_out = out
        self._summary = pd.DataFrame(rows).set_index("variable")
        self.meta = dict(bw=self.bw, n=self.n, n_classes=int(self.Y.max() + 1),
                         categorical=self.categorical, condition=condition)
        return self

    def summary(self) -> pd.DataFrame:
        """Mean local MI / CMI per predictor, ranks, and the reliability check."""
        if self.df_out is None:
            raise RuntimeError("Call fit() first.")
        s = self._summary.copy()
        s["rank_mi"] = s["mean_mi"].rank(ascending=False).astype(int)
        if "mean_cmi" in s:
            s["rank_cmi"] = s["mean_cmi"].rank(ascending=False).astype("Int64")
        return s

    def plot_maps(self, kind: str = "mi", smooth: bool = True, s: float = 3.0, ncols: int = 2,
                  panel_size: float = 4.5, cmap: str = "magma", save: str = None):
        """Maps of local MI (kind="mi") or CMI (kind="cmi") on the coordinates."""
        if self.df_out is None:
            raise RuntimeError("Call fit() first.")
        suf = f"_{kind}" + ("_smooth" if smooth else "")
        cols = [x for x in self.xlist if f"{x}{suf}" in self.df_out]
        if not cols:
            raise ValueError(f"no {kind} maps available (fit with condition=... for cmi)")
        nrows = int(np.ceil(len(cols) / ncols))
        fig, axes = plt.subplots(nrows, min(ncols, len(cols)),
                                 figsize=(panel_size * 1.4 * min(ncols, len(cols)), panel_size * nrows),
                                 squeeze=False)
        u = self.df_raw[self.u_col].to_numpy(float)
        v = self.df_raw[self.v_col].to_numpy(float)
        hi = max(np.nanpercentile(self.df_out[f"{c}{suf}"], 98) for c in cols)
        for i, c in enumerate(cols):
            ax = axes[i // ncols, i % ncols]
            sc = ax.scatter(u, v, c=self.df_out[f"{c}{suf}"], s=s, cmap=cmap, vmin=0, vmax=hi)
            given = self._summary.loc[c].get("given") if kind == "cmi" else None
            ax.set_title(f"{c}: local {kind.upper()}" + (f" | {given}" if isinstance(given, str) else ""))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            plt.colorbar(sc, ax=ax, fraction=0.04, label="nats")
        for k in range(len(cols), nrows * min(ncols, len(cols))):
            axes[k // ncols, k % ncols].axis("off")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        return fig

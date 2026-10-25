"""
lcmi — Local Conditional Mutual Information for interpretable, nonlinear
spatially varying effects.

Two-stage estimator
-------------------
Stage 1  local conditional mutual information I(x_j; y | other x) in a k-NN
         window around every location (KSG estimator)  →  {v}_lws
         and global partial-residual curves G_j(x_j).
Stage 2  varying-coefficient model  y = b_0(s) + Σ_j b_j(s) · g_j(x_j)
         b_j(s): tensor-product P-spline surfaces (GCV-tuned penalty)
         g_j   : learned transforms, initialised from G_j and re-estimated
                 jointly with b(s) by backfitting
         →  {v}_esd  (scale-identified coefficient: Y change per 1 SD of g_j)
            {v}_lin  (linear-equivalent dY/dx_j), {v}_vc (raw multiplier)

Quick start
-----------
    from geocmi import lcmi

    model = lcmi.CMI(df, u="u", v="v", y="y", xlist=["x1", "x2", "x3"]).fit()
    model.summary()
    model.coef_table()                 # summary of the coefficient surfaces
    model.plot_coefficients()          # maps of {v}_esd
    model.plot_transforms()            # learned g_j curves (read signs with these)
    out = model.df_out                 # data + *_lws, *_esd, *_lin, *_vc, y_pred_vc
"""



from __future__ import annotations

__version__ = "0.2.0"

import functools
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, t as t_dist
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import lstsq as np_lstsq



def _quiet_matmul(fn):
    """Silence spurious 'encountered in matmul' RuntimeWarnings raised by some
    NumPy ≥ 2 / BLAS builds (e.g. macOS Accelerate) on finite inputs.
    Other warnings are left untouched."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*encountered in matmul",
                                    category=RuntimeWarning)
            return fn(*args, **kwargs)

    return wrapper


# ── Result container ──────────────────────────────────────────────────


@dataclass
class CMIResult:
    df_out: pd.DataFrame = field(default=None)
    alpha: np.ndarray = field(default=None)
    y_local_mean: np.ndarray = field(default=None)
    H: np.ndarray = field(default=None)
    G: np.ndarray = field(default=None)
    G_raw: np.ndarray = field(
        default=None
    )  # raw-scale partial-residual LOWESS features
    beta_wls: np.ndarray = field(default=None)  # local WLS beta estimates (n, p)
    beta_vc: np.ndarray = field(default=None)  # P-spline VC coefficient surfaces (n, p)
    intercept_vc: np.ndarray = field(default=None)  # P-spline VC intercept surface (n,)
    G_vc: np.ndarray = field(default=None)  # backfitted transforms g_j(x_j) (n, p)
    cmi_contrib: np.ndarray = field(default=None)
    cmi_surfaces: dict = field(default_factory=dict)
    pdp_curves: dict = field(default_factory=dict)
    k_search_table: Optional[pd.DataFrame] = field(default=None)
    bw_search_table: Optional[pd.DataFrame] = field(default=None)
    meta: dict = field(default_factory=dict)


# ── Numerical helpers ─────────────────────────────────────────────────


def _batch_ols_resid(
    T_nb: np.ndarray, Z_nb: np.ndarray, ridge_lambda: float
) -> np.ndarray:
    T_nb = T_nb.astype(np.float32, copy=False)
    Z_nb = Z_nb.astype(np.float32, copy=False)
    n_pts, bw_sz = T_nb.shape

    Tm = T_nb.mean(axis=1, keepdims=True)
    Ts = T_nb.std(axis=1, keepdims=True) + np.float32(1e-7)
    Tt = (T_nb - Tm) / Ts

    Zm = Z_nb.mean(axis=1, keepdims=True)
    Zs = Z_nb.std(axis=1, keepdims=True) + np.float32(1e-7)
    Zn = (Z_nb - Zm) / Zs

    ones = np.ones((n_pts, bw_sz, 1), dtype=np.float32)
    D = np.concatenate([ones, Zn], axis=2)

    DtD = np.einsum("ijk,ijl->ikl", D, D).astype(np.float64)
    Dty = np.einsum("ijk,ij->ik", D, Tt).astype(np.float64)

    p_int = DtD.shape[-1]
    Rpen = np.eye(p_int, dtype=np.float64)
    Rpen[0, 0] = 0.0
    lam = max(float(ridge_lambda), 0.0)
    epsI = np.float64(1e-12) * np.eye(p_int, dtype=np.float64)
    beta = np.linalg.solve(DtD + lam * Rpen + epsI, Dty[:, :, np.newaxis])[
        :, :, 0
    ].astype(np.float32)
    resid = Tt - np.einsum("ijk,ik->ij", D, beta)
    rs = resid.std(axis=1, keepdims=True) + np.float32(1e-7)
    return resid / rs


def _batch_ols_resid_chunked(
    T_nb: np.ndarray,
    Z_nb: np.ndarray,
    ridge_lambda: float,
    chunk_size: int = 2000,
) -> np.ndarray:
    """
    Memory-safe wrapper around _batch_ols_resid.

    Splits the n_pts axis into chunks of `chunk_size` rows and processes
    each chunk independently, then concatenates results.  Peak memory is
    O(chunk_size × bw × p) instead of O(n × bw × p), so n_jobs=-1 is
    safe even for large n and wide bw.
    """
    n_pts = T_nb.shape[0]
    if n_pts <= chunk_size:
        return _batch_ols_resid(T_nb, Z_nb, ridge_lambda)

    parts = []
    for start in range(0, n_pts, chunk_size):
        end = min(start + chunk_size, n_pts)
        parts.append(_batch_ols_resid(T_nb[start:end], Z_nb[start:end], ridge_lambda))
    return np.concatenate(parts, axis=0)


def _mi_ksg_sorted(Xr: np.ndarray, Yr: np.ndarray, k: int = 8) -> float:
    Xr, Yr = np.asarray(Xr, dtype=np.float64), np.asarray(Yr, dtype=np.float64)
    n_pts = len(Xr)
    if n_pts < k + 2:
        return 0.0

    XY = np.stack([Xr, Yr], axis=1)
    dists, _ = cKDTree(XY).query(XY, k=k + 1, p=np.inf)
    eps = dists[:, k]

    # KSG algorithm 1: n_x(i) = #{j != i : |x_j - x_i| < eps_i} (strict).
    # Inclusive counting adds the k-th neighbour itself to one marginal and
    # biases MI downward, most strongly for small windows / weak dependence.
    # Exact ties (eps_i == 0) fall back to the previous inclusive count.
    xs = np.sort(Xr)
    ys = np.sort(Yr)
    tie = eps <= 0
    eps_t = np.where(tie, 1e-12, eps)

    def _count(sorted_v, v):
        hi = np.where(
            tie,
            np.searchsorted(sorted_v, v + eps_t, "right"),
            np.searchsorted(sorted_v, v + eps_t, "left"),
        )
        lo = np.where(
            tie,
            np.searchsorted(sorted_v, v - eps_t, "left"),
            np.searchsorted(sorted_v, v - eps_t, "right"),
        )
        return hi - lo - 1

    cx = _count(xs, Xr)
    cy = _count(ys, Yr)
    cx = np.maximum(cx, 0).astype(np.float64)
    cy = np.maximum(cy, 0).astype(np.float64)

    return max(
        0.0,
        float(digamma(k) + digamma(n_pts) - np.mean(digamma(cx + 1) + digamma(cy + 1))),
    )


def _mi_ksg_global(Xr: np.ndarray, Yr: np.ndarray, k: int = 8) -> float:
    return _mi_ksg_sorted(Xr, Yr, k)


def _ols_resid_scalar(target: np.ndarray, regressors: np.ndarray, ridge_lambda: float):
    target = np.asarray(target, float)
    Zm = np.c_[np.ones(len(target)), np.asarray(regressors, float)]
    for c in range(1, Zm.shape[1]):
        s = Zm[:, c].std()
        if s > 1e-10:
            Zm[:, c] = (Zm[:, c] - Zm[:, c].mean()) / s
    s = target.std()
    if s < 1e-10:
        return None
    ts = (target - target.mean()) / s
    p = Zm.shape[1]
    XtX = Zm.T @ Zm
    Xty = Zm.T @ ts
    Rpen = np.eye(p, dtype=float)
    Rpen[0, 0] = 0.0
    beta = np.linalg.solve(
        XtX + max(float(ridge_lambda), 0.0) * Rpen + 1e-12 * np.eye(p), Xty
    )
    res = ts - Zm @ beta
    sr = res.std()
    return res / sr if sr > 1e-10 else None


def _precompute_residuals(
    Y, ALL_X, col_idx, nl_flag, nn_idx, ridge_lambda: float, chunk_size: int = 2000,
    Z_other=None,
):
    if Z_other is None:
        Z_other = np.delete(ALL_X, col_idx, axis=1)
    Y_nb = Y[nn_idx].astype(np.float32)
    Xj_nb = ALL_X[nn_idx, col_idx].astype(np.float32)
    Z_nb = Z_other[nn_idx].astype(np.float32)

    Yr_all = _batch_ols_resid_chunked(Y_nb, Z_nb, ridge_lambda, chunk_size)

    if nl_flag:
        Xm = Xj_nb.mean(axis=1, keepdims=True)
        Xs = Xj_nb.std(axis=1, keepdims=True) + np.float32(1e-7)
        Xr_all = (Xj_nb - Xm) / Xs
    else:
        Xr_all = _batch_ols_resid_chunked(Xj_nb, Z_nb, ridge_lambda, chunk_size)

    return Yr_all, Xr_all


def _mi_ross(xc: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """Mutual information between a discrete x and a continuous y (Ross 2014,
    PLoS ONE 9:e87357). Invariant to the labelling of x's categories."""
    xc = np.asarray(xc)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < k + 2 or np.unique(xc).size < 2:
        return 0.0
    D = np.abs(y[:, None] - y[None, :])
    same = xc[:, None] == xc[None, :]
    nx = same.sum(1)
    kk = np.minimum(k, nx - 1)
    ok = kk >= 1
    if not ok.any():
        return 0.0
    Ds = np.where(same, D, np.inf)
    np.fill_diagonal(Ds, np.inf)
    Ds.sort(1)
    d = Ds[np.arange(n), np.maximum(kk, 1) - 1]
    m = (D <= d[:, None]).sum(1) - 1
    val = (digamma(n) - np.mean(digamma(nx[ok])) + np.mean(digamma(kk[ok]))
           - np.mean(digamma(np.maximum(m[ok], 1))))
    return max(0.0, float(val))


def _compute_one_surface(args):
    (
        var_name,
        col_idx,
        nl_flag,
        Y,
        ALL_X,
        nn_idx,
        k_sel,
        bw_sel,
        ridge_lambda,
        chunk_size,
        Z_other,
        is_cat,
    ) = args

    Yr_all, Xr_all = _precompute_residuals(
        Y, ALL_X, col_idx, nl_flag, nn_idx, ridge_lambda, chunk_size, Z_other=Z_other
    )
    n = len(Y)
    if is_cat:   # categorical focal variable: label-free discrete–continuous MI
        codes = ALL_X[:, col_idx][nn_idx]
        surf = np.array([_mi_ross(codes[i], Yr_all[i]) for i in range(n)])
    else:
        surf = np.array([_mi_ksg_sorted(Xr_all[i], Yr_all[i], k=k_sel) for i in range(n)])
    return var_name, surf


# ── LOWESS helpers ────────────────────────────────────────────────────


def _lowess_curve(x, y, frac=0.25, n_pts=120):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs, ys = x[np.argsort(x)], y[np.argsort(x)]
    x_eval = np.linspace(xs.min(), xs.max(), n_pts)
    y_eval = np.zeros(n_pts)
    n = len(xs)
    h = min(n, max(int(n * frac), 5))

    for i, xe in enumerate(x_eval):
        idx = np.argpartition(np.abs(xs - xe), h - 1)[:h]
        d_local = np.abs(xs[idx] - xe)
        wgt = (1 - (d_local / (d_local.max() + 1e-8)) ** 3) ** 3
        try:
            sw = np.sqrt(wgt)
            A = np.column_stack([sw, xs[idx] * sw])
            b = ys[idx] * sw
            beta = np.linalg.lstsq(A, b, rcond=None)[0]
            y_eval[i] = beta[0] + beta[1] * xe
        except Exception:
            y_eval[i] = np.average(ys[idx], weights=wgt)

    return x_eval, y_eval


# ── PDP sample helper ─────────────────────────────────────────────────


def _build_pdp_sample_points(ALL_X, xlist, n_bins=100, pts_per_bin=200, seed=42):
    rng = np.random.default_rng(seed)

    per_var_sets = []
    bin_edges = {}

    for j, vn in enumerate(xlist):
        x = ALL_X[:, j]
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        edges = np.linspace(lo, hi, n_bins + 1)
        bin_edges[vn] = edges

        chosen = []
        for b in range(n_bins):
            cand = np.where((x >= edges[b]) & (x < edges[b + 1]))[0]
            if len(cand) == 0:
                continue
            chosen.append(
                rng.choice(cand, size=min(pts_per_bin, len(cand)), replace=False)
            )

        per_var_sets.append(set(np.concatenate(chosen).tolist()) if chosen else set())

    shared = set.union(*per_var_sets) if per_var_sets else set()
    sample_idx = np.array(sorted(shared), dtype=np.int32)
    return sample_idx, bin_edges


# ── Stage 2 (default): penalised-spline varying coefficients ─────────
#
# Model:  y = b_0(s) + Σ_j b_j(s) · g_j(x_j)
#   b_j(s) = tensor-product cubic B-spline surface on (u, v) with a 2nd-order
#            difference penalty; λ_j per term chosen by GCV.
#   g_j    = initialised with stage-1 G_raw, then re-estimated jointly with b(s)
#            by backfitting (weighted local-linear smoothing of r_j / b_j(s)).
# Joint estimation avoids the bias of a plug-in G when the distribution of x_j
# varies over space (stage-1 G_raw estimates E[β_j(S) | x_j]·g_j(x_j)).


def _wlowess(x, t, w, frac=0.10, n_grid=200):
    """Weighted local-linear smoother on a full-range quantile grid, interpolated to x."""
    order = np.argsort(x)
    xs, ts, ws = x[order], t[order], w[order]
    grid = np.unique(np.quantile(xs, np.linspace(0, 1, n_grid)))
    h = max(int(len(xs) * frac), 20)
    out = np.zeros(len(grid))
    for gi, xe in enumerate(grid):
        c = np.searchsorted(xs, xe)
        lo, hi = max(0, c - h // 2), min(len(xs), c + h // 2)
        xx, tt, ww = xs[lo:hi], ts[lo:hi], ws[lo:hi]
        d = np.abs(xx - xe)
        k = ww * (1 - (d / (d.max() + 1e-12)) ** 3) ** 3
        A = np.c_[np.ones_like(xx), xx - xe]
        AtK = A.T * k
        try:
            out[gi] = np.linalg.solve(AtK @ A + 1e-10 * np.eye(2), AtK @ tt)[0]
        except np.linalg.LinAlgError:
            out[gi] = np.average(tt, weights=k + 1e-12)
    return np.interp(x, grid, out)


def _category_means(x, t, w=None):
    """Weighted mean of t within each category of x, mapped back to the rows."""
    w = np.ones_like(t) if w is None else w
    out = np.zeros_like(t, dtype=float)
    for lv in np.unique(x):
        m = x == lv
        out[m] = np.sum(w[m] * t[m]) / (np.sum(w[m]) + 1e-12)
    return out


def _bspline_basis(x, n_basis, degree=3, lo=None, hi=None):
    from scipy.interpolate import BSpline

    lo = float(x.min()) if lo is None else float(lo)
    hi = float(x.max()) if hi is None else float(hi)
    inner = np.linspace(lo, hi, n_basis - degree + 1)
    t = np.r_[[lo] * degree, inner, [hi] * degree]
    return BSpline.design_matrix(np.clip(x, lo, hi), t, degree).toarray()


def _tensor_basis(coords, n_basis=12, bounds=None):
    """bounds: ((u_lo, u_hi), (v_lo, v_hi)) knot range; default = range of coords."""
    (ulo, uhi), (vlo, vhi) = bounds if bounds is not None else ((None, None), (None, None))
    Bu = _bspline_basis(coords[:, 0], n_basis, lo=ulo, hi=uhi)
    Bv = _bspline_basis(coords[:, 1], n_basis, lo=vlo, hi=vhi)
    B = np.einsum("ni,nj->nij", Bu, Bv).reshape(len(coords), -1)
    D = np.diff(np.eye(n_basis), n=2, axis=0)
    Pm = D.T @ D
    I = np.eye(n_basis)
    return B, np.kron(Pm, I) + np.kron(I, Pm)


def _fit_pspline_vc(G, y, B, P, lam_grid, n_sweeps=3):
    n, p = G.shape
    m = B.shape[1]
    X = np.c_[np.ones(n), G]
    q = p + 1
    Z = np.concatenate([B * X[:, [j]] for j in range(q)], axis=1)
    ZtZ, Zty = Z.T @ Z, Z.T @ y
    yty = float(y @ y)

    def solve(lam):
        # One Cholesky factor per lambda; triangular solves via cho_solve (the general LU solve on
        # the factor and the extra full solve for edf dominated the run time). RSS from y'y avoids Z @ theta.
        S = ZtZ.copy()
        for j in range(q):
            S[j * m:(j + 1) * m, j * m:(j + 1) * m] += lam[j] * P
        S[np.diag_indices_from(S)] += 1e-8
        cf = cho_factor(S, lower=True, check_finite=False)
        theta = cho_solve(cf, Zty, check_finite=False)
        edf = float(np.trace(cho_solve(cf, ZtZ, check_finite=False)))
        rss = max(yty - 2.0 * float(theta @ Zty) + float(theta @ ZtZ @ theta), 0.0)
        return theta, n * rss / (n - edf) ** 2, edf

    lam = np.ones(q)
    for _ in range(n_sweeps):
        for j in range(q):
            scores = []
            for lv in lam_grid:
                trial = lam.copy()
                trial[j] = lv
                scores.append((solve(trial)[1], lv))
            lam[j] = min(scores)[1]
    theta, gcv, edf = solve(lam)
    theta = theta.reshape(q, m)
    beta = B @ theta.T  # (n, q)
    return dict(beta=beta, theta=theta, yhat=(X * beta).sum(1), lam=lam, gcv=gcv, edf=edf)


# ── Main class ────────────────────────────────────────────────────────


class CMI:
    """
    Conditional Mutual Information spatial regression model.

    Parameters
    ----------
    df           : pd.DataFrame
    u, v         : str           spatial coordinate column names
    y            : str           response variable column name
    xlist        : list[str]     predictor column names
    k_candidates : list[int]     KSG k values to search (default [4,6,8,10,12])
    bw_candidates: list[int]|None  BW values to search; None = auto
    sigma_smooth : float|None    Gaussian smoothing of CMI surface
    ridge_lambda : float         L2 penalty on slopes (intercept excluded)
    n_jobs       : int           parallel workers (-1 = all cores)
    seed         : int           random seed
    """

    def _log(self, *args, **kwargs):
        if getattr(self, "verbose", True):
            print(*args, **kwargs)

    @_quiet_matmul
    def __init__(
        self,
        df: pd.DataFrame,
        u: str,
        v: str,
        y: str,
        xlist: list,
        k_candidates: list = None,
        bw_candidates: list = None,
        sigma_smooth=None,
        ridge_lambda: float = 1e-4,
        n_jobs: int = -1,
        seed: int = 42,
        wls: bool = False,
        chunk_size: int = 2000,
        stage2: str = "pspline",
        verbose: bool = True,
        categorical: list = None,
    ):
        """
        stage2 : "pspline" (default) — penalised-spline varying-coefficient stage 2
                 with joint backfitting of g_j (outputs *_vc, *_esd, *_lin).
                 "none" — skip it.
        wls    : also run the legacy kernel local-WLS stage 2 of the original
                 IJGIS implementation (outputs *_wls). Default False.
        verbose: print progress.
        categorical : names in ``xlist`` that are nominal (unordered categories).
                 They are one-hot encoded whenever they act as conditioning
                 covariates; their stage-1 local information uses a label-free
                 discrete–continuous MI estimator (Ross 2014); in stage 2 their
                 transform g_j is a learned category-effect profile.
        """
        if stage2 not in ("pspline", "none"):
            raise ValueError("stage2 must be 'pspline' or 'none'")
        self._stage2 = stage2
        self.verbose = bool(verbose)
        self.df_raw = df.copy()
        self.u_col = u
        self.v_col = v
        self.y_col = y
        self.xlist = list(xlist)
        self.sigma_smooth = sigma_smooth
        self._ridge_lambda = float(ridge_lambda)
        self.seed = seed
        self._wls = bool(wls)
        self._chunk_size = max(100, int(chunk_size))
        self.result: Optional[CMIResult] = None
        self.pdp_table: Optional[pd.DataFrame] = None
        self._pdp_plot_cache: dict = {}  # keyed by (vn, col_suffix, frac, n_boot, seed)

        self.n = len(df)
        self.p = len(xlist)

        raw_coords = df[[u, v]].values.astype(float)
        cmin = raw_coords.min(axis=0)
        crange = np.where(
            raw_coords.max(axis=0) - cmin > 1e-10, raw_coords.max(axis=0) - cmin, 1.0
        )
        self._coords = (raw_coords - cmin) / crange
        self._cmin, self._crange = cmin, crange

        self.Y = df[y].values.astype(float)
        self.ALL_X = df[xlist].values.astype(float)

        # Detect discrete variables: unique count < 20
        self._is_discrete = {
            vn: (len(np.unique(self.ALL_X[:, j])) < 20) for j, vn in enumerate(xlist)
        }

        self.k_candidates = (
            k_candidates if k_candidates is not None else [4, 6, 8, 10, 12]
        )

        if bw_candidates is not None:
            self.bw_candidates = bw_candidates
        else:
            bw_hi = min(int(30 * self.p), self.n - 1)
            bw_lo = max(15, min(int(self.p * 10), bw_hi))
            step = 30
            if bw_lo >= bw_hi:
                self.bw_candidates = [max(15, bw_lo)]
            else:
                vals = list(range(bw_lo, bw_hi + 1, step))
                if not vals or vals[-1] < bw_hi:
                    vals.append(bw_hi)
                self.bw_candidates = sorted({max(15, int(v)) for v in vals})

        import os

        self._n_jobs = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        self._n_jobs = min(self._n_jobs, self.p)

        max_bw = min(max(self.bw_candidates) + 20, self.n - 1)
        self._log(
            f"  [CMI.__init__] Building nn_idx at max_bw={max_bw} ...",
            end=" ",
            flush=True,
        )
        t0 = time.perf_counter()
        nn = NearestNeighbors(n_neighbors=max_bw + 1, algorithm="kd_tree")
        nn.fit(self._coords)
        _, self._nn_idx = nn.kneighbors(self._coords)
        self._nn_idx = self._nn_idx.astype(np.int32)
        del nn
        self._log(f"{time.perf_counter()-t0:.2f}s  shape={self._nn_idx.shape}")

        self._log(f"  [CMI.__init__] Building PDP sample points ...", end=" ", flush=True)
        self._pdp_sample_idx, self._pdp_bin_edges = _build_pdp_sample_points(
            self.ALL_X, self.xlist, n_bins=60, pts_per_bin=80, seed=self.seed
        )
        self._log(f"{len(self._pdp_sample_idx)} points selected")

        self.categorical = [c for c in (categorical or []) if c in self.xlist]
        unknown = set(categorical or []) - set(self.xlist)
        if unknown:
            raise ValueError(f"categorical variables not in xlist: {sorted(unknown)}")
        self._is_cat = {vn: vn in self.categorical for vn in self.xlist}
        for vn in self.categorical:      # nominal variables are shown as levels in PDPs
            self._is_discrete[vn] = True
        self._Z_others = {j: self._conditioning_matrix(j) for j in range(self.p)}

    def _conditioning_matrix(self, j: int) -> np.ndarray:
        """Other predictors used for conditioning on x_j: numeric as is,
        categorical as one-hot dummies (first level dropped)."""
        cols = []
        for jj, vn in enumerate(self.xlist):
            if jj == j:
                continue
            x = self.ALL_X[:, jj]
            if self._is_cat.get(vn, False):
                levels = np.unique(x)
                cols.extend((x == lv).astype(float) for lv in levels[1:])
            else:
                cols.append(x)
        return np.column_stack(cols) if cols else np.zeros((self.n, 0))

    # ── k search ─────────────────────────────────────────────────────

    def _search_k(self, col_idx, nl_flag):
        Y_arr = self.Y
        X_arr = self.ALL_X[:, col_idx]
        Z_arr = self._Z_others[col_idx]
        if self._is_cat.get(self.xlist[col_idx], False):
            return 3, {3: np.nan}        # Ross estimator uses a fixed small k

        Yr = _ols_resid_scalar(Y_arr, Z_arr, self._ridge_lambda)
        if Yr is None:
            return self.k_candidates[0], {k: 0.0 for k in self.k_candidates}

        if nl_flag:
            sx = X_arr.std()
            if sx < 1e-10:
                return self.k_candidates[0], {k: 0.0 for k in self.k_candidates}
            Xr = (X_arr - X_arr.mean()) / sx
        else:
            Xr = _ols_resid_scalar(X_arr, Z_arr, self._ridge_lambda)
            if Xr is None:
                return self.k_candidates[0], {k: 0.0 for k in self.k_candidates}

        scores = {int(k): _mi_ksg_global(Xr, Yr, k=int(k)) for k in self.k_candidates}
        return int(max(scores, key=scores.get)), scores

    # ── BW search ────────────────────────────────────────────────────

    def _score_bw(self, col_idx, nl_flag, bw, k_mi, n_probe=120, seed=0):
        rng = np.random.default_rng(seed)
        probes = rng.choice(self.n, size=min(n_probe, self.n), replace=False)
        probe_nn = self._nn_idx[:, : bw + 1][probes]

        Z_other = self._Z_others[col_idx]
        Y_nb = self.Y[probe_nn].astype(np.float32)
        Xj_nb = self.ALL_X[probe_nn, col_idx].astype(np.float32)
        Z_nb = Z_other[probe_nn].astype(np.float32)

        Yr_p = _batch_ols_resid_chunked(
            Y_nb, Z_nb, self._ridge_lambda, self._chunk_size
        )
        if nl_flag:
            Xm = Xj_nb.mean(axis=1, keepdims=True)
            Xs = Xj_nb.std(axis=1, keepdims=True) + np.float32(1e-7)
            Xr_p = (Xj_nb - Xm) / Xs
        else:
            Xr_p = _batch_ols_resid_chunked(
                Xj_nb, Z_nb, self._ridge_lambda, self._chunk_size
            )

        if self._is_cat.get(self.xlist[col_idx], False):
            codes = self.ALL_X[probe_nn, col_idx]
            vals = np.array([_mi_ross(codes[i], Yr_p[i]) for i in range(len(probes))])
        else:
            vals = np.array(
                [_mi_ksg_sorted(Xr_p[i], Yr_p[i], k_mi) for i in range(len(probes))]
            )
        return float(vals.mean() - 0.15 * vals.std())

    def _search_bw(self, col_idx, nl_flag, k_mi, refinement=True, n_probe=120, seed=0):
        cand = sorted({max(15, int(x)) for x in self.bw_candidates})
        scores = {
            bw: self._score_bw(col_idx, nl_flag, bw, k_mi, n_probe, seed) for bw in cand
        }
        best_bw = max(scores, key=scores.get)

        if refinement and len(cand) > 1:
            lo, hi = max(15, best_bw - 60), best_bw + 60
            phi = 0.618
            for _ in range(4):
                c1 = max(15, int(round(hi - phi * (hi - lo))))
                c2 = max(15, int(round(lo + phi * (hi - lo))))
                for c, s in [(c1, seed + 1), (c2, seed + 2)]:
                    if c not in scores:
                        scores[c] = self._score_bw(
                            col_idx, nl_flag, c, k_mi, n_probe, s
                        )
                if scores[c1] >= scores[c2]:
                    hi = c2
                else:
                    lo = c1
            best_bw = max(scores, key=scores.get)

        return int(best_bw), scores

    # ── fit ───────────────────────────────────────────────────────────

    @_quiet_matmul
    def fit(
        self,
        k: int | dict | None = None,
        bw: int | dict | None = None,
        nl_flags: list | None = None,
        refinement: bool = True,
        n_probe: int = 120,
        sigma_smooth=None,
    ) -> "CMI":
        """
        Estimate CMI surfaces and calibrate the prediction model.

        Parameters
        ----------
        k          : None / int / dict  KSG neighbour count
        bw         : None / int / dict  local bandwidth
        nl_flags   : list[bool]         True = nonlinear mode per variable
        refinement : bool               golden-section refinement in BW search
        n_probe    : int                anchor points used in BW scoring
        sigma_smooth                    override instance sigma_smooth
        """
        t0_total = time.perf_counter()
        n, p = self.n, self.p
        xlist = self.xlist
        nl_flags = nl_flags if nl_flags is not None else [False] * p

        k_per_var, bw_per_var = {}, {}
        k_search_records, bw_search_records = [], []

        single_k = isinstance(k, int)
        single_bw = isinstance(bw, int)

        self._log(f"\n{'='*60}")
        self._log(f"CMI.fit()  n={n}  p={p}  vars={xlist}")
        self._log(f"{'='*60}")

        for j, vn in enumerate(xlist):
            nl = nl_flags[j]

            if single_k:
                k_j, k_scores = int(k), None
            elif isinstance(k, dict):
                k_j, k_scores = int(k[vn]), None
            else:
                self._log(f"  k search [{vn}] ...", end=" ", flush=True)
                k_j, k_scores = self._search_k(j, nl)
                self._log(f"best k={k_j}")
                for kv, sc in k_scores.items():
                    k_search_records.append(
                        {"variable": vn, "k_candidate": kv, "score": sc}
                    )

            k_per_var[vn] = k_j

            if single_bw:
                bw_j, bw_scores = int(bw), None
            elif isinstance(bw, dict):
                bw_j, bw_scores = int(bw[vn]), None
            else:
                self._log(f"  BW search [{vn}] k={k_j} ...", end=" ", flush=True)
                bw_j, bw_scores = self._search_bw(
                    j,
                    nl,
                    k_j,
                    refinement=refinement,
                    n_probe=n_probe,
                    seed=self.seed + j,
                )
                self._log(f"best bw={bw_j}")
                for bwv, sc in bw_scores.items():
                    bw_search_records.append(
                        {"variable": vn, "bw_candidate": bwv, "score": sc}
                    )

            bw_per_var[vn] = bw_j

        k_table = (
            pd.DataFrame(k_search_records)
            .pivot(index="k_candidate", columns="variable", values="score")
            .sort_index()
            if k_search_records
            else None
        )
        bw_table = (
            pd.DataFrame(bw_search_records)
            .pivot(index="bw_candidate", columns="variable", values="score")
            .sort_index()
            if bw_search_records
            else None
        )

        self._log(f"\n  Selected k:  { {v: k_per_var[v]  for v in xlist} }")
        self._log(f"  Selected BW: { {v: bw_per_var[v] for v in xlist} }")

        self._log(f"\n  Computing CMI surfaces (n_jobs={self._n_jobs}) ...")
        job_args = []
        for j, vn in enumerate(xlist):
            nl = nl_flags[j]
            nn_idx = self._nn_idx[:, : bw_per_var[vn] + 1]
            job_args.append(
                (
                    vn,
                    j,
                    nl,
                    self.Y,
                    self.ALL_X,
                    nn_idx,
                    k_per_var[vn],
                    bw_per_var[vn],
                    float(self._ridge_lambda),
                    self._chunk_size,
                    self._Z_others[j],
                    self._is_cat.get(vn, False),
                )
            )

        cmi_surfaces = {}
        if self._n_jobs > 1 and p > 1:
            for vn, surf in Parallel(n_jobs=self._n_jobs, backend="loky")(
                delayed(_compute_one_surface)(a) for a in job_args
            ):
                cmi_surfaces[vn] = surf
                self._log(f"    [{vn}] done  mean={surf.mean():.4f}  max={surf.max():.4f}")
        else:
            for args in job_args:
                vn, surf = _compute_one_surface(args)
                cmi_surfaces[vn] = surf
                self._log(f"    [{vn}] done  mean={surf.mean():.4f}  max={surf.max():.4f}")

        Z_sets = self._Z_others
        y_resids = {}
        pdp_curves = {}
        for j, vn in enumerate(xlist):
            Yr = _ols_resid_scalar(self.Y, Z_sets[j], self._ridge_lambda)
            y_resids[vn] = Yr
            if Yr is not None and self._is_cat.get(vn, False):
                xg = np.unique(self.ALL_X[:, j])
                yg = np.array([Yr[self.ALL_X[:, j] == lv].mean() for lv in xg])
            elif Yr is not None:
                xg, yg = _lowess_curve(self.ALL_X[:, j], Yr)
            else:
                xg = np.linspace(self.ALL_X[:, j].min(), self.ALL_X[:, j].max(), 120)
                yg = np.zeros(120)
            pdp_curves[vn] = (xg, yg)

        G = np.zeros((n, p))
        for j, vn in enumerate(xlist):
            xg, yg = pdp_curves[vn]
            G[:, j] = np.interp(self.ALL_X[:, j], xg, yg)

        # G_raw: LOWESS of raw-scale (unstandardised) partial residuals on x_j.
        # Unlike G (standardised Yr), G_raw[:,j] is in Y units and captures
        # f_j(x_j) = the nonlinear transform shape at the correct scale.
        # Used by fit_local_beta() as features for distance-weighted local OLS.
        G_raw = np.zeros((n, p), dtype=np.float32)
        for j in range(p):
            Z_oth = self._Z_others[j]
            Zm = np.c_[np.ones(n), Z_oth]
            for c in range(1, Zm.shape[1]):
                s = Zm[:, c].std()
                if s > 1e-10:
                    Zm[:, c] = (Zm[:, c] - Zm[:, c].mean()) / s
            p_ = Zm.shape[1]
            Rp = np.eye(p_, dtype=float)
            Rp[0, 0] = 0.0
            lam = max(float(self._ridge_lambda), 0.0)
            b = np.linalg.solve(
                Zm.T @ Zm + lam * Rp + 1e-12 * np.eye(p_), Zm.T @ self.Y
            )
            yr_raw_j = (self.Y - Zm @ b).astype(
                np.float32
            )  # raw-scale partial residual
            xj = self.ALL_X[:, j]
            if self._is_cat.get(xlist[j], False):    # category means of the partial residual
                G_raw[:, j] = _category_means(xj, yr_raw_j.astype(float)).astype(np.float32)
            else:
                xg_r, yg_r = _lowess_curve(xj, yr_raw_j.astype(float), frac=0.25, n_pts=200)
                G_raw[:, j] = np.interp(xj, xg_r, yg_r).astype(np.float32)

        S = np.maximum(np.column_stack([cmi_surfaces[vn] for vn in xlist]), 0.0)
        S_means = S.mean(axis=0)
        S_means[S_means < 1e-12] = 1.0
        H = (S / S_means[np.newaxis, :]) * G

        max_bw_sel = max(bw_per_var.values())
        nn_pred = NearestNeighbors(
            n_neighbors=min(max_bw_sel + 1, n), algorithm="kd_tree"
        )
        nn_pred.fit(self._coords)
        _, nn_idx_pred = nn_pred.kneighbors(self._coords)

        y_local_mean = np.array(
            [self.Y[nn_idx_pred[i][nn_idx_pred[i] != i]].mean() for i in range(n)]
        )

        alpha = np_lstsq(H, self.Y - y_local_mean, rcond=None)[0]
        y_hat_cmi = y_local_mean + H @ alpha
        cmi_contrib = H * alpha[np.newaxis, :]

        ss_res = np.sum((self.Y - y_hat_cmi) ** 2)
        ss_tot = np.sum((self.Y - self.Y.mean()) ** 2)
        r2 = max(0.0, 1 - ss_res / (ss_tot + 1e-12))

        self._log(f"\n  alpha = {np.round(alpha, 4)}")
        self._log(f"  R2    = {r2:.4f}")
        self._log(f"  total time = {time.perf_counter()-t0_total:.1f}s")

        df_out = self.df_raw.copy()
        for j, vn in enumerate(xlist):
            df_out[f"{vn}_lws"] = cmi_surfaces[vn]  # local weight signal (CMI surface)
        df_out["y_pred"] = y_hat_cmi

        self.result = CMIResult(
            df_out=df_out,
            alpha=alpha,
            y_local_mean=y_local_mean,
            H=H,
            G=G,
            G_raw=G_raw,
            cmi_contrib=cmi_contrib,
            cmi_surfaces=cmi_surfaces,
            pdp_curves=pdp_curves,
            k_search_table=k_table,
            bw_search_table=bw_table,
            meta=dict(
                k_per_var=k_per_var,
                bw_per_var=bw_per_var,
                nl_flags=nl_flags,
                ridge_lambda=self._ridge_lambda,
                r2=r2,
                y_resids=y_resids,
            ),
        )

        self._build_pdp_table()
        self._pdp_plot_cache.clear()
        if self._wls:
            self._compute_local_beta(bw_per_var)
        if self._stage2 == "pspline":
            self.fit_vc()
        return self

    # ── Stage 2: penalised-spline varying coefficients (default) ─────

    @_quiet_matmul
    def fit_vc(
        self,
        n_basis: int = 12,
        n_outer: int = 4,
        frac: float = 0.10,
        lam_grid=None,
        verbose: bool = True,
    ) -> "CMI":
        """
        Estimate y = b_0(s) + Σ_j b_j(s)·g_j(x_j) jointly (called by fit()).

        b_j(s): tensor-product B-spline surfaces (n_basis² per term) with a
        2nd-order difference penalty, λ_j chosen by GCV. g_j starts from the
        stage-1 curve G_raw and is re-estimated n_outer times by weighted
        local-linear smoothing (span ``frac``) given b_j(s).

        Outputs in df_out
        -----------------
        {v}_vc       b_j(s): multiplier of the learned transform g_j
                     (g_j oriented by its dominant trend: increasing in x_j, or
                     high at the extremes for U-shapes — signs of b_j are
                     relative to this; see meta["orient_vc"])
        {v}_esd      b_j(s)·sd(g_j): Y change per 1 SD of g_j(x_j)  (scale-identified)
        {v}_lin      b_j(s)·slope(g_j ~ x_j): linear-equivalent dY/dx_j
                     (meaningful for monotone, near-linear effects)
        vc_intercept b_0(s);  y_pred_vc fitted values
        meta: r2_vc, gcv_vc, edf_vc, lam_vc, n_basis_vc
        """
        if self.result is None:
            raise RuntimeError("Call fit() first.")
        t0 = time.perf_counter()
        res = self.result
        y = self.Y
        G = res.G_raw.astype(float).copy()
        lam_grid = np.logspace(-3, 4, 15) if lam_grid is None else np.asarray(lam_grid)
        B, P = _tensor_basis(self._coords, n_basis)
        fit = None
        for it in range(n_outer + 1):
            fit = _fit_pspline_vc(G, y, B, P, lam_grid)
            if it == n_outer:
                break
            beta = fit["beta"]
            X = np.c_[np.ones(self.n), G]
            for j in range(self.p):
                r = y - (X * beta).sum(1) + beta[:, j + 1] * G[:, j]
                bj = beta[:, j + 1]
                ok = np.abs(bj) > 1e-3 * (np.abs(bj).max() + 1e-12)
                t = np.where(ok, r / np.where(ok, bj, 1.0), 0.0)
                wj = np.where(ok, bj**2, 0.0)
                if self._is_cat.get(self.xlist[j], False):
                    G[:, j] = _category_means(self.ALL_X[:, j], t, wj)
                else:
                    G[:, j] = _wlowess(self.ALL_X[:, j], t, wj, frac=frac)
                X[:, j + 1] = G[:, j]
        beta = fit["beta"].copy()
        r2_vc = 1 - np.sum((y - fit["yhat"]) ** 2) / np.sum((y - y.mean()) ** 2)

        # Orientation convention (b·g = (−b)(−g)): flip g_j and b_j together.
        # Use whichever describes g_j better — its linear trend or its U-shape:
        #   |corr(g, x)| ≥ |corr(g, (x−x̄)²)|  →  g increases with x on average
        #   otherwise                          →  g is high at the extremes (U)
        # Signs of b_j(s) are read relative to this orientation (always report
        # the g_j curve with the map); predictions are unchanged.
        orient = np.ones(self.p)
        for j in range(self.p):
            x = self.ALL_X[:, j]
            if self._is_cat.get(self.xlist[j], False):
                # nominal: no order, so orient the category profile so that the
                # average local multiplier is positive; b_j(s) < 0 then means the
                # local category ranking is reversed relative to the global one
                c = float(np.mean(beta[:, j + 1]))
            else:
                c_lin = np.corrcoef(G[:, j], x)[0, 1]
                c_u = np.corrcoef(G[:, j], (x - x.mean()) ** 2)[0, 1]
                c = c_lin if abs(c_lin) >= abs(c_u) else c_u
            orient[j] = np.sign(c) if np.isfinite(c) and c != 0 else 1.0
            G[:, j] *= orient[j]
            beta[:, j + 1] *= orient[j]
        theta = fit["theta"].copy()
        theta[1:] *= orient[:, None]
        self._vc_state = dict(
            theta=theta, n_basis=n_basis,
            bounds=tuple((float(self._coords[:, a].min()), float(self._coords[:, a].max()))
                         for a in range(2)),
            x_sorted=[np.sort(self.ALL_X[:, j]) for j in range(self.p)],
            g_sorted=[G[np.argsort(self.ALL_X[:, j]), j] for j in range(self.p)],
        )

        res.beta_vc = beta[:, 1:]
        res.intercept_vc = beta[:, 0]
        res.G_vc = G
        res.meta.update(dict(r2_vc=r2_vc, gcv_vc=fit["gcv"], edf_vc=fit["edf"],
                             lam_vc=fit["lam"], n_basis_vc=n_basis, orient_vc=orient))
        for j, vn in enumerate(self.xlist):
            g, x = G[:, j], self.ALL_X[:, j]
            slope = (np.nan if self._is_cat.get(vn, False)
                     else np.cov(g, x)[0, 1] / (x.var() + 1e-12))
            res.df_out[f"{vn}_vc"] = beta[:, j + 1]
            res.df_out[f"{vn}_esd"] = beta[:, j + 1] * g.std()
            res.df_out[f"{vn}_lin"] = beta[:, j + 1] * slope
        res.df_out["vc_intercept"] = beta[:, 0]
        res.df_out["y_pred_vc"] = fit["yhat"]
        if verbose:
            self._log(f"  [stage 2 P-spline VC] R2={r2_vc:.4f}  edf={fit['edf']:.1f}  "
                  f"lambda={np.round(fit['lam'], 3).tolist()}  {time.perf_counter()-t0:.1f}s")
        return self

    @_quiet_matmul
    def predict(self, df_new: pd.DataFrame, return_components: bool = False):
        """Predict y at new locations / predictor values with the stage-2 model.

        b_j(s) is evaluated from the fitted spline surfaces (locations outside the
        training bounding box are clamped to its edge) and g_j(x) by linear
        interpolation of the learned transforms (x outside the training range is
        clamped). With ``return_components=True`` a DataFrame with the predicted
        coefficient surfaces ({v}_vc, {v}_esd, vc_intercept) and y_pred is returned.
        """
        if getattr(self, "_vc_state", None) is None:
            raise RuntimeError("Call fit() with stage2='pspline' first.")
        st = self._vc_state
        c = (df_new[[self.u_col, self.v_col]].to_numpy(float) - self._cmin) / self._crange
        B, _ = _tensor_basis(c, st["n_basis"], bounds=st["bounds"])
        b = B @ st["theta"].T                                   # (n_new, p + 1)
        G = np.ones_like(b)
        for j, vn in enumerate(self.xlist):
            G[:, j + 1] = np.interp(df_new[vn].to_numpy(float), st["x_sorted"][j], st["g_sorted"][j])
        y_pred = (b * G).sum(1)
        if not return_components:
            return y_pred
        out = pd.DataFrame({"vc_intercept": b[:, 0]}, index=df_new.index)
        for j, vn in enumerate(self.xlist):
            out[f"{vn}_vc"] = b[:, j + 1]
            out[f"{vn}_esd"] = b[:, j + 1] * self.result.G_vc[:, j].std()
        out["y_pred"] = y_pred
        return out

    # ── PDP table ─────────────────────────────────────────────────────

    def _build_pdp_table(self):
        if self.result is None:
            raise RuntimeError("Call fit() before building PDP table.")

        res = self.result
        idx = self._pdp_sample_idx
        p = self.p

        # ── Three-part decomposition (all centred for PDP comparison) ──
        #
        # The fitted model for variable j at observation i is:
        #   T_ij = alpha_j * W_ij * G_ij          (cmi_contrib)
        # where W_ij = S_ij / mean(S_j) is the local CMI weight and
        # G_ij is the partial-residual baseline curve (LOWESS of Yr ~ x_j).
        #
        # Decompose into:
        #   base_raw  = alpha_j * G_ij             (baseline: no spatial modulation)
        #   resid_raw = alpha_j * (W_ij - 1) * G_ij  (local CMI adjustment beyond baseline)
        #   total_raw = alpha_j * W_ij * G_ij      = cmi_contrib  (full model contribution)
        #
        # Centre each column so all three are mean-zero (contribution scale).

        total_raw = res.cmi_contrib  # (n, p)
        base_raw = res.G * res.alpha[np.newaxis, :]  # alpha_j * G_ij
        resid_raw = total_raw - base_raw  # alpha_j * (W-1) * G_ij

        def _centre(M):
            return M - M.mean(axis=0, keepdims=True)

        base_contrib = _centre(base_raw)
        resid_contrib = _centre(resid_raw)
        total_contrib = _centre(total_raw)

        rows = {"obs_idx": idx}
        for j, vn in enumerate(self.xlist):
            rows[f"{vn}_x"] = self.ALL_X[idx, j]
            rows[f"{vn}_baseline_contrib"] = base_contrib[idx, j]
            rows[f"{vn}_yresid_contrib"] = resid_contrib[idx, j]
            rows[f"{vn}_total_contrib"] = total_contrib[idx, j]
        rows["y_obs"] = self.Y[idx]
        rows["y_pred"] = (res.y_local_mean + res.H @ res.alpha)[idx]

        self.pdp_table = pd.DataFrame(rows)

    # ── plot_pdp ──────────────────────────────────────────────────────

    def plot_pdp(
        self,
        frac: float = 0.30,
        n_boot: int = 40,
        ci_level: float = 0.80,
        ci_vars: list | None = None,
        fast_mode: bool = False,
        figsize=None,
        panel_side_inches: float = 3.0,
        dpi: int = 150,
        save: str = None,
        y_clip_quantiles: tuple | None = (0.5, 99.5),
        title_fontsize: float = 12.0,
        label_fontsize: float = 12.0,
        tick_labelsize: float = 12.0,
        legend_fontsize: float = 12.0,
        suptitle_fontsize: float = 12.0,
        show_suptitle: bool = False,
        separated_legend: bool = False,
        panel_aspect_hw: float = 0.8,
    ):
        """
        PDP plot: one row per variable, four columns per row.

        Col 1 — Baseline (alpha * G)
        Col 2 — Nonlinear residual (CMI modulation)
        Col 3 — Total contribution
        Col 4 — Distribution (full df vs PDP sample x marginal)

        Parameters
        ----------
        frac        : LOWESS bandwidth fraction.
        n_boot      : bootstrap resamples for CI band (continuous vars).
        ci_level    : Student-t CI level for discrete bar charts.
        ci_vars     : variables to draw bootstrap CI band for; None = all continuous.
                      Variables not in this list get a single LOWESS line (faster).
        fast_mode   : if True, override n_boot=10 and frac=0.25 for quick previews.
        y_clip_quantiles : (lo, hi) percentiles for y-axis clipping per panel.
                      Cols 1-3 within the same row share the same y range.
        figsize     : ``(width, height)`` in inches. If given, overrides ``panel_side_inches``.
        panel_side_inches : Panel width in inches (one column). Row height is
                      ``panel_side_inches * panel_aspect_hw`` (default height:width = 0.8:1).
                      Figure width ``4 * panel_side_inches``; figure height from row count,
                      optional bottom legend band, and padding.
        dpi         : Figure dots-per-inch for both on-screen display and ``save`` output.
        title_fontsize : Subplot titles (row 0) and left-column y-axis variable names.
        label_fontsize : Reserved (x-axis labels under panels are omitted by default).
        tick_labelsize : Tick labels and small marginal-axis labels (``density``, etc.).
        legend_fontsize : Legend text.
        suptitle_fontsize : Figure super-title (only if ``show_suptitle``).
        show_suptitle : If True, draw the figure super-title.
        separated_legend : If True, omit per-axis legends and add a bottom row aligned with
                      the four columns: (1) continuous PDP baseline legend, (2) discrete PDP
                      baseline legend, (3) continuous marginal KDE legend, (4) discrete marginal
                      frequency legend. Missing variable types leave that cell empty.
        panel_aspect_hw : Subplot box height/width (display), default ``0.8`` (high:wide = 0.8:1).
        """
        if self.result is None:
            raise RuntimeError("Call fit() first.")
        if self.pdp_table is None:
            self._build_pdp_table()

        if fast_mode:
            n_boot, frac = 10, 0.25

        xlist = self.xlist
        p = self.p
        tbl = self.pdp_table
        rng = np.random.default_rng(self.seed)
        ci_set = set(xlist) if ci_vars is None else set(ci_vars)

        col_specs = [
            ("Baseline", "baseline_contrib", "#2F6DB3"),
            ("Nonlinear", "yresid_contrib", "#D85A30"),
            ("Total", "total_contrib", "#7A1FA2"),
        ]

        # ── Pre-compute discrete group stats (cached) ─────────────────
        # key: (vn, col_suffix)  →  (levels, means, yerr, x_pos, label)
        disc_stats: dict = {}
        for vn in xlist:
            if not self._is_discrete[vn]:
                continue
            x_arr = tbl[f"{vn}_x"].values
            levels = np.unique(x_arr.astype(float))
            levels.sort()
            x_pos = np.arange(len(levels))
            pct = int(round(100 * ci_level))
            for _, col_suffix, _ in col_specs:
                cache_key = (vn, col_suffix, ci_level)
                if cache_key in self._pdp_plot_cache:
                    disc_stats[(vn, col_suffix)] = self._pdp_plot_cache[cache_key]
                    continue
                y_arr = tbl[f"{vn}_{col_suffix}"].values
                means, el, eh = [], [], []
                for lv in levels:
                    yg = y_arr[np.isclose(x_arr, lv, rtol=0, atol=1e-8)]
                    ng = len(yg)
                    m = float(np.mean(yg))
                    means.append(m)
                    if ng < 2:
                        el.append(0.0)
                        eh.append(0.0)
                    else:
                        se = float(np.std(yg, ddof=1) / np.sqrt(ng))
                        tc = t_dist.ppf((1 + ci_level) / 2, ng - 1)
                        el.append(tc * se)
                        eh.append(tc * se)
                entry = (levels, np.array(means), np.array([el, eh]), x_pos, pct)
                disc_stats[(vn, col_suffix)] = entry
                self._pdp_plot_cache[cache_key] = entry

        # ── Pre-compute LOWESS / bootstrap (cached) ───────────────────
        # key: (vn, col_suffix, frac, n_boot, seed)
        cont_curves: dict = {}
        for vn in xlist:
            if self._is_discrete[vn]:
                continue
            x_arr = tbl[f"{vn}_x"].values
            want_ci = vn in ci_set
            for _, col_suffix, _ in col_specs:
                cache_key = (vn, col_suffix, frac, n_boot if want_ci else 0, self.seed)
                if cache_key in self._pdp_plot_cache:
                    cont_curves[(vn, col_suffix)] = self._pdp_plot_cache[cache_key]
                    continue
                y_arr = tbl[f"{vn}_{col_suffix}"].values
                boot_seed = abs(hash((vn, col_suffix, self.seed))) % (2**31)
                if want_ci:
                    result = self._boot_lowess(
                        x_arr,
                        y_arr,
                        frac=frac,
                        n_pts=120,
                        n_boot=n_boot,
                        seed=boot_seed,
                    )
                else:
                    xev, ymed = _lowess_curve(x_arr, y_arr, frac=frac, n_pts=120)
                    result = (xev, ymed, ymed, ymed)  # no band
                cont_curves[(vn, col_suffix)] = result
                self._pdp_plot_cache[cache_key] = result

        s = float(panel_side_inches)
        ah = float(panel_aspect_hw)
        legend_band = (0.5 * s) if separated_legend else 0.0
        top_pad = 0.55 if show_suptitle else 0.25
        if figsize is not None:
            use_figsize = figsize
        else:
            row_h = ah * s
            use_figsize = (
                4.0 * s,
                float(p) * row_h + legend_band + top_pad,
            )

        # ── Draw ──────────────────────────────────────────────────────
        fig = plt.figure(figsize=use_figsize)
        if separated_legend:
            # Legend row height / one PDP row height = (0.5*s) / (ah*s) => ratio 0.5/ah .
            gs = mgridspec.GridSpec(
                p + 1,
                4,
                figure=fig,
                height_ratios=[1.0] * p + [0.5 / ah],
                hspace=0.38,
                wspace=0.28,
            )
            axes = np.empty((p, 4), dtype=object)
            for ri in range(p):
                for ci in range(4):
                    axes[ri, ci] = fig.add_subplot(gs[ri, ci])
            axes_leg = [fig.add_subplot(gs[p, k]) for k in range(4)]
            for axl in axes_leg:
                axl.axis("off")
        else:
            gs = mgridspec.GridSpec(p, 4, figure=fig, hspace=0.38, wspace=0.28)
            axes = np.empty((p, 4), dtype=object)
            for ri in range(p):
                for ci in range(4):
                    axes[ri, ci] = fig.add_subplot(gs[ri, ci])
            axes_leg = None

        for ri, vn in enumerate(xlist):
            discrete = self._is_discrete[vn]
            x_arr = tbl[f"{vn}_x"].values
            want_ci = vn in ci_set

            # Collect y data for all 3 PDP cols to compute shared y range
            all_y_pool = []

            panel_data = []  # (ax, col_label, col_suffix, color, ymed, ylo, yhi)
            for ci, (col_label, col_suffix, color) in enumerate(col_specs):
                ax = axes[ri, ci]
                y_arr = tbl[f"{vn}_{col_suffix}"].values

                if discrete:
                    levels, means, yerr, x_pos, pct = disc_stats[(vn, col_suffix)]
                    ymed = means
                    ylo = means - yerr[0]
                    yhi = means + yerr[1]
                else:
                    xev, ymed, ylo, yhi = cont_curves[(vn, col_suffix)]

                all_y_pool.append(np.asarray(y_arr, float))
                all_y_pool.append(np.asarray(ymed, float))
                all_y_pool.append(np.asarray(ylo, float))
                all_y_pool.append(np.asarray(yhi, float))
                panel_data.append(
                    (
                        ax,
                        col_label,
                        col_suffix,
                        color,
                        y_arr,
                        ymed,
                        ylo,
                        yhi,
                        levels if discrete else None,
                        means if discrete else None,
                        yerr if discrete else None,
                        x_pos if discrete else None,
                        pct if discrete else None,
                        xev if not discrete else None,
                    )
                )

            # Shared y range for cols 0-2
            shared_ylim = None
            if y_clip_quantiles is not None:
                pool = np.concatenate([a.ravel() for a in all_y_pool])
                pool = pool[np.isfinite(pool)]
                if pool.size > 0:
                    ya, yb = np.nanpercentile(pool, list(y_clip_quantiles))
                    if yb > ya:
                        pad = (yb - ya) * 0.08 + 1e-12
                        shared_ylim = (ya - pad, yb + pad)

            for ci, (
                ax,
                col_label,
                col_suffix,
                color,
                y_arr,
                ymed,
                ylo,
                yhi,
                levels,
                means,
                yerr,
                x_pos,
                pct,
                xev,
            ) in enumerate(panel_data):

                if discrete:
                    ax.bar(
                        x_pos,
                        means,
                        color=color,
                        alpha=0.85,
                        edgecolor="white",
                        linewidth=0.6,
                        yerr=yerr,
                        capsize=3,
                        error_kw=dict(ecolor="#333", linewidth=1.0, capthick=1.0),
                        label=f"mean ± t-CI {pct}%",
                    )
                    sc_idx = rng.choice(
                        len(x_arr), size=min(300, len(x_arr)), replace=False
                    )
                    lv_arr = levels.astype(float)
                    x_plot = np.array(
                        [
                            int(np.argmin(np.abs(lv_arr - float(x_arr[k]))))
                            + rng.uniform(-0.28, 0.28)
                            for k in sc_idx
                        ]
                    )
                    ax.scatter(
                        x_plot,
                        y_arr[sc_idx],
                        s=5,
                        alpha=0.15,
                        color="#888",
                        rasterized=True,
                        zorder=0,
                        label=("sample" if separated_legend else None),
                    )
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(
                        [
                            (
                                str(int(v))
                                if abs(v - round(float(v))) < 1e-7
                                else f"{float(v):.4g}"
                            )
                            for v in levels
                        ],
                        fontsize=tick_labelsize,
                    )
                else:
                    if want_ci:
                        ax.fill_between(
                            xev, ylo, yhi, alpha=0.20, color=color, label="80% CI"
                        )
                    ax.plot(xev, ymed, color=color, lw=2.2, label="LOWESS")
                    sc_idx = rng.choice(
                        len(x_arr), size=min(300, len(x_arr)), replace=False
                    )
                    ax.scatter(
                        x_arr[sc_idx],
                        y_arr[sc_idx],
                        s=4,
                        alpha=0.12,
                        color="#888",
                        rasterized=True,
                        label=("sample" if separated_legend else None),
                    )

                ax.axhline(0, color="#aaa", lw=0.7, ls="--")
                if shared_ylim is not None:
                    ax.set_ylim(*shared_ylim)

                if ci == 0:
                    ax.set_ylabel(vn, fontsize=title_fontsize, fontweight="bold")
                if ri == 0:
                    ax.set_title(col_label, fontsize=title_fontsize, fontweight="bold")
                if not separated_legend:
                    ax.legend(fontsize=legend_fontsize)
                ax.tick_params(labelsize=tick_labelsize)

            # ── Col 3: x marginal ─────────────────────────────────────
            ax = axes[ri, 3]
            j = self.xlist.index(vn)
            xf_ok = self.ALL_X[:, j][np.isfinite(self.ALL_X[:, j])]
            xs_ok = x_arr[np.isfinite(x_arr)]

            if discrete:
                lev = np.unique(np.concatenate([np.unique(xf_ok), np.unique(xs_ok)]))
                lev.sort()
                c_f = np.array(
                    [np.sum(np.isclose(xf_ok, v, rtol=0, atol=1e-8)) for v in lev],
                    float,
                )
                c_s = np.array(
                    [np.sum(np.isclose(xs_ok, v, rtol=0, atol=1e-8)) for v in lev],
                    float,
                )
                c_f /= c_f.sum() or 1.0
                c_s /= c_s.sum() or 1.0
                xp = np.arange(len(lev))
                w = 0.38
                ax.bar(
                    xp - w / 2,
                    c_f,
                    width=w,
                    alpha=0.85,
                    color="#4C72B0",
                    label="full df",
                )
                ax.bar(
                    xp + w / 2,
                    c_s,
                    width=w,
                    alpha=0.85,
                    color="#DD8452",
                    label="PDP sample",
                )
                ax.set_xticks(xp)
                ax.set_xticklabels(
                    [
                        (
                            str(int(v))
                            if abs(v - round(float(v))) < 1e-7
                            else f"{float(v):.4g}"
                        )
                        for v in lev
                    ],
                    fontsize=tick_labelsize,
                )
                ax.set_ylabel("norm. freq.", fontsize=tick_labelsize)
            else:
                lo = float(min(xf_ok.min(), xs_ok.min()))
                hi = float(max(xf_ok.max(), xs_ok.max()))
                if hi <= lo:
                    hi = lo + 1.0
                bins = int(min(80, max(25, len(xf_ok) ** 0.45)))
                ax.hist(
                    xf_ok,
                    bins=bins,
                    range=(lo, hi),
                    density=True,
                    alpha=0.40,
                    color="#4C72B0",
                    label="full hist",
                )
                ax.hist(
                    xs_ok,
                    bins=bins,
                    range=(lo, hi),
                    density=True,
                    alpha=0.40,
                    color="#DD8452",
                    label="PDP hist",
                )
                grid = np.linspace(lo, hi, 300)
                if xf_ok.size > 1 and np.ptp(xf_ok) > 1e-15:
                    ax.plot(
                        grid,
                        gaussian_kde(xf_ok)(grid),
                        color="#2a4d8f",
                        lw=1.7,
                        label="KDE full",
                    )
                if xs_ok.size > 1 and np.ptp(xs_ok) > 1e-15:
                    ax.plot(
                        grid,
                        gaussian_kde(xs_ok)(grid),
                        color="#b45f1a",
                        lw=1.7,
                        ls="--",
                        label="KDE PDP",
                    )
                ax.set_ylabel("density", fontsize=tick_labelsize)
                ax.set_xlim(lo, hi)

            if ri == 0:
                ax.set_title(
                    "Distribution",
                    fontsize=title_fontsize,
                    fontweight="bold",
                )
            if not separated_legend:
                ax.legend(fontsize=legend_fontsize, loc="best")
            ax.tick_params(labelsize=tick_labelsize)

            for _ci in range(4):
                axes[ri, _ci].set_box_aspect(ah)

        if separated_legend:
            idx_disc = next(
                (i for i, vni in enumerate(xlist) if self._is_discrete[vni]),
                None,
            )
            idx_cont = next(
                (i for i, vni in enumerate(xlist) if not self._is_discrete[vni]),
                None,
            )

            def _legend_center(ax_cell, handles, labels):
                if handles:
                    ax_cell.legend(
                        handles,
                        labels,
                        loc="center",
                        fontsize=legend_fontsize,
                        frameon=False,
                    )

            # Col 0 — continuous PDP (baseline column): LOWESS / CI / sample
            if idx_cont is not None:
                h, lb = axes[idx_cont, 0].get_legend_handles_labels()
                _legend_center(axes_leg[0], list(h), list(lb))

            # Col 1 — discrete PDP (baseline column): bar + sample
            if idx_disc is not None:
                h, lb = axes[idx_disc, 0].get_legend_handles_labels()
                _legend_center(axes_leg[1], list(h), list(lb))

            # Col 2 — continuous marginal: KDE full & KDE PDP only
            if idx_cont is not None:
                h, lb = axes[idx_cont, 3].get_legend_handles_labels()
                pairs = [
                    (hi, li)
                    for hi, li in zip(h, lb)
                    if li in ("KDE full", "KDE PDP")
                ]
                if pairs:
                    hs, ls = zip(*pairs)
                    _legend_center(axes_leg[2], list(hs), list(ls))

            # Col 3 — discrete marginal: normalized frequency bars (full df vs PDP sample)
            if idx_disc is not None:
                h, lb = axes[idx_disc, 3].get_legend_handles_labels()
                pairs = [
                    (hi, li)
                    for hi, li in zip(h, lb)
                    if li in ("full df", "PDP sample")
                ]
                if pairs:
                    hs, ls = zip(*pairs)
                    _legend_center(axes_leg[3], list(hs), list(ls))

        if show_suptitle:
            fig.suptitle(
                "CMI — Partial Dependence Plots",
                fontsize=suptitle_fontsize,
                fontweight="bold",
            )
        with warnings.catch_warnings():  # box-aspect axes are not tight_layout-compatible
            warnings.simplefilter("ignore", UserWarning)
            if show_suptitle:
                plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
            else:
                plt.tight_layout()
        fig.set_dpi(float(dpi))
        if save:
            fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _boot_lowess(x, y, frac=0.25, n_pts=120, n_boot=40, seed=0):
        rng = np.random.default_rng(seed)
        curves = []
        for _ in range(n_boot):
            idx = rng.choice(len(x), size=len(x), replace=True)
            try:
                _, yc = _lowess_curve(x[idx], y[idx], frac, n_pts)
                curves.append(yc)
            except Exception:
                pass
        xev = np.linspace(x.min(), x.max(), n_pts)
        if not curves:
            _, ym = _lowess_curve(x, y, frac, n_pts)
            return xev, ym, ym, ym
        curves = np.array(curves)
        return (
            xev,
            np.median(curves, 0),
            np.percentile(curves, 10, 0),
            np.percentile(curves, 90, 0),
        )

    # ── _compute_local_beta (called automatically by fit()) ──────────────

    def _compute_local_beta(
        self, bw_per_var: dict, bw_dist: float = None, n_neighbors: int = None
    ):
        """Internal: compute WLS local beta and populate _wls columns in df_out."""
        res = self.result
        n, p = self.n, self.p
        G_raw = res.G_raw
        lws_mat = np.column_stack([res.cmi_surfaces[vn] for vn in self.xlist])

        k_wls = min(n_neighbors or max(bw_per_var.values()), n - 2)
        median_bw = float(np.median(list(bw_per_var.values())))
        bw_dist_use = (
            float(bw_dist)
            if bw_dist is not None
            else float(np.sqrt(median_bw / (np.pi * n)))
        )

        nn = NearestNeighbors(n_neighbors=k_wls + 1, algorithm="kd_tree")
        nn.fit(self._coords)
        dists_all, idx_all = nn.kneighbors(self._coords)

        self._log(
            f"  [local beta] bw_dist={bw_dist_use:.4f}{'(auto)' if bw_dist is None else '(manual)'}  k={k_wls}"
        )

        beta_wls = np.zeros((n, p), dtype=np.float32)
        # Local fitted level c_i and weighted local mean of G_raw. The model is
        # stored in centred form  y ≈ c_i + Σ_j β_ij (G_ij − Ḡ_ij), so c_i is
        # unaffected by the slope scale correction and smoothing below.
        c_wls = res.y_local_mean.astype(np.float64).copy()
        g_bar = np.zeros((n, p), dtype=np.float64)
        for i in range(n):
            idx = idx_all[i, 1 : k_wls + 1]
            d = dists_all[i, 1 : k_wls + 1]
            w_d = np.exp(-(d**2) / (2.0 * bw_dist_use**2)).astype(np.float32)
            w_c = lws_mat[idx].mean(axis=1).astype(np.float32) + np.float32(1e-6)
            w = w_d * w_c
            ws = w.sum()
            if ws < 1e-15:
                continue
            w /= ws
            Xi = np.c_[np.ones(k_wls, dtype=np.float32), G_raw[idx]]
            Yi = self.Y[idx].astype(np.float32)
            XtW = Xi.T @ np.diag(w)
            try:
                b = np.linalg.solve(
                    (XtW @ Xi).astype(np.float64) + 1e-8 * np.eye(p + 1),
                    (XtW @ Yi).astype(np.float64),
                )
                beta_wls[i] = b[1:].astype(np.float32)
                g_bar[i] = w @ G_raw[idx]
                c_wls[i] = b[0] + g_bar[i] @ b[1:]
            except Exception:
                pass

        # Per-variable global scale correction: scale_j = G_raw_j' Yr_raw_j / G_raw_j' G_raw_j
        scale_wls = np.ones(p, dtype=float)
        for j in range(p):
            g = G_raw[:, j].astype(float)
            Z_oth = self._Z_others[j]
            Zm = np.c_[np.ones(n), Z_oth]
            for c in range(1, Zm.shape[1]):
                s = Zm[:, c].std()
                if s > 1e-10:
                    Zm[:, c] = (Zm[:, c] - Zm[:, c].mean()) / s
            p_ = Zm.shape[1]
            Rp = np.eye(p_)
            Rp[0, 0] = 0.0
            lam = max(float(self._ridge_lambda), 0.0)
            b_z = np.linalg.solve(
                Zm.T @ Zm + lam * Rp + 1e-12 * np.eye(p_), Zm.T @ self.Y
            )
            yr_raw_j = self.Y - Zm @ b_z
            denom = float(np.dot(g, g))
            if denom > 1e-10:
                scale_wls[j] = float(np.dot(g, yr_raw_j)) / denom

        beta_wls_scaled = beta_wls * scale_wls[np.newaxis, :].astype(np.float32)

        # CMI-weighted spatial smoothing:
        # Weight neighbours by their CMI surface value for each variable j.
        # Points with high CMI (strong local signal) dominate the average,
        # so noisy low-CMI patches are pulled toward reliable high-CMI neighbours.
        k_smooth = min(max(5, int(median_bw / 8)), n - 1)
        nn_sm = NearestNeighbors(n_neighbors=k_smooth + 1, algorithm="kd_tree")
        nn_sm.fit(self._coords)
        d_sm, idx_sm = nn_sm.kneighbors(self._coords)

        beta_wls_smooth = np.zeros_like(beta_wls_scaled)
        for j in range(p):
            lws_j = lws_mat[:, j]  # CMI surface for variable j
            for i in range(n):
                nb = idx_sm[i, 1:]  # k_smooth neighbours
                w = lws_j[nb] + 1e-6  # CMI weight of each neighbour
                w /= w.sum()
                beta_wls_smooth[i, j] = np.dot(w, beta_wls_scaled[nb, j])
        beta_wls_scaled = beta_wls_smooth

        # Prediction keeps the local WLS intercept (in centred form). Previously
        # the intercept was dropped and replaced by y_local_mean, which offsets
        # the prediction by Σ β_ij Ḡ_ij whenever G_raw has a non-zero local mean.
        b_final = beta_wls_scaled.astype(float)
        y_hat_wls = c_wls + (b_final * (G_raw.astype(float) - g_bar)).sum(axis=1)
        wls_intercept = c_wls - (b_final * g_bar).sum(axis=1)
        ss_res = float(np.sum((self.Y - y_hat_wls) ** 2))
        ss_tot = float(np.sum((self.Y - self.Y.mean()) ** 2))
        r2_wls = max(0.0, 1.0 - ss_res / (ss_tot + 1e-12))

        self._log(f"  [local beta] scale={np.round(scale_wls,4)}  R2_wls={r2_wls:.4f}")

        res.beta_wls = beta_wls_scaled
        res.meta.update(
            dict(bw_dist_wls=bw_dist_use, scale_wls=scale_wls, r2_wls=r2_wls)
        )
        for j, vn in enumerate(self.xlist):
            res.df_out[f"{vn}_wls"] = beta_wls_scaled[:, j].astype(float)
        res.df_out["wls_intercept"] = wls_intercept
        res.df_out["y_pred_wls"] = y_hat_wls

    # ── fit_local_beta (public wrapper — re-run with custom bw_dist) ──

    @_quiet_matmul
    def fit_local_beta(self, bw_dist: float = None, n_neighbors: int = None) -> "CMI":
        """
        Re-run local WLS beta estimation with a custom bw_dist or n_neighbors.
        Under normal usage this is NOT needed — fit() calls this automatically.

        Use only when you want to experiment with a different spatial bandwidth,
        e.g. model.fit_local_beta(bw_dist=0.08).

        Parameters
        ----------
        bw_dist     : Gaussian kernel bandwidth in normalised [0,1]² space.
                      None = auto from bw_per_var (same as fit()).
        n_neighbors : WLS window size. None = use max(bw_per_var).
        """
        if self.result is None:
            raise RuntimeError("Call fit() first.")
        self._compute_local_beta(
            self.result.meta["bw_per_var"], bw_dist=bw_dist, n_neighbors=n_neighbors
        )
        return self

    # ── contribution_summary ─────────────────────────────────────────

    def contribution_summary(self, print_table: bool = True) -> pd.DataFrame:
        """
        Per-variable summary table of contribution magnitudes (full sample).

        Columns
        -------
        mean_abs_baseline  : mean |alpha_j * G_ij|            global linear effect size
        mean_abs_residual  : mean |alpha_j * (W_ij-1) * G_ij| CMI nonlinear adjustment
        mean_abs_total     : mean |alpha_j * W_ij * G_ij|     total model contribution
        residual_share     : mean_abs_residual / (mean_abs_total + 1e-12)
                             share of total effect from CMI local modulation
        """
        if self.result is None:
            raise RuntimeError("Call fit() first.")

        res = self.result
        total_raw = res.cmi_contrib  # (n, p)
        base_raw = res.G * res.alpha[np.newaxis, :]  # alpha_j * G_ij
        resid_raw = total_raw - base_raw

        def _centre(M):
            return M - M.mean(axis=0, keepdims=True)

        base_c = _centre(base_raw)
        resid_c = _centre(resid_raw)
        total_c = _centre(total_raw)

        rows = []
        for j, vn in enumerate(self.xlist):
            mab = float(np.mean(np.abs(base_c[:, j])))
            mar = float(np.mean(np.abs(resid_c[:, j])))
            mat = float(np.mean(np.abs(total_c[:, j])))
            rows.append(
                {
                    "variable": vn,
                    "mean_abs_baseline": round(mab, 6),
                    "mean_abs_residual": round(mar, 6),
                    "mean_abs_total": round(mat, 6),
                    "residual_share": round(mar / (mat + 1e-12), 4),
                }
            )

        df_summary = pd.DataFrame(rows).set_index("variable")

        if print_table:
            print(f"\n{'='*62}")
            print(f"CMI Contribution Summary  (n={self.n}, full sample)")
            print(f"{'='*62}")
            print(df_summary.to_string(float_format=lambda x: f"{x:.4f}"))
            print(f"{'='*62}\n")

        return df_summary

    # ── Properties ───────────────────────────────────────────────────

    @property
    def df_out(self) -> pd.DataFrame:
        if self.result is None:
            raise RuntimeError("Call fit() first.")
        return self.result.df_out

    @property
    def k_search_table(self) -> Optional[pd.DataFrame]:
        if self.result is None:
            raise RuntimeError("Call fit() first.")
        return self.result.k_search_table

    @property
    def bw_search_table(self) -> Optional[pd.DataFrame]:
        if self.result is None:
            raise RuntimeError("Call fit() first.")
        return self.result.bw_search_table

    def summary(self):
        if self.result is None:
            print("Not fitted yet.")
            return
        m = self.result.meta
        print(f"\n{'='*50}")
        print(f"CMI Model Summary")
        print(f"{'='*50}")
        print(f"  Variables : {self.xlist}")
        print(f"  n         : {self.n}")
        print(f"  R2        : {m['r2']:.4f}")
        if "r2_vc" in m:
            print(f"  R2 stage 2 (P-spline VC): {m['r2_vc']:.4f}")
        print(f"  alpha     : {np.round(self.result.alpha, 4)}")
        print(f"  k per var : {m['k_per_var']}")
        print(f"  BW per var: {m['bw_per_var']}")
        print(f"  ridge lambda: {m.get('ridge_lambda', 'n/a')}")
        print(f"{'='*50}")

    # ── Convenience: coefficient table and plots ─────────────────────

    def coef_table(self, kind: str = "esd") -> pd.DataFrame:
        """Summary statistics of the stage-2 coefficient surfaces.

        kind : "esd" (default, scale-identified), "lin" or "vc".
        Columns: mean, sd, min, q05, median, q95, max, share_positive.
        """
        if self.result is None or self.result.beta_vc is None:
            raise RuntimeError("Call fit() with stage2='pspline' first.")
        rows = []
        for vn in self.xlist:
            b = self.result.df_out[f"{vn}_{kind}"].to_numpy(float)
            q = np.percentile(b, [5, 50, 95])
            rows.append(dict(variable=vn, mean=b.mean(), sd=b.std(), min=b.min(),
                             q05=q[0], median=q[1], q95=q[2], max=b.max(),
                             share_positive=float(np.mean(b > 0))))
        return pd.DataFrame(rows).set_index("variable")

    def plot_coefficients(self, kind: str = "esd", cmap: str = "RdBu_r",
                          symmetric: bool = True, s: float = 2.0, ncols: int = 3,
                          panel_size: float = 3.6, save: str = None):
        """Map the stage-2 coefficient surfaces on the original coordinates.

        kind      : "esd" (default), "lin", "vc", or "lws" (stage-1 CMI)
        symmetric : centre the colour scale at 0 (diverging maps for signed effects)
        """
        if self.result is None:
            raise RuntimeError("Call fit() first.")
        u = self.df_raw[self.u_col].to_numpy(float)
        v = self.df_raw[self.v_col].to_numpy(float)
        p = self.p
        nrows = int(np.ceil(p / ncols))
        fig, axes = plt.subplots(nrows, min(ncols, p),
                                 figsize=(panel_size * min(ncols, p), panel_size * nrows),
                                 squeeze=False)
        for j, vn in enumerate(self.xlist):
            ax = axes[j // ncols, j % ncols]
            z = self.result.df_out[f"{vn}_{kind}"].to_numpy(float)
            if symmetric and kind != "lws":
                lim = np.nanpercentile(np.abs(z), 99) or 1.0
                sc = ax.scatter(u, v, c=z, s=s, cmap=cmap, vmin=-lim, vmax=lim)
            else:
                sc = ax.scatter(u, v, c=z, s=s, cmap="viridis" if kind == "lws" else cmap)
            ax.set_title(f"{vn} ({kind})", fontsize=11)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            plt.colorbar(sc, ax=ax, fraction=0.046)
        for k in range(p, nrows * min(ncols, p)):
            axes[k // ncols, k % ncols].axis("off")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        return fig

    def plot_transforms(self, ncols: int = 3, panel_size: float = 3.2, save: str = None):
        """Plot the learned transforms g_j(x_j) (stage 2, after orientation).

        The sign of b_j(s) is read relative to these curves: b_j > 0 means the
        local effect has the shape shown; b_j < 0 means its mirror image.
        """
        if self.result is None or self.result.G_vc is None:
            raise RuntimeError("Call fit() with stage2='pspline' first.")
        p = self.p
        nrows = int(np.ceil(p / ncols))
        fig, axes = plt.subplots(nrows, min(ncols, p),
                                 figsize=(panel_size * min(ncols, p), panel_size * 0.8 * nrows),
                                 squeeze=False)
        for j, vn in enumerate(self.xlist):
            ax = axes[j // ncols, j % ncols]
            x = self.ALL_X[:, j]
            g = self.result.G_vc[:, j]
            o = np.argsort(x)
            ax.plot(x[o], (g - g.mean())[o] / (g.std() + 1e-12), color="#2F6DB3", lw=2)
            ax.axhline(0, color="#aaa", lw=0.7, ls="--")
            ax.set_title(f"g({vn})  (standardised)", fontsize=10)
            ax.set_xlabel(vn)
        for k in range(p, nrows * min(ncols, p)):
            axes[k // ncols, k % ncols].axis("off")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        return fig

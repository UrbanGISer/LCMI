"""
Alternative stage-2 estimators for LCMI (experimental, lcmi_ur only).

Stage 2 model:  y ≈ b_0(s) + Σ_j b_j(s) · G_j(x_j),   G_j = LCMI's global
partial-residual curves (G_raw). The packaged stage 2 uses one bandwidth
tied to the CMI search, multiplies the spatial kernel by noisy CMI weights,
and then smooths β with CMI weights again. Both CMI-weighting steps and the
single bandwidth can create patchy ("blob") surfaces.

Here:
  fit_common(G, y, coords)   one adaptive bisquare bandwidth for all terms,
                             chosen by leave-one-out CV (GWR on G)
  fit_multiscale(G, y, coords)
                             one bandwidth per term (incl. intercept), chosen by
                             LOO-CV inside a backfitting loop (MGWR on G)
All local fits use an adaptive bisquare kernel over the k nearest neighbours,
and report leave-one-out predictions (self excluded) for honest accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

K_GRID = (30, 45, 70, 100, 150, 220, 330, 500, 750, 1100, 1600, 2400)


class Neighbours:
    def __init__(self, coords, kmax):
        kmax = min(kmax, len(coords) - 1)
        d, i = NearestNeighbors(n_neighbors=kmax + 1).fit(coords).kneighbors(coords)
        self.d, self.i, self.kmax = d.astype(np.float32), i.astype(np.int32), kmax

    def weights(self, k, include_self):
        """Adaptive bisquare weights over the k nearest neighbours (n, k+1)."""
        d = self.d[:, : k + 1]
        h = self.d[:, k : k + 1] + 1e-12
        w = (1 - (d / h) ** 2) ** 2
        w[:, -1] = 0.0  # k-th neighbour sits on the kernel edge
        if not include_self:
            w[:, 0] = 0.0
        return w, self.i[:, : k + 1]


def _local_fit(X, y, nb, k, include_self, ridge=1e-8, chunk=2000):
    """Batched local WLS. X (n, q), y (n,). Returns coefficients (n, q)."""
    n, q = X.shape
    w, idx = nb.weights(k, include_self)
    out = np.zeros((n, q))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        Xi = X[idx[s:e]]                      # (m, k+1, q)
        yi = y[idx[s:e]]                      # (m, k+1)
        wi = w[s:e][..., None]
        A = np.einsum("mkq,mkr->mqr", Xi * wi, Xi) + ridge * np.eye(q)
        bvec = np.einsum("mkq,mk->mq", Xi * wi, yi)
        out[s:e] = np.linalg.solve(A, bvec[..., None])[..., 0]
    return out


def fit_common(G, y, coords, k_grid=K_GRID, nb=None):
    X = np.c_[np.ones(len(y)), G]
    nb = nb or Neighbours(coords, max(k_grid) + 1)
    scores = {}
    for k in k_grid:
        if k >= nb.kmax:
            continue
        B = _local_fit(X, y, nb, k, include_self=False)
        scores[k] = float(np.mean((y - (X * B).sum(1)) ** 2))
    k_best = min(scores, key=scores.get)
    B_loo = _local_fit(X, y, nb, k_best, include_self=False)
    B = _local_fit(X, y, nb, k_best, include_self=True)
    return dict(B=B, k=k_best, cv=scores, yhat_loo=(X * B_loo).sum(1),
                yhat=(X * B).sum(1))


def fit_multiscale(G, y, coords, k_grid=K_GRID, n_iter=30, tol=1e-5, init=None,
                   nb=None, verbose=False):
    """MGWR-style backfitting on [1, G]: one CV-chosen bandwidth per term."""
    n, p = G.shape
    X = np.c_[np.ones(n), G]
    nb = nb or Neighbours(coords, max(k_grid) + 1)
    if init is None:
        init = fit_common(G, y, coords, k_grid, nb)
    B = init["B"].copy()
    ks = [init["k"]] * (p + 1)
    prev = np.inf
    for it in range(n_iter):
        for j in range(p + 1):
            r = y - (X * B).sum(1) + X[:, j] * B[:, j]   # partial residual
            xj = X[:, [j]]
            best = None
            for k in k_grid:
                if k >= nb.kmax:
                    continue
                bj = _local_fit(xj, r, nb, k, include_self=False)[:, 0]
                sse = float(np.mean((r - xj[:, 0] * bj) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, k)
            ks[j] = best[1]
            B[:, j] = _local_fit(xj, r, nb, ks[j], include_self=True)[:, 0]
        rss = float(np.mean((y - (X * B).sum(1)) ** 2))
        if verbose:
            print(f"  multiscale iter {it}: mse={rss:.5f}  k={ks}")
        if abs(prev - rss) / (prev + 1e-12) < tol:
            break
        prev = rss
    # leave-one-out prediction with the selected per-term bandwidths
    B_loo = B.copy()
    for j in range(p + 1):
        r = y - (X * B).sum(1) + X[:, j] * B[:, j]
        B_loo[:, j] = _local_fit(X[:, [j]], r, nb, ks[j], include_self=False)[:, 0]
    return dict(B=B, k=ks, yhat=(X * B).sum(1), yhat_loo=(X * B_loo).sum(1),
                n_iter=it + 1)


def fit_multiscale_backfit_G(G, y, x, coords, n_outer=4, frac=0.10, k_grid=K_GRID, nb=None,
                             verbose=False):
    """Varying-coefficient additive model: alternate multiscale local b(s) and
    weighted-LOWESS updates of each G_j (G_j <- argmin Σ (r_ij - b_j(s_i) G_j(x_ij))²)."""
    from .extensions import _wlowess
    nb = nb or Neighbours(coords, max(k_grid) + 1)
    G = G.copy()
    fit = None
    for it in range(n_outer):
        fit = fit_multiscale(G, y, coords, k_grid=k_grid, nb=nb,
                             init=None if fit is None else dict(B=fit["B"], k=fit["k"][0]))
        B = fit["B"]
        X = np.c_[np.ones(len(y)), G]
        for j in range(G.shape[1]):
            r = y - (X * B).sum(1) + B[:, j + 1] * G[:, j]
            bj = B[:, j + 1]
            ok = np.abs(bj) > 1e-3 * np.abs(bj).max()
            t = np.where(ok, r / np.where(ok, bj, 1.0), 0.0)
            w = np.where(ok, bj ** 2, 0.0)
            g = _wlowess(x[:, j], t, w, frac=frac)
            G[:, j] = g
            X[:, j + 1] = g
        if verbose:
            mse = float(np.mean((y - fit["yhat"]) ** 2))
            print(f"  G-backfit outer {it}: mse={mse:.5f} k={fit['k']}")
    fit = fit_multiscale(G, y, coords, k_grid=k_grid, nb=nb,
                         init=dict(B=fit["B"], k=fit["k"][0]))
    fit["G"] = G
    return fit

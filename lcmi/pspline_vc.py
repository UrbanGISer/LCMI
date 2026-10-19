"""
Penalised-spline varying-coefficient stage 2 (experimental).

    y = Σ_j β_j(s) · G_j(x_j),   G_0 ≡ 1 (intercept),
    β_j(s) = B(s) θ_j,  B = tensor-product cubic B-splines on (u, v),
    minimise ||y − Zθ||² + λ_j θ_jᵀ P θ_j   (P = 2nd-order difference penalty)

Unlike kernel (GWR/kNN) local fits, every coefficient surface is one smooth
global function, so there are no window edges and no per-point noise
("blobs"). λ is chosen per term by GCV (coordinate search on a log grid).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def _bspline_basis(x, n_basis, degree=3):
    lo, hi = x.min(), x.max()
    inner = np.linspace(lo, hi, n_basis - degree + 1)
    t = np.r_[[lo] * degree, inner, [hi] * degree]
    return BSpline.design_matrix(np.clip(x, lo, hi), t, degree).toarray()


def _diff_penalty(m, order=2):
    D = np.diff(np.eye(m), n=order, axis=0)
    return D.T @ D


def tensor_basis(coords, n_basis=12):
    Bu = _bspline_basis(coords[:, 0], n_basis)
    Bv = _bspline_basis(coords[:, 1], n_basis)
    B = np.einsum("ni,nj->nij", Bu, Bv).reshape(len(coords), -1)
    Pm = _diff_penalty(n_basis)
    I = np.eye(n_basis)
    P = np.kron(Pm, I) + np.kron(I, Pm)
    return B, P


def fit_pspline_vc(G, y, coords, n_basis=12, lam_grid=np.logspace(-3, 4, 15), n_sweeps=3,
                   basis=None):
    """basis: optional precomputed (B, P) for these rows (e.g. a CV training subset
    of a basis built on all coordinates, so held-out rows share the same knots)."""
    n, p = G.shape
    B, P = tensor_basis(coords, n_basis) if basis is None else basis
    m = B.shape[1]
    X = np.c_[np.ones(n), G]
    q = p + 1
    Z = np.concatenate([B * X[:, [j]] for j in range(q)], axis=1)   # (n, q*m)
    ZtZ, Zty = Z.T @ Z, Z.T @ y
    lam = np.full(q, 1.0)

    def solve(lam):
        S = ZtZ.copy()
        for j in range(q):
            S[j * m:(j + 1) * m, j * m:(j + 1) * m] += lam[j] * P
        S += 1e-8 * np.eye(q * m)
        L = np.linalg.cholesky(S)
        theta = np.linalg.solve(L.T, np.linalg.solve(L, Zty))
        # effective degrees of freedom = trace(S^-1 ZtZ)
        Sinv_ZtZ = np.linalg.solve(L.T, np.linalg.solve(L, ZtZ))
        edf = np.trace(Sinv_ZtZ)
        rss = float(np.sum((y - Z @ theta) ** 2))
        gcv = n * rss / (n - edf) ** 2
        return theta, gcv, edf

    best = solve(lam)
    for _ in range(n_sweeps):
        for j in range(q):
            cands = []
            for l in lam_grid:
                trial = lam.copy()
                trial[j] = l
                cands.append((solve(trial)[1], l))
            lam[j] = min(cands)[1]
        best = solve(lam)
    theta, gcv, edf = best
    beta = np.column_stack([B @ theta[j * m:(j + 1) * m] for j in range(q)])
    yhat = (X * beta).sum(1)
    return dict(B=beta, yhat=yhat, lam=lam, edf=edf, gcv=gcv,
                theta=theta.reshape(q, m))

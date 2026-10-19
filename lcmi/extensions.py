"""
Experimental extensions on top of lcmi.CMI (stage 2), kept outside lcmi.py.

Background (experiments/scale_diagnosis.py): stage 2 fits
    y ≈ a(s) + Σ_j b_j(s) · G_j(x_j),     G_j = global LOWESS partial-residual curve.
If the truth is y = Σ_j β_j(s) g_j(x_j), then G_j ≈ κ_j g_j with κ_j ≈ the
global average effect, so b_j(s) ≈ β_j(s)/κ_j: a *relative multiplier*
(≈1 = average strength). β_j(s) and g_j are only identified up to
β·g = (cβ)(g/c), so any "absolute" coefficient needs a scale convention.

1. scaled_coefficients(model, beta=None)
      {v}_mult : b_j(s) / weighted-mean(b_j)            (relative to global effect)
      {v}_esd  : b_j(s) · sd(G_j)   Y-units per 1 SD of the learned transform
      {v}_lin  : b_j(s) · slope(G_j ~ x_j)   linear-equivalent dY/dx_j
                 (exactly β_j(s) when g_j is linear; GWR-comparable)

2. backfit(model, n_iter=3, frac=0.10)
      Alternates (i) local WLS for a(s), b(s) given G and
      (ii) a weighted LOWESS update of each G_j given b(s):
          G_j ← argmin Σ_i (r_ij − b_j(s_i) G_j(x_ij))²
              = LOWESS of r_ij / b_j(s_i) with weights b_j(s_i)²
      This removes the attenuation of G_j caused by fitting it globally
      under spatial heterogeneity (and by a wide LOWESS span).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _wlowess(x, t, w, frac=0.10, n_grid=200):
    """Weighted local-linear smoother evaluated on a grid, interpolated to x."""
    order = np.argsort(x)
    xs, ts, ws = x[order], t[order], w[order]
    # quantile grid over the full range, so sparse tails are fitted, not clamped
    grid = np.unique(np.quantile(xs, np.linspace(0, 1, n_grid)))
    h = max(int(len(xs) * frac), 20)
    out = np.zeros(len(grid))
    for g_i, xe in enumerate(grid):
        c = np.searchsorted(xs, xe)
        lo, hi = max(0, c - h // 2), min(len(xs), c + h // 2)
        xx, tt, ww = xs[lo:hi], ts[lo:hi], ws[lo:hi]
        d = np.abs(xx - xe)
        k = ww * (1 - (d / (d.max() + 1e-12)) ** 3) ** 3
        A = np.c_[np.ones_like(xx), xx - xe]
        AtK = A.T * k
        try:
            out[g_i] = np.linalg.solve(AtK @ A + 1e-10 * np.eye(2), AtK @ tt)[0]
        except np.linalg.LinAlgError:
            out[g_i] = np.average(tt, weights=k + 1e-12)
    return np.interp(x, grid, out)


def _local_wls(model, G, smooth=True):
    """Local WLS with intercept, same kernel/CMI weights as lcmi stage 2."""
    res, meta = model.result, model.result.meta
    n, p = G.shape
    lws = np.column_stack([res.cmi_surfaces[v] for v in model.xlist])
    k = min(max(meta["bw_per_var"].values()), n - 2)
    med_bw = float(np.median(list(meta["bw_per_var"].values())))
    bwd = float(np.sqrt(med_bw / (np.pi * n)))
    if not hasattr(model, "_ext_nn"):
        model._ext_nn = NearestNeighbors(n_neighbors=k + 1).fit(model._coords).kneighbors(model._coords)
    D, I = model._ext_nn
    a = np.zeros(n)
    B = np.zeros((n, p))
    y = model.Y
    for i in range(n):
        idx, d = I[i, 1:], D[i, 1:]
        w = np.exp(-d**2 / (2 * bwd**2)) * (lws[idx].mean(1) + 1e-6)
        w /= w.sum()
        X = np.c_[np.ones(k), G[idx]]
        XtW = X.T * w
        b = np.linalg.solve(XtW @ X + 1e-8 * np.eye(p + 1), XtW @ y[idx])
        a[i], B[i] = b[0], b[1:]
    if smooth:  # same CMI-weighted neighbour smoothing as lcmi stage 2
        ks = min(max(5, int(med_bw / 8)), n - 1)
        _, Is = NearestNeighbors(n_neighbors=ks + 1).fit(model._coords).kneighbors(model._coords)
        nb = Is[:, 1:]
        Gbar = np.einsum("ij,ijk->ik", np.full(nb.shape, 1.0 / ks), G[nb])
        level = a + (B * Gbar).sum(1)  # local fitted level, invariant to smoothing
        for j in range(p):
            wj = lws[nb, j] + 1e-6
            wj /= wj.sum(1, keepdims=True)
            B[:, j] = (wj * B[nb, j]).sum(1)
        a = level - (B * Gbar).sum(1)
    return a, B


def scaled_coefficients(model, B=None, G=None):
    """Scale-identified versions of the stage-2 coefficients."""
    B = model.result.beta_wls.astype(float) if B is None else B
    G = model.result.G_raw.astype(float) if G is None else G
    lws = np.column_stack([model.result.cmi_surfaces[v] for v in model.xlist])
    out = {}
    for j, v in enumerate(model.xlist):
        x = model.ALL_X[:, j]
        g = G[:, j]
        sd = g.std()
        lin = np.cov(g, x)[0, 1] / (x.var() + 1e-12)
        wmean = np.average(B[:, j], weights=lws[:, j] + 1e-6)
        out[f"{v}_mult"] = B[:, j] / (wmean if abs(wmean) > 1e-12 else 1.0)
        out[f"{v}_esd"] = B[:, j] * sd
        out[f"{v}_lin"] = B[:, j] * lin
    return pd.DataFrame(out)


def backfit(model, n_iter=3, frac=0.10, smooth=True, verbose=True):
    """Alternate local WLS (a, b) and weighted-LOWESS updates of G. Returns dict."""
    y = model.Y
    G = model.result.G_raw.astype(float).copy()
    hist = []
    for it in range(n_iter + 1):
        a, B = _local_wls(model, G, smooth=smooth)
        yhat = a + (B * G).sum(1)
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
        hist.append(r2)
        if verbose:
            print(f"  backfit iter {it}: R² = {r2:.4f}")
        if it == n_iter:
            break
        for j in range(G.shape[1]):
            others = (B * G).sum(1) - B[:, j] * G[:, j]
            r = y - a - others
            bj = B[:, j]
            ok = np.abs(bj) > 1e-3 * np.abs(bj).max()
            t = np.where(ok, r / np.where(ok, bj, 1.0), 0.0)
            w = np.where(ok, bj**2, 0.0)
            gnew = _wlowess(model.ALL_X[:, j], t, w, frac=frac)
            # keep the scale convention of the original G_j (same sd), so b
            # stays a multiplier of comparable size
            gnew = (gnew - gnew.mean()) * (G[:, j].std() / (gnew.std() + 1e-12)) + G[:, j].mean()
            G[:, j] = gnew
    return dict(a=a, B=B, G=G, yhat=yhat, r2_history=hist)

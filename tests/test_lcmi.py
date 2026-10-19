"""Quick regression tests: python -m pytest tests -q"""

import numpy as np
import pandas as pd

from lcmi import CMI


def _synthetic(grid=40, mean_shift=1.0, seed=0):
    rng = np.random.default_rng(seed)
    u1 = np.linspace(0, 1, grid)
    U, V = np.meshgrid(u1, u1)
    u, v = U.ravel(), V.ravel()
    b1 = 2.0 * (u + v - 1.0 + mean_shift)          # plane, sign changes if mean_shift < 1
    b2 = np.cos(2 * np.pi * u) + 0.5 * mean_shift   # sign-changing pattern
    x1, x2 = rng.normal(size=(2, u.size))
    y = b1 * x1 + b2 * x2 ** 2 + rng.normal(0, .3, u.size)
    return pd.DataFrame(dict(u=u, v=v, y=y, x1=x1, x2=x2)), (b1, b2)


def _fit(df):
    return CMI(df, u="u", v="v", y="y", xlist=["x1", "x2"], verbose=False).fit(k=8, bw=60)


def test_outputs_and_fit():
    df, _ = _synthetic()
    m = _fit(df)
    for c in ["x1_lws", "x1_esd", "x1_lin", "x1_vc", "vc_intercept", "y_pred_vc"]:
        assert c in m.df_out
    assert m.result.meta["r2_vc"] > 0.9
    assert m.coef_table().shape == (2, 8)


def test_recovers_signed_effects():
    df, (b1, b2) = _synthetic(mean_shift=0.0)        # half-positive / half-negative effects
    m = _fit(df)
    x1, x2 = df.x1.values, df.x2.values
    for beta, est, g in ((b1, m.df_out.x1_esd, x1), (b2, m.df_out.x2_esd, x2 ** 2)):
        truth = beta * g.std()
        big = np.abs(beta) > 0.1 * np.abs(beta).max()
        assert np.corrcoef(truth, est)[0, 1] > 0.95
        assert np.mean(np.sign(est[big]) == np.sign(truth[big])) > 0.95


def test_legacy_wls_option():
    df, _ = _synthetic()
    m = CMI(df, u="u", v="v", y="y", xlist=["x1", "x2"], wls=True, verbose=False).fit(k=8, bw=60)
    assert "x1_wls" in m.df_out and np.isfinite(m.result.meta["r2_wls"])

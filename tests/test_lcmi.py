"""Quick regression tests: python -m pytest tests -q"""

import numpy as np
import pandas as pd

from geocmi import lcmi, lcmi_c, local_mi_mixed, local_mi_nominal


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
    return lcmi(df, u="u", v="v", y="y", xlist=["x1", "x2"], verbose=False).fit(k=8, bw=60)


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
    m = lcmi(df, u="u", v="v", y="y", xlist=["x1", "x2"], wls=True, verbose=False).fit(k=8, bw=60)
    assert "x1_wls" in m.df_out and np.isfinite(m.result.meta["r2_wls"])


def test_predict_matches_fit_and_generalises():
    df, _ = _synthetic(grid=40, seed=1)
    m = _fit(df)
    # in-sample prediction reproduces the fitted values
    assert np.allclose(m.predict(df), m.df_out.y_pred_vc, atol=1e-6)
    # held-out rows: refit on 80 %, predict the rest
    rng = np.random.default_rng(0)
    te = rng.random(len(df)) < 0.2
    mt = _fit(df[~te].reset_index(drop=True))
    yp = mt.predict(df[te])
    y = df.y[te].to_numpy()
    r2 = 1 - np.sum((y - yp) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2 > 0.85
    comp = mt.predict(df[te], return_components=True)
    assert {"x1_esd", "x2_esd", "vc_intercept", "y_pred"} <= set(comp.columns)


def test_categorical_predictor_label_invariant():
    rng = np.random.default_rng(3)
    g = 35
    u1 = np.linspace(0, 1, g)
    U, V = np.meshgrid(u1, u1)
    u, v = U.ravel(), V.ravel()
    x1 = rng.normal(size=u.size)
    c = rng.integers(0, 4, u.size)
    mu = np.array([0.9, -0.9, 0.4, -0.4])
    b2 = 0.5 + 1.5 * u
    y = (1 + v) * x1 + b2 * mu[c] + rng.normal(0, .3, u.size)
    perm = np.array([2, 0, 3, 1])
    fits = []
    for codes in (c, perm[c]):
        df = pd.DataFrame(dict(u=u, v=v, y=y, x1=x1, c=codes))
        m = lcmi(df, u="u", v="v", y="y", xlist=["x1", "c"], categorical=["c"],
                verbose=False).fit(bw=60)
        fits.append(m.df_out["c_esd"].to_numpy())
    truth = b2 * mu[c].std()
    assert abs(np.corrcoef(fits[0], fits[1])[0, 1]) > 0.9999      # relabelling-invariant
    assert abs(np.corrcoef(fits[0], truth)[0, 1]) > 0.95
    assert np.isnan(m.df_out["c_lin"]).all()


def test_shared_categorical_estimators():
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(0)
    n = 900
    coords = rng.random((n, 2))
    nb = NearestNeighbors(n_neighbors=150).fit(coords).kneighbors(coords)[1]
    x = rng.integers(0, 4, n)
    y_dep = np.where(rng.random(n) < 0.6, x, rng.integers(0, 4, n))   # dependent
    y_ind = rng.integers(0, 4, n)                                      # independent
    dep, ind = local_mi_nominal(x, y_dep, nb), local_mi_nominal(x, y_ind, nb)
    assert dep.mean() > 0.2 and abs(ind.mean()) < 0.02                 # debiased null ~ 0
    perm = np.array([3, 1, 0, 2])
    assert np.allclose(local_mi_nominal(perm[x], y_dep, nb), dep)      # label-invariant
    xc = rng.normal(size=n)
    yc = (xc + 0.5 * rng.normal(size=n) > 0).astype(int)
    a, b = local_mi_mixed(yc, xc, nb), local_mi_mixed(1 - yc, xc, nb)
    assert a.mean() > 0.2 and np.allclose(a, b)


def test_cmic_nominal_outcome():
    rng = np.random.default_rng(1)
    g = 40
    u1 = np.linspace(0, 1, g)
    U, V = np.meshgrid(u1, u1)
    u, v = U.ravel(), V.ravel()
    n = u.size
    x2 = rng.integers(0, 4, n)
    x1 = np.where(rng.random(n) < 0.6, x2, rng.integers(0, 4, n))   # copy of x2, no effect
    xc = rng.normal(size=n)                                          # continuous predictor
    w = 0.5 + 2.5 * u                                                # x2 effect grows eastwards
    logit = w * (x2 == 1) - w * (x2 == 3) + 0.8 * xc
    y = (logit + rng.normal(size=n) > 0.5).astype(int)
    df = pd.DataFrame(dict(u=u, v=v, y=y, x1=x1, x2=x2, xc=xc))
    m = lcmi_c(df, "u", "v", "y", ["x1", "x2", "xc"], categorical=["x1", "x2"], bw=300,
             verbose=False).fit(condition="auto", levels_z=4)
    s = m.summary()
    assert s.loc["x1", "given"] == "x2"
    assert abs(s.loc["x1", "mean_cmi"]) < 0.01 < s.loc["x1", "mean_mi"]   # spurious pairwise, ~0 given x2
    assert np.corrcoef(m.df_out["x2_mi"], w)[0, 1] > 0.6                  # follows the spatial weight
    assert s.loc["xc", "mean_mi"] > 0.05
    df2 = df.assign(y=1 - df.y, x2=(df.x2 + 1) % 4)                        # relabel classes and categories
    m2 = lcmi_c(df2, "u", "v", "y", ["x1", "x2", "xc"], categorical=["x1", "x2"], bw=300,
              verbose=False).fit()
    assert np.allclose(m2.df_out["x2_mi"], m.df_out["x2_mi"])

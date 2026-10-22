# LCMI

**Residualized Local Conditional Mutual Information (LCMI): A Scalable Information-Theoretic Framework for Interpretable and Nonlinear Spatially Varying Effect Modelling**

![LCMI overview](https://github.com/user-attachments/assets/0ea2948e-6be8-4f3c-abe0-eb702a633223)

LCMI estimates how the effect of each predictor varies across space, without assuming the effect is linear. It works in two stages.
1. **Local dependence (stage 1).** At every location, LCMI measures the conditional mutual information between each predictor and the response within a local neighbourhood. This gives an unsigned "where does x_j matter" surface.
2. **Spatially varying effects (stage 2).** LCMI then fits a varying-coefficient model
   `y = b0(s) + Σj bj(s) · gj(xj)`.
   - Each `gj` is a learned nonlinear transform.
   - Each `bj(s)` is a smooth coefficient surface (penalised splines). It is reported on an interpretable scale and with a well-defined sign.

## Installation

```bash
pip install geocmi
```

The PyPI distribution is called `geocmi`; the module is imported as `lcmi`:

```python
from lcmi import CMI, CMIc
```

From source (this repository):

```bash
pip install git+https://github.com/UrbanGISer/LCMI.git
```

Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, joblib.

## Quick start

```python
from lcmi import CMI

model = CMI(df, u="lon", v="lat", y="y", xlist=["x1", "x2", "x3"]).fit()

model.summary()               # R² of both stages, selected k / bandwidth
model.coef_table()            # summary of the coefficient surfaces
model.plot_coefficients()     # maps of the spatially varying effects
model.plot_transforms()       # learned shapes g_j(x_j)
model.plot_pdp()              # partial-dependence decomposition (stage 1)

out = model.df_out            # your data + all LCMI columns
y_new = model.predict(new_df) # predict at new locations / predictor values
```

## Outputs (`model.df_out`)

| column | meaning |
|---|---|
| `{x}_esd` | **Spatially varying effect** of `x`: change in `y` per 1 SD of the learned transform `g(x)`. Signed, comparable across variables. Use this for maps. |
| `{x}_lin` | Linear-equivalent local slope `dy/dx`. Comparable to GWR coefficients for monotone, roughly linear effects. |
| `{x}_vc` | Raw multiplier `b(s)` of `g(x)`. |
| `{x}_lws` | Stage-1 local conditional mutual information (dependence strength, unsigned). |
| `vc_intercept`, `y_pred_vc` | Intercept surface and fitted values (stage 2). |
| `y_pred` | Stage-1 fitted values. |

**Categorical predictors.** Variables listed in `categorical=` are one-hot encoded whenever they act as conditioning covariates. Their local information `{x}_lws` uses a label-free discrete–continuous estimator (Ross 2014), and their stage-2 transform is a learned category-effect profile, so results do not depend on how categories are coded. For a categorical `x`, `b(s) > 0` means the local ranking of categories matches the overall profile, and `b(s) < 0` means it is reversed. `{x}_lin` is not defined (NaN).

**Reading signs.** A coefficient `b(s) > 0` means the local effect has the shape shown by `plot_transforms()`; `b(s) < 0` means the mirror image. Each `g(x)` is oriented to increase with `x`, or, for U-shaped effects, to be high at the extremes.

## Categorical data

| outcome | predictors | use |
|---|---|---|
| continuous | numeric and/or categorical | `CMI(..., categorical=[...])` → signed effect maps `{x}_esd` |
| **nominal (classes)** | categorical and/or continuous | `CMIc` (module `lcmi_c`) → local (conditional) mutual information maps |

```python
from lcmi import CMIc, quantile_codes

df["edu_q5"] = quantile_codes(df["edu"], 5)            # optional: coarse classes for a numeric variable
m = CMIc(df, u="lon", v="lat", y="vote_class",
         xlist=["edu_q5", "black_q5", "income"],
         categorical=["edu_q5", "black_q5"], bw=440)
m.fit(condition="auto")                                # or {x: [columns]}; None = pairwise only
m.summary()                                            # mean MI / CMI, ranking, reliability check
m.plot_maps(kind="cmi")                                # {x}_mi, {x}_cmi (+ _smooth) in m.df_out
```

How the estimators work:
- **Nominal x vs nominal y.** Debiased local plug-in MI: Miller–Madow corrected tables minus a permutation null, so the result is essentially unbiased in small windows.
- **Continuous x vs nominal y.** Discrete–continuous MI (Ross 2014) on x adjusted for the other predictors, with nominal ones one-hot encoded.

All of these are invariant to category labels.
- **Conditioning.** `condition=` gives the local conditional MI Σ_z p(z) I(x; y | z), which separates a predictor's own information from information shared with an associated predictor.
- **Reliability check.** The `retention_check` column re-computes the estimate with a shuffled z. Values well below 0.8 mean the local strata are too sparse; use coarser classes or a larger `bw`.
- **Helper functions.** `local_mi_nominal`, `local_mi_mixed`, `onehot`, `conditioning_matrix`, `cramers_v`, `quantile_codes`.

## Common options

```python
CMI(df, u, v, y, xlist,
    k_candidates=[4, 6, 8, 10, 12],  # KSG neighbours searched in stage 1
    bw_candidates=None,              # stage-1 window sizes (None = automatic)
    n_jobs=-1,
    verbose=True,
    categorical=["land_use"],        # nominal predictors (unordered categories)
    wls=False)                       # also run the original kernel-WLS stage 2 (*_wls)

model.fit()                          # automatic k / bandwidth search
model.fit(k=8, bw=200)               # fixed values (int, or dict per variable)
model.fit_vc(n_basis=16)             # refit stage 2 with a finer spline basis
model.predict(new_df, return_components=True)   # coefficients + y at new rows
model.result.meta                    # r2, r2_vc, edf_vc, lam_vc, k_per_var, bw_per_var, ...
```

## Examples

[`notebooks/LCMI_examples.ipynb`](https://github.com/UrbanGISer/LCMI/blob/main/notebooks/LCMI_examples.ipynb) covers:
1. **Synthetic data:** recovery of known spatially varying, nonlinear effects.
2. **King County house prices:** an empirical application.
3. **Benchmark:** comparison with GWR, MGWR, GeoShapley and XGeoML on the synthetic data. Needs `pip install mgwr geoshapley XGeoML`.
4. **Robustness:** out-of-sample prediction, and data outside LCMI's model class.

[`notebooks/LCMI_categorical_examples.ipynb`](https://github.com/UrbanGISer/LCMI/blob/main/notebooks/LCMI_categorical_examples.ipynb) covers categorical data:
1. **Synthetic, categorical predictors with a continuous outcome:** sign-changing category effects, label invariance.
2. **Synthetic, nominal outcome:** the fully nominal benchmark, correlated predictors and conditioning, and a continuous predictor.
3. **2016 U.S. election (3,108 counties):** pairwise and conditional maps of education, Black share, Hispanic share and density for a three-class vote outcome.

The data are in `data/us_election_2016_counties.csv` (public GeoDa Center data).

## Method

![Illustration of LCMI](https://github.com/user-attachments/assets/9536aec7-4aa4-411d-bc68-5d7f124104f0)

![Two-stage estimation](https://github.com/user-attachments/assets/d9d39ede-0b86-4e6e-a26e-689e3fc7c341)

**Changes relative to the code published with the paper.** The paper code in [`lcmi-ijgis/`](https://github.com/UrbanGISer/LCMI/tree/main/lcmi-ijgis) is kept exactly as published; the package differs as follows:
- **Stage 2 is now a penalised-spline varying-coefficient model** with the transforms `g_j` re-estimated jointly. It replaces the kernel local WLS and gives more accurate, smoother coefficient surfaces. Signs are now absolute: the original `*_wls` signs were relative to the global average effect.
- **Stage-1 KSG estimator uses strict neighbour counts**, which reduces a downward bias in small windows.
- The original stage 2 is still available with `wls=True`.

## Model assumptions

- **Separable effects.** Stage 2 represents each effect as `bj(s) · gj(xj)`: one nonlinear shape whose strength and sign vary over space. When effects interact or change shape across locations, the maps give the best separable summary of them. Comparing `model.predict` out of sample with a flexible learner is a simple check.
- **Smooth surfaces.** Coefficient surfaces are penalised splines, so very sharp spatial boundaries are smoothed. For strongly irregular study areas, a larger `n_basis` can help.
- **Prediction range.** `predict` clamps new locations to the training bounding box, and new predictor values to the training range.

## Citation

Liu, L. Residualized Local Conditional Mutual Information (LCMI): A Scalable Information-Theoretic Framework for Interpretable and Nonlinear Spatially Varying Effect Modelling. *International Journal of Geographical Information Science*.

## License

MIT

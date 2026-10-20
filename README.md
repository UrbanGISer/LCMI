# LCMI

**Local Conditional Mutual Information (LCMI): A Scalable Information-Theoretic Framework for Interpretable and Nonlinear Spatially Varying Effect Modelling**

<img width="1422" height="282" alt="LCMI overview" src="https://github.com/user-attachments/assets/0ea2948e-6be8-4f3c-abe0-eb702a633223" />

LCMI estimates how the effect of each predictor varies across space, without assuming the effect is linear. It works in two stages.
1. **Local dependence (stage 1).** At every location, LCMI measures the conditional mutual information between each predictor and the response within a local neighbourhood. This gives an unsigned "where does x_j matter" surface.
2. **Spatially varying effects (stage 2).** LCMI then fits a varying-coefficient model
   `y = b0(s) + Σj bj(s) · gj(xj)`.
   - Each `gj` is a learned nonlinear transform.
   - Each `bj(s)` is a smooth coefficient surface (penalised splines). It is reported on an interpretable scale and with a well-defined sign.

## Installation

```bash
pip install lcmi
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

**Reading signs.** A coefficient `b(s) > 0` means the local effect has the shape shown by `plot_transforms()`; `b(s) < 0` means the mirror image. Each `g(x)` is oriented to increase with `x`, or, for U-shaped effects, to be high at the extremes.

## Common options

```python
CMI(df, u, v, y, xlist,
    k_candidates=[4, 6, 8, 10, 12],  # KSG neighbours searched in stage 1
    bw_candidates=None,              # stage-1 window sizes (None = automatic)
    n_jobs=-1,
    verbose=True,
    wls=False)                       # also run the original kernel-WLS stage 2 (*_wls)

model.fit()                          # automatic k / bandwidth search
model.fit(k=8, bw=200)               # fixed values (int, or dict per variable)
model.fit_vc(n_basis=16)             # refit stage 2 with a finer spline basis
model.predict(new_df, return_components=True)   # coefficients + y at new rows
model.result.meta                    # r2, r2_vc, edf_vc, lam_vc, k_per_var, bw_per_var, ...
```

## Examples

[`notebooks/LCMI_examples.ipynb`](notebooks/LCMI_examples.ipynb) covers:
1. **Synthetic data:** recovery of known spatially varying, nonlinear effects.
2. **King County house prices:** an empirical application.
3. **Benchmark:** comparison with GWR, MGWR, GeoShapley and XGeoML on the synthetic data. Needs `pip install mgwr geoshapley XGeoML`.
4. **Robustness:** out-of-sample prediction, and data outside LCMI's model class.

## Method

<img width="1629" height="957" alt="Illustration of LCMI" src="https://github.com/user-attachments/assets/9536aec7-4aa4-411d-bc68-5d7f124104f0" />

<img width="1803" height="1109" alt="Two-stage estimation" src="https://github.com/user-attachments/assets/d9d39ede-0b86-4e6e-a26e-689e3fc7c341" />

**Changes relative to the code published with the paper.** The paper code in [`lcmi-ijgis/`](lcmi-ijgis/) is kept exactly as published; the package differs as follows:
- **Stage 2 is now a penalised-spline varying-coefficient model** with the transforms `g_j` re-estimated jointly. It replaces the kernel local WLS and gives more accurate, smoother coefficient surfaces. Signs are now absolute: the original `*_wls` signs were relative to the global average effect.
- **Stage-1 KSG estimator uses strict neighbour counts**, which reduces a downward bias in small windows.
- The original stage 2 is still available with `wls=True`.

## When to trust the maps

LCMI's stage 2 assumes that every effect has the separable form `bj(s) · gj(xj)`: one global shape, scaled up, down or flipped by location.

| synthetic data (100 × 100, noise SD 0.5) | noise ceiling R² | LCMI out-of-fold R² | gradient boosting [x, u, v] out-of-fold R² |
|---|---|---|---|
| separable effects (the benchmark above) | 0.970 | 0.967 | 0.881 |
| shape changes with location + interaction | 0.974 | 0.849 | 0.902 |

- **When the assumption holds,** LCMI generalises out of sample almost up to the noise ceiling. There is no leakage or overfitting, and on data with no spatial variation it does not invent any.
- **The benchmark generator satisfies the assumption.** LCMI's near-perfect scores there partly reflect a matched model class.
- **When effects interact or change shape over space,** a flexible model predicts better. LCMI's maps are then the best *separable summary* of the effects, not their full structure.
- **Practical check:** compare `model.predict` out of sample with a flexible model. A large gap means the maps should be read as summaries.

## Limitations
- Coefficient surfaces are smooth by construction. Very sharp spatial boundaries will be rounded.
- The spline basis is a regular grid over the bounding box of the coordinates. Strongly irregular study areas may need a larger `n_basis`.
- Rare binary predictors are weakly identified.
- Effects that interact or change shape over space are only approximated (see above).
- `predict` clamps locations outside the training bounding box and predictor values outside the training range.
- No standard errors / confidence bands yet.

## Citation

Liu, L. Local Conditional Mutual Information (LCMI): A Scalable Information-Theoretic Framework for Interpretable and Nonlinear Spatially Varying Effect Modelling. *International Journal of Geographical Information Science* (accepted).

## License

MIT

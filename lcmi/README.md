# lcmi (package seed)

Snapshot of the LCMI code to grow into an installable package. Development and benchmarks live in the `lcmi_ur` exploration folder.

## Model
1. **Stage 1:** local conditional mutual information surfaces `{v}_lws` (KSG with strict counts), plus global partial-residual curves `G_raw`.
2. **Stage 2 (default, `stage2="pspline"`):** a varying-coefficient model
   `y = b_0(s) + Σ_j b_j(s)·g_j(x_j)`.
   - `b_j(s)` are tensor-product B-spline surfaces with a difference penalty (λ_j by GCV).
   - `g_j` starts from `G_raw` and is re-estimated jointly with `b(s)` (backfitting). This avoids the bias of a plug-in curve when x_j's distribution varies over space.
3. **Legacy stage 2 (`wls=True`, on by default for now):** the kernel local WLS of the IJGIS paper (`{v}_wls`, intercept bug fixed).

## Outputs (`model.df_out`)
| column | meaning |
|---|---|
| `{v}_lws` | stage-1 local CMI (dependence strength, unsigned) |
| `{v}_esd` | **recommended coefficient**: Y change per 1 SD of the learned transform g_j (scale-identified) |
| `{v}_lin` | linear-equivalent dY/dx_j (for monotone, near-linear effects; GWR-comparable) |
| `{v}_vc` | raw multiplier b_j(s) of g_j |
| `vc_intercept`, `y_pred_vc` | intercept surface, stage-2 fitted values |
| `{v}_wls`, `wls_intercept`, `y_pred_wls` | legacy kernel stage 2 |

`model.result.meta`: `r2`, `r2_vc`, `gcv_vc`, `edf_vc`, `lam_vc`, `r2_wls`.

```python
from lcmi import CMI
m = CMI(df, u="u", v="v", y="y", xlist=["x1", "x2"], wls=False).fit()   # P-spline stage 2 only
m.df_out[["x1_esd", "x2_esd"]]
m.fit_vc(n_basis=16)          # refit stage 2 with a finer spline basis
```

## Benchmarks (lcmi_ur)
| | synthetic case 1 coef RMSE (X1/X2/X3) | X depends on location (X1/X2/X3) | King County R² stage 2 |
|---|---|---|---|
| legacy WLS | 0.053 / 0.086 / 0.092 | — | 0.879 |
| P-spline VC + backfit | **0.023 / 0.039 / 0.040** | 0.240 / 0.144 / 0.483 (plug-in G: 0.876 / 0.512 / 1.235) | **0.896** (5-fold out-of-sample 0.888) |

## Known limitations / to do
- The spline basis is a fixed `n_basis × n_basis` grid on the bounding box. Irregular or elongated study areas may need adaptive or thin-plate bases; select λ by REML.
- Sparse binary predictors (e.g. waterfront) are weakly identified; consider a separate penalty or ridge floor.
- Smooth global surfaces blur truly sharp boundaries; a local residual correction may be needed there.
- No prediction at new locations yet; no standard errors.
- Packaging (pyproject, tests, docs) is not set up yet.

# LCMI — Code & Data (IJGIS)

This folder contains the **Local Conditional Mutual Information (LCMI)** implementation and notebooks used to produce the manuscript figures/tables. The core estimator is implemented in **`lcmi.py`** (class **`CMI`**).

---

## Directory layout

| Path | Description |
|------|-------------|
| **`lcmi.py`** | LCMI model: `CMI(df, u, v, y, xlist, ...)`, `.fit()`, `.plot_pdp()`, outputs `df_out`, `pdp_table`, surfaces, optional WLS local coefficients. |
| **`data/kc_house_data.csv`** | King County house sales (2015). **Source:** [GeoDa Center — King County house sales](https://geodacenter.github.io/data-and-lab/KingCounty-HouseSales2015/). Place the CSV here or set `data_path` in the notebook. |
| **`results/`** | Figures and CSVs written by notebooks (`PREFIX` + suffix). Created automatically when you run the notebooks. |
| **`lcmi_case1_synthetic_comparison _10K.ipynb`** | Synthetic benchmark: LCMI vs GWR / MGWR / GeoShapley; surfaces, PDP comparison, comparison table. |
| **`lcmi_case2_KingCounty _20k.ipynb`** | Empirical application: preprocessing, LCMI fit, maps (lws / WLS), PDPs, GeoShapley comparison, importance table. |
| **`LCMI_Google_Colab_Loop_1M.ipynb`** | Optional scalability stress test on a grid (Google Colab-oriented). |

---

## Environment

**Required (for `lcmi.py`):**  
`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `joblib`

**Optional (for benchmark notebooks):**  
`geoshapley`, `mgwr`, `shap`, `pygam` — only where those cells are executed (GeoShapley / GWR / MGWR / GAM curves).

---

## Reproduction 

1. Keep **`kc_house_data.csv`** under **`data/`** (or edit `data_path`).
2. Open Jupyter in **`notebook/case_CMI/lcmi/`** (or equivalent repo path).
3. Run **`lcmi_case1_synthetic_comparison _10K.ipynb`** → synthetic figures/tables under **`results/`** with prefix `lcmi_case1`.
4. Run **`lcmi_case2_KingCounty _20k.ipynb`** → King County outputs with prefix **`results/lcmi_case2`** (see table below).
Runtime depends on hardware and optional models (GeoShapley / MGWR).

---

## Outputs ↔ manuscript (reference)

**Synthetic notebook (`results/lcmi_case1_*`):**

| Output file | Role |
|-------------|------|
| `lcmi_case1_fig1_surfaces.png` | Fig. 1 — surfaces & validation |
| `lcmi_case1_fig2_scatter.png` | Fig. 2 — scatter diagnostics |
| `lcmi_case1_fig3_pdp_compare.png` | Fig. 3 — PDP comparison |
| `lcmi_case1_comparison_table.csv` | Table 1 — model comparison |

**King County notebook (`results/lcmi_case2_*`):**

| Output file | Role |
|-------------|------|
| `lcmi_case2_KingCounty_LCMI_WLS.png` | Fig. 4 — WLS coefficient surfaces |
| `lcmi_case2_KingCounty_LCMI_PDP.png` | Fig. 5 — LCMI PDP decomposition |
| `lcmi_case2_KingCounty_Importance_Table.csv` | Table 2 — importance summary |
| `lcmi_case2_KingCounty_Original.png` | Fig. S1 — focal variables on map |
| `lcmi_case2_KingCounty_LCMI_lws.png` | Fig. S2 — raw CMI (lws) surfaces |
| `lcmi_case2_KingCounty_Geoshapley_GWR.png` | Fig. S3 — GeoShapley–GWR-style surfaces |
| `lcmi_case2_KingCounty_geoshapley_pdp.png` | Fig. S4 — GeoShapley PDPs |


---

## Minimal LCMI usage (API sketch)

```python
from lcmi import CMI

model = CMI(df, u="long", v="lat", y="log_price", xlist=[...], bw_candidates=[...])
model.fit()
df_out = model.df_out          # predictions + per-variable columns (e.g. *_lws, *_wls)
pdp = model.pdp_table          # wide PDP decomposition table
model.plot_pdp(separated_legend=True)  # optional styling; see docstring in lcmi.py
```

See **`lcmi.py`** docstrings for `fit`, `plot_pdp`, `fit_local_beta`, and result fields (`cmi_surfaces`, `result.cmi_contrib`, etc.).

---

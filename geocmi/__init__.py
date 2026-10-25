"""geocmi — Residualized Local Conditional Mutual Information (LCMI) for interpretable,
nonlinear spatially varying effects.

    import geocmi
    m = geocmi.lcmi(df, u="lon", v="lat", y="y", xlist=["x1", "x2"]).fit()          # continuous y

    from geocmi import lcmi, lcmi_c
    mc = lcmi_c(df, u="lon", v="lat", y="cls", xlist=["a", "b"], categorical=["a"]).fit()  # classes

lcmi    continuous outcome; numeric and categorical predictors → signed effect maps
lcmi_c  nominal outcome (classes) → local (conditional) mutual information maps
Helpers for lcmi_c: quantile_codes, cramers_v, onehot, conditioning_matrix,
local_mi_mixed, local_mi_nominal.
"""

from ._lcmi import lcmi, __version__
from ._lcmi_c import (lcmi_c, quantile_codes, cramers_v, onehot, conditioning_matrix,
                      local_mi_mixed, local_mi_nominal)

__all__ = ["lcmi", "lcmi_c", "quantile_codes", "cramers_v", "onehot", "conditioning_matrix",
           "local_mi_mixed", "local_mi_nominal", "__version__"]

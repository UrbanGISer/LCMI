"""geocmi — Residualized Local Conditional Mutual Information (LCMI) for interpretable,
nonlinear spatially varying effects.

Two modules:
    geocmi.lcmi    lcmi.CMI    continuous outcome; numeric and categorical predictors
    geocmi.lcmi_c  lcmi_c.CMIc nominal outcome (classes); local (conditional) MI maps

    import geocmi
    m = geocmi.lcmi.CMI(df, u="lon", v="lat", y="y", xlist=["x1", "x2"]).fit()

    from geocmi import lcmi, lcmi_c
    m = lcmi_c.CMIc(df, u="lon", v="lat", y="cls", xlist=["a", "b"], categorical=["a"]).fit()
"""

from . import lcmi, lcmi_c
from .lcmi import __version__

__all__ = ["lcmi", "lcmi_c", "__version__"]

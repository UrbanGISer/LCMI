"""LCMI — Local Conditional Mutual Information for interpretable, nonlinear
spatially varying effects.

lcmi.lcmi   CMI  : continuous outcome; numeric and categorical predictors
lcmi.lcmi_c CMIc : nominal outcome (classes); local (conditional) MI maps
"""

from .lcmi import CMI, CMIResult, __version__
from .lcmi_c import (CMIc, quantile_codes, cramers_v, onehot, conditioning_matrix,
                     local_mi_mixed, local_mi_nominal)

__all__ = ["CMI", "CMIResult", "__version__", "CMIc", "quantile_codes", "cramers_v",
           "onehot", "conditioning_matrix", "local_mi_mixed", "local_mi_nominal"]

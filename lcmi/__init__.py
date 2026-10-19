"""LCMI — Local Conditional Mutual Information (package seed, work in progress).

Two-stage estimator (lcmi.CMI):
  stage 1  local CMI surfaces ({v}_lws) and global partial-residual curves G_raw
  stage 2  (default) penalised-spline varying-coefficient model with joint
           backfitting of g_j:  y = b_0(s) + Σ_j b_j(s)·g_j(x_j)
           outputs {v}_vc, {v}_esd (scale-identified), {v}_lin, vc_intercept, y_pred_vc
  legacy   kernel local WLS ({v}_wls), kept for reproducing the IJGIS paper (wls=True)

Experimental modules: extensions (scaled_coefficients for legacy *_wls),
stage2 (kernel CV / multiscale variants), pspline_vc (stand-alone P-spline VC).
"""

from .lcmi import CMI, CMIResult
from .extensions import scaled_coefficients

__all__ = ["CMI", "CMIResult", "scaled_coefficients"]

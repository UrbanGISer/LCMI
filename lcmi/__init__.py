"""LCMI — Local Conditional Mutual Information for interpretable, nonlinear
spatially varying effects. See ``lcmi.lcmi`` for the model description."""

from .lcmi import CMI, CMIResult, __version__

__all__ = ["CMI", "CMIResult", "__version__"]

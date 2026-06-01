"""Shared utilities for the paid-media-playbook case studies.

Submodules:
- ``adstock``  : geometric + delayed adstock transforms
- ``saturation``: Hill and log-saturation response curves
- ``simulate`` : panel and path simulators used across case studies
- ``credit``   : click-vs-view credit-weighting helpers (SEM + display)
"""

from . import adstock, credit, saturation, simulate  # noqa: F401

__all__ = ["adstock", "saturation", "simulate", "credit"]
__version__ = "0.1.0"

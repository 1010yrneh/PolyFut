"""PolyFut CV Pipeline v2 — ball-anchored, single-player.

Design: docs/pipeline-v2.md. This package is built incrementally, stage by
stage, alongside the shipped v1 ``polyfut_video`` package (which it reuses for
decode / shot-filter / deadtime / ball-smoothing). v1 is left untouched.

Build progress:
  Step 1 [done]  Stages 1-3  — continuous ball trajectory
  Step 2 [done]  Stage 4     — kinematic contact candidates
  Step 3 [done]  Stages 0,5,6 — seed + sparse player detect + color filter
  Step 4 [todo]  Stages 0,7  — appearance gallery + orbital scoring
  Step 5 [todo]  Stages 8,9  — review montage + hotspot assembly
"""

__all__ = ["__version__"]

__version__ = "2.0.0-step3"

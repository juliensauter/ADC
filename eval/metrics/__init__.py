"""Metric wrappers (KID, LPIPS, DreamSim, Boundary IoU).

Each metric is implemented as a small adapter so the rest of the pipeline can
treat them uniformly. See `metrics_implementation_plan.md` §A.2–A.5 for
specs.

Placeholders below — real implementations land in subsequent commits.
"""

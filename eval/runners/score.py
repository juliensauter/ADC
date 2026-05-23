"""Generated batch + reference → all available metrics → records.

Handles paired metrics (LPIPS, DreamSim, Boundary IoU) and distributional
metrics (KID) separately per `metrics_implementation_plan.md` §A.6.
Implementation lands in commit A.6.
"""

# TODO(A.6): implement scoring runner with paired-vs-distributional split.

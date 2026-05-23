"""ADC evaluation pipeline.

See `working/notes/metrics_implementation_plan.md` for the executable plan and
`working/notes/metrics_kid_lpips.md` / `working/notes/metrics_mask_faithfulness.md`
for metric reference details.

Public API:
- `eval.cli` — command-line entrypoint (`python -m eval.cli ...`)
- `eval.metrics` — metric wrappers (KID, LPIPS, DreamSim, Boundary IoU)
- `eval.runners` — generation + scoring runners
"""

__version__ = "0.1.0"

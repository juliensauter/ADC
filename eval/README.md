# ADC eval module

Inference-time evaluation pipeline for ADC checkpoints. Computes KID-DINOv2,
LPIPS-AlexNet, DreamSim and (when a segmenter is available) Mask-IoU +
Boundary IoU. Writes JSON + CSV records suitable for A/B comparison across
checkpoints and feature flags.

## Usage (sketch — full implementation lands in Phase A.6)

```bash
# Single checkpoint
python -m eval.cli \
    --ckpt runs/paper_faithful_v2_polyp/version_0/checkpoints/last.ckpt \
    --preset paper_faithful_v2_polyp \
    --seed 42 --batch-size 2 --n-batches 4 --ddim-steps 50 \
    --out eval_runs/<timestamp>.json

# A/B comparison across branches
python -m eval.cli --ab \
    --ckpt-a <ckpt> --branch-a <pre-A6 commit> \
    --ckpt-b <ckpt> --branch-b <post-A6 commit> \
    --seed 42 --batch-size 2 --n-batches 4 \
    --out eval_runs/<timestamp>_ab.json
```

## Module layout

```
eval/
├── __init__.py
├── README.md          # this file
├── metrics/           # metric wrappers
│   ├── kid.py         # KID with DINOv2 ViT-L/14 features
│   ├── lpips.py       # LPIPS-AlexNet (torchmetrics)
│   ├── dreamsim.py    # DreamSim (NIGHTS-calibrated, NeurIPS 2023)
│   └── boundary_iou.py# Boundary IoU @ 5px (dormant until segmenter)
├── runners/
│   ├── generate.py    # checkpoint → seeded sample batch
│   └── score.py       # batch + reference → all metrics
└── cli.py             # argparse entrypoint
```

## Reference documents

- Plan: `working/notes/metrics_implementation_plan.md`
- KID / LPIPS / DreamSim background: `working/notes/metrics_kid_lpips.md`
- Mask-IoU / Boundary IoU background: `working/notes/metrics_mask_faithfulness.md`

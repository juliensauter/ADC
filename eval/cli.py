"""ADC eval CLI — `python -m eval.cli ...`.

Argparse entrypoint for single-checkpoint and A/B-branch evaluation runs.
See `working/notes/metrics_implementation_plan.md` §A.6 for the full
interface spec; for now this scaffolds `--help` so the module is importable
and the test `python -m eval.cli --help` succeeds.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser. Real flags fleshed out in Phase A.6."""
    parser = argparse.ArgumentParser(
        prog="eval.cli",
        description="ADC evaluation pipeline (KID-DINOv2, LPIPS, DreamSim, [Mask-IoU]).",
    )
    parser.add_argument("--ckpt", type=str, help="Path to ADC checkpoint (single-run mode).")
    parser.add_argument("--preset", type=str, help="ADC training preset name (e.g. paper_faithful_v2_polyp).")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed (default 42).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-batches", type=int, default=1)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--out", type=str, help="Output JSON path.")
    parser.add_argument("--ab", action="store_true", help="A/B mode (use --ckpt-a/--ckpt-b instead of --ckpt).")
    parser.add_argument("--ckpt-a", type=str, help="A/B mode: checkpoint A.")
    parser.add_argument("--branch-a", type=str, help="A/B mode: git branch / commit for side A.")
    parser.add_argument("--ckpt-b", type=str, help="A/B mode: checkpoint B.")
    parser.add_argument("--branch-b", type=str, help="A/B mode: git branch / commit for side B.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Phase A.1 scaffold: parsing only, no execution yet.
    print("eval.cli scaffold — Phase A.1 (not yet wired). Args:", vars(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())

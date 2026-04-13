"""
analyze_runs.py — Post-training analysis for ADC runs.

Scans runs/ directory, collects metrics, checkpoints, images, and generates
a comprehensive Markdown report at runs/training_report.md.

All text generation is template-based — no LLMs.

Usage:
    uv run python analyze_runs.py              # analyze all runs
    uv run python analyze_runs.py scratch      # analyze specific preset(s)
"""

import csv
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────────────
# Preset metadata (must match tutorial_train_single_gpu.py)
# ──────────────────────────────────────────────────────────────────────────────
PRESET_INFO = {
    "scratch":                   {"max_steps": 20000, "phase": "1",  "source": "SD v1.5"},
    "polyp_transfer":            {"max_steps": 20000, "phase": "1",  "source": "ADC polyp"},
    "scratch_unlocked":          {"max_steps": 10000, "phase": "1b", "source": "scratch"},
    "polyp_unlocked":            {"max_steps": 10000, "phase": "1b", "source": "polyp_transfer"},
    "polyp_stage2":              {"max_steps": 10000, "phase": "2",  "source": "polyp_transfer"},
    "scratch_stage2":            {"max_steps": 10000, "phase": "2",  "source": "scratch_unlocked"},
    "polyp_stage2_from_unlocked":{"max_steps": 10000, "phase": "2",  "source": "polyp_unlocked"},
}


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_dir_size(path: str) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def read_metrics(preset_name: str) -> dict:
    """Read CSVLogger metrics for a preset.

    Returns dict with keys: steps (list), losses (list), first_step, last_step,
    first_loss, last_loss, min_loss, epochs.
    """
    result = {
        "steps": [], "losses": [], "epochs": set(),
        "first_step": None, "last_step": None,
        "first_loss": None, "last_loss": None, "min_loss": None,
    }

    # Find metrics CSVs (may be in version_0, version_1, etc.)
    csv_files = sorted(glob.glob(f"runs/{preset_name}/version_*/metrics.csv"))
    if not csv_files:
        # Try legacy path
        csv_files = sorted(glob.glob(f"runs/{preset_name}/*/metrics.csv"))
    if not csv_files:
        return result

    for csv_path in csv_files:
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    step = int(row.get("step", 0))
                    loss = row.get("train/loss_simple") or row.get("train/loss")
                    epoch = row.get("epoch")
                    if loss:
                        try:
                            loss_val = float(loss)
                            result["steps"].append(step)
                            result["losses"].append(loss_val)
                        except ValueError:
                            pass
                    if epoch:
                        try:
                            result["epochs"].add(int(float(epoch)))
                        except ValueError:
                            pass
        except (OSError, csv.Error):
            pass

    if result["steps"]:
        result["first_step"] = result["steps"][0]
        result["last_step"] = result["steps"][-1]
        result["first_loss"] = result["losses"][0]
        result["last_loss"] = result["losses"][-1]
        result["min_loss"] = min(result["losses"])
    result["epochs"] = len(result["epochs"])

    return result


def analyze_preset(preset_name: str) -> dict:
    """Analyze a single preset's outputs.

    Returns a dict with all analysis data.
    """
    info = PRESET_INFO.get(preset_name, {"max_steps": 0, "phase": "?", "source": "?"})
    run_dir = f"runs/{preset_name}"

    analysis = {
        "name": preset_name,
        "exists": os.path.isdir(run_dir),
        "phase": info["phase"],
        "source": info["source"],
        "max_steps": info["max_steps"],
    }

    if not analysis["exists"]:
        analysis["status"] = "not started"
        return analysis

    # Checkpoints
    ckpt_files = sorted(glob.glob(f"{run_dir}/*/checkpoints/*.ckpt"))
    analysis["checkpoints"] = ckpt_files
    analysis["checkpoint_count"] = len(ckpt_files)
    analysis["checkpoint_size"] = sum(os.path.getsize(f) for f in ckpt_files if os.path.isfile(f))

    last_ckpt = None
    for c in ckpt_files:
        if c.endswith("last.ckpt"):
            last_ckpt = c
    analysis["last_ckpt"] = last_ckpt

    # Images
    image_files = sorted(glob.glob(f"{run_dir}/image_log/train/*.png"))
    analysis["image_count"] = len(image_files)
    analysis["image_size"] = sum(os.path.getsize(f) for f in image_files if os.path.isfile(f))

    # Timestamps
    if image_files:
        first_img_time = os.path.getmtime(image_files[0])
        last_img_time = os.path.getmtime(image_files[-1])
        analysis["first_image_time"] = datetime.fromtimestamp(first_img_time)
        analysis["last_image_time"] = datetime.fromtimestamp(last_img_time)
        analysis["training_duration_min"] = (last_img_time - first_img_time) / 60
    else:
        analysis["training_duration_min"] = 0

    # Metrics
    metrics = read_metrics(preset_name)
    analysis.update(metrics)

    # Total size
    analysis["total_size"] = get_dir_size(run_dir)

    # Status
    if metrics["last_step"] is not None and metrics["last_step"] >= info["max_steps"]:
        analysis["status"] = "complete"
    elif metrics["last_step"] is not None:
        analysis["status"] = f"in progress ({metrics['last_step']}/{info['max_steps']} steps)"
    elif last_ckpt:
        analysis["status"] = "has checkpoint (no metrics)"
    else:
        analysis["status"] = "started (no checkpoint yet)"

    return analysis


def generate_report(analyses: list[dict]) -> str:
    """Generate Markdown report from analysis data."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append(f"# ADC Training Report")
    lines.append(f"")
    lines.append(f"Generated: {now}")
    lines.append(f"")

    # ── Summary table ──
    lines.append("## Summary")
    lines.append("")
    lines.append("| Preset | Phase | Status | Steps | Final Loss | Min Loss | Images | Disk |")
    lines.append("|--------|-------|--------|-------|------------|----------|--------|------|")

    for a in analyses:
        if not a["exists"]:
            lines.append(f"| {a['name']} | {a['phase']} | not started | — | — | — | — | — |")
            continue

        steps = f"{a.get('last_step', '—')}/{a['max_steps']}" if a.get('last_step') else "—"
        final_loss = f"{a['last_loss']:.4f}" if a.get('last_loss') is not None else "—"
        min_loss = f"{a['min_loss']:.4f}" if a.get('min_loss') is not None else "—"
        images = str(a.get('image_count', 0))
        disk = format_size(a.get('total_size', 0))

        lines.append(f"| {a['name']} | {a['phase']} | {a['status']} | {steps} | {final_loss} | {min_loss} | {images} | {disk} |")

    lines.append("")

    # ── Dependency graph ──
    lines.append("## Training Paths")
    lines.append("")
    lines.append("```")
    lines.append("Phase 1 (base)          Phase 1b (unlock)        Phase 2 (full ADC)")
    lines.append("─────────────────       ──────────────────       ──────────────────")
    lines.append("scratch ──────────────→ scratch_unlocked ──────→ scratch_stage2")
    lines.append("polyp_transfer ──────→ polyp_unlocked ────────→ polyp_stage2_from_unlocked")
    lines.append("           └──────────────────────────────────→ polyp_stage2")
    lines.append("```")
    lines.append("")

    # ── Per-preset details ──
    lines.append("## Preset Details")
    lines.append("")

    for a in analyses:
        lines.append(f"### {a['name']}")
        lines.append("")

        if not a["exists"]:
            lines.append("Not started.")
            lines.append("")
            continue

        lines.append(f"- **Status:** {a['status']}")
        lines.append(f"- **Phase:** {a['phase']} | **Source:** {a['source']}")
        lines.append(f"- **Max steps:** {a['max_steps']}")

        if a.get('last_step') is not None:
            lines.append(f"- **Steps completed:** {a['last_step']}")
            lines.append(f"- **Epochs:** {a.get('epochs', '?')}")

        if a.get('first_loss') is not None:
            lines.append(f"- **Loss:** {a['first_loss']:.4f} (start) → {a['last_loss']:.4f} (end) | min: {a['min_loss']:.4f}")

        lines.append(f"- **Checkpoints:** {a.get('checkpoint_count', 0)} files ({format_size(a.get('checkpoint_size', 0))})")
        lines.append(f"- **Images:** {a.get('image_count', 0)} ({format_size(a.get('image_size', 0))})")

        if a.get('training_duration_min', 0) > 0:
            hrs = a['training_duration_min'] / 60
            lines.append(f"- **Training time:** ~{hrs:.1f} hours")

        lines.append(f"- **Total disk:** {format_size(a.get('total_size', 0))}")

        if a.get('last_ckpt'):
            lines.append(f"- **Latest checkpoint:** `{a['last_ckpt']}`")

        lines.append("")

    # ── File inventory ──
    lines.append("## File Inventory")
    lines.append("")
    lines.append("```")

    total_disk = 0
    for a in analyses:
        if not a["exists"]:
            continue
        lines.append(f"runs/{a['name']}/")
        # List version directories
        versions = sorted(glob.glob(f"runs/{a['name']}/version_*"))
        for v in versions:
            vname = os.path.basename(v)
            # Checkpoints
            ckpts = sorted(glob.glob(f"{v}/checkpoints/*.ckpt"))
            if ckpts:
                lines.append(f"  {vname}/checkpoints/")
                for c in ckpts:
                    size = format_size(os.path.getsize(c))
                    lines.append(f"    {os.path.basename(c)}  ({size})")
            # Metrics
            metrics_csv = f"{v}/metrics.csv"
            if os.path.isfile(metrics_csv):
                lines.append(f"  {vname}/metrics.csv")
        # Image log
        img_dir = f"runs/{a['name']}/image_log/train"
        if os.path.isdir(img_dir):
            img_count = len(glob.glob(f"{img_dir}/*.png"))
            img_size = format_size(sum(os.path.getsize(f) for f in glob.glob(f"{img_dir}/*.png")))
            lines.append(f"  image_log/train/  ({img_count} images, {img_size})")
        lines.append("")
        total_disk += a.get('total_size', 0)

    lines.append(f"Total disk usage: {format_size(total_disk)}")
    lines.append("```")
    lines.append("")

    # ── Cross-preset comparison ──
    completed = [a for a in analyses if a.get('status') == 'complete']
    if len(completed) >= 2:
        lines.append("## Cross-Preset Comparison")
        lines.append("")
        lines.append("Completed presets ranked by final loss (lower is better):")
        lines.append("")
        ranked = sorted(completed, key=lambda a: a.get('last_loss', float('inf')))
        for i, a in enumerate(ranked, 1):
            lines.append(f"{i}. **{a['name']}** — final loss: {a['last_loss']:.4f}, min: {a['min_loss']:.4f}")
        lines.append("")

    return "\n".join(lines)


def main():
    # Determine which presets to analyze
    if len(sys.argv) > 1:
        preset_names = sys.argv[1:]
    else:
        preset_names = list(PRESET_INFO.keys())

    print(f"Analyzing {len(preset_names)} presets...")

    analyses = []
    for name in preset_names:
        analysis = analyze_preset(name)
        analyses.append(analysis)
        status_icon = {"complete": "✓", "not started": "·"}.get(analysis["status"], "…")
        print(f"  {status_icon} {name}: {analysis['status']}")

    # Generate report
    report = generate_report(analyses)

    # Write report
    os.makedirs("runs", exist_ok=True)
    report_path = "runs/training_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport written to: {report_path}")
    print(f"  {len(analyses)} presets analyzed")
    print(f"  {sum(1 for a in analyses if a.get('status') == 'complete')} complete")
    print(f"  {sum(1 for a in analyses if a.get('status') == 'not started')} not started")


if __name__ == "__main__":
    main()

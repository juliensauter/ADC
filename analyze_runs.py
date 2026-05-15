"""
analyze_runs.py — Post-training analysis for ADC runs.

Scans runs/ directory, collects metrics, checkpoints, images, and generates
a comprehensive Markdown report at runs/training_report.md.

Fold-aware behavior:
  - single runs are read from runs/<preset>/version_*/metrics.csv
  - k-fold runs are read from runs/<preset>/fold_<i>/version_*/metrics.csv
  - validation metrics (val/kid, val/lpips, val/composite) are summarized per
    run and aggregated across folds with mean/std

All text generation is template-based — no LLMs.

Usage:
    uv run python analyze_runs.py              # analyze all known presets
    uv run python analyze_runs.py scratch      # analyze specific preset(s)
"""

import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean, stdev

from experiment_config import PRESET_MAX_STEPS

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────────────
# Preset metadata (must match experiment_config.py / tutorial_train_single_gpu.py)
# ──────────────────────────────────────────────────────────────────────────────
PRESET_INFO = {
    "scratch": {"phase": "1", "source": "SD v1.5"},
    "polyp_transfer": {"phase": "1", "source": "ADC polyp"},
    "scratch_unlocked": {"phase": "1b", "source": "scratch"},
    "polyp_unlocked": {"phase": "1b", "source": "polyp_transfer"},
    "polyp_stage2": {"phase": "2", "source": "polyp_transfer"},
    "scratch_stage2": {"phase": "2", "source": "scratch_unlocked"},
    "polyp_stage2_from_unlocked": {"phase": "2", "source": "polyp_unlocked"},
    "paper_faithful_polyp": {"phase": "paper-faithful", "source": "ADC polyp"},
    "paper_faithful_scratch": {"phase": "paper-faithful", "source": "SD v1.5"},
    "paper_faithful_v2_polyp": {"phase": "paper-faithful", "source": "ADC polyp"},
}

METRIC_COLUMNS = ("train/loss_simple", "train/loss", "val/kid", "val/lpips", "val/composite")


def get_preset_metadata(preset_name: str) -> dict:
    info = PRESET_INFO.get(preset_name, {})
    return {
        "phase": info.get("phase", "?"),
        "source": info.get("source", "?"),
        "max_steps": PRESET_MAX_STEPS.get(preset_name, 0),
    }


def is_sanity_path(path: Path) -> bool:
    return "sanity" in path.parts


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def parse_step(value: str | None) -> int | None:
    numeric = parse_float(value)
    if numeric is None:
        return None
    return int(numeric)


def summarize_values(values: list[float]) -> dict:
    clean_values = [value for value in values if isinstance(value, (int, float)) and not math.isnan(value)]
    if not clean_values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(clean_values),
        "mean": fmean(clean_values),
        "std": stdev(clean_values) if len(clean_values) > 1 else 0.0,
        "min": min(clean_values),
        "max": max(clean_values),
    }


def format_mean_std(mean_value: float | None, std_value: float | None, digits: int = 4) -> str:
    if mean_value is None:
        return "—"
    if std_value is None or std_value == 0:
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} ± {std_value:.{digits}f}"


def format_metric_stat(stat: dict | None, key: str = "best", digits: int = 4) -> str:
    if not stat:
        return "—"
    value = stat.get(key)
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


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


def run_root_from_metrics(metrics_path: Path) -> Path:
    return metrics_path.parent.parent


def run_root_from_checkpoint(ckpt_path: Path) -> Path:
    return ckpt_path.parent.parent


def run_root_from_image(path: Path, marker: str) -> Path:
    marker_index = path.parts.index(marker)
    return Path(*path.parts[:marker_index])


def collect_run_roots(preset_dir: Path) -> list[Path]:
    roots: set[Path] = set()
    if not preset_dir.exists():
        return []

    for metrics_path in preset_dir.rglob("metrics.csv"):
        if is_sanity_path(metrics_path):
            continue
        roots.add(run_root_from_metrics(metrics_path))

    for ckpt_path in preset_dir.rglob("*.ckpt"):
        if is_sanity_path(ckpt_path):
            continue
        roots.add(run_root_from_checkpoint(ckpt_path))

    for image_path in preset_dir.rglob("image_log/train/*.png"):
        if is_sanity_path(image_path):
            continue
        roots.add(run_root_from_image(image_path, "image_log"))

    for image_path in preset_dir.rglob("validation_metrics/**/*.png"):
        if is_sanity_path(image_path):
            continue
        roots.add(run_root_from_image(image_path, "validation_metrics"))

    for child in preset_dir.iterdir():
        if child.is_dir() and child.name.startswith("fold_"):
            roots.add(child)

    if not roots:
        roots.add(preset_dir)

    return sorted(roots)


def read_metrics(run_root: Path) -> dict:
    """Read CSVLogger metrics for one logical run root."""
    result = {
        "steps": [],
        "losses": [],
        "epochs": set(),
        "first_step": None,
        "last_step": None,
        "first_loss": None,
        "last_loss": None,
        "min_loss": None,
        "metric_stats": {},
        "loss_stats": None,
    }

    csv_files = sorted(run_root.glob("version_*/metrics.csv"))
    if not csv_files:
        csv_files = [path for path in sorted(run_root.glob("*/metrics.csv")) if not is_sanity_path(path)]
    if not csv_files:
        result["epochs"] = 0
        return result

    metric_series: dict[str, list[tuple[int | None, float]]] = defaultdict(list)
    all_steps: list[int] = []
    loss_key: str | None = None

    for csv_path in csv_files:
        try:
            with open(csv_path) as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    step = parse_step(row.get("step"))
                    if step is not None:
                        all_steps.append(step)

                    epoch = parse_step(row.get("epoch"))
                    if epoch is not None:
                        result["epochs"].add(epoch)

                    row_loss_key = None
                    row_loss_value = parse_float(row.get("train/loss_simple"))
                    if row_loss_value is not None:
                        row_loss_key = "train/loss_simple"
                    else:
                        row_loss_value = parse_float(row.get("train/loss"))
                        if row_loss_value is not None:
                            row_loss_key = "train/loss"

                    if row_loss_value is not None and row_loss_key is not None:
                        loss_key = loss_key or row_loss_key
                        metric_series[row_loss_key].append((step, row_loss_value))
                        result["steps"].append(step if step is not None else 0)
                        result["losses"].append(row_loss_value)

                    for metric_name in METRIC_COLUMNS[2:]:
                        metric_value = parse_float(row.get(metric_name))
                        if metric_value is not None:
                            metric_series[metric_name].append((step, metric_value))
        except (OSError, csv.Error):
            pass

    if all_steps:
        result["first_step"] = min(all_steps)
        result["last_step"] = max(all_steps)

    if result["steps"]:
        result["first_loss"] = result["losses"][0]
        result["last_loss"] = result["losses"][-1]
        result["min_loss"] = min(result["losses"])

    result["epochs"] = len(result["epochs"])

    metric_stats = {}
    for metric_name, series in metric_series.items():
        values = [value for _, value in series]
        if not values:
            continue
        best_index = min(range(len(values)), key=lambda index: values[index])
        metric_stats[metric_name] = {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "best": values[best_index],
            "best_step": series[best_index][0],
        }

    result["metric_stats"] = metric_stats
    if loss_key and loss_key in metric_stats:
        result["loss_stats"] = metric_stats[loss_key]
    elif metric_stats.get("train/loss_simple"):
        result["loss_stats"] = metric_stats["train/loss_simple"]
    else:
        result["loss_stats"] = metric_stats.get("train/loss")

    return result


def run_name_for_root(run_root: Path, preset_dir: Path, preset_name: str) -> str:
    try:
        relative = run_root.relative_to(preset_dir)
    except ValueError:
        return run_root.name
    return preset_name if str(relative) == "." else str(relative)


def analyze_run_root(preset_name: str, run_root: Path) -> dict:
    info = get_preset_metadata(preset_name)
    preset_dir = Path("runs") / preset_name
    run_name = run_name_for_root(run_root, preset_dir, preset_name)

    analysis = {
        "name": run_name,
        "run_root": str(run_root),
        "exists": run_root.is_dir(),
        "phase": info["phase"],
        "source": info["source"],
        "max_steps": info["max_steps"],
    }

    if not analysis["exists"]:
        analysis["status"] = "not started"
        return analysis

    ckpt_files = sorted(run_root.glob("*/checkpoints/*.ckpt"))
    train_images = sorted(run_root.glob("image_log/train/*.png"))
    validation_images = sorted(run_root.glob("validation_metrics/**/*.png"))

    analysis["checkpoints"] = [str(path) for path in ckpt_files]
    analysis["checkpoint_count"] = len(ckpt_files)
    analysis["checkpoint_size"] = sum(path.stat().st_size for path in ckpt_files if path.is_file())
    analysis["last_ckpt"] = next((str(path) for path in ckpt_files if path.name == "last.ckpt"), None)
    analysis["best_ckpt"] = next((str(path) for path in ckpt_files if path.name == "best.ckpt"), None)

    analysis["image_count"] = len(train_images)
    analysis["image_size"] = sum(path.stat().st_size for path in train_images if path.is_file())
    analysis["validation_image_count"] = len(validation_images)
    analysis["validation_image_size"] = sum(path.stat().st_size for path in validation_images if path.is_file())

    if train_images:
        first_img_time = train_images[0].stat().st_mtime
        last_img_time = train_images[-1].stat().st_mtime
        analysis["first_image_time"] = datetime.fromtimestamp(first_img_time)
        analysis["last_image_time"] = datetime.fromtimestamp(last_img_time)
        analysis["training_duration_min"] = (last_img_time - first_img_time) / 60
    else:
        analysis["training_duration_min"] = 0

    metrics = read_metrics(run_root)
    analysis.update(metrics)
    analysis["total_size"] = get_dir_size(str(run_root))

    if analysis.get("last_step") is not None and analysis["last_step"] >= info["max_steps"]:
        analysis["status"] = "complete"
    elif analysis.get("last_step") is not None:
        analysis["status"] = f"in progress ({analysis['last_step']}/{info['max_steps']} steps)"
    elif analysis["checkpoint_count"] > 0:
        analysis["status"] = "has checkpoint (no metrics)"
    else:
        analysis["status"] = "started (no checkpoint yet)"

    return analysis


def aggregate_metric_stats(run_analyses: list[dict], metric_name: str, stat_name: str = "best") -> dict:
    values = []
    for analysis in run_analyses:
        metric_stats = analysis.get("metric_stats", {})
        metric_stat = metric_stats.get(metric_name)
        if metric_stat is None:
            continue
        value = metric_stat.get(stat_name)
        if value is not None:
            values.append(value)
    return summarize_values(values)


def aggregate_loss_stats(run_analyses: list[dict], stat_name: str = "last") -> dict:
    values = []
    for analysis in run_analyses:
        loss_stat = analysis.get("loss_stats")
        if not loss_stat:
            continue
        value = loss_stat.get(stat_name)
        if value is not None:
            values.append(value)
    return summarize_values(values)


def aggregate_run_analyses(run_analyses: list[dict]) -> dict:
    aggregate = {
        "run_count": len(run_analyses),
        "fold_count": sum(1 for analysis in run_analyses if analysis["name"].startswith("fold_")),
        "complete_count": sum(1 for analysis in run_analyses if analysis.get("status") == "complete"),
        "checkpoint_count": sum(analysis.get("checkpoint_count", 0) for analysis in run_analyses),
        "checkpoint_size": sum(analysis.get("checkpoint_size", 0) for analysis in run_analyses),
        "image_count": sum(analysis.get("image_count", 0) for analysis in run_analyses),
        "image_size": sum(analysis.get("image_size", 0) for analysis in run_analyses),
        "validation_image_count": sum(analysis.get("validation_image_count", 0) for analysis in run_analyses),
        "validation_image_size": sum(analysis.get("validation_image_size", 0) for analysis in run_analyses),
        "total_size": sum(analysis.get("total_size", 0) for analysis in run_analyses),
        "steps": summarize_values([analysis.get("last_step") for analysis in run_analyses if analysis.get("last_step") is not None]),
        "epochs": summarize_values([analysis.get("epochs") for analysis in run_analyses if analysis.get("epochs") is not None]),
        "final_loss": aggregate_loss_stats(run_analyses, "last"),
        "min_loss": aggregate_loss_stats(run_analyses, "min"),
        "best_val_kid": aggregate_metric_stats(run_analyses, "val/kid", "best"),
        "last_val_kid": aggregate_metric_stats(run_analyses, "val/kid", "last"),
        "best_val_lpips": aggregate_metric_stats(run_analyses, "val/lpips", "best"),
        "last_val_lpips": aggregate_metric_stats(run_analyses, "val/lpips", "last"),
        "best_val_composite": aggregate_metric_stats(run_analyses, "val/composite", "best"),
        "last_val_composite": aggregate_metric_stats(run_analyses, "val/composite", "last"),
    }

    if not run_analyses:
        aggregate["status"] = "not started"
    elif aggregate["complete_count"] == aggregate["run_count"]:
        aggregate["status"] = "complete"
    elif aggregate["complete_count"] > 0:
        aggregate["status"] = f"in progress ({aggregate['complete_count']}/{aggregate['run_count']} runs complete)"
    elif any(analysis.get("checkpoint_count", 0) > 0 or analysis.get("last_step") is not None for analysis in run_analyses):
        aggregate["status"] = "in progress"
    else:
        aggregate["status"] = "started (no checkpoint yet)"

    return aggregate


def analyze_preset(preset_name: str) -> dict:
    """Analyze a preset's outputs, including all fold-specific run roots."""
    preset_dir = Path("runs") / preset_name
    info = get_preset_metadata(preset_name)
    run_roots = collect_run_roots(preset_dir)
    run_analyses = [analyze_run_root(preset_name, run_root) for run_root in run_roots]

    analysis = {
        "name": preset_name,
        "exists": preset_dir.exists(),
        "phase": info["phase"],
        "source": info["source"],
        "max_steps": info["max_steps"],
        "runs": run_analyses,
    }
    analysis.update(aggregate_run_analyses(run_analyses))
    return analysis


def report_stat(stat: dict | None, digits: int = 4) -> str:
    if not stat:
        return "—"
    return format_mean_std(stat.get("mean"), stat.get("std"), digits=digits)


def best_score_for_comparison(analysis: dict) -> float:
    aggregate = analysis.get("best_val_composite") or {}
    score = aggregate.get("mean")
    if score is not None:
        return score
    aggregate = analysis.get("final_loss") or {}
    score = aggregate.get("mean")
    if score is not None:
        return score
    return float("inf")


def generate_report(analyses: list[dict]) -> str:
    """Generate Markdown report from analysis data."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# ADC Training Report")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Preset | Phase | Status | Runs | Steps | Final Loss | Best Val Comp | Best Val KID | Best Val LPIPS | Disk |")
    lines.append("|--------|-------|--------|------|-------|------------|---------------|--------------|---------------|------|")

    for a in analyses:
        if not a["exists"]:
            lines.append(f"| {a['name']} | {a['phase']} | not started | 0 | — | — | — | — | — | — |")
            continue

        aggregate = a.get("steps", {})
        steps = report_stat(aggregate)
        final_loss = report_stat(a.get("final_loss"))
        best_val_composite = report_stat(a.get("best_val_composite"))
        best_val_kid = report_stat(a.get("best_val_kid"))
        best_val_lpips = report_stat(a.get("best_val_lpips"))
        disk = format_size(a.get("total_size", 0))

        lines.append(
            f"| {a['name']} | {a['phase']} | {a['status']} | {a.get('run_count', 0)} | {steps} | {final_loss} | {best_val_composite} | {best_val_kid} | {best_val_lpips} | {disk} |"
        )

    lines.append("")

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
        lines.append(f"- **Runs:** {a.get('run_count', 0)} | **Fold runs:** {a.get('fold_count', 0)}")

        if a.get("steps", {}).get("mean") is not None:
            lines.append(f"- **Steps completed:** {report_stat(a.get('steps'))}")

        if a.get("epochs", {}).get("mean") is not None:
            lines.append(f"- **Epochs:** {report_stat(a.get('epochs'), digits=1)}")

        if a.get("final_loss", {}).get("mean") is not None:
            lines.append(f"- **Loss:** {report_stat(a.get('final_loss'))} (final) | min: {report_stat(a.get('min_loss'))}")

        if a.get("best_val_composite", {}).get("mean") is not None:
            lines.append(f"- **Best val/composite:** {report_stat(a.get('best_val_composite'))}")
        if a.get("best_val_kid", {}).get("mean") is not None:
            lines.append(f"- **Best val/KID:** {report_stat(a.get('best_val_kid'))}")
        if a.get("best_val_lpips", {}).get("mean") is not None:
            lines.append(f"- **Best val/LPIPS:** {report_stat(a.get('best_val_lpips'))}")

        lines.append(f"- **Checkpoints:** {a.get('checkpoint_count', 0)} files ({format_size(a.get('checkpoint_size', 0))})")
        lines.append(f"- **Training images:** {a.get('image_count', 0)} ({format_size(a.get('image_size', 0))})")
        lines.append(f"- **Validation images:** {a.get('validation_image_count', 0)} ({format_size(a.get('validation_image_size', 0))})")

        if a.get("training_duration_min", 0) > 0:
            hrs = a["training_duration_min"] / 60
            lines.append(f"- **Training time:** ~{hrs:.1f} hours")

        lines.append(f"- **Total disk:** {format_size(a.get('total_size', 0))}")

        if a.get("last_ckpt"):
            lines.append(f"- **Last checkpoint:** {a['last_ckpt']}")
        if a.get("best_ckpt"):
            lines.append(f"- **Best checkpoint:** {a['best_ckpt']}")

        lines.append("")

        for run in a.get("runs", []):
            lines.append(f"#### {run['name']}")
            lines.append("")
            lines.append(f"- **Run root:** {run['run_root']}")
            lines.append(f"- **Status:** {run['status']}")
            if run.get("last_step") is not None:
                lines.append(f"- **Steps completed:** {run['last_step']}")
            if run.get("epochs") is not None:
                lines.append(f"- **Epochs:** {run.get('epochs', '?')}")
            if run.get("loss_stats"):
                loss_stats = run["loss_stats"]
                lines.append(
                    f"- **Loss:** {format_metric_stat(loss_stats, 'first')} (start) → {format_metric_stat(loss_stats, 'last')} (end) | min: {format_metric_stat(loss_stats, 'min')}"
                )
            for metric_name, label in (("val/composite", "val/composite"), ("val/kid", "val/KID"), ("val/lpips", "val/LPIPS")):
                metric_stats = run.get("metric_stats", {}).get(metric_name)
                if metric_stats:
                    lines.append(
                        f"- **{label}:** {format_metric_stat(metric_stats, 'first')} → {format_metric_stat(metric_stats, 'last')} | best: {format_metric_stat(metric_stats, 'best')} @ step {metric_stats.get('best_step', '—')}"
                    )
            lines.append(f"- **Checkpoints:** {run.get('checkpoint_count', 0)} files ({format_size(run.get('checkpoint_size', 0))})")
            lines.append(f"- **Training images:** {run.get('image_count', 0)} ({format_size(run.get('image_size', 0))})")
            lines.append(f"- **Validation images:** {run.get('validation_image_count', 0)} ({format_size(run.get('validation_image_size', 0))})")
            if run.get("last_ckpt"):
                lines.append(f"- **Last checkpoint:** {run['last_ckpt']}")
            if run.get("best_ckpt"):
                lines.append(f"- **Best checkpoint:** {run['best_ckpt']}")
            lines.append("")

    lines.append("## File Inventory")
    lines.append("")
    lines.append("```")

    total_disk = 0
    for a in analyses:
        if not a["exists"]:
            continue

        lines.append(f"runs/{a['name']}/")
        for run in a.get("runs", []):
            run_root = Path(run["run_root"])
            run_prefix = f"  {run['name']}/" if run['name'] != a['name'] else "  "
            lines.append(f"{run_prefix}{run_root.name if run_root.name != a['name'] else ''}")

            versions = sorted(run_root.glob("version_*"))
            for version_dir in versions:
                version_name = version_dir.name
                ckpts = sorted(version_dir.glob("checkpoints/*.ckpt"))
                if ckpts:
                    lines.append(f"    {version_name}/checkpoints/")
                    for ckpt_path in ckpts:
                        lines.append(f"      {ckpt_path.name}  ({format_size(ckpt_path.stat().st_size)})")
                metrics_csv = version_dir / "metrics.csv"
                if metrics_csv.is_file():
                    lines.append(f"    {version_name}/metrics.csv")

            if run.get("image_count", 0) > 0:
                lines.append(f"    image_log/train/  ({run['image_count']} images, {format_size(run.get('image_size', 0))})")
            if run.get("validation_image_count", 0) > 0:
                lines.append(f"    validation_metrics/  ({run['validation_image_count']} images, {format_size(run.get('validation_image_size', 0))})")
            lines.append("")

            total_disk += run.get("total_size", 0)

        lines.append("")

    lines.append(f"Total disk usage: {format_size(total_disk)}")
    lines.append("```")
    lines.append("")

    completed = [a for a in analyses if a.get("aggregate", a).get("status") == "complete"]
    if len(completed) >= 2:
        lines.append("## Cross-Preset Comparison")
        lines.append("")
        lines.append("Completed presets ranked by best val/composite (lower is better):")
        lines.append("")
        ranked = sorted(completed, key=best_score_for_comparison)
        for index, a in enumerate(ranked, 1):
            best_comp = report_stat(a.get("best_val_composite"))
            final_loss = report_stat(a.get("final_loss"))
            lines.append(f"{index}. **{a['name']}** — best val/composite: {best_comp}, final loss: {final_loss}")
        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        preset_names = sys.argv[1:]
    else:
        preset_names = list(PRESET_INFO.keys())

    print(f"Analyzing {len(preset_names)} presets...")

    analyses = []
    for name in preset_names:
        analysis = analyze_preset(name)
        analyses.append(analysis)
        status_icon = {"complete": "✓", "not started": "·"}.get(analysis.get("status", ""), "…")
        print(f"  {status_icon} {name}: {analysis.get('status', 'unknown')}")

    report = generate_report(analyses)

    os.makedirs("runs", exist_ok=True)
    report_path = "runs/training_report.md"
    with open(report_path, "w") as handle:
        handle.write(report)

    print(f"\nReport written to: {report_path}")
    print(f"  {len(analyses)} presets analyzed")
    print(f"  {sum(1 for a in analyses if a.get('status') == 'complete')} complete")
    print(f"  {sum(1 for a in analyses if a.get('status') == 'not started')} not started")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze hits/visibility CSV logs produced during training."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def load_rows(csv_path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with csv_path.open("r", encoding="ascii", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(
                {
                    "split": row["split"],
                    "global_step": _to_float(row, "global_step"),
                    "hits_sum": _to_float(row, "hits_sum"),
                    "visible_gaussians": _to_float(row, "visible_gaussians"),
                    "visible_gaussians_frac": _to_float(row, "visible_gaussians_frac"),
                    "forward_render_ms": _to_float(row, "forward_render_ms"),
                }
            )
    return rows


def summarize_split(split: str, rows: list[dict[str, float | str]]) -> str:
    steps = [int(row["global_step"]) for row in rows]
    hits = [float(row["hits_sum"]) for row in rows]
    visible = [float(row["visible_gaussians"]) for row in rows]
    frac = [float(row["visible_gaussians_frac"]) for row in rows]
    render_ms = [float(row["forward_render_ms"]) for row in rows if not math.isnan(float(row["forward_render_ms"]))]

    last = rows[-1]
    lines = [
        f"[{split}] rows={len(rows)} steps={steps[0]}..{steps[-1]}",
        (
            "  hits_sum:"
            f" mean={mean(hits):.3f} min={min(hits):.3f} max={max(hits):.3f} last={float(last['hits_sum']):.3f}"
        ),
        (
            "  visible_gaussians:"
            f" mean={mean(visible):.3f} min={min(visible):.3f} max={max(visible):.3f}"
            f" last={float(last['visible_gaussians']):.3f}"
        ),
        (
            "  visible_gaussians_frac:"
            f" mean={mean(frac):.6f} min={min(frac):.6f} max={max(frac):.6f}"
            f" last={float(last['visible_gaussians_frac']):.6f}"
        ),
    ]
    if render_ms:
        lines.append(
            "  forward_render_ms:"
            f" mean={mean(render_ms):.3f} min={min(render_ms):.3f} max={max(render_ms):.3f}"
            f" last={float(last['forward_render_ms']):.3f}"
        )
    return "\n".join(lines)


def maybe_plot(rows_by_split: dict[str, list[dict[str, float | str]]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"Plotting requested, but matplotlib is not installed: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    series = [
        ("hits_sum", "Hits Sum"),
        ("visible_gaussians", "Visible Gaussians"),
        ("visible_gaussians_frac", "Visible Gaussian Fraction"),
        ("forward_render_ms", "Forward Render (ms)"),
    ]

    for key, title in series:
        plt.figure(figsize=(8, 4.5))
        plotted = False
        for split, rows in rows_by_split.items():
            xs = [float(row["global_step"]) for row in rows]
            ys = [float(row[key]) for row in rows]
            if all(math.isnan(y) for y in ys):
                continue
            plt.plot(xs, ys, marker="o", markersize=2, linewidth=1.2, label=split)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.title(title)
        plt.xlabel("Global Step")
        plt.ylabel(key)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to hits_visibility.csv")
    parser.add_argument("--split", choices=["training", "validation"], help="Only analyze one split")
    parser.add_argument("--plot-dir", type=Path, help="Optional directory to save plots")
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    if not rows:
        raise SystemExit(f"No data rows found in {args.csv_path}")

    rows_by_split: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        rows_by_split[str(row["split"])].append(row)

    splits = [args.split] if args.split else sorted(rows_by_split)
    for split in splits:
        split_rows = rows_by_split.get(split, [])
        if not split_rows:
            print(f"[{split}] no rows found")
            continue
        print(summarize_split(split, split_rows))

    if args.plot_dir is not None:
        filtered = {split: rows_by_split[split] for split in splits if split in rows_by_split}
        maybe_plot(filtered, args.plot_dir)
        print(f"Saved plots to {args.plot_dir}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Plot decode throughput vs batch size (one line per device) from sweep CSVs.

  python bench/plot_sweep.py --csv results/sweep_cuda.csv --csv results/sweep_cpu.csv --out results/sweep.png

Colors are the dataviz reference palette's validated categorical slots 1 (blue)
and 2 (orange); GPU is the hero series.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
COLORS = {"cuda": "#2a78d6", "cpu": "#eb6834"}   # validated slot 1 / slot 2
LABELS = {"cuda": "GPU (H100)", "cpu": "CPU"}


def load(path: str):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    dev = rows[0]["device"]
    xs = [int(r["batch_size"]) for r in rows]
    ys = [float(r["tokens_per_s"]) for r in rows]
    return dev, xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="sweep CSV; repeat per device")
    ap.add_argument("--out", default="results/sweep.png")
    ap.add_argument("--title", default="Decode throughput vs batch size — GPU scales, CPU plateaus\nQwen3-0.6B, 32 new tokens/request")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    all_x: set[int] = set()
    for path in args.csv:
        dev, xs, ys = load(path)
        all_x.update(xs)
        c = COLORS.get(dev, "#888888")
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=LABELS.get(dev, dev), zorder=3)
        # selective direct label: endpoint throughput only
        ax.annotate(f"{ys[-1]:,.0f} tok/s", xy=(xs[-1], ys[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=c, fontsize=10, fontweight="bold", va="center")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    xticks = sorted(all_x)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlim(right=xticks[-1] * 1.7)
    ax.set_xlabel("Batch size (requests in one forward)", color=INK2, fontsize=11)
    ax.set_ylabel("Decode throughput (tokens/s)", color=INK2, fontsize=11)
    ax.set_title(args.title, color=INK, fontsize=12.5, fontweight="bold", pad=12, loc="left")
    ax.grid(True, which="major", color=GRID, lw=1, zorder=0)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

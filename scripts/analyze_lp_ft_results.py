#!/usr/bin/env python3
"""Compare VPT linear-probe and fine-tune compiled result CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


FIELDNAMES = ["Model Name", "Acc 1", "Acc 2", "Acc 3", "Avg Acc"]


def read_results(path: Path):
    rows = {}
    total = None
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"{path} has unexpected header: {reader.fieldnames}")
        for row in reader:
            name = row["Model Name"]
            values = {
                "acc1": float(row["Acc 1"]),
                "acc2": float(row["Acc 2"]),
                "acc3": float(row["Acc 3"]),
                "avg": float(row["Avg Acc"]),
            }
            if name == "Total Average":
                total = values
            else:
                rows[name] = values
    return rows, total


def family(name: str) -> str:
    special = [
        "beit3",
        "beitv2",
        "beit",
        "convnextv2",
        "convnext",
        "efficientformerv2",
        "efficientformer",
        "efficientnet",
        "efficientvit",
        "mobilenetv4",
        "mobilenetv3",
        "mobilenetv2",
        "mobilevitv2",
        "mobilevit",
        "maxxvitv2",
        "maxvit",
        "swinv2",
        "swin",
        "eva02",
        "eva",
        "deit3",
        "deit",
        "vit",
        "resnext",
        "resnest",
        "resnetrs",
        "resnet",
        "wide_resnet",
        "repvgg",
        "repvit",
        "repghostnet",
        "poolformerv2",
        "poolformer",
        "tf_efficientnetv2",
        "tf_efficientnet",
        "tf_mobilenetv3",
    ]
    for prefix in special:
        if name.startswith(prefix):
            return prefix
    return name.split("_", 1)[0]


def input_size(name: str) -> str:
    sizes = re.findall(r"(?:_|r)(196|224|240|256|288|336|384|448|512|560)(?:\.|_|$)", name)
    return sizes[-1] if sizes else "unknown"


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def median(values):
    return statistics.median(values) if values else float("nan")


def pstdev(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (den_x * den_y) if den_x and den_y else float("nan")


def ranks(values):
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[indexed[k][0]] = rank
        i = j
    return out


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def top_lines(rows, key, n=10, reverse=True, fields=("model", "ft_avg", "lp_avg", "delta")):
    selected = sorted(rows, key=key, reverse=reverse)[:n]
    lines = []
    for row in selected:
        vals = []
        for field in fields:
            value = row[field]
            vals.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lp", type=Path, required=True)
    parser.add_argument("--ft", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    lp, lp_total = read_results(args.lp)
    ft, ft_total = read_results(args.ft)
    common = sorted(set(lp) & set(ft))
    only_lp = sorted(set(lp) - set(ft))
    only_ft = sorted(set(ft) - set(lp))

    rows = []
    for model in common:
        lp_vals = lp[model]
        ft_vals = ft[model]
        row = {
            "model": model,
            "family": family(model),
            "input_size": input_size(model),
            "lp_avg": lp_vals["avg"],
            "ft_avg": ft_vals["avg"],
            "delta": ft_vals["avg"] - lp_vals["avg"],
            "lp_run_std": pstdev([lp_vals["acc1"], lp_vals["acc2"], lp_vals["acc3"]]),
            "ft_run_std": pstdev([ft_vals["acc1"], ft_vals["acc2"], ft_vals["acc3"]]),
            "lp_run_range": max(lp_vals["acc1"], lp_vals["acc2"], lp_vals["acc3"]) - min(lp_vals["acc1"], lp_vals["acc2"], lp_vals["acc3"]),
            "ft_run_range": max(ft_vals["acc1"], ft_vals["acc2"], ft_vals["acc3"]) - min(ft_vals["acc1"], ft_vals["acc2"], ft_vals["acc3"]),
        }
        rows.append(row)

    args.outdir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.outdir / "vpt1_v18_lp_vs_ft_comparison.csv"
    with comparison_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lp_avgs = [r["lp_avg"] for r in rows]
    ft_avgs = [r["ft_avg"] for r in rows]
    deltas = [r["delta"] for r in rows]

    by_family = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    for fam, fam_rows in grouped.items():
        if len(fam_rows) < 4:
            continue
        by_family.append({
            "family": fam,
            "n": len(fam_rows),
            "lp_mean": mean([r["lp_avg"] for r in fam_rows]),
            "ft_mean": mean([r["ft_avg"] for r in fam_rows]),
            "delta_mean": mean([r["delta"] for r in fam_rows]),
            "ft_std_mean": mean([r["ft_run_std"] for r in fam_rows]),
        })
    by_family.sort(key=lambda r: r["delta_mean"], reverse=True)

    by_size = []
    size_grouped = defaultdict(list)
    for row in rows:
        size_grouped[row["input_size"]].append(row)
    for size, size_rows in size_grouped.items():
        if len(size_rows) < 8:
            continue
        by_size.append({
            "input_size": size,
            "n": len(size_rows),
            "lp_mean": mean([r["lp_avg"] for r in size_rows]),
            "ft_mean": mean([r["ft_avg"] for r in size_rows]),
            "delta_mean": mean([r["delta"] for r in size_rows]),
        })
    by_size.sort(key=lambda r: r["ft_mean"], reverse=True)

    summary_path = args.outdir / "vpt1_v18_lp_ft_analysis.md"
    lines = [
        "# VPT1 v18 LP vs FT Analysis",
        "",
        "Source: compiled three-run result tables pasted from the VPT1 v18 run.",
        "",
        "## Files",
        "",
        f"- Linear probe CSV: `{args.lp}`",
        f"- Fine-tune CSV: `{args.ft}`",
        f"- Pairwise comparison CSV: `{comparison_path}`",
        "",
        "## Headline",
        "",
        f"- LP table: {len(lp)} models; total average {lp_total['avg']:.3f}%.",
        f"- FT table: {len(ft)} models; total average {ft_total['avg']:.3f}%.",
        f"- Shared models: {len(common)}.",
        f"- Paired mean delta FT-LP: {mean(deltas):+.3f} percentage points; median {median(deltas):+.3f}; std {pstdev(deltas):.3f}.",
        f"- Pearson LP/FT correlation on shared models: {pearson(lp_avgs, ft_avgs):.3f}; Spearman rank correlation: {spearman(lp_avgs, ft_avgs):.3f}.",
        f"- FT beats LP on {sum(d > 0 for d in deltas)}/{len(deltas)} shared models ({100 * sum(d > 0 for d in deltas) / len(deltas):.1f}%).",
        "",
        "Interpretation: FT worked mechanically, but the gain is modest relative to the training capacity. The weak LP/FT rank correlation means fine-tuning changes which families/models look best; it is not just the LP leaderboard shifted upward.",
        "",
        "## Important Caveat",
        "",
        "`VPT_code/VPT/run_accel_finetune.py` currently evaluates on the test split each epoch and stores the epoch with best test accuracy. That makes the FT table useful for debugging/model search, but optimistic for thesis reporting. For clean reporting, split train into train/val, select by val, and evaluate once on held-out test.",
        "",
        "## Top FT Models",
        "",
        "| model | ft_avg | lp_avg | delta |",
        "| --- | ---: | ---: | ---: |",
        *top_lines(rows, key=lambda r: r["ft_avg"]),
        "",
        "## Top LP Models",
        "",
        "| model | lp_avg | ft_avg | delta |",
        "| --- | ---: | ---: | ---: |",
        *top_lines(rows, key=lambda r: r["lp_avg"], fields=("model", "lp_avg", "ft_avg", "delta")),
        "",
        "## Biggest FT Gains",
        "",
        "| model | ft_avg | lp_avg | delta |",
        "| --- | ---: | ---: | ---: |",
        *top_lines(rows, key=lambda r: r["delta"]),
        "",
        "## Biggest FT Drops",
        "",
        "| model | ft_avg | lp_avg | delta |",
        "| --- | ---: | ---: | ---: |",
        *top_lines(rows, key=lambda r: r["delta"], reverse=False),
        "",
        "## Highest FT Run Variance",
        "",
        "| model | ft_avg | ft_run_std | ft_run_range |",
        "| --- | ---: | ---: | ---: |",
        *top_lines(rows, key=lambda r: r["ft_run_std"], fields=("model", "ft_avg", "ft_run_std", "ft_run_range")),
        "",
        "## Family Patterns",
        "",
        "| family | n | lp_mean | ft_mean | delta_mean | ft_std_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in by_family[:24]:
        lines.append(
            f"| {row['family']} | {row['n']} | {row['lp_mean']:.3f} | {row['ft_mean']:.3f} | {row['delta_mean']:+.3f} | {row['ft_std_mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Input Size Pattern",
        "",
        "| inferred_size | n | lp_mean | ft_mean | delta_mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in by_size:
        lines.append(
            f"| {row['input_size']} | {row['n']} | {row['lp_mean']:.3f} | {row['ft_mean']:.3f} | {row['delta_mean']:+.3f} |"
        )
    lines.extend([
        "",
        "## Model Set Mismatch",
        "",
        f"- Only in LP: {len(only_lp)} models.",
        f"- Only in FT: {len(only_ft)} models.",
    ])
    if only_lp:
        lines.append(f"- LP-only examples: {', '.join(only_lp[:12])}.")
    if only_ft:
        lines.append(f"- FT-only examples: {', '.join(only_ft[:12])}.")

    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

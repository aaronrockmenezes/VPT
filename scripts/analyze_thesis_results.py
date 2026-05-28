#!/usr/bin/env python3
"""Analyze VPT thesis LP/FT result tables across tasks.

Outputs:
  - matched per-task CSVs with ImageNet top-1 and parameter counts
  - summary markdown
  - interactive Plotly HTML scatter plots
  - static PNG scatter plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vpt")

import matplotlib.pyplot as plt


FIELDNAMES = ["Model Name", "Acc 1", "Acc 2", "Acc 3", "Avg Acc"]


TASKS = {
    "vpt1_v18": {
        "label": "VPT1 v18",
        "outdir": "vpt1_v18",
        "lp": "docs/results/vpt1_v18/vpt1_v18_linear_probe_results.csv",
        "ft": "docs/results/vpt1_v18/vpt1_v18_finetune_results.csv",
    },
    "vpt1_v18_depth": {
        "label": "VPT1 v18 Depth",
        "outdir": "vpt1_v18_depth",
        "lp": "docs/results/vpt1_v18_depth/vpt1_v18_depth_linear_probe_results.csv",
        "ft": "docs/results/vpt1_v18_depth/vpt1_v18_depth_finetune_results.csv",
    },
    "vpt2_v4": {
        "label": "VPT2 v4",
        "outdir": "vpt2_v4",
        "lp": "docs/results/vpt2_v4/vpt2_v4_linear_probe_results.csv",
        "ft": "docs/results/vpt2_v4/vpt2_v4_finetune_results.csv",
    },
}


COLORS = {
    "lp": "#2E6FBB",
    "ft": "#D45243",
}


def parse_float(value: str) -> float:
    return float(value.replace(",", ""))


def read_result_csv(path: Path):
    rows = {}
    total = None
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"{path} has unexpected header: {reader.fieldnames}")
        for row in reader:
            vals = {
                "acc1": parse_float(row["Acc 1"]),
                "acc2": parse_float(row["Acc 2"]),
                "acc3": parse_float(row["Acc 3"]),
                "avg": parse_float(row["Avg Acc"]),
            }
            name = row["Model Name"]
            if name == "Total Average":
                total = vals
            else:
                rows[name] = vals
    if total is None:
        raise ValueError(f"{path} has no Total Average row")
    return rows, total


def read_imagenet(path: Path):
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["model"]] = {
                "imagenet_top1": parse_float(row["top1"]),
                "imagenet_top5": parse_float(row["top5"]),
                "param_count_m": parse_float(row["param_count"]),
                "img_size": row.get("img_size", ""),
            }
    return out


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


def family(name: str) -> str:
    prefixes = [
        "tf_efficientnetv2",
        "tf_efficientnet",
        "tf_mobilenetv3",
        "wide_resnet",
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
        "poolformerv2",
        "poolformer",
        "maxxvitv2",
        "maxvit",
        "swinv2",
        "swin",
        "beitv2",
        "beit3",
        "beit",
        "eva02",
        "eva",
        "deit3",
        "deit",
        "resnext",
        "resnest",
        "resnetrs",
        "resnet",
        "repghostnet",
        "repvgg",
        "repvit",
        "vit",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            return prefix
    return name.split("_", 1)[0]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def build_task_rows(task_key, spec, imagenet):
    lp, lp_total = read_result_csv(Path(spec["lp"]))
    ft, ft_total = read_result_csv(Path(spec["ft"]))
    rows = []
    for model in sorted(set(lp) & set(ft) & set(imagenet)):
        lp_vals = lp[model]
        ft_vals = ft[model]
        im = imagenet[model]
        rows.append(
            {
                "task_key": task_key,
                "task": spec["label"],
                "model": model,
                "family": family(model),
                "lp_avg": lp_vals["avg"],
                "ft_avg": ft_vals["avg"],
                "delta_ft_minus_lp": ft_vals["avg"] - lp_vals["avg"],
                "lp_run_range": max(lp_vals["acc1"], lp_vals["acc2"], lp_vals["acc3"])
                - min(lp_vals["acc1"], lp_vals["acc2"], lp_vals["acc3"]),
                "ft_run_range": max(ft_vals["acc1"], ft_vals["acc2"], ft_vals["acc3"])
                - min(ft_vals["acc1"], ft_vals["acc2"], ft_vals["acc3"]),
                "lp_run_std": pstdev([lp_vals["acc1"], lp_vals["acc2"], lp_vals["acc3"]]),
                "ft_run_std": pstdev([ft_vals["acc1"], ft_vals["acc2"], ft_vals["acc3"]]),
                **im,
            }
        )
    meta = {
        "lp_total": lp_total["avg"],
        "ft_total": ft_total["avg"],
        "lp_models": len(lp),
        "ft_models": len(ft),
        "matched": len(rows),
        "lp_only": sorted(set(lp) - set(ft)),
        "ft_only": sorted(set(ft) - set(lp)),
        "missing_imagenet": sorted((set(lp) & set(ft)) - set(imagenet)),
    }
    return rows, meta


def corr(rows, x_key, y_key):
    xs = [r[x_key] for r in rows]
    ys = [r[y_key] for r in rows]
    return pearson(xs, ys), spearman(xs, ys)


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_static(path: Path, rows, x_key, x_label, title, log_x=False):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=False)
    for ax, (task_key, spec) in zip(axes, TASKS.items()):
        subset = [r for r in rows if r["task_key"] == task_key]
        ax.scatter(
            [r[x_key] for r in subset],
            [r["lp_avg"] for r in subset],
            s=28,
            alpha=0.72,
            c=COLORS["lp"],
            label="LP",
            edgecolors="white",
            linewidths=0.3,
        )
        ax.scatter(
            [r[x_key] for r in subset],
            [r["ft_avg"] for r in subset],
            s=28,
            alpha=0.72,
            c=COLORS["ft"],
            label="FT",
            edgecolors="white",
            linewidths=0.3,
        )
        if log_x:
            ax.set_xscale("log")
        lp_r = corr(subset, x_key, "lp_avg")[0]
        ft_r = corr(subset, x_key, "ft_avg")[0]
        ax.set_title(f"{spec['label']}\nLP r={lp_r:.2f}, FT r={ft_r:.2f}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Task accuracy (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_task_static(path: Path, rows, task_key, x_key, x_label, title, log_x=False):
    subset = [r for r in rows if r["task_key"] == task_key]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.scatter([r[x_key] for r in subset], [r["lp_avg"] for r in subset], s=34, alpha=0.72, c=COLORS["lp"], label="LP")
    ax.scatter([r[x_key] for r in subset], [r["ft_avg"] for r in subset], s=34, alpha=0.72, c=COLORS["ft"], label="FT")
    if log_x:
        ax.set_xscale("log")
    lp_r = corr(subset, x_key, "lp_avg")[0]
    ft_r = corr(subset, x_key, "ft_avg")[0]
    ax.set_title(f"{title}\nLP r={lp_r:.2f}, FT r={ft_r:.2f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Task accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plotly_scatter_trace(rows, y_key, name, color, x_key):
    return {
        "type": "scattergl",
        "mode": "markers",
        "name": name,
        "x": [r[x_key] for r in rows],
        "y": [r[y_key] for r in rows],
        "text": [r["model"] for r in rows],
        "customdata": [
            [
                r["task"],
                r["family"],
                r["imagenet_top1"],
                r["param_count_m"],
                r["lp_avg"],
                r["ft_avg"],
                r["delta_ft_minus_lp"],
                r["lp_run_range"],
                r["ft_run_range"],
            ]
            for r in rows
        ],
        "marker": {
            "color": color,
            "size": 7,
            "opacity": 0.74,
            "line": {"width": 0.4, "color": "white"},
        },
        "hovertemplate": (
            "<b>%{text}</b><br>"
            "task=%{customdata[0]}<br>"
            "family=%{customdata[1]}<br>"
            "x=%{x:.2f}<br>"
            f"{name}=%{{y:.2f}}<br>"
            "ImageNet top1=%{customdata[2]:.2f}<br>"
            "params=%{customdata[3]:.2f}M<br>"
            "LP=%{customdata[4]:.2f}<br>"
            "FT=%{customdata[5]:.2f}<br>"
            "FT-LP=%{customdata[6]:+.2f}<br>"
            "LP range=%{customdata[7]:.2f}<br>"
            "FT range=%{customdata[8]:.2f}<extra></extra>"
        ),
    }


def write_plotly_html(path: Path, traces, layout):
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{layout.get("title", {}).get("text", "VPT thesis plot")}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const traces = {json.dumps(traces)};
    const layout = {json.dumps(layout)};
    Plotly.newPlot("plot", traces, layout, {{responsive: true, displaylogo: false}});
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def write_interactive(path: Path, rows, x_key, x_label, title, log_x=False):
    traces = []
    for task_key, spec in TASKS.items():
        subset = [r for r in rows if r["task_key"] == task_key]
        traces.append(plotly_scatter_trace(subset, "lp_avg", f"{spec['label']} LP", COLORS["lp"], x_key))
        traces.append(plotly_scatter_trace(subset, "ft_avg", f"{spec['label']} FT", COLORS["ft"], x_key))
    layout = {
        "title": {"text": title, "x": 0.03, "xanchor": "left"},
        "template": "plotly_white",
        "xaxis": {"title": x_label, "type": "log" if log_x else "linear"},
        "yaxis": {"title": "Task accuracy (%)"},
        "legend": {"orientation": "h", "y": 1.1, "x": 0.01},
        "hovermode": "closest",
        "margin": {"l": 70, "r": 30, "t": 95, "b": 70},
    }
    write_plotly_html(path, traces, layout)


def write_task_interactive(path: Path, rows, task_key, x_key, x_label, title, log_x=False):
    subset = [r for r in rows if r["task_key"] == task_key]
    traces = [
        plotly_scatter_trace(subset, "lp_avg", "LP", COLORS["lp"], x_key),
        plotly_scatter_trace(subset, "ft_avg", "FT", COLORS["ft"], x_key),
    ]
    layout = {
        "title": {"text": title, "x": 0.03, "xanchor": "left"},
        "template": "plotly_white",
        "xaxis": {"title": x_label, "type": "log" if log_x else "linear"},
        "yaxis": {"title": "Task accuracy (%)"},
        "legend": {"orientation": "h", "y": 1.08, "x": 0.01},
        "hovermode": "closest",
        "margin": {"l": 70, "r": 30, "t": 90, "b": 70},
    }
    write_plotly_html(path, traces, layout)


def markdown_table(rows, fields):
    header = "| " + " | ".join(label for _, label, _ in fields) + " |"
    sep = "| " + " | ".join("---:" if numeric else "---" for _, _, numeric in fields) + " |"
    out = [header, sep]
    for row in rows:
        cells = []
        for key, _, numeric in fields:
            val = row[key]
            cells.append(f"{val:.3f}" if numeric and isinstance(val, float) else str(val))
        out.append("| " + " | ".join(cells) + " |")
    return out


def summarize(args):
    imagenet = read_imagenet(args.imagenet)
    all_rows = []
    metas = {}
    for task_key, spec in TASKS.items():
        rows, meta = build_task_rows(task_key, spec, imagenet)
        metas[task_key] = meta
        all_rows.extend(rows)
        outdir = args.outdir / spec["outdir"]
        write_csv(outdir / f"{task_key}_matched_results.csv", rows)

    write_csv(args.outdir / "all_task_matched_results.csv", all_rows)

    plot_static(
        args.outdir / "plots" / "all_tasks_imagenet_top1_vs_accuracy.png",
        all_rows,
        "imagenet_top1",
        "ImageNet top-1 accuracy (%)",
        "VPT task accuracy vs ImageNet top-1",
    )
    write_interactive(
        args.outdir / "plots" / "all_tasks_imagenet_top1_vs_accuracy.html",
        all_rows,
        "imagenet_top1",
        "ImageNet top-1 accuracy (%)",
        "VPT task accuracy vs ImageNet top-1",
    )
    plot_static(
        args.outdir / "plots" / "all_tasks_param_count_vs_accuracy.png",
        all_rows,
        "param_count_m",
        "Parameter count (M, log scale)",
        "VPT task accuracy vs parameter count",
        log_x=True,
    )
    write_interactive(
        args.outdir / "plots" / "all_tasks_param_count_vs_accuracy.html",
        all_rows,
        "param_count_m",
        "Parameter count (M, log scale)",
        "VPT task accuracy vs parameter count",
        log_x=True,
    )

    for task_key, spec in TASKS.items():
        task_rows = [r for r in all_rows if r["task_key"] == task_key]
        plot_task_static(
            args.outdir / "plots" / f"{task_key}_imagenet_top1_vs_accuracy.png",
            all_rows,
            task_key,
            "imagenet_top1",
            "ImageNet top-1 accuracy (%)",
            f"{spec['label']} accuracy vs ImageNet top-1",
        )
        write_task_interactive(
            args.outdir / "plots" / f"{task_key}_imagenet_top1_vs_accuracy.html",
            all_rows,
            task_key,
            "imagenet_top1",
            "ImageNet top-1 accuracy (%)",
            f"{spec['label']} accuracy vs ImageNet top-1",
        )
        plot_task_static(
            args.outdir / "plots" / f"{task_key}_param_count_vs_accuracy.png",
            all_rows,
            task_key,
            "param_count_m",
            "Parameter count (M, log scale)",
            f"{spec['label']} accuracy vs parameter count",
            log_x=True,
        )
        write_task_interactive(
            args.outdir / "plots" / f"{task_key}_param_count_vs_accuracy.html",
            all_rows,
            task_key,
            "param_count_m",
            "Parameter count (M, log scale)",
            f"{spec['label']} accuracy vs parameter count",
            log_x=True,
        )

    lines = [
        "# VPT Thesis Results Analysis",
        "",
        "Source tables are the compiled three-run LP/FT result CSVs saved under `docs/results/`.",
        "ImageNet top-1 and parameter counts come from `results-imagenet.csv`.",
        "",
        "## Output Files",
        "",
        "- `all_task_matched_results.csv`: all matched model/task rows with ImageNet and parameter metadata.",
        "- `plots/all_tasks_imagenet_top1_vs_accuracy.{html,png}`",
        "- `plots/all_tasks_param_count_vs_accuracy.{html,png}`",
        "- Per-task matched CSVs and per-task plots under each task/plot filename.",
        "",
        "## Headline Totals",
        "",
        "| task | LP total avg | FT total avg | FT-LP | LP models | FT models | ImageNet matched |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task_key, spec in TASKS.items():
        meta = metas[task_key]
        lines.append(
            f"| {spec['label']} | {meta['lp_total']:.3f} | {meta['ft_total']:.3f} | "
            f"{meta['ft_total'] - meta['lp_total']:+.3f} | {meta['lp_models']} | "
            f"{meta['ft_models']} | {meta['matched']} |"
        )

    lines += [
        "",
        "## Correlations",
        "",
        "| task | metric | ImageNet Pearson | ImageNet Spearman | params Pearson | params Spearman |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for task_key, spec in TASKS.items():
        rows = [r for r in all_rows if r["task_key"] == task_key]
        for metric, label in [("lp_avg", "LP"), ("ft_avg", "FT")]:
            im_p, im_s = corr(rows, "imagenet_top1", metric)
            pc_p, pc_s = corr(rows, "param_count_m", metric)
            lines.append(
                f"| {spec['label']} | {label} | {im_p:.3f} | {im_s:.3f} | {pc_p:.3f} | {pc_s:.3f} |"
            )

    lines += [
        "",
        "## Top Models By Task",
        "",
    ]
    for task_key, spec in TASKS.items():
        rows = [r for r in all_rows if r["task_key"] == task_key]
        lines += [
            f"### {spec['label']}",
            "",
            "Top FT:",
            "",
            *markdown_table(
                sorted(rows, key=lambda r: r["ft_avg"], reverse=True)[:10],
                [
                    ("model", "model", False),
                    ("ft_avg", "FT", True),
                    ("lp_avg", "LP", True),
                    ("delta_ft_minus_lp", "FT-LP", True),
                    ("imagenet_top1", "ImageNet", True),
                    ("param_count_m", "params M", True),
                    ("ft_run_range", "FT range", True),
                ],
            ),
            "",
            "Top LP:",
            "",
            *markdown_table(
                sorted(rows, key=lambda r: r["lp_avg"], reverse=True)[:10],
                [
                    ("model", "model", False),
                    ("lp_avg", "LP", True),
                    ("ft_avg", "FT", True),
                    ("delta_ft_minus_lp", "FT-LP", True),
                    ("imagenet_top1", "ImageNet", True),
                    ("param_count_m", "params M", True),
                ],
            ),
            "",
            "Largest FT run instability:",
            "",
            *markdown_table(
                sorted(rows, key=lambda r: r["ft_run_range"], reverse=True)[:8],
                [
                    ("model", "model", False),
                    ("ft_avg", "FT", True),
                    ("ft_run_range", "FT range", True),
                    ("lp_avg", "LP", True),
                ],
            ),
            "",
        ]

    lines += [
        "## Cross-Task Notes",
        "",
        "- VPT1 v18 LP is near the high-50s and FT only modestly improves it, so the normal RGB task remains difficult under this setup.",
        "- VPT1 depth LP is already strong, and FT jumps sharply; this is the cleanest current task separation.",
        "- VPT2 LP is near chance, but FT jumps high. That gap is useful but should be treated carefully because many VPT2 FT models show large run-to-run instability.",
        "- Parameter count is not a reliable standalone predictor here; check the parameter scatter and the correlation table before using size as a model-selection rule.",
        "- ImageNet top-1 is most useful as a weak prior, not as a direct thesis-task predictor.",
        "",
        "## Caveat",
        "",
        "If the FT training code still selects checkpoints by test accuracy, the FT numbers are optimistic for final thesis reporting. Use these for exploration/model selection, then rerun finalists with train/val selection and one held-out test evaluation.",
        "",
    ]

    (args.outdir / "thesis_results_analysis.md").write_text("\n".join(lines))
    return all_rows, metas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet", type=Path, default=Path("results-imagenet.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("docs/results/thesis_analysis"))
    args = parser.parse_args()
    rows, metas = summarize(args)
    print(f"Wrote analysis for {len(rows)} matched rows across {len(TASKS)} tasks")
    for task_key, spec in TASKS.items():
        meta = metas[task_key]
        print(
            f"{spec['label']}: LP={meta['lp_total']:.3f} FT={meta['ft_total']:.3f} "
            f"matched={meta['matched']}"
        )


if __name__ == "__main__":
    main()

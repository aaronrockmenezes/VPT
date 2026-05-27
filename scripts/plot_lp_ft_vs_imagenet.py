#!/usr/bin/env python3
"""Build a Plotly HTML scatter of VPT1 v18 LP/FT accuracy vs ImageNet top-1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_result_csv(path: Path):
    rows = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["Model Name"]
            if model == "Total Average":
                continue
            rows[model] = float(row["Avg Acc"])
    return rows


def read_imagenet_csv(path: Path):
    rows = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["model"]] = {
                "top1": float(row["top1"]),
                "img_size": row.get("img_size", ""),
                "param_count": row.get("param_count", ""),
            }
    return rows


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (den_x * den_y) if den_x and den_y else float("nan")


def linear_fit(xs, ys):
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if not den:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    intercept = my - slope * mx
    return slope, intercept


def fit_trace(xs, ys, name, color):
    slope, intercept = linear_fit(xs, ys)
    x0, x1 = min(xs), max(xs)
    return {
        "type": "scatter",
        "mode": "lines",
        "name": f"{name} trend",
        "x": [x0, x1],
        "y": [slope * x0 + intercept, slope * x1 + intercept],
        "line": {"color": color, "dash": "dash", "width": 2},
        "hovertemplate": f"{name} trend<br>ImageNet top-1=%{{x:.2f}}<br>VPT acc=%{{y:.2f}}<extra></extra>",
    }


def scatter_trace(rows, metric, name, color):
    return {
        "type": "scattergl",
        "mode": "markers",
        "name": name,
        "x": [r["imagenet_top1"] for r in rows],
        "y": [r[metric] for r in rows],
        "text": [r["model"] for r in rows],
        "customdata": [
            [r["lp_avg"], r["ft_avg"], r["delta_ft_minus_lp"], r["img_size"], r["param_count"]]
            for r in rows
        ],
        "marker": {
            "color": color,
            "size": 8,
            "opacity": 0.78,
            "line": {"width": 0.5, "color": "white"},
        },
        "hovertemplate": (
            "<b>%{text}</b><br>"
            "ImageNet top-1=%{x:.2f}<br>"
            f"{name}=%{{y:.2f}}<br>"
            "LP=%{customdata[0]:.2f}<br>"
            "FT=%{customdata[1]:.2f}<br>"
            "FT-LP=%{customdata[2]:+.2f}<br>"
            "img_size=%{customdata[3]}<br>"
            "params=%{customdata[4]}M"
            "<extra></extra>"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lp", type=Path, required=True)
    parser.add_argument("--ft", type=Path, required=True)
    parser.add_argument("--imagenet", type=Path, default=Path("results-imagenet.csv"))
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    lp = read_result_csv(args.lp)
    ft = read_result_csv(args.ft)
    imagenet = read_imagenet_csv(args.imagenet)

    rows = []
    for model in sorted(set(lp) & set(ft) & set(imagenet)):
        rows.append({
            "model": model,
            "imagenet_top1": imagenet[model]["top1"],
            "lp_avg": lp[model],
            "ft_avg": ft[model],
            "delta_ft_minus_lp": ft[model] - lp[model],
            "img_size": imagenet[model]["img_size"],
            "param_count": imagenet[model]["param_count"],
        })

    if not rows:
        raise SystemExit("No shared models across LP, FT, and ImageNet CSVs.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    matched_csv = args.outdir / "vpt1_v18_lp_ft_imagenet_matched.csv"
    with matched_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = [r["imagenet_top1"] for r in rows]
    lp_y = [r["lp_avg"] for r in rows]
    ft_y = [r["ft_avg"] for r in rows]
    lp_r = pearson(x, lp_y)
    ft_r = pearson(x, ft_y)

    traces = [
        scatter_trace(rows, "lp_avg", "Linear probe", "#2E6FBB"),
        scatter_trace(rows, "ft_avg", "Fine-tune", "#D45243"),
        fit_trace(x, lp_y, "Linear probe", "#2E6FBB"),
        fit_trace(x, ft_y, "Fine-tune", "#D45243"),
    ]
    layout = {
        "title": {
            "text": (
                "VPT1 v18 Accuracy vs ImageNet Top-1"
                f"<br><sup>n={len(rows)} shared models; "
                f"LP r={lp_r:.3f}; FT r={ft_r:.3f}</sup>"
            ),
            "x": 0.03,
            "xanchor": "left",
        },
        "template": "plotly_white",
        "xaxis": {
            "title": "ImageNet top-1 accuracy (%)",
            "showgrid": True,
            "zeroline": False,
        },
        "yaxis": {
            "title": "VPT1 v18 accuracy (%)",
            "showgrid": True,
            "zeroline": False,
        },
        "legend": {"orientation": "h", "y": 1.08, "x": 0.02},
        "hovermode": "closest",
        "margin": {"l": 70, "r": 30, "t": 100, "b": 70},
        "annotations": [
            {
                "text": "FT and LP are plotted as separate points for each matched model.",
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.18,
                "font": {"size": 12, "color": "#555"},
            }
        ],
    }

    html_path = args.outdir / "vpt1_v18_lp_ft_vs_imagenet.html"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>VPT1 v18 LP/FT vs ImageNet</title>
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
    html_path.write_text(html)

    print(f"Matched models: {len(rows)}")
    print(f"LP/ImageNet Pearson r: {lp_r:.3f}")
    print(f"FT/ImageNet Pearson r: {ft_r:.3f}")
    print(f"Wrote {matched_csv}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()

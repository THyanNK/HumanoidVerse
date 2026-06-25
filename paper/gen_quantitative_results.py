#!/usr/bin/env python3
"""Generate a clean quantitative bar chart for the paper."""

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
})


conditions = [
    "Stage III\n$\\alpha=0$",
    "Stage III\n$\\alpha=1$",
    "Stage III\n$\\alpha=2$",
    "Stage II\n$\\alpha=2$",
]

metrics = [
    {
        "xlabel": "Survival rate (%)",
        "values": np.array([100.0, 100.0, 100.0, 0.0]),
        "fmt": "{:.0f}",
        "xlim": (0, 105),
        "xticks": [0, 25, 50, 75, 100],
    },
    {
        "xlabel": "Tracking error $E_{v_x}$ (m/s)",
        "values": np.array([0.0228, 0.0225, 0.0245, 0.1401]),
        "fmt": "{:.3f}",
        "xlim": (0, 0.16),
        "xticks": [0.00, 0.04, 0.08, 0.12, 0.16],
    },
    {
        "xlabel": "Max body tilt (deg)",
        "values": np.array([4.67, 7.81, 9.12, 89.93]),
        "fmt": "{:.1f}",
        "xlim": (0, 100),
        "xticks": [0, 25, 50, 75, 100],
    },
    {
        "xlabel": "Total falls",
        "values": np.array([0.0, 0.0, 0.0, 95.0]),
        "fmt": "{:.0f}",
        "xlim": (0, 100),
        "xticks": [0, 25, 50, 75, 100],
    },
]

colors = ["#2A9D8F", "#2A9D8F", "#2A9D8F", "#D55E00"]
y = np.arange(len(conditions))

fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.2), sharey=True)
axes = axes.ravel()

for ax, metric in zip(axes, metrics):
    xlabel = metric["xlabel"]
    values = metric["values"]
    fmt = metric["fmt"]
    xmin, xmax = metric["xlim"]
    bars = ax.barh(y, values, height=0.58, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_xlabel(xlabel)
    ax.set_yticks(y)
    ax.set_yticklabels(conditions)
    ax.invert_yaxis()
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(metric["xticks"])

    for bar, value in zip(bars, values):
        label = fmt.format(value)
        offset = (xmax - xmin) * 0.025
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=7.5,
            color="#333333",
        )

fig.text(0.02, 0.97, "Stage III robust policy", color="#2A9D8F", fontsize=8.5)
fig.text(0.31, 0.97, "Stage II baseline", color="#D55E00", fontsize=8.5)
fig.subplots_adjust(left=0.17, right=0.98, top=0.91, bottom=0.12, wspace=0.30, hspace=0.42)

fig.savefig("quantitative_results.pdf")
fig.savefig("quantitative_results.png")

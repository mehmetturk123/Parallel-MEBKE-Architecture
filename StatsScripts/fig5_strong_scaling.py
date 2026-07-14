"""Figure 5 of the main paper: strong scaling of PMA with individual task
dispatch, three representative instances, ideal diagonal, per-instance
W_max bounds, +/-1 SD error bars propagated to the speedups.

Requires: matplotlib. Data: data_benchmarks.py (same directory).
Outputs figure5_strong_scaling.{png,svg,pdf} into the working directory.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_benchmarks as db

INSTANCES = {
    (30, 15): ("C(30,15)", "#2a78d6", "o"),
    (35, 18): ("C(35,18)", "#1baf7a", "s"),
    (40, 20): ("C(40,20)", "#eda100", "^"),
}
THREADS = [1, 2, 4, 8, 16, 32, 56]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "legend.frameon": True, "legend.fancybox": False,
    "legend.edgecolor": "black", "legend.framealpha": 1.0,
})

fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.plot([0, 58], [0, 58], ls="--", lw=0.8, color="black", zorder=1, label="Ideal")

bounds = {k: math.comb(k[0], k[1]) / math.comb(k[0] - 5, k[1] - 5) for k in INSTANCES}
topmost = max(bounds.values())
for k, (label, color, marker) in INSTANCES.items():
    b = bounds[k]
    ax.plot([36, 57.5], [b, b], ls=(0, (4, 3)), lw=0.9, color=color, zorder=2)
    ax.text(35.2, b, f"{b:.1f}", color=color, fontsize=7, ha="right", va="center")
    if b == topmost:
        ax.text(46.5, b + 1.0, r"$C(n,r)/W_{\max}$", color="black",
                fontsize=7, ha="center", va="bottom")

for k, (label, color, marker) in INSTANCES.items():
    t1, _, s1, _ = db.PARALLEL[(k[0], k[1], 1)]
    S, err = [], []
    for p in THREADS:
        tp, _, sp, _ = db.PARALLEL[(k[0], k[1], p)]
        s = t1 / tp
        S.append(s)
        err.append(s * math.sqrt((s1 / t1) ** 2 + (sp / tp) ** 2))
    ax.errorbar(THREADS, S, yerr=err, color=color, marker=marker, ms=4.5, lw=1.2,
                capsize=2, elinewidth=0.8, markeredgecolor="white",
                markeredgewidth=0.4, zorder=3, label=label)

ax.set_xlim(0, 58)
ax.set_ylim(0, 58)
ax.set_xticks(range(0, 57, 8))
ax.set_yticks(range(0, 57, 8))
ax.set_xlabel(r"Threads $p$")
ax.set_ylabel("Speedup")
ax.legend(loc="upper left", fontsize=7, borderpad=0.6, handlelength=2.2)

for ext in ("png", "svg", "pdf"):
    fig.savefig(f"figure5_strong_scaling.{ext}",
                dpi=300 if ext == "png" else None, bbox_inches="tight")
print("written: figure5_strong_scaling.png/.svg/.pdf")

"""Figure 6 of the main paper: sensitivity of PMA to the prefix depth D at
p = 56 threads (median of 30 runs; logarithmic time axis; +/-1 SD error bars,
lower bars clipped where they would cross zero; faint horizontal gridlines at
the decade ticks).

Requires: matplotlib. Data: data_benchmarks.py (same directory).
Outputs figure6_depth_sweep.{png,svg,pdf} into the working directory.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_benchmarks as db

SERIES = [
    (30, 15, "C(30,15)", "#2a78d6", "o"),
    (32, 16, "C(32,16)", "#1baf7a", "s"),
    (34, 17, "C(34,17)", "#eda100", "^"),
    (36, 18, "C(36,18)", "#008300", "D"),
    (38, 19, "C(38,19)", "#4a3aa7", "v"),
    (40, 20, "C(40,20)", "#e34948", "P"),
]
DS = [5, 6, 7, 8, 9]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "ytick.minor.size": 2.0,
    "legend.frameon": True, "legend.fancybox": False,
    "legend.edgecolor": "black", "legend.framealpha": 1.0,
})

fig, ax = plt.subplots(figsize=(3.4, 2.8))
for n, r, label, color, marker in SERIES:
    med = [db.DEPTH[(n, r, D)][0] for D in DS]
    sd = [db.DEPTH[(n, r, D)][2] for D in DS]
    lo = [min(s, 0.8 * m) for m, s in zip(med, sd)]   # clamp for the log axis
    ax.errorbar(DS, med, yerr=[lo, sd], color=color, marker=marker, ms=4.0,
                lw=1.2, capsize=2, elinewidth=0.8, markeredgecolor="white",
                markeredgewidth=0.4, label=label, zorder=3)

ax.set_yscale("log")
ax.set_xlim(4.7, 9.3)
ax.set_xticks(DS)
ax.set_axisbelow(True)
ax.grid(True, axis="y", which="major", color="#d9d9d9", linewidth=0.6)
ax.set_xlabel(r"Prefix depth $D$")
ax.set_ylabel("Median running time (s)")
ax.legend(loc="upper right", fontsize=6.5, borderpad=0.5, handlelength=1.8,
          labelspacing=0.35)

for ext in ("png", "svg", "pdf"):
    fig.savefig(f"figure6_depth_sweep.{ext}",
                dpi=300 if ext == "png" else None, bbox_inches="tight")
print("written: figure6_depth_sweep.png/.svg/.pdf")

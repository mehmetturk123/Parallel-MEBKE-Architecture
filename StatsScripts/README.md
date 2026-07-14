# Statistical analyses of Section 5.9 (Parallel MEBKE Architecture)

Standalone, self-contained Python scripts that reproduce **every statistical
test and derived quantity** quoted in Section 5.9 of the paper. All measurement
data are embedded in `data_benchmarks.py` (generated from the raw benchmark
files, which are also in this repository), so the scripts run without any
external input. Each script prints the supporting numbers and **asserts** the
exact values quoted in the paper; `run_all.py` executes all of them and exits
non-zero if any reproduction fails.

```
python run_all.py
```

| Script | Paper claim it reproduces |
|---|---|
| `stat1_cross_machine_wilcoxon.py` | "the TRUBA-to-Section-4 ratio ranges from 0.82 to 1.20 with a geometric mean of 1.015 ... a two-sided Wilcoxon signed-rank test on the paired log running times does not reject equal performance (W = 28, exact p = 0.70; a paired t-test on the log ratios agrees, t(10) = 0.42)" — exact p computed both by `scipy.stats.wilcoxon(method="exact")` and by a from-scratch enumeration of all 2¹¹ sign assignments |
| `stat2_serial_vs_parallel_welch.py` | "the serial engine is significantly slower than the parallel engine executed with a single thread — by 38% to 86% (median ratio 1.47; Welch's unequal-variance t-test on the two 30-run samples rejects equality with \|t\| ≥ 142, p ≪ 0.001, on every instance)" and "the largest instance C(40,20) improves from 333.566 s to 4.958 s — a factor of 67.3" |
| `stat3_depth_sweep_welch.py` | "At D = 5 every instance runs between 4.4× and 6.4× slower than at its best depth within the sweep (Welch's t-test against the per-instance best rejects equality with p < 10⁻⁵ on every instance) ... 20.589 s at D = 5 to 3.231 s at D = 9 — 71.6× ... and 103×" |
| `stat4_workload_bounds.py` | the exact combinatorial quantities (pure standard library): T_tasks = C(n−r+D, D), the partition identity ΣW = C(n,r), the head-portion share ω₀ = 10.5–12.1% ⇒ 1/ω₀ = 8.3–9.5 vs the pilot plateau 8.8–10.3, the heaviest-task share 2.1–2.7% ⇒ bounds 37.1–47.5 vs observed S(56) (ratio 1.05–1.13), the depth bounds (≥ 99 at D = 6, ≥ 240 at D = 7) and task-list growth (53,130 → 10,015,005) |
| `stat5_scaling_and_variance.py` | the scaling summaries (S(8) = 7.9–8.1, S(16) = 15.8–16.5, S(32) = 31.1–33.0 at 97–103% efficiency, S(56) = 40.6–51.0; C(40,20): 46.6× at 83%), the Figure-5 error propagation σ_S = S·√((σ₁/t₁)² + (σ_p/t_p)²) with largest bar ±0.5, the CoV census (76 of 77 configurations < 3.5%, one at 7.0%) and the 0.7% single-thread parity between the chunk-64 and chunk-1 campaigns |
| `fig5_strong_scaling.py` | regenerates Figure 5 (strong scaling with error bars and W_max bounds) |
| `fig6_depth_sweep.py` | regenerates Figure 6 (depth sensitivity, log time axis, decade gridlines) |

**Requirements:** Python ≥ 3.9. `scipy` for `stat1`–`stat3` (p-values);
`matplotlib` for the two figure scripts; `stat4`, `stat5` and the embedded
data need the standard library only.

**Data provenance** (all TRUBA *orfoz* nodes, 2× Intel Xeon Platinum 8480+,
112 cores, SMT disabled; g++ 14.1.0 `-std=c++20 -O3 -fopenmp`; 30 measured
runs after 3 warm-ups per configuration):
`benchmark_results_30C15to40C20_OMP1.csv` (main campaign, job 6043673),
`benchmark_results_iMEBKE.xlsx` (serial baseline, job 6043876),
`stats_dimension_6065957.xlsx` (depth sweep, job 6065957),
pilot chunk-64 campaign (job 6030114; p = 1 and p = 16 medians embedded).
Section-4 values are the iMEBKE column of Table 2 of the main paper.

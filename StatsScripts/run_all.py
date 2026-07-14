"""Run every statistical analysis of Section 5.9 and verify that each
paper-quoted value is reproduced (any assertion failure exits non-zero).

Usage:  python run_all.py
"""
import stat1_cross_machine_wilcoxon
import stat2_serial_vs_parallel_welch
import stat3_depth_sweep_welch
import stat4_workload_bounds
import stat5_scaling_and_variance

STEPS = [
    ("1. cross-machine Wilcoxon / paired t", stat1_cross_machine_wilcoxon.main),
    ("2. serial vs parallel p=1 (Welch)", stat2_serial_vs_parallel_welch.main),
    ("3. depth sweep (Welch vs best D)", stat3_depth_sweep_welch.main),
    ("4. exact workload bounds", stat4_workload_bounds.main),
    ("5. scaling and variance", stat5_scaling_and_variance.main),
]

for title, fn in STEPS:
    print("=" * 72)
    print(title)
    print("=" * 72)
    fn()
    print()

print("ALL ANALYSES REPRODUCED THE PAPER-QUOTED VALUES.")

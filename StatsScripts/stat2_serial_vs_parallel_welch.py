"""Serial iMEBKE vs the parallel engine at p = 1 (Section 5.9, Table 7;
Supplementary Material II, Section D.14).

Reproduces the paper sentence:
  "On every instance the serial engine is significantly slower than the parallel
   engine executed with a single thread -- by 38% to 86% (median ratio 1.47;
   Welch's unequal-variance t-test on the two 30-run samples rejects equality
   with |t| >= 142, p << 0.001, on every instance)."
and the consequence:
  "measured against the published serial iMEBKE, the largest instance C(40, 20)
   improves from 333.566 s to 4.958 s -- a factor of 67.3."

Method: for each instance, Welch's unequal-variance t-test from the summary
statistics (mean, sample SD, n = 30) of the serial and the parallel p = 1
samples (scipy.stats.ttest_ind_from_stats, equal_var=False).

Requires: scipy. Data: data_benchmarks.py (same directory).
"""
from statistics import median

from scipy import stats

import data_benchmarks as db


def main():
    keys = sorted(db.SERIAL)
    ratios = []
    min_abs_t, max_p = float("inf"), 0.0
    print("instance   serial_med  par1_med   ratio   Welch_t      df     p")
    for k in keys:
        s_med, s_mean, s_sd, _ = db.SERIAL[k]
        p_med, p_mean, p_sd, _ = db.PARALLEL[(k[0], k[1], 1)]
        r = s_med / p_med
        ratios.append(r)
        t, p = stats.ttest_ind_from_stats(s_mean, s_sd, db.N_RUNS,
                                          p_mean, p_sd, db.N_RUNS,
                                          equal_var=False)
        # Welch-Satterthwaite degrees of freedom, for reference
        v1, v2 = s_sd**2 / db.N_RUNS, p_sd**2 / db.N_RUNS
        df = (v1 + v2) ** 2 / (v1**2 / (db.N_RUNS - 1) + v2**2 / (db.N_RUNS - 1))
        min_abs_t = min(min_abs_t, abs(t))
        max_p = max(max_p, p)
        print(f"C({k[0]},{k[1]})  {s_med:10.5f} {p_med:9.5f}  {r:.3f}  {t:8.1f}  {df:5.1f}  {p:.2e}")

    print(f"ratio range {min(ratios):.2f}-{max(ratios):.2f} "
          f"(slower by {100*(min(ratios)-1):.0f}%-{100*(max(ratios)-1):.0f}%); "
          f"median ratio {median(ratios):.2f}")
    print(f"min |t| = {min_abs_t:.1f}; max p = {max_p:.2e}")

    e2e = db.SERIAL[(40, 20)][0] / db.PARALLEL[(40, 20, 56)][0]
    print(f"end-to-end C(40,20): {db.SERIAL[(40,20)][0]:.3f} s / "
          f"{db.PARALLEL[(40,20,56)][0]:.5f} s = {e2e:.1f}x")

    # values as quoted in the paper
    assert round(min(ratios), 2) == 1.38 and round(max(ratios), 2) == 1.86
    assert round(median(ratios), 2) == 1.47
    assert min_abs_t >= 142
    assert max_p < 1e-3          # "p << 0.001" (in fact < 1e-30)
    assert round(e2e, 1) == 67.3
    print("all paper-quoted values reproduced. OK")


if __name__ == "__main__":
    main()

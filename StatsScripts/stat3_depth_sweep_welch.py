"""Depth sensitivity of PMA at p = 56 (Section 5.9, Figure 6).

Reproduces the paper sentence:
  "At D = 5 every instance runs between 4.4x and 6.4x slower than at its best
   depth within the sweep (Welch's t-test against the per-instance best rejects
   equality with p < 10^-5 on every instance); the best depth is 7, 8 or 9
   depending on the instance, and the largest instance C(40, 20) improves from
   20.589 s at D = 5 to 3.231 s at D = 9 -- 71.6x relative to the single-thread
   reference of Table 5 and 103x relative to the serial iMEBKE of Table 7."

Method: within each instance, the depth with the smallest median is the
baseline; Welch's unequal-variance t-test compares the D = 5 sample against the
best-depth sample from the summary statistics (mean, sample SD, n = 30).

Requires: scipy. Data: data_benchmarks.py (same directory).
"""
from scipy import stats

import data_benchmarks as db


def main():
    instances = sorted(set((n, r) for n, r, D in db.DEPTH))
    ratios = []
    max_p = 0.0
    print("instance   best_D  best_med    D5_med     D5/best   Welch_t     p")
    for n, r in instances:
        meds = {D: db.DEPTH[(n, r, D)][0] for D in range(5, 10)}
        best_d = min(meds, key=meds.get)
        ratio = meds[5] / meds[best_d]
        ratios.append(ratio)
        m5, s5 = db.DEPTH[(n, r, 5)][1], db.DEPTH[(n, r, 5)][2]
        mb, sb = db.DEPTH[(n, r, best_d)][1], db.DEPTH[(n, r, best_d)][2]
        t, p = stats.ttest_ind_from_stats(mb, sb, db.N_RUNS, m5, s5, db.N_RUNS,
                                          equal_var=False)
        max_p = max(max_p, p)
        print(f"C({n},{r})   D={best_d}   {meds[best_d]:9.4f} {meds[5]:10.4f}   "
              f"{ratio:5.2f}x  {t:9.1f}  {p:.2e}")

    print(f"D=5-to-best ratio range: {min(ratios):.2f}-{max(ratios):.2f}; "
          f"max p = {max_p:.2e}")

    d5 = db.DEPTH[(40, 20, 5)][0]
    d9 = db.DEPTH[(40, 20, 9)][0]
    vs_p1 = db.PARALLEL[(40, 20, 1)][0] / d9
    vs_serial = db.SERIAL[(40, 20)][0] / d9
    print(f"C(40,20): {d5:.3f} s (D=5) -> {d9:.3f} s (D=9); "
          f"{vs_p1:.1f}x vs p=1 reference; {vs_serial:.0f}x vs serial iMEBKE")

    # values as quoted in the paper
    assert round(min(ratios), 2) == 4.45 and round(max(ratios), 2) == 6.37
    assert max_p < 1e-5
    assert round(d5, 3) == 20.589 and round(d9, 3) == 3.231
    assert round(vs_p1, 1) == 71.6
    assert round(vs_serial) == 103
    print("all paper-quoted values reproduced. OK")


if __name__ == "__main__":
    main()

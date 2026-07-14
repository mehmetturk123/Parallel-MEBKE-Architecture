"""Strong-scaling summaries and variance reporting (Section 5.9: Table 5,
Figure 5, methodology paragraph).

Reproduces:
  - speedups/efficiencies of the main (chunk = 1) campaign: "the speedup at
    p = 32 lies between 31.1x and 33.0x (parallel efficiency 97-103%) ...
    (7.9-8.1x at p = 8, 15.8-16.5x at p = 16). At p = 56 the speedups reach
    40.6x-51.0x; the largest instance C(40, 20) drops from 231.270 s to
    4.958 s, a speedup of 46.6x at 83% efficiency.";
  - the Figure 5 error bars: sigma_S = S*sqrt((s1/t1)^2 + (sp/tp)^2), "smaller
    than the plotted markers throughout (the largest is +/-0.5 at p = 56 for
    C(30,15))";
  - the stability sentence: "the coefficient of variation is below 3.5% in 76
    of the 77 configurations, and the single exception (7.0% ...)";
  - the queue-cost sentence: "the single-thread running times of the two
    configurations agree within 0.7%".

Requires: Python >= 3.9 only. Data: data_benchmarks.py (same directory).
"""
import math

import data_benchmarks as db

THREADS = [1, 2, 4, 8, 16, 32, 56]


def speedup(n, r, p):
    return db.PARALLEL[(n, r, 1)][0] / db.PARALLEL[(n, r, p)][0]


def main():
    for p in [8, 16, 32, 56]:
        ss = [speedup(n, r, p) for n, r in db.INSTANCES]
        print(f"S({p}): {min(ss):.2f}-{max(ss):.2f}  "
              f"(efficiency {100*min(ss)/p:.0f}-{100*max(ss)/p:.0f}%)")
    s56_40 = speedup(40, 20, 56)
    print(f"C(40,20): {db.PARALLEL[(40,20,1)][0]:.3f} s -> "
          f"{db.PARALLEL[(40,20,56)][0]:.3f} s; S(56) = {s56_40:.1f} "
          f"({100*s56_40/56:.0f}% efficiency)")

    # Figure 5 error propagation for the three plotted instances
    max_bar, arg = 0.0, None
    for n, r in [(30, 15), (35, 18), (40, 20)]:
        t1, _, s1, _ = db.PARALLEL[(n, r, 1)]
        for p in THREADS:
            tp, _, sp, _ = db.PARALLEL[(n, r, p)]
            S = t1 / tp
            sig = S * math.sqrt((s1 / t1) ** 2 + (sp / tp) ** 2)
            if sig > max_bar:
                max_bar, arg = sig, (n, r, p)
    print(f"largest Figure-5 error bar: +/-{max_bar:.3f} at C({arg[0]},{arg[1]}), p={arg[2]}")

    covs = [db.PARALLEL[k][3] for k in db.PARALLEL]
    below = sum(1 for c in covs if c < 3.5)
    print(f"CoV census: {below} of {len(covs)} configurations < 3.5%; max = {max(covs):.2f}%")

    parity = (db.PARALLEL[(40, 20, 1)][0] - db.PILOT_P1[(40, 20)]) / db.PILOT_P1[(40, 20)]
    print(f"single-thread parity, chunk=1 vs chunk=64 campaign (C(40,20)): {100*parity:+.2f}%")

    # values as quoted in the paper
    s32 = [speedup(n, r, 32) for n, r in db.INSTANCES]
    s56 = [speedup(n, r, 56) for n, r in db.INSTANCES]
    s8 = [speedup(n, r, 8) for n, r in db.INSTANCES]
    s16 = [speedup(n, r, 16) for n, r in db.INSTANCES]
    assert (round(min(s8), 1), round(max(s8), 1)) == (7.9, 8.1)
    assert (round(min(s16), 1), round(max(s16), 1)) == (15.8, 16.5)
    assert (round(min(s32), 1), round(max(s32), 1)) == (31.1, 33.0)
    assert 97 <= 100 * min(s32) / 32 and 100 * max(s32) / 32 <= 103.5
    assert (round(min(s56), 1), round(max(s56), 1)) == (40.6, 51.0)
    assert round(s56_40, 1) == 46.6 and round(100 * s56_40 / 56) == 83
    assert round(max_bar, 1) == 0.5 and arg == (30, 15, 56)
    assert below == 76 and len(covs) == 77 and round(max(covs), 1) == 7.0
    assert abs(parity) < 0.007
    print("all paper-quoted values reproduced. OK")


if __name__ == "__main__":
    main()

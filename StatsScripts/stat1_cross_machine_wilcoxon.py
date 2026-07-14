"""Cross-machine consistency of the serial iMEBKE baseline (Section 5.9, Table 7).

Reproduces the paper sentence:
  "Across the eleven instances the TRUBA-to-Section-4 ratio ranges from 0.82 to
   1.20 with a geometric mean of 1.015 and no systematic direction: a two-sided
   Wilcoxon signed-rank test on the paired log running times does not reject
   equal performance (W = 28, exact p = 0.70; a paired t-test on the log ratios
   agrees, t(10) = 0.42)."

Method: the eleven instances give paired observations (TRUBA serial median,
Section-4 reported time). The test is performed on log(TRUBA/Section4), i.e.
H0: the paired log running times have symmetric differences about zero.
The exact two-sided p-value is computed twice: by scipy (method="exact") and by
a from-scratch enumeration of all 2^11 sign assignments, which agree.

Requires: scipy. Data: data_benchmarks.py (same directory).
"""
import math
from itertools import product

from scipy import stats

import data_benchmarks as db


def wilcoxon_exact_enumeration(values):
    """Exact two-sided signed-rank p by full enumeration (no ties, no zeros)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: abs(values[i]))
    rank = [0] * n
    for rk, i in enumerate(order, start=1):
        rank[i] = rk
    w_plus = sum(rank[i] for i in range(n) if values[i] > 0)
    total = n * (n + 1) // 2
    center = total / 2.0
    dev = abs(w_plus - center)
    count = 0
    for signs in product((0, 1), repeat=n):
        w = sum((i + 1) for i in range(n) if signs[i])
        if abs(w - center) >= dev - 1e-12:
            count += 1
    return w_plus, total - w_plus, count / (1 << n)


def main():
    keys = sorted(db.SERIAL)
    ratios = [db.SERIAL[k][0] / db.SECTION4[k] for k in keys]
    logr = [math.log(x) for x in ratios]
    gm = math.exp(sum(logr) / len(logr))

    print("instance   TRUBA_med   Section4   ratio")
    for k, x in zip(keys, ratios):
        print(f"C({k[0]},{k[1]})  {db.SERIAL[k][0]:10.5f} {db.SECTION4[k]:10.3f}  {x:.3f}")
    print(f"ratio range {min(ratios):.2f}-{max(ratios):.2f}; geometric mean {gm:.3f}")

    w_plus, w_minus, p_enum = wilcoxon_exact_enumeration(logr)
    res = stats.wilcoxon(logr, alternative="two-sided", method="exact")
    t_res = stats.ttest_1samp(logr, 0.0)

    print(f"Wilcoxon signed-rank: W+ = {w_plus}, W- = {w_minus}, "
          f"W = min = {min(w_plus, w_minus)}")
    print(f"exact two-sided p: enumeration = {p_enum:.3f}, scipy = {res.pvalue:.3f}")
    print(f"paired t on log-ratios: t({len(logr)-1}) = {t_res.statistic:.2f}, "
          f"p = {t_res.pvalue:.2f}")
    print(f"max serial CoV = {max(db.SERIAL[k][3] for k in keys):.2f}%")

    # values as quoted in the paper
    assert round(min(ratios), 2) == 0.82 and round(max(ratios), 2) == 1.20
    assert round(gm, 3) == 1.015
    assert min(w_plus, w_minus) == 28
    assert round(p_enum, 3) == 0.700 and round(res.pvalue, 3) == 0.700
    assert round(t_res.statistic, 2) == 0.42
    assert max(db.SERIAL[k][3] for k in keys) <= 0.97
    print("all paper-quoted values reproduced. OK")


if __name__ == "__main__":
    main()

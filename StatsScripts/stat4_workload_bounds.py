"""Exact workload analysis of the prefix-space partition (Sections 5.3-5.5,
5.9; Tables 4 and 6; the pilot-campaign and depth-sweep bounds).

Reproduces, from the closed forms alone (pure standard library):
  - T_tasks = C(n-r+D, D) and the partition identity  sum W(c_D) = C(n, r);
  - the head-portion share  omega_0  of the 64 lexicographically first tasks
    ("omega_0 ranges between 10.5% and 12.1% ... bounds the speedup by
    1/omega_0 = 8.3-9.5", matched against the pilot campaign's observed S(16)
    plateau of 8.8-10.3);
  - the heaviest-task share W_max/C(n,r) = C(n-D, r-D)/C(n,r)
    ("between 2.1% and 2.7%") and the bound C(n,r)/W_max
    ("between 37.1 and 47.5"), compared with the observed S(56)
    ("exceed it by a stable margin of 5-13%");
  - the depth-dependence of the bound ("already exceeds 99 at D = 6 and 240 at
    D = 7") and the task-list growth ("from 53,130 tasks at D = 5 to
    10,015,005 at D = 9 for C(40,20)").

Requires: Python >= 3.9 only. Data: data_benchmarks.py (same directory).
"""
import math

import data_benchmarks as db

CHUNK = db.OMP_CHUNK_PILOT  # 64: portion size of the pilot campaign


def lex_prefixes(n, r, D):
    """Yield all depth-D prefixes in lexicographical order (Algorithm 9)."""
    U = n - r + D
    P = list(range(1, D + 1))
    while True:
        yield tuple(P)
        j = D - 1
        while j >= 0 and P[j] == U - (D - 1 - j):
            j -= 1
        if j < 0:
            return
        P[j] += 1
        for i in range(j + 1, D):
            P[i] = P[i - 1] + 1


def main():
    D = 5
    print("instance   T_tasks   sumW==C(n,r)  omega0%  1/omega0  S16_obs  "
          "Wmax%   bound   S56_obs  obs/bound")
    om_list, b1_list, s16_list, wm_list, bound_list, ratio_list = [], [], [], [], [], []
    for n, r in db.INSTANCES:
        total = math.comb(n, r)
        head = 0
        sum_w = 0
        count = 0
        for P in lex_prefixes(n, r, D):
            w = math.comb(n - P[-1], r - D)
            sum_w += w
            count += 1
            if count <= CHUNK:
                head += w
        assert count == math.comb(n - r + D, D)      # T_tasks closed form
        assert sum_w == total                        # partition identity
        omega0 = 100.0 * head / total
        s16 = db.PILOT_P1[(n, r)] / db.PILOT_P16[(n, r)]
        wmax = 100.0 * math.comb(n - D, r - D) / total
        bound = total / math.comb(n - D, r - D)
        s56 = db.PARALLEL[(n, r, 1)][0] / db.PARALLEL[(n, r, 56)][0]
        om_list.append(omega0); b1_list.append(100.0 / omega0)
        s16_list.append(s16); wm_list.append(wmax)
        bound_list.append(bound); ratio_list.append(s56 / bound)
        print(f"C({n},{r})  {count:8,}   ok           {omega0:5.2f}   "
              f"{100.0/omega0:6.2f}   {s16:6.2f}  {wmax:5.2f}  {bound:6.1f}  "
              f"{s56:6.2f}   {s56/bound:.3f}")

    print(f"omega0 range {min(om_list):.1f}-{max(om_list):.1f}%; "
          f"1/omega0 range {min(b1_list):.1f}-{max(b1_list):.1f}; "
          f"pilot S(16) observed {min(s16_list):.1f}-{max(s16_list):.1f}")
    print(f"Wmax share {min(wm_list):.1f}-{max(wm_list):.1f}%; "
          f"bound {min(bound_list):.1f}-{max(bound_list):.1f}; "
          f"observed/bound {min(ratio_list):.2f}-{max(ratio_list):.2f}")

    # depth dependence of the bound ("on the six instances of the sweep it
    # already exceeds 99 at D = 6 and 240 at D = 7") and the task-list size
    swept = sorted(set((n, r) for n, r, _ in db.DEPTH))
    min_b6 = min(math.comb(n, r) / math.comb(n - 6, r - 6) for n, r in swept)
    min_b7 = min(math.comb(n, r) / math.comb(n - 7, r - 7) for n, r in swept)
    print(f"minimum bound over the six swept instances: D=6: {min_b6:.1f}; D=7: {min_b7:.1f}")
    all6 = min(math.comb(n, r) / math.comb(n - 6, r - 6) for n, r in db.INSTANCES)
    all7 = min(math.comb(n, r) / math.comb(n - 7, r - 7) for n, r in db.INSTANCES)
    print(f"(for reference, over the full 11-instance ladder: D=6: {all6:.1f}; "
          f"D=7: {all7:.1f} -- still far above p = 56)")
    print(f"T_tasks C(40,20): D=5: {math.comb(25,5):,}; D=9: {math.comb(29,9):,}")

    # values as quoted in the paper
    assert (round(min(om_list), 1), round(max(om_list), 1)) == (10.5, 12.1)
    assert (round(min(b1_list), 1), round(max(b1_list), 1)) == (8.3, 9.5)
    assert (round(min(s16_list), 1), round(max(s16_list), 1)) == (8.8, 10.3)
    assert (round(min(wm_list), 1), round(max(wm_list), 1)) == (2.1, 2.7)
    assert (round(min(bound_list), 1), round(max(bound_list), 1)) == (37.1, 47.5)
    assert (round(min(ratio_list), 2), round(max(ratio_list), 2)) == (1.05, 1.13)
    assert min_b6 >= 99 and min_b7 >= 240
    assert math.comb(25, 5) == 53130 and math.comb(29, 9) == 10015005
    print("all paper-quoted values reproduced. OK")


if __name__ == "__main__":
    main()

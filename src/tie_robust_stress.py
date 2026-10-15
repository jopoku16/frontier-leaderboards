"""Least-favourable check of the gated tie-robust set.

Gating tests the accuracy hypothesis A_ij only once the cost hypothesis C_ij has been rejected. When failed tasks cost more,
the cost difference and the accuracy difference of a pair are positively dependent, and conditioning on the cost test having
rejected could push the accuracy deviation up where the true cost gap is about as large as the critical value. This script
puts every system at the same true accuracy (so every system is in F and any exclusion is an error), spreads true costs so
that gaps are of the order of the critical value, makes failed tasks cost more, and reports how often each set keeps all of F.

usage: python tie_robust_stress.py SPREAD_COST LAM [R] [K] [T]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
import validate_frontier as vf  # noqa: E402
from honestboard.frontier import pareto_frontier  # noqa: E402


def main(spread_cost, lam, R=300, K=10, T=150):
    base = {"corr": 0.0, "spread_acc": 0.0, "spread_cost": spread_cost, "cost_cv": 0.8, "lam": lam, "K": K, "T": T, "label": "stress"}
    Lpool = vf.rng.normal(0.0, 1.2, 2000)
    kept = {"std": 0, "intervals": 0, "gated": 0, "stepdown": 0}
    size = {k: [] for k in kept}
    for r in range(R):
        world = vf.make_world(base, Lpool)
        board, costs, _g = vf.draw_board(world, base, T, Lpool)
        names = board.names
        for key, kw in (("std", {}), ("intervals", {"tie_robust": "intervals"}), ("gated", {"tie_robust": "gated"}), ("stepdown", {"tie_robust": "stepdown"})):
            res = pareto_frontier(board, costs, alpha=0.05, B=800, seed=r, **kw)
            kept[key] += len(res["frontier_set"]) == K                      # every system is in F, so keeping all of F is keeping all
            size[key].append(len(res["frontier_set"]))
    print(f"spread_cost {spread_cost} lam {lam} K={K} T={T} R={R}: " + " | ".join(f"{k} kept-all {kept[k] / R:.3f} size {np.mean(size[k]):.2f}" for k in kept), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]), float(a[1]), int(a[2]) if len(a) > 2 else 300, int(a[3]) if len(a) > 3 else 10, int(a[4]) if len(a) > 4 else 150)

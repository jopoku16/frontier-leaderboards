"""Prototype of a less conservative tie-robust frontier set, compared with the interval version of honestboard 0.5.1.

Version 1 (honestboard 0.5.1): a rival j is "sure" when the cost intervals of j and i (each mean +- c_cost standard errors,
c_cost at alpha/2) do not overlap; the margin is the (1 - alpha/2) quantile of the largest studentised accuracy deviation
over every ordered pair.

Version 2 (this file): (a) the cost comparison is PAIRED: j is a sure rival of i when the mean cost difference exceeds
c times the paired standard error of that difference, which is much smaller than the sum of the two unpaired standard
errors when costs move together across tasks; (b) ONE joint critical value c is the (1 - alpha) quantile of the largest of
all studentised cost-difference deviations and all studentised accuracy-difference deviations, so no alpha-split is needed;
(c) optionally a Romano-Wolf step-down: pairs already excluded with a sure rival are removed from the maximum and c is
recomputed until nothing changes.

Validity (sketch): let E be the event that every pairwise cost difference lies within c paired standard errors of its
mean and every pairwise accuracy deviation is at most c. P(E) >= 1 - alpha - o(1) because c is the bootstrap quantile of the
largest of exactly these deviations. On E a sure rival is a true rival, and for a truly undominated i every true rival has
G_ij <= Z_ij <= c, so i is kept. The step-down is the usual Romano-Wolf argument: until a true null is first rejected the
remaining set contains every true null, so the critical value is at least the quantile of the maximum over the true nulls.

usage: python tie_robust_experiment.py [R]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
import validate_frontier as vf  # noqa: E402
from honestboard.frontier import _bootstrap_pairs, observed_frontier, pareto_frontier  # noqa: E402


def tie_robust_v3(Y, C, alpha=0.05, B=1200, seed=0, return_info=False):
    """Gated step-down (Romano-Wolf) on two families of one-sided hypotheses for every ordered pair (i, j):
         C_ij: j is not strictly cheaper than i   (rejected when (cbar_i - cbar_j) / se_c > c)
         A_ij: j is not strictly more accurate    (tested only once C_ij is rejected; rejected when G_ij > c)
    c is the (1 - alpha) bootstrap quantile of the largest centred deviation over the hypotheses still standing; the
    rejection sets grow and c shrinks until nothing changes. System i is excluded when some j has both C_ij and A_ij rejected."""
    K, T = Y.shape
    rng = np.random.default_rng(seed)
    acc, cost = Y.mean(axis=1), C.mean(axis=1)
    S, SC = _bootstrap_pairs(Y, C, B, rng)
    eye = np.eye(K, dtype=bool)

    def paired_se(M):
        Sig = np.cov(M) / T
        v = np.diag(Sig)[:, None] + np.diag(Sig)[None, :] - 2 * Sig
        return np.sqrt(np.maximum(v, 0.0))

    se_a = paired_se(Y)
    se_a[se_a == 0] = 1e-12
    se_c = paired_se(C)
    scale = np.abs(cost[:, None]) + np.abs(cost[None, :]) + 1e-300
    se_c = np.where(se_c <= 1e-9 * scale, 1e-12, se_c)
    dev_a = ((S[None, :, :] - S[:, None, :]) - (acc[None, :] - acc[:, None])[:, :, None]) / se_a[:, :, None]   # [i, j, b]: j above i
    dev_c = ((SC[:, None, :] - SC[None, :, :]) - (cost[:, None] - cost[None, :])[:, :, None]) / se_c[:, :, None]  # [i, j, b]: j cheaper than i
    stat_a = (acc[None, :] - acc[:, None]) / se_a
    stat_c = (cost[:, None] - cost[None, :]) / se_c
    rc = np.zeros((K, K), dtype=bool)
    ra = np.zeros((K, K), dtype=bool)
    steps = 0
    while True:
        steps += 1
        remc = ~eye & ~rc
        rema = rc & ~ra
        m = np.full(B, -np.inf)
        if remc.any():
            m = np.maximum(m, np.where(remc[:, :, None], dev_c, -np.inf).max(axis=(0, 1)))
        if rema.any():
            m = np.maximum(m, np.where(rema[:, :, None], dev_a, -np.inf).max(axis=(0, 1)))
        c = float(np.quantile(m, 1 - alpha)) if np.isfinite(m).any() else 0.0
        new_c = rc | (remc & (stat_c > c))
        new_a = ra | ((new_c & ~ra) & (stat_a > c))
        if (new_c == rc).all() and (new_a == ra).all():
            break
        rc, ra = new_c, new_a
    excluded = (rc & ra).any(axis=1)
    in_set = ~excluded
    if return_info:
        iu = np.triu_indices(K, 1)
        return in_set, {"c": c, "steps": steps, "sure_share": float((rc[iu] | rc.T[iu]).mean())}
    return in_set


def run(R=100, only=None):
    base = {"corr": 0.6, "spread_acc": 0.8, "spread_cost": 1.0, "cost_cv": 0.8}
    scenarios = [
        {**base, "K": 20, "T": 150, "cost_cv": 0.0, "label": "fixed costs"},
        {**base, "K": 20, "T": 150, "spread_acc": 0.0, "label": "equal accuracies"},
        {**base, "K": 10, "T": 100, "spread_acc": 0.0, "label": "equal accuracies small"},
        {**base, "K": 20, "T": 150, "label": "base"},
        {**base, "K": 8, "T": 50, "label": "base small"},
        {**base, "K": 20, "T": 150, "spread_cost": 0.05, "label": "near-tied costs"},
        {**base, "K": 20, "T": 150, "n_copy": 6, "copy_flip": 0.1, "copy_cost_mult": 0.97, "label": "cheaper near-copies"},
        {**base, "K": 20, "T": 150, "n_copy": 6, "copy_flip": 0.1, "copy_cost_mult": 1.03, "label": "costlier near-copies"},
        {**base, "K": 60, "T": 45, "label": "crowded"},
    ]
    Lpool = vf.rng.normal(0.0, 1.2, 2000)
    for sc in scenarios:
        if only and only not in sc['label']:
            continue
        kept = {k: 0 for k in ("std", "v1", "v3", "sd")}
        size = {k: [] for k in kept}
        sure = []
        for r in range(R):
            world = vf.make_world(sc, Lpool)
            th, ga = world["theta"], world["gamma"]
            rival = (ga[None, :] <= ga[:, None]) & ~np.eye(len(th), dtype=bool)
            truth = ~((th[None, :] > th[:, None]) & rival).any(axis=1)          # F: no true rival is strictly more accurate (the set the propositions cover)
            board, costs, _g = vf.draw_board(world, sc, sc["T"], Lpool)
            names = board.names
            Y = np.asarray(board.Y, float)
            C = np.array([costs[n] for n in names])
            std = pareto_frontier(board, costs, alpha=0.05, B=1000, seed=r)
            v1 = pareto_frontier(board, costs, alpha=0.05, B=1000, seed=r, tie_robust=True)
            m = {"std": np.array([n in set(std["frontier_set"]) for n in names]), "v1": np.array([n in set(v1["frontier_set"]) for n in names])}
            g3 = pareto_frontier(board, costs, alpha=0.05, B=1000, seed=r, tie_robust="gated")
            sd = pareto_frontier(board, costs, alpha=0.05, B=1000, seed=r, tie_robust="stepdown")
            m["v3"] = np.array([n in set(g3["frontier_set"]) for n in names]); m["sd"] = np.array([n in set(sd["frontier_set"]) for n in names])
            info = {"sure_share": g3.get("cost_ordered_share") or 0.0}
            for k in kept:
                kept[k] += bool(np.all(m[k][truth]))
                size[k].append(int(m[k].sum()))
            sure.append(info["sure_share"])
        print(f"{sc['label']:22s} K={sc['K']:2d} T={sc['T']:3d} | " + " | ".join(f"{k} {kept[k] / R:.2f} size {np.mean(size[k]):.1f}" for k in kept) + f" | sure share {np.mean(sure):.2f}", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 100, sys.argv[2] if len(sys.argv) > 2 else None)

"""Does the frontier winner's-curse correction work? Known-truth simulation.

Boards are simulated as in validate_frontier.py (shared task difficulty, lognormal per-task costs). On each board,
every system on the observed frontier that has at least one cheaper-or-equal rival is a "frontier winner": it was
selected for being the most accurate system of its pool. For each such system we record the error of four estimates
of its true accuracy (reported, plug-in bootstrap, iterated bootstrap, conditional) and whether each interval covers
the truth. Costs enter by their mean per task (fixed pool), as the method assumes.

usage: python validate_curse.py [R] [--quick]
Writes results/validation_curse.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from honestboard.frontier import frontier_winner, observed_frontier  # noqa: E402
import validate_frontier as vf  # noqa: E402

METHODS = ("naive", "bootstrap", "iterated", "conditional")


def run(R, quick):
    vf.rng = np.random.default_rng(7)
    Lpool = vf.rng.normal(0.0, 1.2, 2000)
    scenarios = [{"K": K, "T": T, "corr": 0.6, "spread_acc": 0.8, "spread_cost": 1.0}
                 for K in (8, 20) for T in ((50, 150) if quick else (50, 150, 500))]
    B, B_out, B_in = (400, 60, 60) if quick else (1000, 100, 100)
    out = {"R": R, "alpha": 0.05, "B": B, "B_outer": B_out, "B_inner": B_in, "scenarios": []}
    for sc in scenarios:
        t0 = time.time()
        K, T = sc["K"], sc["T"]
        rows = []
        n_members = n_cheapest = 0
        for r in range(R):
            ability, mu = vf.make_truth(K, sc["spread_acc"], sc["spread_cost"], sc["corr"])
            theta = vf.true_accuracy(ability, Lpool)
            board, costs = vf.simulate_board(ability, mu, T, Lpool)
            names = board.names
            acc = board.scores()
            cost = np.array([float(np.mean(costs[n])) for n in names])
            onf = observed_frontier(acc, cost)
            for k in np.where(onf)[0]:
                n_members += 1
                pool = np.where(cost <= cost[k])[0]
                if len(pool) == 1:
                    n_cheapest += 1
                    continue
                res = frontier_winner(board, costs, names[k], alpha=0.05, B=B, seed=1000 * r + int(k), iterate=False)
                it = frontier_winner(board, costs, names[k], alpha=0.05, B=B, seed=1000 * r + int(k), iterate=True, B_outer=B_out, B_inner=B_in)
                truth = float(theta[k])
                ests = {"naive": (res["observed"], res["naive_ci"]), "bootstrap": (res["bootstrap"]["estimate"], res["bootstrap"]["ci"]),
                        "iterated": (it["bootstrap"]["estimate"], it["bootstrap"]["ci"]),
                        "conditional": (res["conditional"]["estimate"], res["conditional"]["ci"])}
                row = {"pool": int(len(pool)), "truly_undominated": bool(observed_frontier(theta, mu)[k])}
                for m, (est, ci) in ests.items():
                    lo, hi = ci
                    ok = np.isfinite(est) and not (np.isnan(lo) or np.isnan(hi))
                    row[m] = {"err": float(est - truth) if ok else np.nan, "cov": float(lo <= truth <= hi) if ok else np.nan,
                              "len": float(hi - lo) if ok and np.isfinite(lo) and np.isfinite(hi) else (np.inf if ok else np.nan)}
                rows.append(row)
        entry = {**sc, "n_frontier_members": n_members, "n_cheapest_no_rival": n_cheapest, "n_winners": len(rows),
                 "share_truly_undominated": float(np.mean([r["truly_undominated"] for r in rows])) if rows else None,
                 "mean_pool": float(np.mean([r["pool"] for r in rows])) if rows else None, "methods": {}}
        E = {m: np.array([r[m]["err"] for r in rows], dtype=float) for m in METHODS}
        common = np.all([np.isfinite(E[m]) for m in METHODS], axis=0) if rows else np.array([], bool)
        for m in METHODS:
            err, cov = E[m], np.array([r[m]["cov"] for r in rows], dtype=float)
            ln = np.array([r[m]["len"] for r in rows], dtype=float)
            fin = np.isfinite(err)
            nf = int(fin.sum())
            entry["methods"][m] = {
                "finite": nf, "bias_pts": float(err[fin].mean() * 100) if nf else None,
                "se_bias_pts": float(err[fin].std(ddof=1) / np.sqrt(nf) * 100) if nf > 1 else None,
                "median_err_pts": float(np.median(err[fin]) * 100) if nf else None,
                "rmse_pts": float(np.sqrt((err[fin] ** 2).mean()) * 100) if nf else None,
                "coverage": float(np.nanmean(cov)) if np.isfinite(cov).any() else None,
                "median_len_pts": float(np.median(ln[np.isfinite(ln)]) * 100) if np.isfinite(ln).any() else None,
                "bias_common_pts": float(err[common].mean() * 100) if common.any() else None}
        entry["n_common"] = int(common.sum())
        entry["seconds"] = round(time.time() - t0)
        out["scenarios"].append(entry)
        print(f"K={K:2d} T={T:3d}: winners {len(rows)} (cheapest {n_cheapest}), pool {entry['mean_pool']:.1f}, truly undominated {entry['share_truly_undominated']:.2f}", flush=True)
        for m in METHODS:
            v = entry["methods"][m]
            cov = "n/a" if v["coverage"] is None else f"{v['coverage']:.3f}"
            print(f"   {m:12s} bias {v['bias_pts']:+.2f} (se {v['se_bias_pts']:.2f}) median {v['median_err_pts']:+.2f} rmse {v['rmse_pts']:.2f} "
                  f"cov {cov} len {v['median_len_pts']:.1f} finite {v['finite']}/{len(rows)}  common bias {v['bias_common_pts']:+.2f}", flush=True)
        print(f"   ({entry['seconds']}s)", flush=True)
    (ROOT / "results" / "validation_curse.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
    run(R, "--quick" in sys.argv)

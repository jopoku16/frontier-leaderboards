"""Held-out check on the real HAL boards: select on one half of the tasks, score on the other half.

For every board with at least 100 tasks, the tasks are split at random into two halves R times. On half A the
observed frontier, the frontier confidence set and the frontier winner's-curse estimates are computed exactly as in
the audit (per-task costs resampled where they exist). Half B then plays the role of fresh data:

  curse      for each system on A's observed frontier with at least one cheaper rival, the error of the A-half
             reported accuracy, the iterated-bootstrap estimate and the conditional estimate against the B-half
             accuracy (a held-out, selection-free measurement of the same system), and the coverage of each interval;
  frontier   how many systems on B's observed frontier (with B's costs) are kept in A's frontier set, against how many
             are on A's observed frontier: the set should keep nearly all of them, the plain frontier fewer.

Half-to-half differences in difficulty cancel in paired comparisons but not in a single system's accuracy, so the
B-half accuracy is adjusted by the mean over all systems of (B accuracy - A accuracy), as in the companion paper.

usage: python validate_heldout.py [R]
Writes results/validation_heldout.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from honestboard import Board  # noqa: E402
from honestboard.frontier import frontier_winner, observed_frontier, pareto_frontier  # noqa: E402

BOARDS = ROOT / "data" / "boards"
METHODS = ("naive", "iterated", "conditional")


def sub_board(bj, task_idx):
    tasks = [bj["tasks"][i] for i in task_idx]
    systems = [{"name": s["name"], "date": s.get("date"), "rates": {t: s["rates"][t] for t in tasks}} for s in bj["systems"]]
    return Board.from_rates(tasks, systems), tasks


def main(R):
    rng = np.random.default_rng(19)
    out = {"R": R, "boards": {}}
    paths = sorted(p for p in BOARDS.glob("hal_*.json") if not p.name.endswith("_costs.json") and p.name != "hal_summary.json")
    for p in paths:
        bj = json.loads(p.read_text(encoding="utf-8"))
        T = len(bj["tasks"])
        if T < 100:
            continue
        cost_map = json.loads(p.with_name(p.stem + "_costs.json").read_text(encoding="utf-8"))
        fixed = bj.get("cost_source") == "run_total_even"
        t0 = time.time()
        rows, kept_obsB, onA_obsB, nB = [], 0, 0, 0
        for r in range(R):
            perm = rng.permutation(T)
            A, B = sorted(perm[: T // 2]), sorted(perm[T // 2:])
            bA, tA = sub_board(bj, A)
            bB, tB = sub_board(bj, B)
            names = bA.names
            cA = {n: (float(np.mean([cost_map[n][t] for t in bj["tasks"]])) if fixed else np.array([cost_map[n][t] for t in tA])) for n in names if n in cost_map}
            cB = {n: (float(np.mean([cost_map[n][t] for t in bj["tasks"]])) if fixed else np.array([cost_map[n][t] for t in tB])) for n in names if n in cost_map}
            res = pareto_frontier(bA, cA, alpha=0.05, B=1000, seed=r)
            accA, accB = bA.scores(), bB.scores()
            shift = float(np.mean(accB - accA))                       # half-to-half difficulty difference, field-wide
            costA = np.array([float(np.mean(cA[n])) for n in names])
            costB = np.array([float(np.mean(cB[n])) for n in names])
            onA = observed_frontier(accA, costA)
            onB = observed_frontier(accB, costB)
            in_set = {row["system"]: row["in_frontier_set"] for row in res["table"]}
            for k in np.where(onB)[0]:
                nB += 1
                kept_obsB += bool(in_set[names[k]])
                onA_obsB += bool(onA[k])
            for k in np.where(onA)[0]:
                pool = np.where(costA <= costA[k])[0]
                if len(pool) == 1:
                    continue
                w = frontier_winner(bA, cA, names[k], alpha=0.05, B=1000, seed=100 * r + int(k), iterate=True, B_outer=80, B_inner=80)
                truth = float(accB[k] - shift)
                ests = {"naive": (w["observed"], w["naive_ci"]), "iterated": (w["bootstrap"]["estimate"], w["bootstrap"]["ci"]),
                        "conditional": (w["conditional"]["estimate"], w["conditional"]["ci"])}
                row = {}
                for m, (est, ci) in ests.items():
                    ok = np.isfinite(est) and not (np.isnan(ci[0]) or np.isnan(ci[1]))
                    row[m] = {"err": float(est - truth) if ok else np.nan, "cov": float(ci[0] <= truth <= ci[1]) if ok else np.nan}
                rows.append(row)
        entry = {"tasks": T, "systems": bj["systems"].__len__(), "cost_fixed": fixed, "n_winners": len(rows),
                 "obsB_members": nB, "obsB_kept_in_setA": kept_obsB / max(1, nB), "obsB_on_obsA": onA_obsB / max(1, nB), "methods": {}}
        for m in METHODS:
            err = np.array([row[m]["err"] for row in rows], dtype=float)
            cov = np.array([row[m]["cov"] for row in rows], dtype=float)
            fin = np.isfinite(err)
            entry["methods"][m] = {"bias_pts": float(err[fin].mean() * 100) if fin.any() else None,
                                   "se_bias_pts": float(err[fin].std(ddof=1) / np.sqrt(fin.sum()) * 100) if fin.sum() > 1 else None,
                                   "rmse_pts": float(np.sqrt((err[fin] ** 2).mean()) * 100) if fin.any() else None,
                                   "coverage": float(np.nanmean(cov)) if np.isfinite(cov).any() else None, "finite": int(fin.sum())}
        entry["seconds"] = round(time.time() - t0)
        out["boards"][p.stem] = entry
        print(f"{p.stem:34s} T={T:4d} winners {len(rows):4d} | B-frontier members kept in A's set {entry['obsB_kept_in_setA']:.2f}, "
              f"on A's plain frontier {entry['obsB_on_obsA']:.2f} | bias naive {entry['methods']['naive']['bias_pts']:+.1f} "
              f"iterated {entry['methods']['iterated']['bias_pts']:+.1f} conditional {entry['methods']['conditional']['bias_pts']:+.1f} | "
              f"cov {entry['methods']['naive']['coverage']:.2f}/{entry['methods']['iterated']['coverage']:.2f}/{entry['methods']['conditional']['coverage']:.2f} ({entry['seconds']}s)", flush=True)
    (ROOT / "results" / "validation_heldout.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)

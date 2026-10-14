"""Run-to-run noise from HAL's repeated runs of one configuration.

HAL holds a few configurations (same agent, same model and settings, same benchmark) that were run more than once.
Each pair measures how far two runs of one system differ on the same tasks.

A single run's standard error already contains independent within-task run noise: it comes from the sample variance of the
run's own per-task outcomes, and a per-task outcome that is a fresh Bernoulli draw each run contributes its Bernoulli
variance to that sample variance. If outcomes flip independently across tasks, two runs' accuracies differ with variance
pi / T, pi being the share of tasks on which the two runs disagree. Only variation BEYOND that (a run-level component:
a lucky or unlucky run, API drift, a shared seed, which moves many tasks together) is missing from the standard errors.
Per pair, the excess variance of a run is (d^2 - pi / T) / 2; pooled, sd_excess = sqrt(max(0, mean excess)).
The evidence is weak (eight pairs, six from one 50-task board), so the sensitivity below is labelled as such.

Sensitivity: add 2 * sd_excess^2 to every paired variance (both systems carry a run-level effect), on every board,
and recompute the frontier set. Reports set sizes before and after.

usage: python rerun_noise.py
Writes results/rerun_noise.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from honestboard import Board  # noqa: E402
from honestboard.frontier import pareto_frontier  # noqa: E402

BOARDS = ROOT / "data" / "boards"


def config_key(name):
    return name.rstrip("_").rsplit("__", 1)[0]


def main():
    paths = sorted(p for p in BOARDS.glob("hal_*.json") if not p.name.endswith("_costs.json") and p.name != "hal_summary.json")
    pairs, boards = [], {}
    for p in paths:
        bj = json.loads(p.read_text(encoding="utf-8"))
        board = Board.from_json(str(p))
        cost_map = json.loads(p.with_name(p.stem + "_costs.json").read_text(encoding="utf-8"))
        tasks = bj["tasks"]
        boards[p.stem] = (bj, board, cost_map, tasks)
        groups = {}
        for s in bj["systems"]:
            groups.setdefault(config_key(s["name"]), []).append(s["name"])
        for key, names in groups.items():
            if len(names) < 2:
                continue
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    ia, ib = board.index(names[a]), board.index(names[b])
                    ya, yb = board.Y[ia].astype(float), board.Y[ib].astype(float)
                    ca = float(np.mean([cost_map[names[a]][t] for t in tasks])); cb = float(np.mean([cost_map[names[b]][t] for t in tasks]))
                    agree = float(np.mean(ya == yb)) if board.is_binary else None
                    pairs.append({"board": p.stem, "config": key, "runs": [names[a], names[b]], "tasks": board.T,
                                  "acc": [float(ya.mean()), float(yb.mean())], "acc_diff": float(ya.mean() - yb.mean()),
                                  "task_agreement": agree, "task_se": float(ya.std(ddof=1) / np.sqrt(board.T)),
                                  "cost": [ca, cb], "cost_ratio": float(max(ca, cb) / max(1e-12, min(ca, cb)))})
    pairs = [q for q in pairs if q["task_agreement"] is not None]
    d = np.array([q["acc_diff"] for q in pairs])
    T = np.array([q["tasks"] for q in pairs], float)
    pi = np.array([1 - q["task_agreement"] for q in pairs])
    var_ind = pi / T                                              # variance of a difference of two runs under independent flips
    rms_obs, rms_ind = float(np.sqrt(np.mean(d ** 2))), float(np.sqrt(np.mean(var_ind)))
    chi = float(np.sum(d ** 2 / var_ind))
    pval = float(1 - stats.chi2.cdf(chi, len(d)))
    sd_excess = float(np.sqrt(max(0.0, np.mean(d ** 2 - var_ind) / 2)))
    out = {"pairs": pairs, "n_pairs": len(pairs), "n_boards_with_pairs": len({q["board"] for q in pairs}),
           "pairs_per_board": {b: sum(1 for q in pairs if q["board"] == b) for b in {q["board"] for q in pairs}},
           "rms_diff_observed": rms_obs, "rms_diff_independent_flips": rms_ind, "chi2": chi, "chi2_df": len(d), "chi2_p": pval,
           "sd_excess_per_run": sd_excess, "mean_abs_acc_diff": float(np.mean(np.abs(d))), "max_abs_acc_diff": float(np.max(np.abs(d))),
           "median_task_se_single_run": float(np.median([q["task_se"] for q in pairs])),
           "median_cost_ratio": float(np.median([q["cost_ratio"] for q in pairs])), "max_cost_ratio": float(max(q["cost_ratio"] for q in pairs)),
           "boards": {}}
    print(f"{len(pairs)} rerun pairs on {out['n_boards_with_pairs']} boards ({out['pairs_per_board']}); observed RMS difference {100 * rms_obs:.1f} pt "
          f"vs {100 * rms_ind:.1f} pt under independent flips; chi-square {chi:.1f} on {len(d)} df, p = {pval:.3f}; excess run-level sd {100 * sd_excess:.1f} pt; "
          f"task se of one run (median) {100 * out['median_task_se_single_run']:.1f} pt; cost ratio median {out['median_cost_ratio']:.2f}, max {out['max_cost_ratio']:.2f}")
    for stem, (bj, board, cost_map, tasks) in boards.items():
        costs = {s: np.array([cost_map[s][t] for t in tasks]) for s in board.names if s in cost_map}
        base = pareto_frontier(board, costs, alpha=0.05, B=4000, seed=0)
        # keep a system unless its strongest cheaper rival's lead, studentised with the paired variance PLUS 2 sd_excess^2,
        # still exceeds the board's margin
        keep = []
        for row in base["table"]:
            if row["rival_lead_se"] is None or row["rival_lead"] is None or row["rival_lead_se"] <= 0:
                keep.append(row["system"]); continue
            se_pt = abs(row["rival_lead"]) / row["rival_lead_se"]
            lead_new = row["rival_lead"] / np.sqrt(se_pt ** 2 + 2 * sd_excess ** 2)
            if not (lead_new > base["margin_se"]):
                keep.append(row["system"])
        out["boards"][stem] = {"set": len(base["frontier_set"]), "set_with_run_level_noise": len(keep), "systems": board.K}
        print(f"   {stem:34s} set {len(base['frontier_set']):3d} -> {len(keep):3d} with a run-level sd of {100 * sd_excess:.1f} pt")
    (ROOT / "results" / "rerun_noise.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Run-to-run noise from HAL's repeated runs of one configuration, and what it does to the frontier set.

HAL holds a few configurations (same agent, same model and settings, same benchmark) that were run more than once.
Each pair gives one direct measurement of run-to-run variation, which the task-sampling model does not include.
For each pair: the accuracy difference, the per-task agreement rate, and the ratio of mean costs. Pooled over pairs,
the run-to-run standard deviation of accuracy is sd_run = sqrt(mean(d^2) / 2), to be compared with the task-sampling
standard error of a single run's accuracy at the board's task count.

Sensitivity: run-to-run noise is carried to every board through the per-task disagreement rate pi between reruns
(the share of tasks whose outcome differs between two runs of one configuration, pooled over the pairs): if outcomes
flip independently across tasks, two runs' accuracies differ with variance pi/T, so one run's run-to-run standard
deviation is sd_run(T) = sqrt(pi / (2T)), which shrinks with the task count as the task-sampling error does. On every
board the paired standard errors are inflated to carry that variance for both systems, se'^2 = se^2 + 2 sd_run(T)^2,
and the frontier set is recomputed with the inflated margins. Reports the set sizes before and after.

usage: python rerun_noise.py
Writes results/rerun_noise.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from honestboard import Board  # noqa: E402
from honestboard.frontier import pareto_frontier  # noqa: E402

BOARDS = ROOT / "data" / "boards"


def config_key(name):
    return name.rstrip("_").rsplit("__", 1)[0]


def main():
    paths = sorted(p for p in BOARDS.glob("hal_*.json") if not p.name.endswith("_costs.json") and p.name != "hal_summary.json")
    pairs = []
    boards = {}
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
                    pairs.append({"board": p.stem, "config": key, "runs": [names[a], names[b]], "tasks": board.T,
                                  "acc": [float(ya.mean()), float(yb.mean())], "acc_diff": float(ya.mean() - yb.mean()),
                                  "task_agreement": float(np.mean(ya == yb)) if board.is_binary else None,
                                  "cost": [ca, cb], "cost_ratio": float(max(ca, cb) / max(1e-12, min(ca, cb))),
                                  "task_se_single_run": float(np.sqrt(max(ya.var(ddof=1), yb.var(ddof=1)) / board.T))})
    d = np.array([q["acc_diff"] for q in pairs])
    sd_run = float(np.sqrt(np.mean(d ** 2) / 2)) if len(d) else float("nan")
    agree_rates = [q["task_agreement"] for q in pairs if q["task_agreement"] is not None]
    pi = float(np.mean([1 - a for a in agree_rates])) if agree_rates else float("nan")      # per-task disagreement between reruns
    sd_model_at_pairs = float(np.mean([np.sqrt(pi / (2 * q["tasks"])) for q in pairs])) if pairs else float("nan")
    out = {"pairs": pairs, "n_pairs": len(pairs), "n_boards_with_pairs": len({q["board"] for q in pairs}),
           "sd_run_accuracy": sd_run, "mean_abs_acc_diff": float(np.mean(np.abs(d))) if len(d) else None,
           "task_disagreement_rate": pi, "sd_run_model_at_pair_boards": sd_model_at_pairs,
           "median_cost_ratio": float(np.median([q["cost_ratio"] for q in pairs])) if pairs else None,
           "max_cost_ratio": float(max(q["cost_ratio"] for q in pairs)) if pairs else None,
           "median_task_se_single_run": float(np.median([q["task_se_single_run"] for q in pairs])) if pairs else None,
           "boards": {}}
    print(f"{len(pairs)} rerun pairs on {out['n_boards_with_pairs']} boards; sd_run (accuracy) {100 * sd_run:.1f} pt (flip model at those boards: {100 * sd_model_at_pairs:.1f} pt, pi {pi:.2f}); "
          f"mean |diff| {100 * out['mean_abs_acc_diff']:.1f} pt; task se of one run (median) {100 * out['median_task_se_single_run']:.1f} pt; "
          f"cost ratio median {out['median_cost_ratio']:.2f}, max {out['max_cost_ratio']:.2f}")
    for q in pairs:
        print(f"   {q['board']:30s} {q['config'][:40]:40s} acc {100 * q['acc'][0]:5.1f} vs {100 * q['acc'][1]:5.1f} (diff {100 * q['acc_diff']:+5.1f}) "
              f"agree {q['task_agreement'] if q['task_agreement'] is None else round(q['task_agreement'], 2)}  cost ratio {q['cost_ratio']:.2f}")
    # sensitivity: inflate the paired standard errors by the run-to-run variance of both systems
    for stem, (bj, board, cost_map, tasks) in boards.items():
        costs = {s: np.array([cost_map[s][t] for t in tasks]) for s in board.names if s in cost_map}
        base = pareto_frontier(board, costs, alpha=0.05, B=4000, seed=0)
        T = board.T
        # inflating every se by the same factor f = sqrt(1 + 2 sd_run^2 / se_med^2) is equivalent to dividing the
        # margin by f; the set is recomputed from the table's rival leads in se units
        se_med = np.median([abs(row["rival_lead"] / row["rival_lead_se"]) for row in base["table"]
                            if row["rival_lead_se"] not in (None, 0) and row["rival_lead"] is not None]) if base["table"] else float("nan")
        sd_T = float(np.sqrt(pi / (2 * T)))                                   # run-to-run sd of one run's accuracy at this board's T
        f = float(np.sqrt(1 + 2 * sd_T ** 2 / se_med ** 2)) if se_med and np.isfinite(se_med) else 1.0
        in_set_inflated = [row["system"] for row in base["table"]
                           if row["rival_lead_se"] is None or row["rival_lead_se"] / f <= base["margin_se"]]
        out["boards"][stem] = {"set": len(base["frontier_set"]), "set_with_run_noise": len(in_set_inflated), "systems": board.K,
                               "se_median_points": float(100 * se_med) if np.isfinite(se_med) else None, "inflation_factor": f,
                               "sd_run_points": float(100 * sd_T)}
        print(f"   {stem:34s} set {len(base['frontier_set']):3d} -> {len(in_set_inflated):3d} with run noise (se x {f:.2f})")
    (ROOT / "results" / "rerun_noise.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

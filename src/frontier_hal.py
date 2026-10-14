"""Run the frontier analysis on every HAL board built by build_hal_boards.py.

For each benchmark: load the board and its per-task costs, compute the observed frontier, the confidence set for the
frontier, each system's undominated frequency, the confidence set of the best and the cheapest plausible best, and
write results/frontier_hal.json plus a short printed table.

usage: python frontier_hal.py [--alpha 0.05] [--B 4000] [benchmark-slug ...]
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


def main(argv):
    alpha = float(argv[argv.index("--alpha") + 1]) if "--alpha" in argv else 0.05
    B = int(argv[argv.index("--B") + 1]) if "--B" in argv else 4000
    slugs = [a for a in argv if not a.startswith("--") and not a.replace(".", "").isdigit()]
    paths = sorted(BOARDS.glob("hal_*.json"))
    paths = [p for p in paths if not p.name.endswith("_costs.json") and p.name != "hal_summary.json"]
    if slugs:
        paths = [p for p in paths if any(s in p.name for s in slugs)]
    out = {}
    for p in paths:
        board = Board.from_json(str(p))
        cost_map = json.loads(p.with_name(p.stem + "_costs.json").read_text(encoding="utf-8"))
        bj = json.loads(p.read_text(encoding="utf-8"))
        tasks = bj["tasks"]
        if bj.get("cost_source") == "run_total_even":                      # run totals only: one fixed number per system
            costs = {s: float(sum(cost_map[s][t] for t in tasks)) / len(tasks) for s in board.names if s in cost_map}   # mean per task
        else:
            costs = {s: np.array([cost_map[s][t] for t in tasks]) for s in board.names if s in cost_map}
        if len(costs) < 2:
            print(f"{p.stem}: fewer than two systems with costs; skipped")
            continue
        r = pareto_frontier(board, costs, alpha=alpha, B=B, seed=0)
        r["cost_source"] = bj.get("cost_source", "per_task_calls")       # "run_total_even": cost fixed, not resampled
        if r["per_task_costs"]:                                           # the same set with costs fixed at their means
            fixed = {s: float(np.mean(c)) for s, c in costs.items()}
            rf = pareto_frontier(board, fixed, alpha=alpha, B=B, seed=0)
            r["frontier_set_fixed"] = rf["frontier_set"]
            r["margin_se_fixed"] = rf["margin_se"]
        else:
            r["frontier_set_fixed"] = r["frontier_set"]
            r["margin_se_fixed"] = r["margin_se"]
        out[p.stem] = r
        print(f"\n{p.stem}: {r['systems']} systems, {r['tasks']} tasks; observed frontier {len(r['observed_frontier'])}, "
              f"frontier set {len(r['frontier_set'])}, best set {len(r['best_set'])}; margin {r['margin_se']:.2f} se"
              + ("  [cost fixed: run totals]" if r["cost_source"] == "run_total_even" else ""))
        print(f"  top: {r['top']['system']} ({100*r['top']['accuracy']:.1f}%, ${r['top']['cost']:.3f}/task); "
              f"cheapest plausible best: {r['cheapest_plausible_best']['system']} ({100*r['cheapest_plausible_best']['accuracy']:.1f}%, ${r['cheapest_plausible_best']['cost']:.3f}/task)")
        for row in r["table"]:
            flag = ("F" if row["on_observed_frontier"] else "-") + ("S" if row["in_frontier_set"] else "-")
            print(f"  {flag} {row['system'][:58]:58s} acc {100*row['accuracy']:5.1f}  $/task {row['cost']:7.3f}  undominated {row['undominated_frequency']:.2f}")
    (ROOT / "results" / "frontier_hal.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

"""How much do the audit's frontiers depend on defensible costing choices?

Three costings of the SAME runs and outcomes, from the stored per-task token counts (each model call counted once):
  baseline      HAL's current price table; cache reads and writes at the table's cache price; a task with an outcome but no
                logged model call imputed at its run's mean costed task (the audit's rule)
  full cache    cache reads and writes priced at the full input price (no cache discount), as HAL itself did for some
                early runs
  no imputation a task with no logged model call costs nothing (the rule the audit replaced)
and, from frontier_flips.py, HAL's reported totals and the once-counted totals at historical prices (the counting and the
price-date contrasts). Online-Mind2Web has run totals only and is left out of the first two.

For each board: the plain observed frontier and the joint frontier set under each costing, and how many systems change
membership against the baseline.

usage: python hal_sensitivity.py
Writes results/hal_sensitivity.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import hal_extract as hx  # noqa: E402
from honestboard import Board  # noqa: E402
from honestboard.frontier import observed_frontier, pareto_frontier  # noqa: E402

BOARDS = ROOT / "data" / "boards"
RUNS = ROOT / "data" / "hal" / "runs"
CACHE_BACKUP = dict(hx.CACHE_PRICES)


def run_costs(rec, tasks, full_cache, impute):
    """Per-task costs of one run under a costing rule; None for a task with no priced call."""
    hx.CACHE_PRICES = {} if full_cache else CACHE_BACKUP
    out = {}
    for t in tasks:
        toks = (rec["tasks"].get(t) or {}).get("tokens") or {}
        total, ok = 0.0, bool(toks)
        for m, tk in toks.items():
            pc = hx._price(m, tk.get("prompt", 0.0), tk.get("cached", 0.0), tk.get("cache_creation", 0.0), tk.get("completion", 0.0))
            if pc is None:
                ok = False
            else:
                total += pc
        out[t] = total if ok else None
    hx.CACHE_PRICES = CACHE_BACKUP
    costed = [v for v in out.values() if v is not None]
    mean_c = float(np.mean(costed)) if costed else 0.0
    return {t: (v if v is not None else (mean_c if impute else 0.0)) for t, v in out.items()}


def main():
    base = json.loads((ROOT / "results" / "frontier_hal.json").read_text(encoding="utf-8"))
    out = {}
    paths = sorted(p for p in BOARDS.glob("hal_*.json") if not p.name.endswith("_costs.json") and p.name != "hal_summary.json")
    for p in paths:
        bj = json.loads(p.read_text(encoding="utf-8"))
        if bj.get("cost_source") == "run_total_even":
            continue
        board = Board.from_json(str(p))
        tasks = bj["tasks"]
        recs = {s["name"]: json.loads((RUNS / (s["source"][:-4] + ".json")).read_text(encoding="utf-8")) for s in bj["systems"]}
        b0 = base[p.stem]
        row = {"systems": board.K, "tasks": len(tasks), "baseline_frontier": len(b0["observed_frontier"]), "baseline_set": len(b0["frontier_set"]), "variants": {}}
        for label, full_cache, impute in (("full_cache", True, True), ("no_imputation", False, False)):
            rc = {n: run_costs(recs[n], tasks, full_cache, impute) for n in board.names}                  # once per system
            costs = {n: np.array([rc[n][t] for t in tasks]) for n in board.names}
            rb = {n: run_costs(recs[n], tasks, False, True) for n in board.names}
            base_means = [float(np.mean([rb[n][t] for t in tasks])) for n in board.names]
            res = pareto_frontier(board, costs, alpha=0.05, B=2000, seed=0)
            f_base, f_new = set(b0["observed_frontier"]), set(res["observed_frontier"])
            s_base, s_new = set(b0["frontier_set"]), set(res["frontier_set"])
            row["variants"][label] = {"frontier": len(f_new), "frontier_changed": len(f_base ^ f_new), "frontier_left": len(f_base - f_new), "frontier_joined": len(f_new - f_base),
                                      "set": len(s_new), "set_changed": len(s_base ^ s_new),
                                      "mean_cost_ratio": float(np.mean([costs[n].mean() for n in board.names]) / np.mean(base_means))}
        out[p.stem] = row
        v1, v2 = row["variants"]["full_cache"], row["variants"]["no_imputation"]
        print(f"{p.stem:34s} frontier {row['baseline_frontier']:2d}, set {row['baseline_set']:3d} | full cache price: frontier changes {v1['frontier_changed']} "
              f"(set {v1['set']}, changes {v1['set_changed']}), costs x{v1['mean_cost_ratio']:.2f} | no imputation: frontier changes {v2['frontier_changed']} "
              f"(set {v2['set']}, changes {v2['set_changed']})", flush=True)
    (ROOT / "results" / "hal_sensitivity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

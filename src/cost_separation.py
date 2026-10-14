"""How many pairs of systems can be told apart on cost?

The tie-robust set of Section 3.3 excludes a system only on the evidence of rivals that are surely no costlier, so the share of
system pairs whose cost intervals (mean +- c standard errors, with the bootstrap c of the board) do not overlap says how far a
board is from the nearly tied case. For each HAL board with per-task costs the share is computed from the stored costs; for
reference it is computed on simulated boards of the base design and of the nearly tied scenario of validate_frontier.py.

usage: python cost_separation.py [R]
Writes results/cost_separation.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import validate_frontier as vf  # noqa: E402


def separated_share(C, c):
    mean = C.mean(axis=1)
    sg = C.std(axis=1, ddof=1) / np.sqrt(C.shape[1])
    lo, hi = mean - c * sg, mean + c * sg
    sep = (hi[None, :] <= lo[:, None]) | (hi[:, None] <= lo[None, :])
    iu = np.triu_indices(len(mean), 1)
    return float(sep[iu].mean())


def main(R):
    audit = json.loads((ROOT / "results" / "frontier_hal.json").read_text(encoding="utf-8"))
    out = {"hal": {}, "sim": {}, "R": R}
    for k, v in audit.items():
        if not v.get("per_task_costs"):
            continue
        p = ROOT / "data" / "boards" / f"{k}.json"
        bj = json.loads(p.read_text(encoding="utf-8"))
        cost_map = json.loads(p.with_name(p.stem + "_costs.json").read_text(encoding="utf-8"))
        names = [s["name"] for s in bj["systems"] if s["name"] in cost_map]
        C = np.array([[cost_map[n][t] for t in bj["tasks"]] for n in names])
        out["hal"][k] = {"systems": len(names), "tasks": len(bj["tasks"]), "separated_share": separated_share(C, v["cost_interval_c"]), "c": v["cost_interval_c"]}
        print(f"{k:36s} {out['hal'][k]['separated_share']:.2f}", flush=True)
    Lpool = vf.rng.normal(0.0, 1.2, 2000)
    base = {"corr": 0.6, "spread_acc": 0.8, "spread_cost": 1.0, "cost_cv": 0.8, "K": 20, "T": 150, "label": "base"}
    for label, sc in (("base", base), ("near-tied costs", {**base, "spread_cost": 0.05, "label": "near-tied costs"})):
        shares = []
        for _ in range(R):
            world = vf.make_world(sc, Lpool)
            board, costs, _g = vf.draw_board(world, sc, sc["T"], Lpool)
            C = np.array([costs[n] for n in board.names])
            shares.append(separated_share(C, 3.4))
        out["sim"][label] = float(np.mean(shares))
        print(f"simulated {label}: {out['sim'][label]:.2f}", flush=True)
    (ROOT / "results" / "cost_separation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)

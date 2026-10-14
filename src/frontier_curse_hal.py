"""The frontier winner's curse on HAL: selection-adjusted accuracy for every system on an observed frontier.

For each board and each system on the observed frontier, reports the reported accuracy, the conditional
(polyhedral) estimate and interval given that the system is the most accurate of its pool (the systems that cost no
more), and the iterated-bootstrap estimate and interval. Costs are fixed at each system's mean cost per task.
A frontier member whose adjusted accuracy falls below its strongest cheaper rival's reported accuracy is flagged:
at its adjusted accuracy it would not be on the frontier.

usage: python frontier_curse_hal.py [--B 4000]
Writes results/curse_hal.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from honestboard import Board  # noqa: E402
from honestboard.frontier import frontier_curse  # noqa: E402

BOARDS = ROOT / "data" / "boards"


def main(argv):
    B = int(argv[argv.index("--B") + 1]) if "--B" in argv else 4000
    paths = sorted(p for p in BOARDS.glob("hal_*.json") if not p.name.endswith("_costs.json") and p.name != "hal_summary.json")
    out = {}
    for p in paths:
        bj = json.loads(p.read_text(encoding="utf-8"))
        board = Board.from_json(str(p))
        cost_map = json.loads(p.with_name(p.stem + "_costs.json").read_text(encoding="utf-8"))
        tasks = bj["tasks"]
        costs = {s: np.array([cost_map[s][t] for t in tasks]) for s in board.names if s in cost_map}
        if len(costs) < 2:
            continue
        rows = frontier_curse(board, costs, alpha=0.05, B=B, seed=0, iterate=True)
        for r in rows:
            r["below_rival_iterated"] = bool(r["rival_accuracy"] is not None and r["bootstrap"]["estimate"] < r["rival_accuracy"])
            r["below_rival_conditional"] = bool(r["rival_accuracy"] is not None and np.isfinite(r["conditional"]["estimate"])
                                                and r["conditional"]["estimate"] < r["rival_accuracy"])
        out[p.stem] = {"systems": board.K, "tasks": board.T, "cost_source": bj.get("cost_source", "per_task_calls"), "members": rows}
        with_rival = [r for r in rows if r["pool_size"] > 1]
        shr = [100 * r["bootstrap"]["bias"] for r in with_rival]
        print(f"\n{p.stem}: {len(rows)} on the observed frontier, {len(with_rival)} with a cheaper rival; "
              f"mean shrinkage (iterated) {np.mean(shr) if shr else 0:.1f} pt, max {max(shr) if shr else 0:.1f} pt; "
              f"below rival after adjustment: {sum(r['below_rival_iterated'] for r in rows)}")
        for r in rows:
            c = r["conditional"]
            print(f"   {r['system'][:44]:44s} ${r['cost']:7.3f} pool {r['pool_size']:2d} reported {100*r['observed']:5.1f} "
                  f"iterated {100*r['bootstrap']['estimate']:5.1f} [{100*r['bootstrap']['ci'][0]:5.1f},{100*r['bootstrap']['ci'][1]:5.1f}] "
                  f"conditional {100*c['estimate'] if np.isfinite(c['estimate']) else float('nan'):5.1f}"
                  + (f" rival {100*r['rival_accuracy']:5.1f}" if r["rival_accuracy"] is not None else "")
                  + ("  BELOW RIVAL" if r["below_rival_iterated"] else ""))
    (ROOT / "results" / "curse_hal.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

"""How far do the frontier sets move if the standard errors are too small?

The standard errors treat tasks as independent draws and each system's single run as its configuration's average. Two things can
make them too small: tasks that cluster (a design effect) and run-level variation shared across a run's tasks. Either one acts on
the studentised leads as a common inflation factor f on the standard error. The joint set keeps system i unless its strongest
cheaper rival's lead, in standard-error units, exceeds the board's margin c. With every standard error multiplied by f that lead
becomes lead / f, so the set at inflation f is read directly from the stored leads, with no new bootstrap. A design effect d means
f = sqrt(d), and the effective number of tasks is T / f^2.

For each board the script reports the set size at f = 1, 1.25, 1.5 and 2, and the smallest f at which the set is the whole board.

usage: python inflation_sensitivity.py
Writes results/inflation_sensitivity.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID = (1.0, 1.25, 1.5, 2.0)


def board_row(r):
    c = r["margin_se"]
    leads = {x["system"]: x["rival_lead_se"] for x in r["table"]}
    observed = {x["system"] for x in r["table"] if x["on_observed_frontier"]}
    sizes = {}
    for f in GRID:
        inside = [s for s, t in leads.items() if s in observed or t is None or t / f <= c]
        sizes[str(f)] = len(inside)
    worst = [t for s, t in leads.items() if t is not None and s not in observed]
    f_all = max(1.0, max(worst) / c) if worst and c > 0 else 1.0
    return {"systems": r["systems"], "tasks": r["tasks"], "margin_se": c, "set_by_f": sizes, "f_whole_board": float(f_all),
            "effective_tasks_at_whole_board": float(r["tasks"] / f_all ** 2)}


def main():
    out = {"grid": list(GRID), "hal": {}, "helm": {}}
    for prefix, path in (("hal", "frontier_hal.json"), ("helm", "frontier_helm.json")):
        res = json.loads((ROOT / "results" / path).read_text(encoding="utf-8"))
        for stem, r in res.items():
            out[prefix][stem] = board_row(r)
            row = out[prefix][stem]
            print(f"{stem:36s} K={row['systems']:3d} T={row['tasks']:5d}  " + "  ".join(f"f={f:g}: {row['set_by_f'][str(f)]:3d}" for f in GRID)
                  + f"   whole board at f={row['f_whole_board']:.2f}")
    for prefix in ("hal", "helm"):
        tot = {str(f): sum(v["set_by_f"][str(f)] for v in out[prefix].values()) for f in GRID}
        k = sum(v["systems"] for v in out[prefix].values())
        out[prefix + "_total"] = {"systems": k, "set_by_f": tot}
        print(prefix, "total systems", k, tot)
    (ROOT / "results" / "inflation_sensitivity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

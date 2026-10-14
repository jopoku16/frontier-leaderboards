"""Does HAL's own frontier survive counting each call once and pricing every run the same way?

For each board, three plain frontiers from the same accuracies:
  reported   HAL's reported run totals (the costs its pages use);
  once       the reported totals divided by each run's raw/counted token ratio (double counting removed, HAL's
             historical prices kept), which isolates the counting bug;
  ours       the once-counted, repriced per-task costs of the audit.
Reports which systems change frontier status between them. Writes results/frontier_flips.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from honestboard.frontier import observed_frontier  # noqa: E402

BOARDS = ROOT / "data" / "boards"


def main():
    idx = json.loads((ROOT / "data" / "hal" / "index.json").read_text(encoding="utf-8"))
    out = {}
    paths = sorted(p for p in BOARDS.glob("hal_*.json") if not p.name.endswith("_costs.json") and p.name != "hal_summary.json")
    for p in paths:
        bj = json.loads(p.read_text(encoding="utf-8"))
        cost_map = json.loads(p.with_name(p.stem + "_costs.json").read_text(encoding="utf-8"))
        tasks = bj["tasks"]
        T = len(tasks)
        names, acc, c_rep, c_once, c_ours = [], [], [], [], []
        for s in bj["systems"]:
            rep = s.get("total_cost_reported")
            if rep is None or s["name"] not in cost_map:
                continue
            e = idx.get(s["source"], {})
            ratio = (e["tokens_raw"] / e["tokens_counted"]) if e.get("tokens_counted") else 1.0
            names.append(s["name"])
            acc.append(float(np.mean([s["rates"][t] for t in tasks])))
            c_rep.append(float(rep) / T)
            c_once.append(float(rep) / T / ratio)
            c_ours.append(float(np.mean([cost_map[s["name"]][t] for t in tasks])))
        acc = np.array(acc)
        fr = {"reported": observed_frontier(acc, np.array(c_rep)), "once": observed_frontier(acc, np.array(c_once)), "ours": observed_frontier(acc, np.array(c_ours))}
        rows = []
        for i, n in enumerate(names):
            rows.append({"system": n, "accuracy": float(acc[i]), "cost_reported": c_rep[i], "cost_once": c_once[i], "cost_ours": c_ours[i],
                         "on_reported": bool(fr["reported"][i]), "on_once": bool(fr["once"][i]), "on_ours": bool(fr["ours"][i])})
        def flips(a, b):
            return int(np.sum(fr[a] & ~fr[b])), int(np.sum(~fr[a] & fr[b]))      # (lost, gained) going from a to b
        lost_c, gained_c = flips("reported", "once")
        lost_p, gained_p = flips("once", "ours")
        lost, gained = flips("reported", "ours")
        out[p.stem] = {"systems": len(names), "tasks": T, "cost_source": bj.get("cost_source", "per_task_calls"),
                       "frontier_reported": int(fr["reported"].sum()), "frontier_once": int(fr["once"].sum()), "frontier_ours": int(fr["ours"].sum()),
                       "lost_counting": lost_c, "gained_counting": gained_c, "lost_pricing": lost_p, "gained_pricing": gained_p,
                       "lost_total": lost, "gained_total": gained, "changed_total": lost + gained,
                       "reported_total_dollars": float(sum(c * T for c in c_rep)), "ours_total_dollars": float(sum(c * T for c in c_ours)),
                       "table": rows}
        print(f"{p.stem:34s} frontier: reported {fr['reported'].sum():2d}  once-counted {fr['once'].sum():2d}  ours {fr['ours'].sum():2d} | "
              f"changed reported->ours {lost + gained:2d} (lost {lost}, gained {gained}); counting alone {lost_c + gained_c}, pricing after that {lost_p + gained_p}")
    (ROOT / "results" / "frontier_flips.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

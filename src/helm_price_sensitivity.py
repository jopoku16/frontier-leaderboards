"""How much does the HELM audit depend on which host's price is used for the open-weight models?

The audit prices open-weight models at the Together AI list price (HELM ran them through Together) and OpenAI, Anthropic and
Amazon models at their own price. LiteLLM also lists DeepInfra prices for nine of the open-weight models, usually lower. Here
those nine models are repriced at DeepInfra from the same recorded tokens, and for every board we report how many systems
enter or leave the plain frontier and the frontier set.

usage: python helm_price_sensitivity.py
Writes results/helm_price_sensitivity.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from helm_fetch import run_file  # noqa: E402
from honestboard import Board  # noqa: E402
from honestboard.frontier import observed_frontier, pareto_frontier  # noqa: E402

BOARDS = ROOT / "data" / "boards"


def main():
    alt = json.loads((ROOT / "data" / "helm" / "model_prices_deepinfra.json").read_text(encoding="utf-8"))
    out = {"repriced_models": sorted(alt)}
    for scen in ("gpqa", "ifeval", "mmlu_pro", "omni_math"):
        p = BOARDS / f"helm_{scen}.json"
        bj = json.loads(p.read_text(encoding="utf-8"))
        base_costs = json.loads(p.with_name(p.stem + "_costs.json").read_text(encoding="utf-8"))
        board = Board.from_json(str(p))
        tasks = bj["tasks"]
        new_costs = {}
        for s in bj["systems"]:
            n = s["name"]
            if n in alt:
                run = json.loads(run_file(s["source"]).read_text(encoding="utf-8"))["instances"]
                new_costs[n] = np.array([run[t]["p"] * alt[n]["prompt"] + run[t]["c"] * alt[n]["completion"] for t in tasks])
            else:
                new_costs[n] = np.array([base_costs[n][t] for t in tasks])
        base = {n: np.array([base_costs[n][t] for t in tasks]) for n in new_costs}
        rb = pareto_frontier(board, base, alpha=0.05, B=2000, seed=0)
        ra = pareto_frontier(board, new_costs, alpha=0.05, B=2000, seed=0)
        fb, fa = set(rb["observed_frontier"]), set(ra["observed_frontier"])
        sb, sa = set(rb["frontier_set"]), set(ra["frontier_set"])
        ratio = float(np.mean([new_costs[n].mean() / base[n].mean() for n in alt if n in new_costs]))
        out[scen] = {"systems": board.K, "frontier": len(fb), "frontier_alt": len(fa), "frontier_changed": len(fb ^ fa),
                     "set": len(sb), "set_alt": len(sa), "set_changed": len(sb ^ sa), "mean_cost_ratio_repriced": ratio}
        print(f"{scen:10s} frontier {len(fb)} -> {len(fa)} (changed {len(fb ^ fa)}); set {len(sb)} -> {len(sa)} (changed {len(sb ^ sa)}); repriced models cost x{ratio:.2f}", flush=True)
    (ROOT / "results" / "helm_price_sensitivity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

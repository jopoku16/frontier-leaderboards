"""Cluster bootstrap where the tasks have natural groups: SWE-bench Verified mini (repositories) and USACO
(problem difficulty divisions), resampling groups instead of tasks. Reports the frontier set and margin under both.

usage: python cluster_check.py
Writes results/cluster_check.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from honestboard import Board  # noqa: E402
from honestboard.frontier import pareto_frontier  # noqa: E402

BOARDS = ROOT / "data" / "boards"
RULES = {
    "hal_swebench_verified_mini": ("repository", lambda t: t.split("-")[0]),                  # django__django-11790 -> django__django
    "hal_usaco": ("division", lambda t: re.sub(r"^\d+_", "", t).split("_")[0]),             # 1011_bronze_triangles -> bronze
}


def main():
    out = {}
    for stem, (label, fn) in RULES.items():
        p = BOARDS / f"{stem}.json"
        bj = json.loads(p.read_text(encoding="utf-8"))
        board = Board.from_json(str(p))
        cost_map = json.loads(p.with_name(p.stem + "_costs.json").read_text(encoding="utf-8"))
        tasks = bj["tasks"]
        costs = {s: np.array([cost_map[s][t] for t in tasks]) for s in board.names if s in cost_map}
        groups = [fn(t) for t in tasks]
        sizes = {g: groups.count(g) for g in set(groups)}
        base = pareto_frontier(board, costs, alpha=0.05, B=4000, seed=0)
        clus = pareto_frontier(board, costs, alpha=0.05, B=4000, seed=0, clusters=groups)
        out[stem] = {"cluster_label": label, "n_clusters": len(sizes), "largest_cluster": max(sizes.values()), "tasks": len(tasks),
                     "set_tasks": len(base["frontier_set"]), "set_clusters": len(clus["frontier_set"]),
                     "margin_tasks": base["margin_se"], "margin_clusters": clus["margin_se"],
                     "best_set_tasks": len(base["best_set"]), "best_set_clusters": len(clus["best_set"]),
                     "frontier_set_clusters": clus["frontier_set"]}
        print(f"{stem}: {len(sizes)} {label} groups (largest {max(sizes.values())} of {len(tasks)} tasks); frontier set {len(base['frontier_set'])} -> {len(clus['frontier_set'])} "
              f"with cluster resampling (margin {base['margin_se']:.2f} -> {clus['margin_se']:.2f} se); best set {len(base['best_set'])} -> {len(clus['best_set'])}")
    (ROOT / "results" / "cluster_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

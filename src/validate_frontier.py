"""Does the frontier confidence set keep what it promises, how wrong is the plain frontier, and how do the obvious
alternatives compare?

Known-truth simulation. K systems with true accuracies theta_k and true mean costs mu_k; per-task outcomes are
Bernoulli with a shared task difficulty (logit model, as in Paper 1's record-test simulation) and per-task costs are
lognormal around mu_k (so the cost ordering is uncertain too; cost_cv=0 gives the fixed-cost case). On every
simulated board we compute the observed frontier and several candidate sets and compare them with the true frontier:

  joint        the confidence set for the frontier (joint bootstrap margin; the paper's set)
  boardwide    the largest per-system bootstrap margin
  persystem    each system's own bootstrap margin
  exact        one-sided exact paired tests, Bonferroni over each system's rivals (finite-sample, per system)
  exactjoint   the same with Bonferroni over every rival pair on the board (finite-sample, joint)
  unadjusted   one-sided exact paired tests at level alpha with no correction (what a casual analyst might do)
  frequency    systems on the observed frontier in at least alpha of the task resamples (a percentile rule)

Metrics per scenario and set: retention of truly undominated systems (per system and jointly), mean size against
the true frontier's size; and for the plain frontier, the share of its members that are truly dominated and the
share of the true frontier it misses.

usage: python validate_frontier.py [R] [--quick]
Writes results/validation_frontier.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from honestboard import Board  # noqa: E402
from honestboard.frontier import exact_frontier_set, observed_frontier, pareto_frontier  # noqa: E402

rng = np.random.default_rng(2026)
SETS = ("joint", "boardwide", "persystem", "exact", "exactjoint", "unadjusted", "frequency")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_truth(K, spread_acc, spread_cost, corr):
    """True abilities and costs; costlier systems tend to be better (corr), with scatter so the frontier is non-trivial."""
    z = rng.standard_normal(K)
    ability = spread_acc * z
    cost_logit = corr * z + np.sqrt(max(0.0, 1 - corr ** 2)) * rng.standard_normal(K)
    mu_cost = np.exp(spread_cost * cost_logit)                              # mean cost per task, lognormal across systems
    return ability, mu_cost


def true_accuracy(ability, Lpool):
    return np.array([sigmoid(a + Lpool).mean() for a in ability])


def simulate_board(ability, mu_cost, T, Lpool, cost_cv=0.8):
    """cost_cv=0 gives every task the system's mean cost: the fixed-cost case."""
    L = rng.choice(Lpool, T)
    P = sigmoid(ability[:, None] + L[None, :])
    Y = (rng.random(P.shape) < P).astype(float)
    sig = np.sqrt(np.log(1 + cost_cv ** 2))
    C = mu_cost[:, None] * np.exp(sig * rng.standard_normal(P.shape) - sig ** 2 / 2)
    names = [f"s{k}" for k in range(len(ability))]
    tasks = [f"t{t}" for t in range(T)]
    systems = [{"name": n, "date": "20250101", "rates": {t: float(Y[k, i]) for i, t in enumerate(tasks)}} for k, n in enumerate(names)]
    board = Board.from_rates(tasks, systems)
    costs = {n: C[k] for k, n in enumerate(names)}
    return board, costs


def run(R, quick):
    Lpool = rng.normal(0.0, 1.2, 2000)                                        # task difficulties on the logit scale
    scenarios = []
    for K in (8, 20):
        for T in ((50, 150) if quick else (50, 150, 500)):
            scenarios.append({"K": K, "T": T, "corr": 0.6, "spread_acc": 0.8, "spread_cost": 1.0, "cost_cv": 0.8})
    if not quick:
        scenarios.append({"K": 20, "T": 150, "corr": 0.6, "spread_acc": 0.8, "spread_cost": 1.0, "cost_cv": 0.0})     # fixed costs
        scenarios.append({"K": 20, "T": 150, "corr": 0.3, "spread_acc": 0.8, "spread_cost": 1.0, "cost_cv": 0.8})     # weaker cost-ability link
        scenarios.append({"K": 60, "T": 45, "corr": 0.6, "spread_acc": 0.8, "spread_cost": 1.0, "cost_cv": 0.8})      # a crowded small board
    out = {"R": R, "alpha": 0.05, "sets": list(SETS), "scenarios": []}
    for sc in scenarios:
        t0 = time.time()
        K, T = sc["K"], sc["T"]
        kept_joint = {s: 0 for s in SETS}
        retained = {s: [] for s in SETS}
        sizes = {s: [] for s in SETS}
        true_sizes, naive_false, naive_missed, freq_und, freq_dom = [], [], [], [], []
        for r in range(R):
            ability, mu = make_truth(K, sc["spread_acc"], sc["spread_cost"], sc["corr"])
            theta = true_accuracy(ability, Lpool)
            truth = observed_frontier(theta, mu)
            board, costs = simulate_board(ability, mu, T, Lpool, cost_cv=sc.get("cost_cv", 0.8))
            res = pareto_frontier(board, costs, alpha=0.05, B=600 if quick else 1500, seed=r)
            ex = exact_frontier_set(board, costs, alpha=0.05)
            names = board.names
            order = sorted(range(K), key=lambda k: int(names[k][1:]))
            rows = {row["system"]: row for row in res["table"]}
            on_obs = np.array([rows[names[k]]["on_observed_frontier"] for k in range(K)])
            freq = np.array([rows[names[k]]["undominated_frequency"] for k in range(K)])
            member = {
                "joint": np.array([n in set(res["frontier_set"]) for n in names]),
                "boardwide": np.array([n in set(res["frontier_set_boardwide"]) for n in names]),
                "persystem": np.array([n in set(res["frontier_set_persystem"]) for n in names]),
                "exact": np.array([n in set(ex["frontier_set_exact"]) for n in names]),
                "exactjoint": np.array([n in set(ex["frontier_set_exact_joint"]) for n in names]),
                "unadjusted": np.array([ex["min_rival_p"][n] > 0.05 for n in names]),
                "frequency": freq >= 0.05,
            }
            for s in SETS:
                m = member[s]
                kept_joint[s] += bool(np.all(m[truth]))
                retained[s].extend(m[truth].tolist())
                sizes[s].append(int(m.sum()))
            true_sizes.append(int(truth.sum()))
            if on_obs.any():
                naive_false.append(float((on_obs & ~truth).sum() / on_obs.sum()))
            naive_missed.append(float((truth & ~on_obs).sum() / max(1, truth.sum())))
            freq_und.extend(freq[truth].tolist()); freq_dom.extend(freq[~truth].tolist())
        entry = {**sc, "mean_true_frontier_size": float(np.mean(true_sizes)),
                 "naive_false_frontier_share": float(np.mean(naive_false)) if naive_false else None,
                 "naive_missed_share": float(np.mean(naive_missed)),
                 "undominated_freq_truly_undominated": float(np.mean(freq_und)) if freq_und else None,
                 "undominated_freq_truly_dominated": float(np.mean(freq_dom)) if freq_dom else None,
                 "sets": {s: {"joint_retention": kept_joint[s] / R, "per_system_retention": float(np.mean(retained[s])),
                              "mean_set_size": float(np.mean(sizes[s]))} for s in SETS},
                 "seconds": round(time.time() - t0)}
        # the paper's set, flattened for the macros that already exist
        entry["joint_retention"] = entry["sets"]["joint"]["joint_retention"]
        entry["per_system_retention"] = entry["sets"]["joint"]["per_system_retention"]
        entry["mean_set_size"] = entry["sets"]["joint"]["mean_set_size"]
        out["scenarios"].append(entry)
        print(f"K={K:2d} T={T:3d} corr={sc['corr']} cv={sc.get('cost_cv', 0.8)}: true frontier {entry['mean_true_frontier_size']:.1f}; "
              + "; ".join(f"{s} ret {entry['sets'][s]['joint_retention']:.2f}/{entry['sets'][s]['per_system_retention']:.3f} size {entry['sets'][s]['mean_set_size']:.1f}" for s in SETS)
              + f"; naive false {entry['naive_false_frontier_share']:.2f} missed {entry['naive_missed_share']:.2f} ({entry['seconds']}s)", flush=True)
    (ROOT / "results" / "validation_frontier.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
    run(R, "--quick" in sys.argv)

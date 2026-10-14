"""Does the frontier confidence set keep what it promises, and how wrong is the plain frontier?

Known-truth simulation. K systems with true accuracies theta_k and true mean costs mu_k; per-task outcomes are
Bernoulli with a shared task difficulty (logit model, as in Paper 1's record-test simulation) and per-task costs are
lognormal around mu_k (so the cost ordering is uncertain too). On every simulated board we compute the observed
frontier and the frontier confidence set and compare both with the true frontier (from theta and mu).

Metrics per scenario:
  retention      share of truly undominated systems kept in the set (target >= 1 - alpha, per system and jointly)
  set_size       mean size of the set against the true frontier's size
  naive_false    share of systems on the observed frontier that are truly dominated
  naive_missed   share of truly undominated systems that are off the observed frontier
  freq_calib     mean undominated frequency of truly undominated vs truly dominated systems

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
from honestboard.frontier import observed_frontier, pareto_frontier  # noqa: E402

rng = np.random.default_rng(2026)


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
    out = {"R": R, "alpha": 0.05, "scenarios": []}
    for sc in scenarios:
        t0 = time.time()
        K, T = sc["K"], sc["T"]
        kept_joint = 0
        retained = []
        set_sizes, true_sizes = [], []
        naive_false, naive_missed = [], []
        freq_und, freq_dom = [], []
        for r in range(R):
            ability, mu = make_truth(K, sc["spread_acc"], sc["spread_cost"], sc["corr"])
            theta = true_accuracy(ability, Lpool)
            truth = observed_frontier(theta, mu)
            board, costs = simulate_board(ability, mu, T, Lpool, cost_cv=sc.get("cost_cv", 0.8))
            res = pareto_frontier(board, costs, alpha=0.05, B=600 if quick else 1500, seed=r)
            in_set = np.array([row["in_frontier_set"] for row in sorted(res["table"], key=lambda x: int(x["system"][1:]))])
            on_obs = np.array([row["on_observed_frontier"] for row in sorted(res["table"], key=lambda x: int(x["system"][1:]))])
            freq = np.array([row["undominated_frequency"] for row in sorted(res["table"], key=lambda x: int(x["system"][1:]))])
            kept_joint += bool(np.all(in_set[truth]))
            retained.extend(in_set[truth].tolist())
            set_sizes.append(int(in_set.sum())); true_sizes.append(int(truth.sum()))
            if on_obs.any():
                naive_false.append(float((on_obs & ~truth).sum() / on_obs.sum()))
            naive_missed.append(float((truth & ~on_obs).sum() / max(1, truth.sum())))
            freq_und.extend(freq[truth].tolist()); freq_dom.extend(freq[~truth].tolist())
        entry = {**sc, "joint_retention": kept_joint / R, "per_system_retention": float(np.mean(retained)),
                 "mean_set_size": float(np.mean(set_sizes)), "mean_true_frontier_size": float(np.mean(true_sizes)),
                 "naive_false_frontier_share": float(np.mean(naive_false)) if naive_false else None,
                 "naive_missed_share": float(np.mean(naive_missed)),
                 "undominated_freq_truly_undominated": float(np.mean(freq_und)) if freq_und else None,
                 "undominated_freq_truly_dominated": float(np.mean(freq_dom)) if freq_dom else None,
                 "seconds": round(time.time() - t0)}
        out["scenarios"].append(entry)
        print(f"K={K:2d} T={T:3d} corr={sc['corr']} cv={sc.get('cost_cv', 0.8)}: joint retention {entry['joint_retention']:.2f}, per-system {entry['per_system_retention']:.3f}, "
              f"set {entry['mean_set_size']:.1f} vs true {entry['mean_true_frontier_size']:.1f}, naive false {entry['naive_false_frontier_share']:.2f}, "
              f"naive missed {entry['naive_missed_share']:.2f}, freq und/dom {entry['undominated_freq_truly_undominated']:.2f}/{entry['undominated_freq_truly_dominated']:.2f} ({entry['seconds']}s)", flush=True)
    (ROOT / "results" / "validation_frontier.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
    run(R, "--quick" in sys.argv)

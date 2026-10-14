"""Does the frontier confidence set keep what it promises, how wrong is the plain frontier, and how do the obvious
alternatives compare?  Known-truth simulation.

K systems with true accuracies theta_k and true mean costs gamma_k; per-task outcomes are Bernoulli given a shared task
difficulty (logit model), per-task costs are lognormal around each system's mean (cost_cv = 0: fixed costs). On every
simulated board we compute the observed frontier and several candidate sets and compare them with the true frontier
(from theta and gamma).

Scenarios beyond the base grid, each aimed at an assumption the guarantees lean on:
  low accuracy      most systems pass a handful of tasks (the shape of SciCode)
  near-copies       a share of the systems are degraded copies of another system (each pass flipped with probability
                    copy_flip) that cost 3% more: the same agent run with a smaller model, a lower effort, or a rerun
  cost-outcome      failed tasks cost more (cost multiplied by 1 + lam * (1 - outcome)), as when an agent burns tokens on
  dependence        tasks it then fails
  near-tied costs   true mean costs spread over a few percent, below the precision of the mean costs
  clustered tasks   tasks fall in groups sharing a difficulty shift, and each system has its own effect in each group,
                    so tasks are not independent draws; the set is computed with and without cluster resampling

Sets (kept = on the set):
  joint        the confidence set for the frontier (joint bootstrap margin; the paper's set)
  boardwide    the largest per-system bootstrap margin
  persystem    each system's own bootstrap margin
  exact        one-sided exact paired tests, Bonferroni over each system's rivals
  exactjoint   the same with Bonferroni over every rival pair on the board
  unadjusted   one-sided exact paired tests at level alpha with no correction
  frequency    systems on the observed frontier in at least alpha of the task resamples
  tierobust    the joint set with rivals only when surely no costlier (alpha split), valid when true costs tie
  cluster      the joint set with task groups resampled (clustered scenario only)
And the band's coverage of the true curve, and the cheapest unresolved competitor of the best: how often the true best
is in the best set and the competitor costs no more than the true best.

usage: python validate_frontier.py [R] [--quick] [--cell K T]
Writes results/validation_frontier.json (or validation_frontier_cell.json with --cell)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from honestboard import Board  # noqa: E402
from honestboard.frontier import band_at, exact_frontier_set, frontier_band, observed_frontier, pareto_frontier  # noqa: E402

rng = np.random.default_rng(2026)
SETS = ("joint", "boardwide", "persystem", "exact", "exactjoint", "unadjusted", "frequency", "tierobust", "cluster")
ZFIX = np.random.default_rng(7).standard_normal(200000)


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


def make_world(sc, Lpool):
    """Draw the true abilities, costs and the true accuracies and mean costs of one simulated population of systems."""
    K, n_copy = sc["K"], sc.get("n_copy", 0)
    K0 = K - n_copy
    ability, mu = make_truth(K0, sc["spread_acc"], sc["spread_cost"], sc["corr"])
    ability = ability + sc.get("shift", 0.0)
    parent = list(range(K0))
    if n_copy:
        parent = parent + [int(x) for x in rng.choice(K0, size=n_copy, replace=n_copy > K0)]
        ability = np.concatenate([ability, ability[parent[K0:]]])
        mu = np.concatenate([mu, mu[parent[K0:]] * sc.get("copy_cost_mult", 1.03)])
    delta = sc.get("copy_flip", 0.1)
    if sc.get("groups"):                                                     # marginal accuracy over task and group effects
        v = sc["sd_group"] ** 2 + sc["sd_task"] ** 2 + sc["tau"] ** 2
        theta = np.array([sigmoid(a + np.sqrt(v) * ZFIX).mean() for a in ability])
    else:
        theta = true_accuracy(ability, Lpool)
    for c in range(K0, K):                                                   # a copy keeps a share 1 - delta of its parent's passes
        theta[c] = (1 - delta) * theta[parent[c]]
    gamma = mu * (1 + sc.get("lam", 0.0) * (1 - theta))                      # true mean cost, cost-outcome dependence included
    return {"ability": ability, "mu": mu, "parent": parent, "K0": K0, "delta": delta, "theta": theta, "gamma": gamma}


def draw_board(world, sc, T, Lpool):
    K = len(world["ability"])
    G = sc.get("groups")
    if G:
        m = T // G
        groups = np.repeat(np.arange(G), m)
        T = G * m
        L = rng.normal(0, sc["sd_group"], G)[groups] + rng.normal(0, sc["sd_task"], T)
        U = rng.normal(0, sc["tau"], (K, G))[:, groups]
        P = sigmoid(world["ability"][:, None] + L[None, :] + U)
    else:
        groups = None
        L = rng.choice(Lpool, T)
        P = sigmoid(world["ability"][:, None] + L[None, :])
    Y = (rng.random(P.shape) < P).astype(float)
    for c in range(world["K0"], K):                                           # degrade the parent's outcomes
        Y[c] = Y[world["parent"][c]] * (rng.random(T) >= world["delta"])
    cv = sc.get("cost_cv", 0.8)
    sig = np.sqrt(np.log(1 + cv ** 2))
    C = world["mu"][:, None] * np.exp(sig * rng.standard_normal(P.shape) - sig ** 2 / 2) * (1 + sc.get("lam", 0.0) * (1 - Y))
    names = [f"s{k}" for k in range(K)]
    tasks = [f"t{t}" for t in range(T)]
    systems = [{"name": n, "date": "20250101", "rates": {t: float(Y[k, i]) for i, t in enumerate(tasks)}} for k, n in enumerate(names)]
    board = Board.from_rates(tasks, systems)
    costs = {n: C[k] for k, n in enumerate(names)}
    return board, costs, (None if groups is None else groups.tolist())


def run(R, quick, only=None, out_name="validation_frontier.json", part=None):
    Lpool = rng.normal(0.0, 1.2, 2000)                                        # task difficulties on the logit scale
    base = {"corr": 0.6, "spread_acc": 0.8, "spread_cost": 1.0, "cost_cv": 0.8}
    scenarios = []
    for K in (8, 20):
        for T in ((50, 150) if quick else (50, 150, 500)):
            scenarios.append({**base, "K": K, "T": T, "label": "base"})
    if not quick:
        scenarios += [
            {**base, "K": 20, "T": 150, "cost_cv": 0.0, "label": "fixed costs"},
            {**base, "K": 20, "T": 150, "corr": 0.3, "label": "weak cost-ability link"},
            {**base, "K": 60, "T": 45, "label": "crowded board"},
            {**base, "K": 20, "T": 65, "shift": -2.5, "label": "low accuracy"},
            {**base, "K": 20, "T": 150, "n_copy": 6, "copy_flip": 0.1, "copy_cost_mult": 1.03, "label": "near-copies"},
            {**base, "K": 20, "T": 150, "lam": 1.0, "label": "cost-outcome dependence"},
            {**base, "K": 20, "T": 150, "spread_cost": 0.05, "label": "near-tied costs"},
            {**base, "K": 20, "T": 200, "groups": 40, "sd_group": 0.8, "sd_task": 0.9, "tau": 0.7, "label": "clustered tasks"},
            {**base, "K": 20, "T": 150, "n_copy": 6, "copy_flip": 0.1, "copy_cost_mult": 0.97, "label": "cheaper near-copies"},
        ]
    if only is not None:                                                      # one cell, more boards (a Monte Carlo check)
        scenarios = [sc for sc in scenarios if (sc["K"], sc["T"]) == only and sc["label"] == "base"]
    for i_, sc_ in enumerate(scenarios):
        sc_["order"] = i_                                                    # position in the full list, to merge parts
    if part is not None:                                                      # run every m-th scenario (parts run in parallel)
        scenarios = scenarios[part[0]::part[1]]
    out = {"R": R, "alpha": 0.05, "sets": list(SETS), "scenarios": []}
    Bq = 600 if quick else 1200
    for sc in scenarios:
        t0 = time.time()
        K, T = sc["K"], sc["T"]
        use = [s for s in SETS if s != "cluster" or sc.get("groups")]
        kept_joint = {s: 0 for s in use}
        retained = {s: [] for s in use}
        sizes = {s: [] for s in use}
        true_sizes, naive_false, naive_missed, freq_und, freq_dom = [], [], [], [], []
        band_cov, band_widths, best_in, cpb_cost_ok = [], [], [], []
        for r in range(R):
            world = make_world(sc, Lpool)
            theta, gamma = world["theta"], world["gamma"]
            truth = observed_frontier(theta, gamma)
            board, costs, groups = draw_board(world, sc, T, Lpool)
            res = pareto_frontier(board, costs, alpha=0.05, B=Bq, seed=r)
            rob = pareto_frontier(board, costs, alpha=0.05, B=Bq, seed=r, tie_robust=True)
            clu = pareto_frontier(board, costs, alpha=0.05, B=Bq, seed=r, clusters=groups) if groups else None
            ex = exact_frontier_set(board, costs, alpha=0.05)
            band = frontier_band(board, costs, alpha=0.05, B=Bq, seed=r)
            names = board.names
            rows = {row["system"]: row for row in res["table"]}
            on_obs = np.array([rows[n]["on_observed_frontier"] for n in names])
            freq = np.array([rows[n]["undominated_frequency"] for n in names])
            member = {
                "joint": np.array([n in set(res["frontier_set"]) for n in names]),
                "boardwide": np.array([n in set(res["frontier_set_boardwide"]) for n in names]),
                "persystem": np.array([n in set(res["frontier_set_persystem"]) for n in names]),
                "exact": np.array([n in set(ex["frontier_set_exact"]) for n in names]),
                "exactjoint": np.array([n in set(ex["frontier_set_exact_joint"]) for n in names]),
                "unadjusted": np.array([ex["min_rival_p"][n] > 0.05 for n in names]),
                "frequency": freq >= 0.05,
                "tierobust": np.array([n in set(rob["frontier_set"]) for n in names]),
            }
            if clu is not None:
                member["cluster"] = np.array([n in set(clu["frontier_set"]) for n in names])
            for s in use:
                m = member[s]
                kept_joint[s] += bool(np.all(m[truth]))
                retained[s].extend(m[truth].tolist())
                sizes[s].append(int(m.sum()))
            true_sizes.append(int(truth.sum()))
            if on_obs.any():
                naive_false.append(float((on_obs & ~truth).sum() / on_obs.sum()))
            naive_missed.append(float((truth & ~on_obs).sum() / max(1, truth.sum())))
            freq_und.extend(freq[truth].tolist()); freq_dom.extend(freq[~truth].tolist())
            ok = True                                                          # the band against the true curve
            for x in sorted(set(band["grid"]) | {float(g) for g in gamma}):
                bx = band_at(band, x)
                inside = np.where(gamma <= x + 1e-12)[0]
                if bx is None or not inside.size:
                    continue
                a0 = theta[inside].max()
                ok = ok and (bx[0] - 1e-12 <= a0 <= bx[1] + 1e-12)
            band_cov.append(ok); band_widths.append(band["width_points_median"])
            tb = int(np.argmax(theta))                                         # the cheapest unresolved competitor of the best
            best_in.append(names[tb] in set(res["best_set"]))
            cpb = int(res["cheapest_plausible_best"]["system"][1:])
            cpb_cost_ok.append(bool(gamma[cpb] <= gamma[tb] + 1e-12))
        entry = {**sc, "mean_true_frontier_size": float(np.mean(true_sizes)),
                 "naive_false_frontier_share": float(np.mean(naive_false)) if naive_false else None,
                 "naive_missed_share": float(np.mean(naive_missed)),
                 "undominated_freq_truly_undominated": float(np.mean(freq_und)) if freq_und else None,
                 "undominated_freq_truly_dominated": float(np.mean(freq_dom)) if freq_dom else None,
                 "sets": {s: {"joint_retention": kept_joint[s] / R, "per_system_retention": float(np.mean(retained[s])),
                              "mean_set_size": float(np.mean(sizes[s]))} for s in use},
                 "band": {"coverage_true_costs": float(np.mean(band_cov)), "coverage_observed_costs": float(np.mean(band_cov)),
                          "width_points_median": float(np.mean(band_widths))},
                 "best": {"true_best_in_best_set": float(np.mean(best_in)), "cheapest_competitor_costs_no_more": float(np.mean(cpb_cost_ok))},
                 "seconds": round(time.time() - t0)}
        entry["joint_retention"] = entry["sets"]["joint"]["joint_retention"]
        entry["per_system_retention"] = entry["sets"]["joint"]["per_system_retention"]
        entry["mean_set_size"] = entry["sets"]["joint"]["mean_set_size"]
        out["scenarios"].append(entry)
        print(f"{sc['label']:24s} K={K:2d} T={T:3d}: true frontier {entry['mean_true_frontier_size']:.1f}; "
              + "; ".join(f"{s} {entry['sets'][s]['joint_retention']:.2f}/{entry['sets'][s]['per_system_retention']:.3f} size {entry['sets'][s]['mean_set_size']:.1f}" for s in use)
              + f"; naive false {entry['naive_false_frontier_share']:.2f} missed {entry['naive_missed_share']:.2f}; band covers {entry['band']['coverage_true_costs']:.2f}"
              + f" width {entry['band']['width_points_median']:.1f}; best in set {entry['best']['true_best_in_best_set']:.2f}, cheapest competitor no dearer {entry['best']['cheapest_competitor_costs_no_more']:.2f}"
              + f" ({entry['seconds']}s)", flush=True)
        (ROOT / "results" / out_name).write_text(json.dumps(out, indent=1), encoding="utf-8")


def merge():
    """Combine results/validation_frontier_part*.json into results/validation_frontier.json in the original scenario order."""
    parts = sorted((ROOT / "results").glob("validation_frontier_part*.json"))
    by_order, R = {}, None
    old = ROOT / "results" / "validation_frontier.json"
    if old.exists():                                                         # scenarios from an earlier merge stay unless a part replaces them
        d = json.loads(old.read_text(encoding="utf-8"))
        R = d["R"]; by_order = {s["order"]: s for s in d["scenarios"]}
    for f in parts:
        d = json.loads(f.read_text(encoding="utf-8"))
        R = d["R"]; by_order.update({s["order"]: s for s in d["scenarios"]})
    scen = [by_order[k] for k in sorted(by_order)]
    (ROOT / "results" / "validation_frontier.json").write_text(json.dumps({"R": R, "alpha": 0.05, "sets": list(SETS), "scenarios": scen}, indent=1), encoding="utf-8")
    print(f"merged {len(parts)} parts, {len(scen)} scenarios")


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
    if "--merge" in sys.argv:
        merge(); sys.exit(0)
    if "--part" in sys.argv:
        i = sys.argv.index("--part")
        k, m = int(sys.argv[i + 1]), int(sys.argv[i + 2])
        run(R, False, out_name=f"validation_frontier_part{k}.json", part=(k, m))
        sys.exit(0)
    if "--cell" in sys.argv:
        i = sys.argv.index("--cell")
        run(R, False, only=(int(sys.argv[i + 1]), int(sys.argv[i + 2])), out_name="validation_frontier_cell.json")
    else:
        run(R, "--quick" in sys.argv)

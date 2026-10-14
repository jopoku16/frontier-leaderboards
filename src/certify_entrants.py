"""Certify frontier entrants on each HAL board in arrival order (Proposition 2 and Corollary 4 of the paper).

Walk the systems by submission date (best-first within a date, as in honestboard's record chain). When system b
arrives, its cheaper-or-equal rivals are the earlier systems whose mean cost per task is at most b's. If there are
none, b is the cheapest system so far and is on the frontier by definition (no test). Otherwise let j* be the most
accurate rival (ties: the earliest). b is a frontier entrant exactly when it beats j*, and the certification test is
Paper 1's selective record test of b against j*: the exact sign test conditional on the record event for pass/fail
outcomes, the selective sign-flip test for rates, times the per-date Bonferroni factor. Levels: alpha-spending
(alpha * 6 / (pi^2 t^2) for the t-th tested entrant) and the online fallback (spending plus the previous level when
the previous entrant was certified); the unadjusted level alpha is reported for comparison. Costs are fixed on
arrival (the system's mean cost per task).

usage: python certify_entrants.py [--alpha 0.05] [--jitter 200]
Writes results/certify_hal.json.
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from honestboard import Board  # noqa: E402
from honestboard.chain import _step_pvalue, alpha_spending_level, fallback_level  # noqa: E402
from honestboard.frontier import entrant_pvalues_arrays  # noqa: E402

BOARDS = ROOT / "data" / "boards"


def _dates(bj):
    """YYYYMMDD per system: the run's recorded date, else the unix timestamp at the end of its run id."""
    out = []
    for s in bj["systems"]:
        d = (s.get("date") or "").replace("-", "")
        if not d:
            tail = (s.get("run_id") or "").rsplit("_", 1)[-1]
            if tail.isdigit() and len(tail) == 10:
                d = dt.datetime.utcfromtimestamp(int(tail)).strftime("%Y%m%d")
        out.append(d)
    return out


def certify_board(path, alpha=0.05, seed=0):
    bj = json.loads(path.read_text(encoding="utf-8"))
    board = Board.from_json(str(path))
    dates = _dates(bj)
    board.dates = dates
    cost_map = json.loads(path.with_name(path.stem + "_costs.json").read_text(encoding="utf-8"))
    tasks = bj["tasks"]
    cost = np.array([np.mean([cost_map[n][t] for t in tasks]) for n in board.names])
    sc = board.scores()
    undated = [n for n, d in zip(board.names, dates) if not d]
    if undated:
        return {"error": f"{len(undated)} systems without a date", "systems": board.K}
    return certify_order(board, cost, dates, alpha=alpha, seed=seed)


def certify_order(board, cost, dates, alpha=0.05, seed=0, with_pvalues=True):
    """Walk the submissions and test the frontier entrants. The step index t counts DATES with at least one entrant,
    not entrants: every entrant on a date is tested at that date's level, times the per-date factor g, so the level
    of a step is settled before any result of that date is looked at (the within-date order never changes it). The
    fallback carries a date's level to the next date with entrants when any entrant of the date was certified."""
    sc = board.scores()
    Yf = board.Y.astype(float)
    Sigma = np.cov(Yf) / board.T if board.K > 1 else np.zeros((1, 1))
    order = sorted(range(board.K), key=lambda k: (dates[k], -sc[k]))        # best-first within a date
    steps, t = [], 0
    last_level, last_cert = 0.0, False                                       # the previous date with entrants (pairwise chain)
    last_level_iu, last_cert_iu = 0.0, False                                 # the same for the intersection-union chain
    cur_date, cur = None, None
    n_cheapest = n_entrants = n_dominated = 0
    for pos, k in enumerate(order):
        earlier = order[:pos]
        rivals = [j for j in earlier if cost[j] <= cost[k]]
        if not rivals:
            n_cheapest += 1
            continue
        jstar = max(rivals, key=lambda j: (sc[j], -earlier.index(j)))           # most accurate rival, earliest on ties
        if sc[k] <= sc[jstar] + 1e-12:
            n_dominated += 1
            continue
        n_entrants += 1
        if dates[k] != cur_date:                                               # a new date with an entrant: next step
            if cur is not None:
                last_level, last_cert = cur["fallback"], cur["certified"]
                last_level_iu, last_cert_iu = cur["fallback_iu"], cur["certified_iu"]
            t += 1
            cur_date = dates[k]
            cur = {"spending": alpha_spending_level(t, alpha), "fallback": fallback_level(t, alpha, last_level, last_cert), "certified": False,
                   "fallback_iu": fallback_level(t, alpha, last_level_iu, last_cert_iu), "certified_iu": False}
        if not with_pvalues:
            steps.append({"test": t, "date": dates[k], "new": board.names[k], "holder": board.names[jstar]})
            continue
        p_raw = _step_pvalue(board, jstar, k, "exact", None, earlier, seed)
        g = board.date_group_size(k)
        p = float(min(1.0, g * p_raw))
        lvl_sp, lvl_fb = cur["spending"], cur["fallback"]
        cert_fb = bool(p <= lvl_fb)
        if cert_fb:
            cur["certified"] = True
        p_iu_raw, pairs = entrant_pvalues_arrays(sc, Sigma, k, rivals, board.T)   # composite null: every cheaper rival, whole frontier event
        p_iu = float(min(1.0, g * p_iu_raw)) if np.isfinite(p_iu_raw) else 1.0
        lvl_fb_iu = cur["fallback_iu"]
        cert_fb_iu = bool(p_iu <= lvl_fb_iu)
        if cert_fb_iu:
            cur["certified_iu"] = True
        steps.append({"test": t, "position": pos, "date": dates[k], "new": board.names[k], "holder": board.names[jstar],
                      "new_accuracy": float(sc[k]), "holder_accuracy": float(sc[jstar]), "margin": float(sc[k] - sc[jstar]),
                      "new_cost": float(cost[k]), "holder_cost": float(cost[jstar]), "n_rivals": len(rivals),
                      "p_uncorrected": float(p_raw), "date_group": int(g), "p": p,
                      "level_spending": float(lvl_sp), "certified_spending": bool(p <= lvl_sp),
                      "level_fallback": float(lvl_fb), "certified_fallback": cert_fb,
                      "certified_unadjusted": bool(p <= alpha),
                      "p_iu_uncorrected": float(p_iu_raw) if np.isfinite(p_iu_raw) else None, "p_iu": p_iu,
                      "iu_weakest_rival": (board.names[max(pairs, key=pairs.get)] if pairs else None),
                      "certified_iu_spending": bool(p_iu <= lvl_sp), "level_fallback_iu": float(lvl_fb_iu),
                      "certified_iu_fallback": cert_fb_iu, "certified_iu_unadjusted": bool(p_iu <= alpha)})
    return {"alpha": alpha, "systems": board.K, "tasks": board.T, "binary": bool(board.is_binary),
            "n_dates_with_entrants": t,
            "n_cheapest_so_far": n_cheapest, "n_entrants": n_entrants, "n_dominated_on_arrival": n_dominated,
            "n_certified_spending": sum(s.get("certified_spending", False) for s in steps),
            "n_certified_fallback": sum(s.get("certified_fallback", False) for s in steps),
            "n_certified_unadjusted": sum(s.get("certified_unadjusted", False) for s in steps),
            "n_certified_iu_spending": sum(s.get("certified_iu_spending", False) for s in steps),
            "n_certified_iu_fallback": sum(s.get("certified_iu_fallback", False) for s in steps),
            "n_certified_iu_unadjusted": sum(s.get("certified_iu_unadjusted", False) for s in steps),
            "steps": steps}


def jitter_check(board, cost_map, tasks, dates, base, R=200, alpha=0.05, seed=0):
    """Sensitivity to the cost noise that certification ignores: resample each system's per-task costs, rebuild the
    rival sets from the resampled means, and rerun the chain. Reports how often each original entrant is still an
    entrant against the same holder, and the spread of the certified counts."""
    rng = np.random.default_rng(seed)
    names = board.names
    C = np.array([[cost_map[n][t] for t in tasks] for n in names])
    T = C.shape[1]
    stable = {s["new"]: 0 for s in base["steps"]}
    n_cert_sp, n_cert_fb, n_cert_iu, n_entr = [], [], [], []
    for r in range(R):
        ix = rng.integers(0, T, size=T)
        cost_r = C[:, ix].mean(axis=1)
        res = certify_order(board, cost_r, dates, alpha=alpha, seed=seed, with_pvalues=False)
        holders = {s["new"]: s["holder"] for s in res["steps"]}
        for s in base["steps"]:
            stable[s["new"]] += int(holders.get(s["new"]) == s["holder"])
        n_entr.append(res["n_entrants"])
    # certified counts under resampled costs need the p-values: run the full chain on a smaller number of draws
    R2 = min(R, 40)
    for r in range(R2):
        ix = rng.integers(0, T, size=T)
        cost_r = C[:, ix].mean(axis=1)
        res = certify_order(board, cost_r, dates, alpha=alpha, seed=seed)
        n_cert_sp.append(res["n_certified_spending"]); n_cert_fb.append(res["n_certified_fallback"])
        n_cert_iu.append(res["n_certified_iu_spending"])
    return {"R": R, "holder_stable_share": float(np.mean([v / R for v in stable.values()])) if stable else None,
            "entrants_min": int(min(n_entr)), "entrants_max": int(max(n_entr)),
            "R_pvalues": R2, "certified_spending_min": int(min(n_cert_sp)), "certified_spending_max": int(max(n_cert_sp)),
            "certified_fallback_min": int(min(n_cert_fb)), "certified_fallback_max": int(max(n_cert_fb)),
            "certified_iu_min": int(min(n_cert_iu)), "certified_iu_max": int(max(n_cert_iu))}


def main(argv):
    alpha = float(argv[argv.index("--alpha") + 1]) if "--alpha" in argv else 0.05
    jitter = int(argv[argv.index("--jitter") + 1]) if "--jitter" in argv else 0
    paths = sorted(p for p in BOARDS.glob("hal_*.json") if not p.name.endswith("_costs.json") and p.name != "hal_summary.json")
    out = {}
    for p in paths:
        r = certify_board(p, alpha=alpha)
        bj = json.loads(p.read_text(encoding="utf-8"))
        r["cost_source"] = bj.get("cost_source", "per_task_calls")
        out[p.stem] = r
        if "error" in r:
            print(f"{p.stem}: {r['error']}")
            continue
        if jitter and r["cost_source"] == "per_task_calls" and r["steps"]:
            board = Board.from_json(str(p))
            cost_map = json.loads(p.with_name(p.stem + "_costs.json").read_text(encoding="utf-8"))
            r["jitter"] = jitter_check(board, cost_map, bj["tasks"], _dates(bj), r, R=jitter, alpha=alpha)
            j = r["jitter"]
            print(f"    jitter: holder stable {j['holder_stable_share']:.2f}; entrants {j['entrants_min']}-{j['entrants_max']}; "
                  f"certified (spending) {j['certified_spending_min']}-{j['certified_spending_max']}")
        print(f"{p.stem:36s} systems {r['systems']:3d}  cheapest-so-far {r['n_cheapest_so_far']:2d}  entrants {r['n_entrants']:2d}  "
              f"dominated {r['n_dominated_on_arrival']:3d}  certified: unadjusted {r['n_certified_unadjusted']:2d}  "
              f"spending {r['n_certified_spending']:2d}  fallback {r['n_certified_fallback']:2d}  | composite (IU): unadjusted "
              f"{r['n_certified_iu_unadjusted']:2d} spending {r['n_certified_iu_spending']:2d} fallback {r['n_certified_iu_fallback']:2d}")
        for s in r["steps"]:
            print(f"    t={s['test']:2d} {s['date']} {s['new'][:40]:40s} vs {s['holder'][:30]:30s} +{100*s['margin']:5.1f}pt  "
                  f"p={s['p']:.4f} (g={s['date_group']})  spend {s['level_spending']:.4f} {'Y' if s['certified_spending'] else '-'}  "
                  f"fallback {s['level_fallback']:.4f} {'Y' if s['certified_fallback'] else '-'}  IU p={s['p_iu']:.4f} {'Y' if s['certified_iu_spending'] else '-'}")
    (ROOT / "results" / "certify_hal.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

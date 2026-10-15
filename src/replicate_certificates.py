"""Do certified entrants keep their lead on fresh tasks?  A prospective-style test of the certificates of Appendix A.

For each board with at least 100 tasks (HAL and HELM) the tasks are split at random into a discovery half D and a fresh half
V, R times. The arrival chain of Appendix A is run on D alone: systems walk in by date, costs are fixed at their mean on D,
and every frontier entrant is tested against the most accurate cheaper system that came before it. Nothing from V is used. The
entrant's lead over that holder is then measured on V, tasks the whole procedure never saw. Entrants fall in three groups:

  certified     the record claim is certified under alpha-spending over the chain (the family-wise guarantee);
  nominal       the unadjusted per-date-corrected p-value is at most 5% but the entrant is not certified;
  other         neither.

For each group we report how often the lead is positive on V and how often it is significantly positive on V (one-sided
paired test at 5%), with the mean lead on V. A certificate that means anything should replicate far more often than a
selected-but-uncertified entrant, whose lead is inflated by the selection that put it on the frontier.

usage: python replicate_certificates.py [R] [--prefix hal|helm]
Writes results/replication_certificates_<prefix>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from certify_entrants import _dates, certify_order  # noqa: E402
from honestboard import Board  # noqa: E402

BOARDS = ROOT / "data" / "boards"


def one_sided_p(a, b):
    """Paired one-sided p-value that system a beats system b on the fresh tasks."""
    d = a - b
    if np.all(a == a.astype(int)) and np.all(b == b.astype(int)):       # pass/fail: sign test on the discordant tasks
        n_a, n_b = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
        n = n_a + n_b
        return float(stats.binom.sf(n_a - 1, n, 0.5)) if n else 1.0
    sd = d.std(ddof=1)
    if sd == 0:
        return 1.0 if d.mean() <= 0 else 0.0
    return float(stats.t.sf(d.mean() / (sd / np.sqrt(len(d))), len(d) - 1))


def replicate_board(path, R, seed=0):
    bj = json.loads(path.read_text(encoding="utf-8"))
    cost_map = json.loads(path.with_name(path.stem + "_costs.json").read_text(encoding="utf-8"))
    dates = _dates(bj)
    if any(not d for d in dates):
        return None
    tasks = bj["tasks"]
    T = len(tasks)
    names = [s["name"] for s in bj["systems"]]
    Yall = np.array([[s["rates"][t] for t in tasks] for s in bj["systems"]], float)
    Call = np.array([[cost_map[n][t] for t in tasks] for n in names], float)
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(R):
        perm = rng.permutation(T)
        D, V = np.sort(perm[: T // 2]), np.sort(perm[T // 2:])
        sub_tasks = [tasks[i] for i in D]
        board = Board.from_rates(sub_tasks, [{"name": n, "date": dates[k], "rates": {t: float(Yall[k, i]) for t, i in zip(sub_tasks, D)}} for k, n in enumerate(names)])
        board.dates = dates
        cost = Call[:, D].mean(axis=1)
        res = certify_order(board, cost, dates, with_pvalues=True, exact=False, seed=r)
        index = {n: k for k, n in enumerate(names)}
        for st in res["steps"]:
            a, b = Yall[index[st["new"]], V], Yall[index[st["holder"]], V]
            group = "certified" if st["certified_spending"] else ("nominal" if st["certified_unadjusted"] else "other")
            rows.append({"split": r, "group": group, "new": st["new"], "holder": st["holder"], "lead_D": st["margin"], "lead_V": float(a.mean() - b.mean()), "p_V": one_sided_p(a, b)})
    return rows


def summarise(rows, min_lead_D=None):
    out = {}
    for g in ("certified", "nominal", "other"):
        sel = [x for x in rows if x["group"] == g and (min_lead_D is None or x["lead_D"] >= min_lead_D)]
        if not sel:
            out[g] = {"n": 0}
            continue
        pairs = {(x.get("board"), x["new"], x["holder"]) for x in sel}
        out[g] = {"n": len(sel), "distinct_pairs": len(pairs), "positive_on_V": float(np.mean([x["lead_V"] > 0 for x in sel])), "significant_on_V": float(np.mean([x["p_V"] <= 0.05 for x in sel])),
                  "mean_lead_D_pts": float(100 * np.mean([x["lead_D"] for x in sel])), "mean_lead_V_pts": float(100 * np.mean([x["lead_V"] for x in sel]))}
    return out


def main(argv):
    R = int(argv[0]) if argv and argv[0].isdigit() else 40
    prefix = argv[argv.index("--prefix") + 1] if "--prefix" in argv else "hal"
    paths = sorted(p for p in BOARDS.glob(f"{prefix}_*.json") if not p.name.endswith("_costs.json") and p.name != f"{prefix}_summary.json")
    out, allrows = {}, []
    for p in paths:
        bj = json.loads(p.read_text(encoding="utf-8"))
        if len(bj["tasks"]) < 100:
            continue
        rows = replicate_board(p, R)
        if rows is not None:
            for x in rows:
                x["board"] = p.stem
        if rows is None:
            print(f"{p.stem}: systems without dates, skipped")
            continue
        out[p.stem] = {"tasks": len(bj["tasks"]), "systems": len(bj["systems"]), **summarise(rows)}
        allrows += rows
        s = out[p.stem]
        print(f"{p.stem:34s} T={s['tasks']:5d} | " + " | ".join(f"{g}: n={s[g]['n']}" + (f" +V {s[g]['positive_on_V']:.2f} sig {s[g]['significant_on_V']:.2f}" if s[g]["n"] else "") for g in ("certified", "nominal", "other")), flush=True)
    out["all"] = {"R": R, **summarise(allrows)}
    out["large_lead"] = {"min_lead_D_pts": 5, **summarise(allrows, 0.05)}
    print("ALL", json.dumps(out["all"]), flush=True)
    (ROOT / "results" / f"replication_certificates_{prefix}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

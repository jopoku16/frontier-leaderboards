"""Re-derive outcomes and reprice costs in the compact HAL run records without fetching anything again.

Each record keeps per-task token usage per model and the small evaluation sections of the upload, so when the price
lookup or the outcome layouts improve (hal_extract.py), this script applies the current rules to every record in
data/hal/runs/ and rewrites the derived fields: tasks[t].outcome, tasks[t].cost, n_tasks_with_outcome,
n_tasks_with_cost, unpriced_models, total_cost_recomputed, outcome_layout.

usage: python reprocess_runs.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hal_extract as hx  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "data" / "hal" / "runs"


def reprocess(rec):
    # outcomes from the stored evaluation sections, when present
    raw = rec.get("raw_sections") or {}
    if raw:
        outs, layout = hx.outcomes({"results": raw.get("results") or {}, "raw_eval_results": raw.get("raw_eval_results") or {}})
        if outs:
            for t in set(rec["tasks"]) | set(outs):
                rec["tasks"].setdefault(t, {"outcome": None, "cost": None, "tokens": {}})
                rec["tasks"][t]["outcome"] = outs.get(t)
            rec["outcome_layout"] = layout
    # costs from the stored tokens with the current price lookup
    unpriced = set()
    total = 0.0
    for t, v in rec["tasks"].items():
        toks = v.get("tokens") or {}
        if not toks:
            v["cost"] = None
            continue
        c, ok = 0.0, True
        for model, tk in toks.items():
            pc = hx._price(model, tk.get("prompt", 0.0), tk.get("cached", 0.0), tk.get("cache_creation", 0.0), tk.get("completion", 0.0))
            if pc is None:
                unpriced.add(model)
                ok = False
            else:
                c += pc
        v["cost"] = c if ok else None
        if ok:
            total += c
    rec["unpriced_models"] = sorted(unpriced)
    rec["total_cost_recomputed"] = total
    rec["n_tasks_with_outcome"] = sum(1 for v in rec["tasks"].values() if v.get("outcome") is not None)
    rec["n_tasks_with_cost"] = sum(1 for v in rec["tasks"].values() if v.get("cost") is not None)
    return rec


def main():
    n = changed = 0
    unpriced_all = {}
    for p in sorted(RUNS.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        before = (rec.get("n_tasks_with_outcome"), rec.get("n_tasks_with_cost"), round(rec.get("total_cost_recomputed") or 0, 2))
        rec = reprocess(rec)
        after = (rec["n_tasks_with_outcome"], rec["n_tasks_with_cost"], round(rec["total_cost_recomputed"], 2))
        n += 1
        if before != after:
            changed += 1
        for m in rec["unpriced_models"]:
            unpriced_all[m] = unpriced_all.get(m, 0) + 1
        p.write_text(json.dumps(rec), encoding="utf-8")
    print(f"reprocessed {n} records; {changed} changed")
    if unpriced_all:
        print("still unpriced (model: runs):", dict(sorted(unpriced_all.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()

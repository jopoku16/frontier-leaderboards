"""Build one board per HAL benchmark from the compact per-task records in data/hal/runs/.

Each HAL run (one agent, one model configuration, one benchmark) becomes one system. Its per-task outcome is the
benchmark's own score (pass/fail for most benchmarks; a reward in [0, 1] where the benchmark reports one) and its
per-task cost is the priced token usage of the model calls logged under that task (HAL's price table and formula,
see hal_extract.py). Runs that do not cover the benchmark's task set, or whose calls could not all be priced, are
listed in the summary and left out of the board. A kept run's tasks with an outcome but no logged model call (a
logging gap, or a run that failed before its first call) are priced at the run's mean cost over its costed tasks,
and the count is kept per system; a run with no cost on more than half its tasks is dropped. On a benchmark where no run logged per-call usage at all (the
browser agents on Online-Mind2Web), each run's cost is its run total priced from the stored usage with the same
table and spread evenly over the tasks; such a board carries cost_source "run_total_even" and the frontier analysis
then treats cost as known rather than resampled.

Writes, per benchmark: data/boards/hal_<benchmark>.json (honestboard board format, outcomes as rates),
data/boards/hal_<benchmark>_costs.json ({system: {task: cost}}), and data/boards/hal_summary.json.

usage: python build_hal_boards.py [--min-coverage 0.95]
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hal_extract as hx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "hal" / "runs"
OUT = ROOT / "data" / "boards"


def system_name(rec):
    agent = (rec.get("agent_name") or "agent").strip()
    args = rec.get("agent_args") or {}
    model = args.get("model_name") or args.get("agent.model.name") or args.get("model") or ""
    effort = args.get("reasoning_effort") or ""
    run = rec.get("run_id") or rec.get("source") or ""
    tail = run.rsplit("_", 1)[-1] if run else ""
    parts = [agent] + ([str(model)] if model else []) + ([str(effort)] if effort else []) + ([tail] if tail else [])
    return "__".join(re.sub(r"\s+", " ", p) for p in parts)


def main(argv):
    min_cov = float(argv[argv.index("--min-coverage") + 1]) if "--min-coverage" in argv else 0.95
    OUT.mkdir(parents=True, exist_ok=True)
    by_bench = defaultdict(list)
    for p in sorted(RUNS.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if not rec.get("benchmark"):
            continue
        by_bench[rec["benchmark"]].append(rec)
    summary = {}
    for bench, recs in sorted(by_bench.items()):
        # the benchmark's task set: tasks with an outcome in at least half of the runs
        cover = Counter(t for r in recs for t, v in r["tasks"].items() if v.get("outcome") is not None)
        tasks = sorted(t for t, c in cover.items() if c >= len(recs) / 2)
        if not tasks:
            summary[bench] = {"runs": len(recs), "kept": 0, "reason": "no task with outcomes in half the runs"}
            continue
        systems, costs, dropped = [], {}, []
        run_level = all((r.get("n_tasks_with_cost") or 0) == 0 for r in recs)   # no run on this board logged per-call usage
        for r in recs:
            name = system_name(r)
            outs = {t: r["tasks"][t]["outcome"] for t in tasks if t in r["tasks"] and r["tasks"][t].get("outcome") is not None}
            cost = {t: r["tasks"][t]["cost"] for t in tasks if t in r["tasks"] and r["tasks"][t].get("cost") is not None}
            coverage = len(outs) / len(tasks)
            no_calls = sum(1 for t in tasks if t not in cost)                # tasks with an outcome but no logged model call
            if coverage < min_cov:
                dropped.append({"system": name, "reason": f"outcome coverage {coverage:.2f}"})
                continue
            if r.get("unpriced_models"):
                dropped.append({"system": name, "reason": f"unpriced models {r['unpriced_models']}"})
                continue
            if run_level:                                                    # price the run total from its stored usage, spread evenly
                tu = (r.get("raw_sections") or {}).get("total_usage") or {}
                total, ok = 0.0, bool(tu)
                for m, u in tu.items():
                    pc = hx._price(m, *hx._normalize_usage(u))
                    if pc is None:
                        ok = False
                    else:
                        total += pc
                if not ok:
                    dropped.append({"system": name, "reason": "no per-call usage and the run total could not be priced"})
                    continue
                cost = {t: total / len(tasks) for t in tasks}
            elif len(cost) < 0.5 * len(tasks):                               # no usable call log: the run's cost is unknown, not zero
                dropped.append({"system": name, "reason": f"no logged model calls on {no_calls} of {len(tasks)} tasks (cost unknown)"})
                continue
            if bench == "colbench_backend_programming":                       # HAL reports backend accuracy as the share of tasks scored exactly 1.0;
                outs = {t: (1.0 if v >= 1.0 else 0.0) for t, v in outs.items()}   # frontend design keeps its continuous correctness (HAL's own metric there)
            rates = {t: float(outs.get(t, 0.0)) for t in tasks}              # a task without a logged outcome counts as a fail
            while any(s["name"] == name for s in systems):                   # reruns of the same configuration keep their run id
                name += "_"
            systems.append({"name": name, "date": (r.get("date") or "").replace("-", ""), "rates": rates,
                            "agent_name": r.get("agent_name"), "run_id": r.get("run_id"), "source": r.get("source"),
                            "accuracy_reported": r.get("accuracy_reported"), "total_cost_reported": r.get("total_cost_reported"),
                            "total_cost_recomputed": r.get("total_cost_recomputed"), "models": r.get("models"),
                            "tasks_without_calls": 0 if run_level else no_calls, "outcome_coverage": round(coverage, 4),
                            "tasks_without_outcome": len(tasks) - len(outs), "cost_source": "run_total_even" if run_level else "per_task_calls"})
            mean_c = float(sum(cost.values()) / len(cost)) if cost else 0.0
            costs[name] = {t: float(cost.get(t, mean_c)) for t in tasks}     # a task with no logged call is imputed at the run's mean costed task
        board = {"benchmark": f"HAL {bench}", "tasks": tasks, "systems": systems, "cost_source": "run_total_even" if run_level else "per_task_calls"}
        slug = re.sub(r"[^a-z0-9]+", "_", bench.lower()).strip("_")
        (OUT / f"hal_{slug}.json").write_text(json.dumps(board), encoding="utf-8")
        (OUT / f"hal_{slug}_costs.json").write_text(json.dumps(costs), encoding="utf-8")
        summary[bench] = {"runs": len(recs), "kept": len(systems), "tasks": len(tasks), "dropped": dropped,
                          "cost_source": "run_total_even" if run_level else "per_task_calls",
                          "mean_cost_per_run": round(sum(sum(c.values()) for c in costs.values()) / max(1, len(costs)), 2)}
        print(f"{bench:24s} runs {len(recs):3d} kept {len(systems):3d} tasks {len(tasks):4d} dropped {len(dropped)}" + ("  [cost: run totals, fixed]" if run_level else ""))
    (OUT / "hal_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

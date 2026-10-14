"""Turn one decrypted HAL upload into a compact per-task record: outcome and cost for every task.

HAL uploads hold (a) the run's configuration, (b) per-task outcomes in one of several layouts (lists of successful
and failed task ids, or a raw_eval_results map with a score or reward per task), (c) raw_logging_results, the Weave
call log in which every model call carries its task id and its token usage, and (d) the run's total cost as HAL
computed it. Per-task cost = the priced token usage of the calls logged under that task, with the price table the
harness uses (data/hal/model_prices.json), counting each model call ONCE: HAL's own total sums summary.usage over
every Weave call, and a wrapper span (litellm.completion) and the provider call inside it (openai.chat.completions
.create) carry the same usage block, so HAL's totals are exactly doubled on the runs that log both (reported on the
dataset's discussion board, Sept 2026, and confirmed by the HAL team). A call is dropped when its parent carries an
identical usage block (the inner duplicate), or when it has two or more children with usage whose sum equals its own
(a roll-up). The record keeps the raw total too, so the overstatement can be reported per run.

usage: python hal_extract.py <decrypted upload json> [...]   (prints a summary; see hal_fetch.py for the batch path)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_D = ROOT / "data" / "hal"
PRICES = json.loads((_D / "model_prices.json").read_text()) if (_D / "model_prices.json").exists() else {}
CACHE_PRICES = json.loads((_D / "cache_prices.json").read_text()) if (_D / "cache_prices.json").exists() else {}


def _normalize_usage(u):
    """HAL's reading of one usage record: (prompt incl. cached, cached reads, cache writes, completion)."""
    if "prompt_tokens" in u or "completion_tokens" in u:                 # OpenAI style
        prompt = u.get("prompt_tokens", 0) or 0
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0
        creation = 0
        completion = u.get("completion_tokens", 0) or 0
    elif "input_tokens" in u or "output_tokens" in u:                   # Anthropic style
        cached = u.get("cache_read_input_tokens", 0) or 0
        creation = u.get("cache_creation_input_tokens", 0) or 0
        prompt = (u.get("input_tokens", 0) or 0) + cached
        completion = u.get("output_tokens", 0) or 0
    elif "inputTokens" in u or "outputTokens" in u:                     # Bedrock style
        prompt = u.get("inputTokens", 0) or 0
        cached = u.get("cacheReadInputTokens", 0) or 0
        creation = u.get("cacheWriteInputTokens", 0) or 0
        completion = u.get("outputTokens", 0) or 0
    else:
        prompt = cached = creation = completion = 0
    return float(prompt), float(cached), float(creation), float(completion)


ALIASES = {"o3-2025-04-03": "o3-2025-04-16", "deepseek-chat": "deepseek-ai/DeepSeek-V3", "deepseek-r1": "deepseek-ai/DeepSeek-R1"}


def _lookup(table, model):
    """Price-table lookup that tolerates provider prefixes (openrouter/anthropic/x), ':thinking' suffixes, dots for
    dashes, dated snapshots (-20251001) and '-001' suffixes, and a few known aliases; case-insensitive."""
    if model in table:
        return model
    lower = {k.lower(): k for k in table}
    base = model.split(":", 1)[0]
    forms = []
    for c in [base] + [base.split("/", k)[-1] for k in range(1, base.count("/") + 1)]:
        forms.append(c)
        forms.append(ALIASES.get(c.lower(), c))
        stripped = re.sub(r"-(\d{8}|\d{4}-\d{2}-\d{2}|001)$", "", c)
        forms += [stripped, ALIASES.get(stripped.lower(), stripped)]
        forms += [f.replace(".", "-") for f in (c, stripped)] + [f.replace("-", ".") for f in (c, stripped)]
    for f in forms:
        if f in table:
            return f
        if f.lower() in lower:
            return lower[f.lower()]
    return None


def _price(model, prompt, cached, creation, completion):
    """HAL's formula: fresh input at the prompt price, cache reads and writes at the override (or prompt) price."""
    key = _lookup(PRICES, model)
    if key is None:
        return None
    p = PRICES[key]
    ckey = _lookup(CACHE_PRICES, model)
    cache_price = CACHE_PRICES[ckey] if ckey is not None else p["prompt"]
    fresh = prompt - cached
    return fresh * p["prompt"] + creation * cache_price + cached * cache_price + completion * p["completion"]


def _usage_of(call):
    """{model: [prompt, cached, creation, completion]} for one Weave call (summary.usage may be a dict or a list)."""
    out = {}
    summ = call.get("summary") or {}
    usage = summ.get("usage") or {}
    items = usage.items() if isinstance(usage, dict) else [(m, mu) for entry in usage if isinstance(entry, dict) for m, mu in entry.items()]
    for model, u in items:
        if not isinstance(u, dict):
            continue
        vals = _normalize_usage(u)
        acc = out.setdefault(model, [0.0, 0.0, 0.0, 0.0])
        for i in range(4):
            acc[i] += vals[i]
    return out


def _dedupe_calls(calls):
    """Indices of the calls whose usage should be counted (see the module docstring), plus the number dropped."""
    usage = [_usage_of(c) for c in calls]
    by_id = {c.get("id"): i for i, c in enumerate(calls) if c.get("id") is not None}
    children = {}
    for i, c in enumerate(calls):
        par = by_id.get(c.get("parent_id"))
        if par is not None:
            children.setdefault(par, []).append(i)
    keep = []
    dropped = 0
    for i, c in enumerate(calls):
        if not usage[i]:
            continue
        par = by_id.get(c.get("parent_id"))
        if par is not None and usage[par] and usage[par] == usage[i]:
            dropped += 1                                   # inner duplicate of its parent's usage block
            continue
        kids = [k for k in children.get(i, []) if usage[k]]
        if len(kids) >= 2:
            total = {}
            for k in kids:
                for m, v in usage[k].items():
                    acc = total.setdefault(m, [0.0, 0.0, 0.0, 0.0])
                    for j in range(4):
                        acc[j] += v[j]
            if total == usage[i]:
                dropped += 1                               # roll-up of its children's usage
                continue
        keep.append(i)
    return keep, dropped


def _task_id(call):
    tid = call.get("weave_task_id")
    if tid is None:
        tid = ((call.get("attributes") or {}).get("weave_task_id"))
    return None if tid is None else str(tid)


def outcomes(upload):
    """{task_id: outcome in [0,1]} from whichever layout the benchmark used, plus a note on the layout."""
    res = upload.get("results") or {}
    raw = upload.get("raw_eval_results") or {}
    out = {}
    if isinstance(res.get("successful_tasks"), list) or isinstance(res.get("failed_tasks"), list):
        for t in res.get("successful_tasks") or []:
            out[str(t)] = 1.0
        for t in res.get("failed_tasks") or []:
            out[str(t)] = 0.0
        if out:
            return out, "successful_tasks/failed_tasks"
    if isinstance(raw, list) and raw and all(isinstance(v, (bool, int, float)) for v in raw):   # colbench: a score per task in task order
        return {str(i): float(v) for i, v in enumerate(raw)}, "raw_eval_results list"
    if isinstance(raw, dict):
        table = raw.get("eval_result") if isinstance(raw.get("eval_result"), dict) else raw   # scienceagentbench nests per-task results
        for t, v in table.items():
            if isinstance(v, dict):
                for key in ("score", "reward", "correct", "success", "passed", "success_rate", "resolved"):
                    if key in v and isinstance(v[key], (bool, int, float)):
                        out[str(t)] = float(bool(v[key])) if isinstance(v[key], bool) else float(v[key])
                        break
            elif isinstance(v, (bool, int, float)):
                out[str(t)] = float(v)
        if out:
            return out, "raw_eval_results" if table is raw else "raw_eval_results.eval_result"
    return out, "none"


def extract(upload, source_name=""):
    cfg = upload.get("config") or {}
    res = upload.get("results") or {}
    per_task_tokens = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))
    calls = 0
    unpriced_models = set()
    no_task = 0
    call_log = []                                     # compact per-call usage, so runs can be repriced without another fetch
    all_calls = upload.get("raw_logging_results") or []
    keep, n_dup = _dedupe_calls(all_calls)
    raw_tokens = 0.0
    for call in all_calls:
        for vals in _usage_of(call).values():
            raw_tokens += vals[0] + vals[3]
    counted_tokens = 0.0
    for i in keep:
        call = all_calls[i]
        tid = _task_id(call)
        if tid is None:
            no_task += 1
            continue
        calls += 1
        for model, vals in _usage_of(call).items():
            acc = per_task_tokens[tid][model]
            for i4 in range(4):
                acc[i4] += vals[i4]
            counted_tokens += vals[0] + vals[3]
            call_log.append([tid, model, vals[0], vals[1], vals[2], vals[3], (call.get("started_at") or "")[:19]])
    per_task_cost = {}
    for tid, models in per_task_tokens.items():
        c = 0.0
        ok = True
        for model, (pt, cached, creation, ct) in models.items():
            pc = _price(model, pt, cached, creation, ct)
            if pc is None:
                unpriced_models.add(model)
                ok = False
            else:
                c += pc
        per_task_cost[tid] = c if ok else None
    outs, layout = outcomes(upload)
    tasks = sorted(set(outs) | set(per_task_cost))
    recomputed = sum(v for v in per_task_cost.values() if v is not None)
    reported = upload.get("total_cost", res.get("total_cost"))
    # keep the small evaluation sections verbatim, so outcome layouts can be re-derived without another fetch
    keep = {}
    for key in ("results", "raw_eval_results", "total_usage"):
        v = upload.get(key)
        if v is not None and len(json.dumps(v, default=str)) <= 3_000_000:
            keep[key] = v
    return {
        "raw_sections": keep,
        "calls": call_log,
        "source": source_name,
        "run_id": cfg.get("run_id"), "benchmark": cfg.get("benchmark_name"), "agent_name": cfg.get("agent_name"),
        "date": cfg.get("date"), "agent_args": cfg.get("agent_args"),
        "accuracy_reported": res.get("accuracy"),
        "total_cost_reported": reported, "total_cost_recomputed": recomputed,
        "outcome_layout": layout, "n_tasks_with_outcome": len(outs), "n_tasks_with_cost": sum(1 for v in per_task_cost.values() if v is not None),
        "n_calls": calls, "n_calls_without_task": no_task, "unpriced_models": sorted(unpriced_models),
        "n_duplicate_calls": n_dup, "tokens_raw": raw_tokens, "tokens_counted": counted_tokens,
        "extract_version": 2,
        "models": sorted({m for ms in per_task_tokens.values() for m in ms}),
        "tasks": {t: {"outcome": outs.get(t), "cost": per_task_cost.get(t),
                      "tokens": {m: {"prompt": v[0], "cached": v[1], "cache_creation": v[2], "completion": v[3]}
                                 for m, v in per_task_tokens.get(t, {}).items()}} for t in tasks},
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        rec = extract(json.loads(Path(p).read_text(encoding="utf-8")), Path(p).name)
        acc = [v["outcome"] for v in rec["tasks"].values() if v["outcome"] is not None]
        print(f"{rec['run_id']}: {rec['benchmark']} / {rec['agent_name']} | tasks {len(rec['tasks'])} outcomes {len(acc)} "
              f"(mean {sum(acc)/len(acc) if acc else float('nan'):.3f}, reported {rec['accuracy_reported']}) | "
              f"cost reported {rec['total_cost_reported']} recomputed {rec['total_cost_recomputed']:.2f} | with cost {rec['n_tasks_with_cost']} | "
              f"layout {rec['outcome_layout']} | unpriced {rec['unpriced_models']} | calls {rec['n_calls']} (+{rec['n_calls_without_task']} without task)")

"""Generate the supplement's tables from the result files (paper/supp_*.tex), so the supplement never disagrees
with the analysis. Run after make_numbers.py.

S1  per-benchmark system tables: cost per task, accuracy, observed frontier, frontier set, undominated frequency,
    best set, strongest cheaper rival's lead in standard errors   (results/frontier_hal.json)
S2  certification steps per benchmark                             (results/certify_hal.json)
S3  frontier winner's curse per frontier member                   (results/curse_hal.json)
S4  runs left out of the audit, with the reason                   (data/boards/hal_summary.json)
S5  HAL's double counting per benchmark: runs by raw/counted ratio (data/hal/index.json, extract version 2)
S6  the price table used                                          (data/hal/model_prices.json, cache_prices.json)
"""
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PAP = os.path.join(ROOT, "paper")
DATA = os.path.join(ROOT, "data")

NAMES = {"hal_assistantbench": "AssistantBench", "hal_colbench_backend_programming": "ColBench backend programming",
         "hal_colbench_frontend_design": "ColBench frontend design", "hal_corebench_hard": "CORE-Bench hard", "hal_gaia": "GAIA",
         "hal_online_mind2web": "Online-Mind2Web", "hal_scicode": "SciCode", "hal_scienceagentbench": "ScienceAgentBench",
         "hal_swebench_verified_mini": "SWE-bench Verified mini", "hal_taubench_airline": "tau-bench airline", "hal_usaco": "USACO"}


def load(name, base=RES):
    p = os.path.join(base, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def tex(s):
    return re.sub(r"([_&%#$])", r"\\\1", str(s))


def short(name, width=46):
    """The agent-plus-model part of a HAL system name, trimmed."""
    head = name.split("__", 1)[0]
    head = re.sub(r"\s+", " ", head).strip()
    if len(head) > width:
        head = head[:width - 1] + "~"
    return tex(head)


def unique_short(names):
    out, seen = [], {}
    for n in names:
        s = short(n)
        seen[s] = seen.get(s, 0) + 1
        out.append(s)
    counts = {}
    for i, s in enumerate(out):
        if seen[s] > 1:
            counts[s] = counts.get(s, 0) + 1
            out[i] = f"{s} ({counts[s]})"
    return out


def write(fname, text):
    with open(os.path.join(PAP, fname), "w", encoding="utf-8") as f:
        f.write(text)


def longtable(cols, header, rows, caption, label):
    body = "\\begin{longtable}{" + cols + "}\n\\caption{" + caption + "}\\label{" + label + "}\\\\\n\\toprule\n" + header + " \\\\\n\\midrule\n\\endfirsthead\n"
    body += "\\toprule\n" + header + " \\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n"
    body += "".join(r + " \\\\\n" for r in rows)
    body += "\\end{longtable}\n"
    return body


def main():
    aud = load("frontier_hal.json") or {}
    parts = []
    for k, v in aud.items():
        dag = "$^{\\dagger}$" if v.get("cost_source") == "run_total_even" else ""
        tab = v["table"]
        names = unique_short([r["system"] for r in tab])
        bpath = os.path.join(DATA, "boards", k + ".json")
        meta = {sy["name"]: sy for sy in json.load(open(bpath, encoding="utf-8"))["systems"]} if os.path.exists(bpath) else {}
        fixed_set = set(v.get("frontier_set_fixed", v["frontier_set"]))
        rows = []
        for r, nm in zip(tab, names):
            lead = "" if r["rival_lead_se"] is None else f"{r['rival_lead_se']:+.1f}"
            m = meta.get(r["system"], {})
            unc = m.get("tasks_without_calls", 0) or 0
            fx = "yes" if r["system"] in fixed_set else ""
            rows.append(f"{nm} & {r['cost']:.3f} & {unc} & {100 * r['accuracy']:.1f} & {'yes' if r['on_observed_frontier'] else ''} & "
                        f"{'yes' if r['in_frontier_set'] else ''} & {fx} & {r['undominated_frequency']:.2f} & {'yes' if r['in_best_set'] else ''} & {lead}")
        parts.append(longtable("lrrrccccrr", "System & \\$/task & Uncosted & Acc.\\ \\% & Frontier & In set & In set (fixed) & Undom.\\ freq. & Best set & Rival lead (se)", rows,
                               f"{NAMES.get(k, k)}{dag}: {v['systems']} systems on {v['tasks']} tasks, cheapest first. Uncosted: tasks with an outcome but no "
                               f"logged model call, priced at the run's mean costed task. In set (fixed): the frontier set with every cost fixed at its mean. "
                               f"Rival lead: how far the strongest cheaper-or-equal system sits above this one, in paired standard errors (negative when "
                               f"this system is ahead); the simultaneous margin for this board is {v['margin_se']:.2f}.", f"tab:supp-{k}"))
    write("supp_systems.tex", "\n".join(parts))

    cert = load("certify_hal.json") or {}
    parts = []
    for k, v in cert.items():
        if "error" in v or not v["steps"]:
            continue
        rows = []
        for s in v["steps"]:
            rows.append(f"{s['test']} & {s['date'][:4]}-{s['date'][4:6]}-{s['date'][6:]} & {short(s['new'], 34)} & {short(s['holder'], 30)} & {100 * s['margin']:+.1f} & "
                        f"{s['p_uncorrected']:.3f} & {s['date_group']} & {s['p']:.3f} & {s['level_spending']:.4f} & {'yes' if s['certified_spending'] else ''} & "
                        f"{s['level_fallback']:.4f} & {'yes' if s['certified_fallback'] else ''}")
        parts.append(longtable("rlllrrrrrcrc", "$t$ & Date & Entrant & Holder & Margin & $p$ raw & $g$ & $p$ & Spending & Cert. & Fallback & Cert.", rows,
                               f"{NAMES.get(k, k)}: frontier entrants in arrival order. Holder: the most accurate cheaper-or-equal earlier system. "
                               f"$p$ raw: selective record test; $g$: submissions on the entrant's date; $p = \\min(1, g \\cdot p_{{\\mathrm{{raw}}}})$. "
                               f"Levels under alpha-spending and the online fallback, with the certification decision.", f"tab:supp-cert-{k}"))
    write("supp_certify.tex", "\n".join(parts))

    curse = load("curse_hal.json") or {}
    rows = []
    for k, v in curse.items():
        first = True
        for m in v["members"]:
            c, b = m["conditional"], m["bootstrap"]
            cond = f"{100 * c['estimate']:.1f} [{100 * c['ci'][0]:.1f}, {100 * c['ci'][1]:.1f}]" if math.isfinite(c["estimate"]) and all(math.isfinite(x) for x in c["ci"]) else "undefined"
            rows.append(f"{NAMES.get(k, k) if first else ''} & {short(m['system'], 36)} & {m['pool_size']} & {100 * m['observed']:.1f} & "
                        f"{100 * b['estimate']:.1f} [{100 * b['ci'][0]:.1f}, {100 * b['ci'][1]:.1f}] & {cond}")
            first = False
    write("supp_curse.tex", longtable("llrrll", "Benchmark & System & Pool & Reported & Iterated bootstrap [95\\% CI] & Conditional [95\\% CI]", rows,
                                      "Selection-adjusted accuracy of every system on an observed frontier. Pool: the number of systems that cost no more "
                                      "(the system included); a pool of one has no selection and no correction.", "tab:supp-curse"))

    summ = load("hal_summary.json", os.path.join(DATA, "boards")) or {}
    rows = []
    for bench, v in summ.items():
        for d in v.get("dropped", []):
            rows.append(f"{tex(bench)} & {short(d['system'], 44)} & {tex(d['reason'])}")
    write("supp_exclusions.tex", longtable("llp{7cm}", "Benchmark & Run & Reason", rows,
                                           "Runs read from HAL's archives but left out of the audit.", "tab:supp-exclusions"))

    idx = load("index.json", os.path.join(DATA, "hal")) or {}
    by = {}
    for name, e in idx.items():
        if e.get("v") != 2 or not e.get("tokens_counted"):
            continue
        bench = e.get("benchmark") or "?"
        r = e["tokens_raw"] / e["tokens_counted"]
        bucket = "exactly 2" if abs(r - 2.0) < 1e-3 else ("between 1 and 2" if r > 1.001 else "1 (no duplicates)")
        by.setdefault(bench, {}).setdefault(bucket, 0)
        by[bench][bucket] += 1
    rows = []
    tot = {}
    for bench in sorted(by):
        b = by[bench]
        rows.append(f"{tex(bench)} & {b.get('1 (no duplicates)', 0)} & {b.get('between 1 and 2', 0)} & {b.get('exactly 2', 0)}")
        for key in ("1 (no duplicates)", "between 1 and 2", "exactly 2"):
            tot[key] = tot.get(key, 0) + b.get(key, 0)
    rows.append(f"\\midrule\nAll & {tot.get('1 (no duplicates)', 0)} & {tot.get('between 1 and 2', 0)} & {tot.get('exactly 2', 0)}")
    write("supp_doublecount.tex", longtable("lrrr", "Benchmark & Ratio 1 & Between 1 and 2 & Exactly 2", rows,
                                            "HAL's token total divided by the total with each model call counted once, per run. A ratio of exactly 2 "
                                            "means every provider call was logged inside a wrapper span that carried the same usage block.", "tab:supp-doublecount"))

    prices = load("model_prices.json", os.path.join(DATA, "hal")) or {}
    cache = load("cache_prices.json", os.path.join(DATA, "hal")) or {}
    rows = []
    for m in sorted(prices):
        p = prices[m]
        cp = cache.get(m)
        rows.append(f"{tex(m)} & {1e6 * p['prompt']:.3g} & {1e6 * p['completion']:.3g} & {'' if cp is None else f'{1e6 * cp:.3g}'}")
    write("supp_prices.tex", longtable("lrrr", "Model & Input (\\$/M) & Output (\\$/M) & Cache read (\\$/M)", rows,
                                       "The price table used for every run (HAL's current table, dollars per million tokens). Cache reads and writes are "
                                       "priced at the cache rate where one is listed and at the input rate otherwise.", "tab:supp-prices"))
    print("wrote supp_systems, supp_certify, supp_curse, supp_exclusions, supp_doublecount, supp_prices")


if __name__ == "__main__":
    main()

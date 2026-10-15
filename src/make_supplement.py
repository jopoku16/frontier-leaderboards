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

HELM_NAMES = {'helm_gpqa': 'GPQA', 'helm_ifeval': 'IFEval', 'helm_mmlu_pro': 'MMLU-Pro', 'helm_omni_math': 'Omni-MATH'}
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


def unique_short(names, width=46):
    out, seen = [], {}
    for n in names:
        s = short(n, width)
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
        names = unique_short([r["system"] for r in tab], 24)
        bpath = os.path.join(DATA, "boards", k + ".json")
        meta = {sy["name"]: sy for sy in json.load(open(bpath, encoding="utf-8"))["systems"]} if os.path.exists(bpath) else {}
        fixed_set = set(v.get("frontier_set_fixed", v["frontier_set"]))
        rows = []
        for r, nm in zip(tab, names):
            lead = "" if r["rival_lead_se"] is None else f"{r['rival_lead_se']:+.1f}"
            m = meta.get(r["system"], {})
            unc = m.get("tasks_without_calls", 0) or 0
            fx = "yes" if r["system"] in fixed_set else ""
            tr = "yes" if r["system"] in set(v.get("frontier_set_tie_robust", v["frontier_set"])) else ""
            rows.append(f"{nm} & {r['cost']:.3f} & {unc} & {100 * r['accuracy']:.1f} & {'yes' if r['on_observed_frontier'] else ''} & "
                        f"{'yes' if r['in_frontier_set'] else ''} & {fx} & {tr} & {r['undominated_frequency']:.2f} & {'yes' if r['in_best_set'] else ''} & {lead}")
        parts.append(longtable("lrrrcccccrr", "System & \\$/task & Unc. & Acc.\\ \\% & Front. & Set & Fixed & Rob. & Stab. & Best & Lead", rows,
                               f"{NAMES.get(k, k)}{dag}: {v['systems']} systems on {v['tasks']} tasks, cheapest first. Unc.: tasks with an outcome but no "
                               f"logged model call, priced at the run's mean costed task. Front.: on the observed frontier. Set: in the frontier set. Fixed: in the frontier set with every cost fixed at its mean. Rob.: in the tie-robust set. Stab.: share of task resamples in which the system is on the observed frontier. Best: in the best set. "
                               f"Lead: rival lead, how far the strongest cheaper-or-equal system sits above this one, in paired standard errors (negative when "
                               f"this system is ahead); the simultaneous margin for this board is {v['margin_se']:.2f}.", f"tab:supp-{k}"))
    write("supp_systems.tex", "{\\scriptsize\\setlength{\\tabcolsep}{2pt}\n" + "\n".join(parts) + "}\n")

    cert = load("certify_hal.json") or {}
    parts = []
    for k, v in cert.items():
        if "error" in v or not v["steps"]:
            continue
        rows = []
        for s in v["steps"]:
            piu = f"{s['p_iu']:.3f}" if s.get("p_iu") is not None else ""
            pex = f"{s['p_ex']:.3f}" if s.get("p_ex") is not None else ""
            rows.append(f"{s['test']} & {s['date'][:4]}-{s['date'][4:6]}-{s['date'][6:]} & {short(s['new'], 18)} & {short(s['holder'], 16)} & {100 * s['margin']:+.1f} & "
                        f"{s['p_uncorrected']:.3f} & {s['date_group']} & {s['p']:.3f} & {s['level_spending']:.4f} & {'yes' if s['certified_spending'] else ''} & "
                        f"{s['level_fallback']:.4f} & {'yes' if s['certified_fallback'] else ''} & {piu} & {'yes' if s.get('certified_iu_spending') else ''} & {pex} & {'yes' if s.get('certified_ex_spending') else ''}")
        parts.append(longtable("rlllrrrrrcrcrcrc", "$t$ & Date & Entrant & Holder & Margin & $p$ raw & $g$ & $p$ & Spending & Cert. & Fallback & Cert. & $p_{\\mathrm{IU}}$ & Cert. & $p_{\\mathrm{ex}}$ & Cert.", rows,
                               f"{NAMES.get(k, k)}: frontier entrants in arrival order. Holder: the most accurate cheaper-or-equal earlier system. "
                               f"$p$ raw: selective record test; $g$: submissions on the entrant's date; $p = \\min(1, g \\cdot p_{{\\mathrm{{raw}}}})$. "
                               f"Levels under alpha-spending and the online fallback, with the certification decision; the last two columns are the "
                               f"composite-null (intersection-union) $p$-value after the per-date factor and its decision under alpha-spending.", f"tab:supp-cert-{k}"))
    write("supp_certify.tex", "{\\tiny\\setlength{\\tabcolsep}{1.5pt}\n" + "\n".join(parts) + "}\n")

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
    sens = load("hal_sensitivity.json") or {}
    sep = (load("cost_separation.json") or {}).get("hal", {})
    rows = []
    for k, v in sens.items():
        fc, ni = v["variants"]["full_cache"], v["variants"]["no_imputation"]
        share = sep.get(k, {}).get("separated_share")
        rows.append(f"{NAMES.get(k, k)} & {v['systems']} & {v['tasks']} & {v['baseline_frontier']} & {v['baseline_set']} & {fc['mean_cost_ratio']:.2f} & "
                    f"{fc['frontier_changed']} & {fc['set_changed']} & {ni['frontier_changed']} & {ni['set_changed']} & {'' if share is None else f'{share:.2f}'}")
    write("supp_sensitivity.tex", longtable("lrrrrrrrrrr", "Benchmark & Sys. & Tasks & Front. & Set & Cost ratio & Front. moved & Set moved & Front. moved & Set moved & Cost-separated pairs", rows,
                                            "Sensitivity of the audit to two costing rules. Frontier and set: sizes in the audit. Cost ratio: mean cost per task with cache reads and writes "
                                            "priced at the full input price, over the audit's. Frontier moved and set moved: systems that enter or leave the plain frontier or the frontier set "
                                            "under that rule (first pair of columns: full input price for cached tokens; second pair: a task with no logged model call costs nothing instead of the "
                                            "run's mean). Online-Mind2Web has run totals only and is left out of the sensitivity columns. Cost-separated pairs: share of pairs of systems whose cost order the paired step-down of Section 3.3 resolves (boards with per-task costs only).", "tab:supp-sensitivity"))
    rn = load("rerun_noise.json") or {}
    rows = []
    for q in rn.get("pairs", []):
        pi = 1 - q["task_agreement"]
        sd_ind = (pi / q["tasks"]) ** 0.5
        rows.append(f"{NAMES.get(q['board'], q['board'])} & {tex(short(q['config'], 40))} & {q['tasks']} & {100 * q['acc'][0]:.1f} & {100 * q['acc'][1]:.1f} & {100 * q['acc_diff']:+.1f} & "
                    f"{100 * q['task_agreement']:.0f} & {100 * sd_ind:.1f} & {q['acc_diff'] / sd_ind:+.1f} & {q['cost_ratio']:.2f}")
    write("supp_rerun.tex", longtable("llrrrrrrrr", "Benchmark & Configuration & Tasks & Acc. A & Acc. B & Diff. & Agree & Pred. sd & $z$ & Cost ratio", rows,
                                      "The eight pairs of runs of one configuration. Acc. A, Acc. B: accuracy of the two runs, in per cent; Diff.: A minus B in points; Agree: share of tasks on which "
                                      "the two runs have the same outcome; Pred. sd: standard deviation of the difference in points if every task flipped independently at the observed "
                                      "rate of disagreement; $z$: the difference over that standard deviation. The sum of the squared $z$ is the chi-square statistic of the main text.", "tab:supp-rerun"))
    # ---- S12: the four tie-robust constructions on known truth, and the least-favourable check ----
    import glob as _glob
    vf = load("validation_frontier.json") or {"scenarios": []}
    rows = []
    for s_ in vf["scenarios"]:
        sets = s_["sets"]
        cell = lambda key: f"{sets[key]['joint_retention']:.2f} & {sets[key]['mean_set_size']:.1f}" if key in sets else " & "
        short_ = {'cost-outcome dependence': 'Cost-outcome dep.', 'weak cost-ability link': 'Weak cost link', 'cheaper near-copies': 'Cheaper near-copies'}
        rows.append(f"{short_.get(s_['label'], s_['label']).capitalize()} & {s_['K']} & {s_['T']} & {s_['mean_true_frontier_size']:.1f} & {cell('joint')} & {cell('tierobust')} & {cell('tierobust_sd')} & {cell('tierobust_int')}")
    write("supp_tierobust.tex", longtable("lrrrrrrrrrrr", "Scenario & $K$ & $T$ & True & Std kept & size & Gated kept & size & Step-down kept & size & Interval kept & size", rows,
                                          "The standard set and three tie-robust constructions on known truth, 300 boards per scenario. Kept: share of boards on which every truly undominated system is kept; size: mean number of systems kept; True: size of the true frontier. Gated: the gated step-down of Section 3.3; Step-down: the same without the gate, the version proved with no condition on the dependence of cost and accuracy errors; Interval: the interval construction of honestboard 0.5.1.", "tab:supp-tierobust"))
    st_rows = []
    for f_ in sorted(_glob.glob(os.path.join(ROOT, "data", "hal", "logs", "tr_stress2_*.log"))):
        for ln in open(f_, encoding="utf-8"):
            m = re.search(r"spread_cost ([0-9.]+) lam ([0-9.]+) K=(\d+) T=(\d+) R=(\d+): std kept-all ([0-9.]+) size ([0-9.]+) \| intervals kept-all ([0-9.]+) size ([0-9.]+) \| gated kept-all ([0-9.]+) size ([0-9.]+) \| stepdown kept-all ([0-9.]+) size ([0-9.]+)", ln)
            if m:
                g = m.groups()
                st_rows.append(f"{g[0]} & {1 + float(g[1]):.0f} & {g[2]} & {g[3]} & {g[4]} & {g[5]} & {g[7]} & {g[11]} & {g[9]}")
    write("supp_tierobust_stress.tex", longtable("rrrrrrrrr", "Cost spread & Failed-task multiple & $K$ & $T$ & Boards & Standard & Interval & Step-down & Gated", st_rows,
                                                 "Least favourable check: every system has the same true accuracy, so every system is in $F$ and any exclusion is an error; true costs are spread so that their gaps are of the order of the critical value (cost spread, the standard deviation of the log true cost); failed tasks cost the failed-task multiple times as much as passed ones. Columns: share of boards on which the set keeps every system, at the nominal 0.95.", "tab:supp-stress"))
    # ---- S13: the HELM price table and the models left out ----
    hp = load("model_prices.json", os.path.join(DATA, "helm")) or {}
    rows = [f"{tex(m)} & {1e6 * p_['prompt']:.3g} & {1e6 * p_['completion']:.3g} & {tex(p_['source'])}" for m, p_ in sorted(hp.items())]
    write("supp_helm_prices.tex", longtable("lrrp{7cm}", "Model & Input (\\$/M) & Output (\\$/M) & Source", rows,
                                            "The price table of the HELM audit: list prices in dollars per million tokens, with the source key of the price. Models from OpenAI and Anthropic that the HAL audit also prices use the HAL table; the others use a public price list fetched on 14 October 2026.", "tab:supp-helm-prices"))
    hs = load("helm_summary.json", os.path.join(DATA, "boards")) or {}
    first = next((v for v in hs.values() if "dropped" in v), {"dropped": []})
    rows = [f"{tex(d['system'])} & {tex(d['reason'])}" for d in sorted(first["dropped"], key=lambda d: (d["reason"], d["system"]))]
    write("supp_helm_dropped.tex", longtable("lp{7cm}", "Model & Why it is left out", rows,
                                             "HELM Capabilities models left out of every board, with the reason. WildBench is left out of the audit altogether because it records no prompt tokens.", "tab:supp-helm-dropped"))
    rows = []
    for pref, lab in (("hal", "HAL"), ("helm", "HELM")):
        rp = load(f"replication_certificates_{pref}.json") or {}
        for k, v in rp.items():
            if not k.startswith(pref + "_"):
                continue
            for g, gl in (("certified", "certified"), ("nominal", "nominal only"), ("other", "neither")):
                x = v.get(g, {})
                if x.get("n"):
                    rows.append(f"{lab} & {HELM_NAMES.get(k, NAMES.get(k, tex(k)))} & {gl} & {x['n']} & {x.get('distinct_pairs', '')} & {x['mean_lead_D_pts']:.1f} & {x['mean_lead_V_pts']:.1f} & {x['positive_on_V']:.2f} & {x['significant_on_V']:.2f}")
    write("supp_replication.tex", longtable("llrrrrrrr", "Leaderboard & Board & Group & Cases & Distinct pairs & Lead on D & Lead on V & Positive & Significant", rows,
                                           "Fresh-task replication of the entrant certificates, board by board. Cases: entrant-holder pairs tested over the random splits; distinct pairs: how many different entrant-holder pairs they are; leads in points. Groups as in Table 9 of the main text.", "tab:supp-replication"))
    hps = load("helm_price_sensitivity.json") or {}
    rows = []
    for k, nm in (("gpqa", "GPQA"), ("ifeval", "IFEval"), ("mmlu_pro", "MMLU-Pro"), ("omni_math", "Omni-MATH")):
        v = hps.get(k)
        if v:
            rows.append(f"{nm} & {v['systems']} & {v['frontier']} & {v['frontier_alt']} & {v['frontier_changed']} & {v['set']} & {v['set_alt']} & {v['set_changed']} & {v['mean_cost_ratio_repriced']:.2f}")
    write("supp_helm_pricesens.tex", longtable("lrrrrrrrr", "Benchmark & Systems & Frontier & at DeepInfra & moved & Set & at DeepInfra & moved & Cost ratio", rows,
                                              "Sensitivity of the HELM audit to the host of the open-weight prices: the models that LiteLLM also lists at DeepInfra are repriced there from the same recorded tokens. Frontier and set: sizes under the Together AI prices of the audit and at DeepInfra; moved: systems that enter or leave; cost ratio: mean ratio of the repriced models' cost per task.", "tab:supp-helm-pricesens"))
    print("wrote supp_systems, supp_certify, supp_curse, supp_exclusions, supp_doublecount, supp_prices, supp_sensitivity, supp_tierobust, supp_helm_prices, supp_helm_dropped")


if __name__ == "__main__":
    main()

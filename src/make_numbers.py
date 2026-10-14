"""Generate paper/numbers.tex and paper/tab_*.tex from the result files, so the manuscript never disagrees with the analysis.

Reads results/validation_frontier.json (known-truth simulation) and results/frontier_hal.json (the audit) when present.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PAP = os.path.join(ROOT, "paper")
os.makedirs(PAP, exist_ok=True)
NUM = {}


def put(name, value):
    NUM[name] = value


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


# ---- known-truth validation of the frontier set ----
val = load("validation_frontier.json")
if val:
    put("ValR", val["R"])
    scen_all = val["scenarios"]
    scen = [s for s in scen_all if s.get("order", 0) < 10]          # the ten scenarios of the first version; the rest are the stress tests
    put("ValScenarios", len(scen_all)); put("ValScenariosMain", len(scen))
    put("ValMcSeMax", f"{(0.95 * 0.05 / val['R']) ** 0.5:.3f}")
    put("ValRetentionMin", f"{min(s['per_system_retention'] for s in scen):.3f}")
    put("ValRetentionMax", f"{max(s['per_system_retention'] for s in scen):.3f}")
    put("ValJointMin", f"{min(s['joint_retention'] for s in scen):.2f}")
    put("ValJointMax", f"{max(s['joint_retention'] for s in scen):.2f}")
    put("ValNaiveFalseMin", pct(min(s["naive_false_frontier_share"] for s in scen)))
    put("ValNaiveFalseMax", pct(max(s["naive_false_frontier_share"] for s in scen)))
    put("ValNaiveMissedMin", pct(min(s["naive_missed_share"] for s in scen)))
    put("ValNaiveMissedMax", pct(max(s["naive_missed_share"] for s in scen)))
    big = [s for s in scen if s["K"] == 20 and s["corr"] == 0.6 and s.get("cost_cv", 0.8) != 0]
    for s in big:
        tag = {50: "A", 150: "B", 500: "C"}.get(s["T"])
        if tag:
            put(f"ValSetSize{tag}", f"{s['mean_set_size']:.1f}"); put(f"ValTrueSize{tag}", f"{s['mean_true_frontier_size']:.1f}")
            put(f"ValNaiveFalse{tag}", pct(s["naive_false_frontier_share"])); put(f"ValNaiveMissed{tag}", pct(s["naive_missed_share"]))
            put(f"ValRetention{tag}", f"{s['per_system_retention']:.3f}")
            put(f"ValFreqUnd{tag}", f"{s['undominated_freq_truly_undominated']:.2f}"); put(f"ValFreqDom{tag}", f"{s['undominated_freq_truly_dominated']:.2f}")
    with open(os.path.join(PAP, "tab_validation_frontier.tex"), "w") as f:
        f.write("\\begin{tabular}{rrllrrrrrrrr}\n\\toprule\n & & & & \\multicolumn{3}{c}{Frontier set} & Band & \\multicolumn{2}{c}{Plain frontier} & \\multicolumn{2}{c}{Stability} \\\\\n"
                "Systems & Tasks & Costs & Corr. & kept, each & kept, all & size (true) & covers & falsely on & missed & true & dominated \\\\\n\\midrule\n")
        for s_ in scen:
            mode = "fixed" if s_.get("cost_cv", 0.8) == 0 else "per task"
            bc = f"{s_['band']['coverage_true_costs']:.2f}" if s_.get('band') else '--'
            f.write(f"{s_['K']} & {s_['T']} & {mode} & {s_['corr']} & {s_['per_system_retention']:.3f} & {s_['joint_retention']:.2f} & {s_['mean_set_size']:.1f} ({s_['mean_true_frontier_size']:.1f}) & {bc} & "
                    f"{pct(s_['naive_false_frontier_share'])}\\% & {pct(s_['naive_missed_share'])}\\% & {s_['undominated_freq_truly_undominated']:.2f} & {s_['undominated_freq_truly_dominated']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    if scen and "sets" in scen[0]:
        SETS = [("joint", "Confidence set (joint margin)"), ("boardwide", "Largest per-system margin"), ("persystem", "Per-system margins"),
                ("exact", "Exact, per system"), ("exactjoint", "Exact, joint"), ("unadjusted", "Unadjusted pairwise tests"), ("frequency", "Frequency $\\ge\\alpha$")]
        show = [s_ for s_ in scen if (s_["K"] == 20 and s_["corr"] == 0.6 and s_.get("cost_cv", 0.8) != 0) or s_["K"] == 60]
        with open(os.path.join(PAP, "tab_validation_alternatives.tex"), "w") as f:
            f.write("\\begin{tabular}{l" + "rr" * len(show) + "}\n\\toprule\n")
            f.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$K={s_['K']}$, $T={s_['T']}$}}" for s_ in show) + " \\\\\n")
            f.write("Set & " + " & ".join("kept, all & size" for _ in show) + " \\\\\n\\midrule\n")
            for key, label in SETS:
                f.write(label + " & " + " & ".join(f"{s_['sets'][key]['joint_retention']:.2f} & {s_['sets'][key]['mean_set_size']:.1f}" for s_ in show) + " \\\\\n")
            f.write("\\midrule\nTrue frontier & " + " & ".join(f" & {s_['mean_true_frontier_size']:.1f}" for s_ in show) + " \\\\\n\\bottomrule\n\\end{tabular}\n")
        for key, _ in SETS:
            tag = {"joint": "Joint", "boardwide": "Boardwide", "persystem": "Persystem", "exact": "Exact", "exactjoint": "Exactjoint", "unadjusted": "Unadjusted", "frequency": "Frequency"}[key]
            put(f"ValAlt{tag}JointMin", f"{min(s_['sets'][key]['joint_retention'] for s_ in scen):.2f}")
            put(f"ValAlt{tag}JointMax", f"{max(s_['sets'][key]['joint_retention'] for s_ in scen):.2f}")
            put(f"ValAlt{tag}SizeKTwentyFifty", f"{next(s_['sets'][key]['mean_set_size'] for s_ in scen if s_['K'] == 20 and s_['T'] == 50):.1f}")
        put("ValJointRetentionMin", f"{min(s_['sets']['joint']['joint_retention'] for s_ in scen):.2f}")
        put("ValJointRetentionMax", f"{max(s_['sets']['joint']['joint_retention'] for s_ in scen):.2f}")
    put("ValBandNaiveCovMin", "18"); put("ValBandNaiveCovMax", "62")   # first (order-trusting) band, superseded; from data/hal/logs/s9_validate.log
    bands = [s_ for s_ in scen_all if s_.get("band")]
    if bands:
        put("ValBandCovMin", f"{min(s_['band']['coverage_true_costs'] for s_ in bands):.2f}"); put("ValBandCovMax", f"{max(s_['band']['coverage_true_costs'] for s_ in bands):.2f}")
        put("ValBandCovObsMin", f"{min(s_['band']['coverage_observed_costs'] for s_ in bands):.2f}"); put("ValBandCovObsMax", f"{max(s_['band']['coverage_observed_costs'] for s_ in bands):.2f}")
        fx = [s_ for s_ in bands if s_.get("cost_cv", 0.8) == 0]
        if fx:
            put("ValBandCovFixed", f"{fx[0]['band']['coverage_true_costs']:.2f}")
        put("ValBandWidthMin", f"{min(s_['band']['width_points_median'] for s_ in bands):.0f}"); put("ValBandWidthMax", f"{max(s_['band']['width_points_median'] for s_ in bands):.0f}")
    base = [s_ for s_ in scen if s_.get("label") == "base"]
    put("ValScenariosBase", len(base))
    put("ValBaseJointMin", f"{min(s_['joint_retention'] for s_ in base):.2f}"); put("ValBaseJointMax", f"{max(s_['joint_retention'] for s_ in base):.2f}")
    put("ValBaseRobJointMin", f"{min(s_['sets']['tierobust']['joint_retention'] for s_ in base):.2f}")
    put("ValBaseRobJointMax", f"{max(s_['sets']['tierobust']['joint_retention'] for s_ in base):.2f}")
    extra = [s_ for s_ in scen if s_.get("label") != "base"]
    put("ValBestInMin", f"{min(s_['best']['true_best_in_best_set'] for s_ in scen_all):.2f}")
    put("ValBestInMax", f"{max(s_['best']['true_best_in_best_set'] for s_ in scen_all):.2f}")
    put("ValCpbOkMin", f"{min(s_['best']['cheapest_competitor_costs_no_more'] for s_ in scen_all):.2f}")
    put("ValCpbOkMax", f"{max(s_['best']['cheapest_competitor_costs_no_more'] for s_ in scen_all):.2f}")
    # ---- the stress tests: one macro family per scenario label, and a table with Monte Carlo standard errors ----
    import re as _re
    R_ = val["R"]
    se_ = lambda p: (p * (1 - p) / R_) ** 0.5
    for s_ in scen_all:
        if s_.get("order", 0) < 10:
            continue
        slug = "".join(w.capitalize() for w in _re.split(r"[ -]", s_["label"]))
        put(f"ValSc{slug}Joint", f"{s_['sets']['joint']['joint_retention']:.2f}")
        put(f"ValSc{slug}JointSe", f"{se_(s_['sets']['joint']['joint_retention']):.3f}")
        put(f"ValSc{slug}RobJoint", f"{s_['sets']['tierobust']['joint_retention']:.2f}")
        put(f"ValSc{slug}Size", f"{s_['sets']['joint']['mean_set_size']:.1f}"); put(f"ValSc{slug}RobSize", f"{s_['sets']['tierobust']['mean_set_size']:.1f}")
        put(f"ValSc{slug}TrueSize", f"{s_['mean_true_frontier_size']:.1f}")
        put(f"ValSc{slug}Band", f"{s_['band']['coverage_true_costs']:.2f}")
        put(f"ValSc{slug}BestIn", f"{s_['best']['true_best_in_best_set']:.2f}"); put(f"ValSc{slug}CpbOk", f"{s_['best']['cheapest_competitor_costs_no_more']:.2f}")
        put(f"ValSc{slug}PerSystem", f"{s_['sets']['joint']['per_system_retention']:.2f}"); put(f"ValSc{slug}BandWidth", f"{s_['band']['width_points_median']:.0f}")
        if "cluster" in s_["sets"]:
            put(f"ValSc{slug}ClusterJoint", f"{s_['sets']['cluster']['joint_retention']:.2f}"); put(f"ValSc{slug}ClusterSize", f"{s_['sets']['cluster']['mean_set_size']:.1f}")
    notie = [s_ for s_ in scen_all if s_.get("label") != "near-tied costs"]
    put("ValCpbOkOtherMin", f"{min(s_['best']['cheapest_competitor_costs_no_more'] for s_ in notie):.2f}")
    put("ValCpbOkOtherMax", f"{max(s_['best']['cheapest_competitor_costs_no_more'] for s_ in notie):.2f}")
    exc_ = [100 * (s_["sets"]["tierobust"]["mean_set_size"] / s_["sets"]["joint"]["mean_set_size"] - 1) for s_ in notie]
    put("ValRobExcessMin", f"{min(exc_):.0f}"); put("ValRobExcessMax", f"{max(exc_):.0f}")
    ref_ = next((s_ for s_ in scen_all if s_.get("label") == "base" and s_["K"] == 20 and s_["T"] == 150), None)
    if ref_:
        put("ValBaseKTwentyTOneFiftyJoint", f"{ref_['sets']['joint']['joint_retention']:.2f}")
    rows_ = ([("Reference: 20 systems, 150 tasks", ref_)] if ref_ else []) + [(s_["label"].capitalize(), s_) for s_ in scen_all if s_.get("order", 0) >= 10]
    exc4_ = [100 * (s_["sets"]["tierobust"]["mean_set_size"] / s_["sets"]["joint"]["mean_set_size"] - 1) for lbl_, s_ in rows_ if s_.get("label") != "near-tied costs"]
    put("ValRobExcessTabMin", f"{min(exc4_):.0f}"); put("ValRobExcessTabMax", f"{max(exc4_):.0f}")
    with open(os.path.join(PAP, "tab_validation_realism.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{Standard set} & \\multicolumn{2}{c}{Tie-robust set} & True & Band & \\multicolumn{2}{c}{Best set} \\\\\n"
                "Scenario & kept, all & size & kept, all & size & frontier & covers & true best in & competitor no dearer \\\\\n\\midrule\n")
        for label_, s_ in rows_:
            st, rb = s_["sets"]["joint"], s_["sets"]["tierobust"]
            f.write(f"{label_} & {st['joint_retention']:.2f} ({se_(st['joint_retention']):.3f}) & {st['mean_set_size']:.1f} & {rb['joint_retention']:.2f} ({se_(rb['joint_retention']):.3f}) & {rb['mean_set_size']:.1f} & "
                    f"{s_['mean_true_frontier_size']:.1f} & {s_['band']['coverage_true_costs']:.2f} & {s_['best']['true_best_in_best_set']:.2f} & {s_['best']['cheapest_competitor_costs_no_more']:.2f} \\\\\n")
            if "cluster" in s_["sets"]:
                cl = s_["sets"]["cluster"]
                f.write(f"\\quad groups resampled & {cl['joint_retention']:.2f} ({se_(cl['joint_retention']):.3f}) & {cl['mean_set_size']:.1f} & \\multicolumn{{2}}{{c}}{{}} & & & & \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    if extra:
        put("ValRetentionExtraMin", f"{min(s_['per_system_retention'] for s_ in extra):.3f}")
        put("ValRetentionExtraMax", f"{max(s_['per_system_retention'] for s_ in extra):.3f}")
        fixed = [s_ for s_ in extra if s_.get("cost_cv", 0.8) == 0]
        if fixed:
            put("ValRetentionFixed", f"{fixed[0]['per_system_retention']:.3f}"); put("ValSetSizeFixed", f"{fixed[0]['mean_set_size']:.1f}")
        big = [s_ for s_ in extra if s_["K"] == 60]
        if big:
            put("ValRetentionSixty", f"{big[0]['per_system_retention']:.3f}"); put("ValSetSizeSixty", f"{big[0]['mean_set_size']:.1f}")
            put("ValTrueSizeSixty", f"{big[0]['mean_true_frontier_size']:.1f}"); put("ValNaiveFalseSixty", pct(big[0]["naive_false_frontier_share"]))

cell = load("validation_frontier_cell.json")
if cell and cell["scenarios"]:
    c0 = cell["scenarios"][0]
    put("ValCellR", cell["R"]); put("ValCellK", c0["K"]); put("ValCellT", c0["T"])
    put("ValCellJoint", f"{c0['sets']['joint']['joint_retention']:.3f}")
    put("ValCellFrequency", f"{c0['sets']['frequency']['joint_retention']:.3f}")

# ---- the HAL audit ----
aud = load("frontier_hal.json")
if aud:
    boards = [(k, v) for k, v in aud.items()]
    put("HalBoards", len(boards))
    put("HalBoardsPerTask", sum(1 for _, v in boards if v.get("cost_source", "per_task_calls") == "per_task_calls"))
    put("HalBoardsRunTotal", sum(1 for _, v in boards if v.get("cost_source") == "run_total_even"))
    put("HalSystems", sum(v["systems"] for _, v in boards))
    put("HalObservedFrontier", sum(len(v["observed_frontier"]) for _, v in boards))
    put("HalFrontierSet", sum(len(v["frontier_set"]) for _, v in boards))
    put("HalObservedFrontierPct", pct(sum(len(v["observed_frontier"]) for _, v in boards) / max(1, sum(v["systems"] for _, v in boards))))
    put("HalFrontierSetPct", pct(sum(len(v["frontier_set"]) for _, v in boards) / max(1, sum(v["systems"] for _, v in boards))))
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    shares = [len(v["frontier_set"]) / v["systems"] for _, v in boards]
    put("HalSetShareMinPct", pct(min(shares))); put("HalSetShareMaxPct", pct(max(shares)))
    put("HalBoardsSetAll", sum(1 for _, v in boards if len(v["frontier_set"]) == v["systems"]))
    put("HalBoardsBestOne", sum(1 for _, v in boards if len(v["best_set"]) == 1))
    put("HalBoardsCheaperBest", sum(1 for _, v in boards if v["cheapest_plausible_best"]["system"] != v["top"]["system"]))
    put("HalMarginSeMin", f"{min(v['margin_se'] for _, v in boards):.2f}"); put("HalMarginSeMax", f"{max(v['margin_se'] for _, v in boards):.2f}")
    pretty = {"hal_assistantbench": "AssistantBench", "hal_colbench_backend_programming": "ColBench backend", "hal_colbench_frontend_design": "ColBench frontend",
              "hal_corebench_hard": "CORE-Bench hard", "hal_gaia": "GAIA", "hal_online_mind2web": "Online-Mind2Web", "hal_scicode": "SciCode",
              "hal_scienceagentbench": "ScienceAgentBench", "hal_swebench_verified_mini": "SWE-bench Verified mini", "hal_taubench_airline": "tau-bench airline", "hal_usaco": "USACO"}

    def join_names(ks):
        ns = [pretty.get(k, k) for k in ks]
        return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]

    share = {k: len(v["frontier_set"]) / v["systems"] for k, v in boards}
    lo, hi = min(share.values()), max(share.values())
    put("HalSetShareMinName", join_names([k for k in share if abs(share[k] - lo) < 1e-9]))
    put("HalSetShareMaxNames", join_names([k for k in share if abs(share[k] - hi) < 1e-9]))
    pt_boards = [(k, v) for k, v in boards if v.get("per_task_costs")]
    put("HalBoardsSetSameFixed", sum(1 for _, v in pt_boards if set(v.get("frontier_set_fixed", v["frontier_set"])) == set(v["frontier_set"])))
    put("HalSetDiffMax", max([abs(len(v.get("frontier_set_fixed", v["frontier_set"])) - len(v["frontier_set"])) for _, v in pt_boards] or [0]))
    put("HalBoardsBestAll", sum(1 for _, v in boards if len(v["best_set"]) == v["systems"]))
    put("HalBoardsCheaperBestInformative", sum(1 for _, v in boards if v["cheapest_plausible_best"]["system"] != v["top"]["system"] and len(v["best_set"]) < v["systems"]))
    put("HalBoardsBestOneNames", join_names([k for k, v in boards if len(v["best_set"]) == 1]))
    put("HalExactSet", sum(len(v.get("frontier_set_exact", [])) for _, v in boards))
    put("HalExactSetPct", pct(sum(len(v.get("frontier_set_exact", [])) for _, v in boards) / max(1, sum(v["systems"] for _, v in boards))))
    put("HalPerSystemSet", sum(len(v.get("frontier_set_persystem", v["frontier_set"])) for _, v in boards))
    put("HalBoardwideSet", sum(len(v.get("frontier_set_boardwide", v["frontier_set"])) for _, v in boards))
    put("HalExactJointSet", sum(len(v.get("frontier_set_exact_joint", v.get("frontier_set_exact", []))) for _, v in boards))
    bw = [(k, v["band"]) for k, v in boards if v.get("band")]
    if bw:
        put("HalBandWidthMin", f"{min(b['width_points_median'] for _, b in bw):.0f}"); put("HalBandWidthMax", f"{max(b['width_points_median'] for _, b in bw):.0f}")
        put("HalBandWidthMinName", join_names([k for k, b in bw if b['width_points_median'] == min(x['width_points_median'] for _, x in bw)]))
        put("HalBandWidthMaxName", join_names([k for k, b in bw if b['width_points_median'] == max(x['width_points_median'] for _, x in bw)]))
        put("HalBandCMin", f"{min(b['c_band'] for _, b in bw):.2f}"); put("HalBandCMax", f"{max(b['c_band'] for _, b in bw):.2f}")
        for k, b in bw:
            tag = "".join(part.capitalize() for part in k.replace("hal_", "").split("_"))
            tag = "".join(digits.get(ch, ch) for ch in tag)
            put(f"{tag}BandWidth", f"{b['width_points_median']:.0f}")
    put("HalBoardsExactJointLarger", sum(1 for _, v in boards if len(v.get("frontier_set_exact_joint", v.get("frontier_set_exact", []))) > len(v["frontier_set"])))
    put("HalBoardsExactJointSmaller", sum(1 for _, v in boards if len(v.get("frontier_set_exact_joint", v.get("frontier_set_exact", []))) < len(v["frontier_set"])))
    put("HalBoardsExactLarger", sum(1 for _, v in boards if len(v.get("frontier_set_exact", [])) > len(v.get("frontier_set_persystem", v["frontier_set"]))))
    put("HalBoardsExactSmaller", sum(1 for _, v in boards if len(v.get("frontier_set_exact", [])) < len(v.get("frontier_set_persystem", v["frontier_set"]))))
    res_rows = [(k, v) for k, v in boards if v.get("resolution") and v["resolution"].get("tasks_median_to_exclude")]
    if res_rows:
        put("ResBoards", len(res_rows))
        put("ResTasksMedianMin", f"{min(v['resolution']['tasks_median_to_exclude'] for _, v in res_rows):,.0f}")
        put("ResTasksMedianMax", f"{max(v['resolution']['tasks_median_to_exclude'] for _, v in res_rows):,.0f}")
        put("ResTasksFiveMin", f"{min(v['resolution']['tasks_for_five_point_lead'] for _, v in boards if v.get('resolution') and v['resolution'].get('tasks_for_five_point_lead')):,.0f}")
        put("ResTasksFiveMax", f"{max(v['resolution']['tasks_for_five_point_lead'] for _, v in boards if v.get('resolution') and v['resolution'].get('tasks_for_five_point_lead')):,.0f}")
        for k, v in boards:
            tag = "".join(part.capitalize() for part in k.replace("hal_", "").split("_"))
            tag = "".join(digits.get(ch, ch) for ch in tag)
            r_ = v.get("resolution", {})
            if r_.get("tasks_median_to_exclude"):
                put(f"{tag}ResMedian", f"{r_['tasks_median_to_exclude']:,.0f}")
            if r_.get("tasks_for_five_point_lead"):
                put(f"{tag}ResFive", f"{r_['tasks_for_five_point_lead']:,.0f}")
    with open(os.path.join(PAP, "tab_resolution_hal.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n & & & \\multicolumn{5}{c}{Systems kept} & \\multicolumn{2}{c}{Tasks needed} \\\\\n"
                "Benchmark & Tasks & Plain frontier & joint set & largest-margin & per-system & exact, per system & exact, joint & to exclude half & for a 5-point lead \\\\\n\\midrule\n")
        for k, v in boards:
            name = k.replace("hal_", "").replace("_", " ") + (r"$^{\dagger}$" if v.get("cost_source") == "run_total_even" else "")
            r_ = v.get("resolution", {})
            med = f"{r_['tasks_median_to_exclude']:,.0f}" if r_.get("tasks_median_to_exclude") else "--"
            five = f"{r_['tasks_for_five_point_lead']:,.0f}" if r_.get("tasks_for_five_point_lead") else "--"
            f.write(f"{name} & {v['tasks']} & {len(v['observed_frontier'])} & {len(v['frontier_set'])} & {len(v.get('frontier_set_boardwide', v['frontier_set']))} & "
                    f"{len(v.get('frontier_set_persystem', v['frontier_set']))} & {len(v.get('frontier_set_exact', []))} & {len(v.get('frontier_set_exact_joint', v.get('frontier_set_exact', [])))} & {med} & {five} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    put("HalBoardsSetAllNames", join_names([k for k, v in boards if len(v["frontier_set"]) == v["systems"]]))
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    for k, v in boards:                                           # per-benchmark macros, e.g. NumTaubenchAirlineSet (digits spelled out)
        tag = "".join(part.capitalize() for part in k.replace("hal_", "").split("_"))
        tag = "".join(digits.get(ch, ch) for ch in tag)
        top, cb = v["top"], v["cheapest_plausible_best"]
        put(f"{tag}Systems", v["systems"]); put(f"{tag}Tasks", v["tasks"])
        put(f"{tag}Frontier", len(v["observed_frontier"])); put(f"{tag}Set", len(v["frontier_set"])); put(f"{tag}Best", len(v["best_set"]))
        put(f"{tag}TopAcc", f"{100 * top['accuracy']:.1f}"); put(f"{tag}TopCost", f"{top['cost']:.2f}")
        put(f"{tag}CbAcc", f"{100 * cb['accuracy']:.1f}"); put(f"{tag}CbCost", f"{cb['cost']:.2f}")
        put(f"{tag}Margin", f"{v['margin_se']:.2f}")
        put(f"{tag}SetPct", pct(len(v["frontier_set"]) / v["systems"]))
    put("HalTieRobustSet", sum(len(v.get("frontier_set_tie_robust", v["frontier_set"])) for _, v in boards))
    put("HalTieRobustBoardsLarger", sum(1 for _, v in pt_boards if len(v.get("frontier_set_tie_robust", v["frontier_set"])) > len(v["frontier_set"])))
    nv = [(k, v) for k, v in boards if len(v["frontier_set"]) < v["systems"]]
    put("HalFrontierSetPctNoVacuous", pct(sum(len(v["frontier_set"]) for _, v in nv) / max(1, sum(v["systems"] for _, v in nv))))
    with open(os.path.join(PAP, "tab_frontier_hal.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\nBenchmark & Systems & Tasks & Observed frontier & Frontier set (share) & Tie-robust set & Best set & Top system (\\%, \\$) & Cheapest unresolved (\\%, \\$) \\\\\n\\midrule\n")
        for k, v in boards:
            name = k.replace("hal_", "").replace("_", " ") + (r"$^{\dagger}$" if v.get("cost_source") == "run_total_even" else "")
            top, cb = v["top"], v["cheapest_plausible_best"]
            f.write(f"{name} & {v['systems']} & {v['tasks']} & {len(v['observed_frontier'])} & {len(v['frontier_set'])} ({pct(len(v['frontier_set']) / v['systems'])}\\%) & "
                    f"{len(v.get('frontier_set_tie_robust', v['frontier_set']))} & {len(v['best_set'])} & "
                    f"{100 * top['accuracy']:.1f}, {top['cost']:.2f} & {100 * cb['accuracy']:.1f}, {cb['cost']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

# ---- certification of frontier entrants on HAL's timeline ----
cert = load("certify_hal.json")
if cert:
    cb = [(k, v) for k, v in cert.items() if "error" not in v]
    put("CertBoards", len(cb))

    def tot(key):
        return sum(v[key] for _, v in cb)

    put("CertSystems", tot("systems")); put("CertCheapest", tot("n_cheapest_so_far")); put("CertEntrants", tot("n_entrants"))
    put("CertDominated", tot("n_dominated_on_arrival"))
    put("CertUnadjusted", tot("n_certified_unadjusted")); put("CertSpending", tot("n_certified_spending")); put("CertFallback", tot("n_certified_fallback"))
    put("CertIUUnadjusted", tot("n_certified_iu_unadjusted")); put("CertIUSpending", tot("n_certified_iu_spending")); put("CertIUFallback", tot("n_certified_iu_fallback"))
    put("CertExUnadjusted", tot("n_certified_ex_unadjusted")); put("CertExSpending", tot("n_certified_ex_spending")); put("CertExFallback", tot("n_certified_ex_fallback"))
    both = [st for _, v in cb for st in v["steps"] if st.get("certified_iu_spending") and st["certified_spending"]]
    put("CertBothSpending", len(both))
    put("CertIUNotPair", sum(1 for _, v in cb for st in v["steps"] if st.get("certified_iu_spending") and not st["certified_spending"]))
    put("CertPairNotIU", sum(1 for _, v in cb for st in v["steps"] if st["certified_spending"] and not st.get("certified_iu_spending")))
    put("CertEntrantsPct", pct(tot("n_entrants") / max(1, tot("systems"))))
    put("CertSpendingPct", pct(tot("n_certified_spending") / max(1, tot("n_entrants"))))
    put("CertUnadjustedPct", pct(tot("n_certified_unadjusted") / max(1, tot("n_entrants"))))
    steps = [st for _, v in cb for st in v["steps"]]
    certd = [st for st in steps if st["certified_spending"] or st.get("certified_iu_spending")]
    if certd:
        put("CertMinMarginPt", f"{100 * min(st['margin'] for st in certd):.0f}"); put("CertMaxMarginPt", f"{100 * max(st['margin'] for st in certd):.0f}")
        put("CertMinTasks", min(v["tasks"] for _, v in cb if v["n_certified_spending"] > 0 or v.get("n_certified_iu_spending", 0) > 0))
        put("CertEitherSpending", len(certd))
    put("CertMaxAnyRule", max(tot("n_certified_spending"), tot("n_certified_fallback"), tot("n_certified_iu_spending"), tot("n_certified_iu_fallback"), tot("n_certified_ex_spending"), tot("n_certified_ex_fallback")))
    put("CertMultiDate", sum(1 for st in steps if st["date_group"] > 1))
    put("CertMultiDatePct", pct(sum(1 for st in steps if st["date_group"] > 1) / max(1, len(steps))))
    put("CertSmallStep", sum(1 for st in steps if st["margin"] < 0.05))
    tasks_of = {k: v["tasks"] for k, v in cb}
    small_steps = [st for k, v in cb for st in v["steps"] if st["margin"] < 0.05 and v["tasks"] <= 165]
    put("CertSmallStepSmallBoard", len(small_steps))
    put("CertBudgetExhausted", sum(1 for st in steps if st["p"] <= 0.05 and not st["certified_spending"]))
    put("CertDatesWithEntrants", sum(v.get("n_dates_with_entrants", 0) for _, v in cb))
    jit = [v["jitter"] for _, v in cb if v.get("jitter")]
    if jit:
        put("CertJitterR", jit[0]["R"]); put("CertJitterBoards", len(jit))
        put("CertJitterStableMin", f"{min(j['holder_stable_share'] for j in jit):.2f}"); put("CertJitterStableMax", f"{max(j['holder_stable_share'] for j in jit):.2f}")
        base_sp = sum(v["n_certified_spending"] for _, v in cb if v.get("jitter"))
        put("CertJitterSpendingBase", base_sp)
        put("CertJitterSpendingMin", sum(j["certified_spending_min"] for j in jit)); put("CertJitterSpendingMax", sum(j["certified_spending_max"] for j in jit))
        put("CertJitterEntrantsMin", sum(j["entrants_min"] for j in jit)); put("CertJitterEntrantsMax", sum(j["entrants_max"] for j in jit))
        put("CertJitterEntrantsBase", sum(v["n_entrants"] for _, v in cb if v.get("jitter")))
        put("CertJitterIUMin", sum(j.get("certified_iu_min", 0) for j in jit)); put("CertJitterIUMax", sum(j.get("certified_iu_max", 0) for j in jit))
        put("CertJitterIUBase", sum(v["n_certified_iu_spending"] for _, v in cb if v.get("jitter")))
    with open(os.path.join(PAP, "tab_certify_hal.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrrrrrrr}\n\\toprule\n & & & & & \\multicolumn{3}{c}{Record claim (pairwise)} & \\multicolumn{3}{c}{Pareto certificate (normal)} & \\multicolumn{3}{c}{Pareto certificate (exact)} \\\\\n"
                "Benchmark & Systems & Cheapest so far & Entrants & Dominated & unadj. & spending & fallback & unadj. & spending & fallback & unadj. & spending & fallback \\\\\n\\midrule\n")
        for k, v in cb:
            name = k.replace("hal_", "").replace("_", " ") + (r"$^{\dagger}$" if v.get("cost_source") == "run_total_even" else "")
            f.write(f"{name} & {v['systems']} & {v['n_cheapest_so_far']} & {v['n_entrants']} & {v['n_dominated_on_arrival']} & "
                    f"{v['n_certified_unadjusted']} & {v['n_certified_spending']} & {v['n_certified_fallback']} & "
                    f"{v.get('n_certified_iu_unadjusted', 0)} & {v.get('n_certified_iu_spending', 0)} & {v.get('n_certified_iu_fallback', 0)} & "
                    f"{v.get('n_certified_ex_unadjusted', 0)} & {v.get('n_certified_ex_spending', 0)} & {v.get('n_certified_ex_fallback', 0)} \\\\\n")
        f.write(f"\\midrule\nAll & {tot('systems')} & {tot('n_cheapest_so_far')} & {tot('n_entrants')} & {tot('n_dominated_on_arrival')} & "
                f"{tot('n_certified_unadjusted')} & {tot('n_certified_spending')} & {tot('n_certified_fallback')} & "
                f"{tot('n_certified_iu_unadjusted')} & {tot('n_certified_iu_spending')} & {tot('n_certified_iu_fallback')} & "
                f"{tot('n_certified_ex_unadjusted')} & {tot('n_certified_ex_spending')} & {tot('n_certified_ex_fallback')} \\\\\n" + "\\bottomrule\n\\end{tabular}\n")

# ---- what the entrant certificate controls (near-copy rivals) ----
en = load("entrant_null.json")
if en:
    put("EntrantNullR", en["R"])
    put("EntrantNullRExact", en["cases"][0].get("R_exact", 0))
    for test, tag in (("rate_pairwise", ""), ("rate_composite", "IU"), ("rate_exact", "Ex")):
        ind = [c[test] for c in en["cases"] if c["world"] == "independent"]
        cp = [c[test] for c in en["cases"] if c["world"] == "near-copy"]
        bt = [c[test] for c in en["cases"] if c["world"] == "better"]
        put(f"EntrantNull{tag}IndepMin", f"{100 * min(ind):.1f}"); put(f"EntrantNull{tag}IndepMax", f"{100 * max(ind):.1f}")
        put(f"EntrantNull{tag}CopyMin", f"{100 * min(cp):.1f}"); put(f"EntrantNull{tag}CopyMax", f"{100 * max(cp):.1f}")
        if bt:
            put(f"EntrantPower{tag}Min", f"{100 * min(bt):.0f}"); put(f"EntrantPower{tag}Max", f"{100 * max(bt):.0f}")
    with open(os.path.join(PAP, "tab_entrant_null.tex"), "w") as f:
        f.write("\\begin{tabular}{llrrrrrr}\n\\toprule\n & & & & & \\multicolumn{3}{c}{Certified at 5\\%} \\\\\nWorld & Rival shape & Tasks & Boards on frontier & & pairwise & composite & exact composite \\\\\n\\midrule\n")
        for c in en["cases"]:
            shape = {"independent": "independent rivals", "near-copy": f"near-copy, flip {c['delta']}", "better": "entrant truly better"}[c["world"]]
            ex = f"{100 * c['rate_exact']:.1f}\\%" if 'rate_exact' in c else '--'
            f.write(f"{c['world']} & {shape} & {c['T']} & {c['selected']} & & {100 * c['rate_pairwise']:.1f}\\% & {100 * c['rate_composite']:.1f}\\% & {ex} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

# ---- HAL's own frontier under its reported costs vs once-counted, repriced costs ----
fl = load("frontier_flips.json")
if fl:
    fb = [(k, v) for k, v in fl.items()]
    put("FlipBoards", len(fb))
    put("FlipBoardsChanged", sum(1 for _, v in fb if v["changed_total"] > 0))
    put("FlipBoardsChangedCounting", sum(1 for _, v in fb if v["lost_counting"] + v["gained_counting"] > 0))
    put("FlipSystemsChanged", sum(v["changed_total"] for _, v in fb))
    put("FlipLost", sum(v["lost_total"] for _, v in fb)); put("FlipGained", sum(v["gained_total"] for _, v in fb))
    put("FlipCounting", sum(v["lost_counting"] + v["gained_counting"] for _, v in fb))
    put("FlipPricing", sum(v["lost_pricing"] + v["gained_pricing"] for _, v in fb))
    put("FlipFrontierReported", sum(v["frontier_reported"] for _, v in fb)); put("FlipFrontierOurs", sum(v["frontier_ours"] for _, v in fb))
    put("FlipChangedPct", pct(sum(v["changed_total"] for _, v in fb) / max(1, sum(v["frontier_reported"] for _, v in fb))))
    zero = [r for _, v in fb for r in v["table"] if r["cost_reported"] < 1e-3 and r["cost_ours"] > 1e-2]
    put("FlipZeroReported", len(zero)); put("FlipZeroReportedOnFrontier", sum(1 for r in zero if r["on_reported"]))
    nz_on = sum(1 for r in zero if r["on_reported"])
    put("FlipZeroPhrase", "all of them" if zero and nz_on == len(zero) else f"{nz_on} of them")
    with open(os.path.join(PAP, "tab_flips_hal.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Plain frontier size under} & \\multicolumn{2}{c}{Reported to ours} & \\multicolumn{2}{c}{Changes due to} \\\\\n"
                "Benchmark & reported & once-counted & ours & lost & gained & counting & pricing \\\\\n\\midrule\n")
        for k, v in fb:
            name = k.replace("hal_", "").replace("_", " ") + (r"$^{\dagger}$" if v.get("cost_source") == "run_total_even" else "")
            f.write(f"{name} & {v['frontier_reported']} & {v['frontier_once']} & {v['frontier_ours']} & {v['lost_total']} & {v['gained_total']} & "
                    f"{v['lost_counting'] + v['gained_counting']} & {v['lost_pricing'] + v['gained_pricing']} \\\\\n")
        f.write(f"\\midrule\nAll & {sum(v['frontier_reported'] for _, v in fb)} & {sum(v['frontier_once'] for _, v in fb)} & {sum(v['frontier_ours'] for _, v in fb)} & "
                f"{sum(v['lost_total'] for _, v in fb)} & {sum(v['gained_total'] for _, v in fb)} & {sum(v['lost_counting'] + v['gained_counting'] for _, v in fb)} & "
                f"{sum(v['lost_pricing'] + v['gained_pricing'] for _, v in fb)} \\\\\n" + "\\bottomrule\n\\end{tabular}\n")

# ---- the frontier winner's curse: known truth and HAL ----
import math

cv = load("validation_curse.json")
if cv:
    put("CurseValR", cv["R"])
    scen = cv["scenarios"]
    meth = ("naive", "bootstrap", "iterated", "conditional")
    put("CurseValNaiveBiasMin", f"{min(sc['methods']['naive']['bias_pts'] for sc in scen):+.1f}")
    put("CurseValNaiveBiasMax", f"{max(sc['methods']['naive']['bias_pts'] for sc in scen):+.1f}")
    put("CurseValIterBiasAbsMax", f"{max(abs(sc['methods']['iterated']['bias_pts']) for sc in scen):.1f}")
    put("CurseValCondBiasAbsMax", f"{max(abs(sc['methods']['conditional']['bias_pts']) for sc in scen):.1f}")
    put("CurseValNaiveCovMin", f"{min(sc['methods']['naive']['coverage'] for sc in scen):.2f}")
    put("CurseValNaiveCovMax", f"{max(sc['methods']['naive']['coverage'] for sc in scen):.2f}")
    put("CurseValBootCovMin", f"{min(sc['methods']['bootstrap']['coverage'] for sc in scen):.2f}")
    put("CurseValBootCovMax", f"{max(sc['methods']['bootstrap']['coverage'] for sc in scen):.2f}")
    put("CurseValIterCovMin", f"{min(sc['methods']['iterated']['coverage'] for sc in scen):.2f}")
    put("CurseValIterCovMax", f"{max(sc['methods']['iterated']['coverage'] for sc in scen):.2f}")
    put("CurseValCondCovMin", f"{min(sc['methods']['conditional']['coverage'] for sc in scen):.2f}")
    put("CurseValCondCovMax", f"{max(sc['methods']['conditional']['coverage'] for sc in scen):.2f}")
    big = [sc for sc in scen if sc["K"] == 20]
    put("CurseValNaiveBiasKTwentyMin", f"{min(sc['methods']['naive']['bias_pts'] for sc in big):+.1f}")
    put("CurseValNaiveBiasKTwentyMax", f"{max(sc['methods']['naive']['bias_pts'] for sc in big):+.1f}")
    put("CurseValTrulyUndMin", pct(min(sc["share_truly_undominated"] for sc in scen)))
    put("CurseValTrulyUndMax", pct(max(sc["share_truly_undominated"] for sc in scen)))
    with open(os.path.join(PAP, "tab_validation_curse.tex"), "w") as f:
        f.write("\\begin{tabular}{rrrr" + "r" * 4 + "r" * 4 + "}\n\\toprule\n & & & & \\multicolumn{4}{c}{Bias, points (standard error)} & \\multicolumn{4}{c}{Coverage of the 95\\% interval} \\\\\n"
                "Systems & Tasks & Winners & Pool & reported & bootstrap & iterated & conditional & reported & bootstrap & iterated & conditional \\\\\n\\midrule\n")
        for sc in scen:
            m = sc["methods"]
            f.write(f"{sc['K']} & {sc['T']} & {sc['n_winners']} & {sc['mean_pool']:.1f}")
            for k in meth:
                f.write(f" & {m[k]['bias_pts']:+.1f} ({m[k]['se_bias_pts']:.1f})")
            for k in meth:
                f.write(f" & {m[k]['coverage']:.2f}")
            f.write(" \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

def nz(x):
    return 0.0 if abs(x) < 0.05 else x   # no '-0.0' in tables


ch = load("curse_hal.json")
if ch:
    boards_c = [(k, v) for k, v in ch.items()]
    members = [m for _, v in boards_c for m in v["members"]]
    with_rival = [m for m in members if m["pool_size"] > 1]
    put("CurseMembers", len(members)); put("CurseWithRival", len(with_rival))
    shr = [100 * m["bootstrap"]["bias"] for m in with_rival]
    cshr = [100 * (m["observed"] - m["conditional"]["estimate"]) for m in with_rival if math.isfinite(m["conditional"]["estimate"])]
    put("CurseMeanShrinkPt", f"{sum(shr) / len(shr):.1f}"); put("CurseMaxShrinkPt", f"{max(shr):.1f}")
    put("CurseMeanShrinkCondPt", f"{sum(cshr) / len(cshr):.1f}" if cshr else "n/a")
    put("CurseDegenerate", sum(1 for m in with_rival if not math.isfinite(m["conditional"]["estimate"])))
    put("CurseBelowRival", sum(1 for m in members if m["below_rival_iterated"]))
    put("CurseBelowRivalCond", sum(1 for m in members if m["below_rival_conditional"]))
    put("CurseBoardsBelow", sum(1 for _, v in boards_c if any(m["below_rival_iterated"] for m in v["members"])))
    put("CurseShrinkOverTwoPt", sum(1 for x in shr if x >= 2.0))
    with open(os.path.join(PAP, "tab_curse_hal.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n & & & \\multicolumn{2}{c}{Shrinkage, iterated bootstrap (pt)} & Shrinkage, conditional (pt) & Below strongest \\\\\n"
                "Benchmark & On frontier & With a cheaper rival & mean & max & mean & cheaper rival after adjustment \\\\\n\\midrule\n")
        for k, v in boards_c:
            name = k.replace("hal_", "").replace("_", " ") + (r"$^{\dagger}$" if v.get("cost_source") == "run_total_even" else "")
            wr = [m for m in v["members"] if m["pool_size"] > 1]
            s1 = [100 * m["bootstrap"]["bias"] for m in wr]
            s2 = [100 * (m["observed"] - m["conditional"]["estimate"]) for m in wr if math.isfinite(m["conditional"]["estimate"])]
            f.write(f"{name} & {len(v['members'])} & {len(wr)} & {nz(sum(s1) / len(s1)) if s1 else 0:.1f} & {nz(max(s1)) if s1 else 0:.1f} & "
                    f"{nz(sum(s2) / len(s2)) if s2 else 0:.1f} & {sum(1 for m in v['members'] if m['below_rival_iterated'])} \\\\\n")
        f.write(f"\\midrule\nAll & {len(members)} & {len(with_rival)} & {sum(shr) / len(shr):.1f} & {max(shr):.1f} & "
                f"{sum(cshr) / len(cshr) if cshr else 0:.1f} & {sum(1 for m in members if m['below_rival_iterated'])} \\\\\n" + "\\bottomrule\n\\end{tabular}\n")

# ---- held-out halves of the real boards ----
ho = load("validation_heldout.json")
if ho:
    hb = [(k, v) for k, v in ho["boards"].items()]
    put("HeldoutR", ho["R"]); put("HeldoutBoards", len(hb))
    put("HeldoutKeptMin", f"{min(v['obsB_kept_in_setA'] for _, v in hb):.2f}"); put("HeldoutKeptMax", f"{max(v['obsB_kept_in_setA'] for _, v in hb):.2f}")
    put("HeldoutPlainMin", f"{min(v['obsB_on_obsA'] for _, v in hb):.2f}"); put("HeldoutPlainMax", f"{max(v['obsB_on_obsA'] for _, v in hb):.2f}")
    put("HeldoutNaiveBiasMin", f"{min(v['methods']['naive']['bias_pts'] for _, v in hb):+.1f}")
    put("HeldoutNaiveBiasMax", f"{max(v['methods']['naive']['bias_pts'] for _, v in hb):+.1f}")
    put("HeldoutIterBiasAbsMax", f"{max(abs(v['methods']['iterated']['bias_pts']) for _, v in hb):.1f}")
    put("HeldoutCondBiasAbsMax", f"{max(abs(v['methods']['conditional']['bias_pts']) for _, v in hb):.1f}")
    put("HeldoutIterCovMin", f"{min(v['methods']['iterated']['coverage'] for _, v in hb):.2f}")
    put("HeldoutCondCovMin", f"{min(v['methods']['conditional']['coverage'] for _, v in hb):.2f}")
    put("HeldoutWinners", sum(v["n_winners"] for _, v in hb))
    names_ho = {"hal_assistantbench": "AssistantBench", "hal_colbench_backend_programming": "ColBench backend", "hal_colbench_frontend_design": "ColBench frontend",
                "hal_corebench_hard": "CORE-Bench hard", "hal_gaia": "GAIA", "hal_online_mind2web": "Online-Mind2Web", "hal_scicode": "SciCode",
                "hal_scienceagentbench": "ScienceAgentBench", "hal_swebench_verified_mini": "SWE-bench Verified mini", "hal_taubench_airline": "tau-bench airline", "hal_usaco": "USACO"}
    put("HeldoutBeatRandomMin", f"{min(v['obsB_kept_in_setA'] - v['obsB_kept_in_size_matched_random'] for _, v in hb):.2f}")
    put("HeldoutBeatRandomMax", f"{max(v['obsB_kept_in_setA'] - v['obsB_kept_in_size_matched_random'] for _, v in hb):.2f}")
    put("HeldoutRandomMin", f"{min(v['obsB_kept_in_size_matched_random'] for _, v in hb):.2f}"); put("HeldoutRandomMax", f"{max(v['obsB_kept_in_size_matched_random'] for _, v in hb):.2f}")
    put("HeldoutSetShareMin", pct(min(v["mean_set_size"] / v["systems"] for _, v in hb))); put("HeldoutSetShareMax", pct(max(v["mean_set_size"] / v["systems"] for _, v in hb)))
    put("HeldoutPlaceboMin", f"{min(v['placebo_naive_coverage'] for _, v in hb):.2f}"); put("HeldoutPlaceboMax", f"{max(v['placebo_naive_coverage'] for _, v in hb):.2f}")
    put("HeldoutNaiveCovMin", f"{min(v['methods']['naive']['coverage'] for _, v in hb):.2f}"); put("HeldoutNaiveCovMax", f"{max(v['methods']['naive']['coverage'] for _, v in hb):.2f}")
    put("HeldoutIterCovMax", f"{max(v['methods']['iterated']['coverage'] for _, v in hb):.2f}"); put("HeldoutCondCovMax", f"{max(v['methods']['conditional']['coverage'] for _, v in hb):.2f}")
    put("HeldoutKeptMean", f"{sum(v['obsB_kept_in_setA'] for _, v in hb) / len(hb):.2f}"); put("HeldoutPlainMean", f"{sum(v['obsB_on_obsA'] for _, v in hb) / len(hb):.2f}")
    put("HeldoutRandomMean", f"{sum(v['obsB_kept_in_size_matched_random'] for _, v in hb) / len(hb):.2f}")
    with open(os.path.join(PAP, "tab_validation_heldout.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & & & & \\multicolumn{3}{c}{B's frontier members kept by} & \\\\\n"
                "Benchmark & Tasks & Systems & Mean set size & A's frontier set & A's plain frontier & size-matched random & \\\\\n\\midrule\n")
        for k, v in hb:
            name = names_ho.get(k, k) + (r"$^{\dagger}$" if v.get("cost_fixed") else "")
            f.write(f"{name} & {v['tasks']} & {v['systems']} & {v['mean_set_size']:.1f} & {v['obsB_kept_in_setA']:.2f} & {v['obsB_on_obsA']:.2f} & {v['obsB_kept_in_size_matched_random']:.2f} & \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(PAP, "tab_heldout_curse.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n & & \\multicolumn{3}{c}{Bias, points} & \\multicolumn{3}{c}{Coverage of the 95\\% interval} & Placebo \\\\\n"
                "Benchmark & Winners & reported & iterated & conditional & reported & iterated & conditional & coverage \\\\\n\\midrule\n")
        for k, v in hb:
            m = v["methods"]
            name = names_ho.get(k, k) + (r"$^{\dagger}$" if v.get("cost_fixed") else "")
            f.write(f"{name} & {v['n_winners']} & {m['naive']['bias_pts']:+.1f} & {m['iterated']['bias_pts']:+.1f} & {m['conditional']['bias_pts']:+.1f} & "
                    f"{m['naive']['coverage']:.2f} & {m['iterated']['coverage']:.2f} & {m['conditional']['coverage']:.2f} & {v['placebo_naive_coverage']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

# ---- board-level data facts ----
import glob as _glob
bfiles = [f for f in _glob.glob(os.path.join(ROOT, "data", "boards", "hal_*.json")) if not f.endswith("_costs.json") and not f.endswith("hal_summary.json")]
if bfiles:
    systems_all = [sy for f in bfiles for sy in json.load(open(f, encoding="utf-8"))["systems"]]
    n_tasks = {os.path.basename(f): len(json.load(open(f, encoding="utf-8"))["tasks"]) for f in bfiles}
    put("HalSystemsMissingOutcome", sum(1 for sy in systems_all if sy.get("tasks_without_outcome", 0) > 0))
    put("HalSystemsUncosted", sum(1 for sy in systems_all if sy.get("tasks_without_calls", 0) > 0))
    tot_tasks = sum(len(json.load(open(f, encoding="utf-8"))["tasks"]) * len(json.load(open(f, encoding="utf-8"))["systems"]) for f in bfiles)
    put("HalUncostedTasksPct", f"{100 * sum(sy.get('tasks_without_calls', 0) for sy in systems_all) / max(1, tot_tasks):.1f}")
    put("HalSystemsUncostedOverTenPct", sum(1 for f in bfiles for sy in json.load(open(f, encoding="utf-8"))["systems"]
                                            if sy.get("tasks_without_calls", 0) > 0.1 * len(json.load(open(f, encoding="utf-8"))["tasks"])))
    put("HalReruns", sum(1 for sy in systems_all if sy["name"].endswith("_")))

# ---- run-to-run noise from rerun pairs, and the cluster bootstrap ----
rn = load("rerun_noise.json")
if rn and rn["n_pairs"]:
    put("RerunPairs", rn["n_pairs"]); put("RerunBoards", rn["n_boards_with_pairs"])
    put("RerunMeanAbsDiffPt", f"{100 * rn['mean_abs_acc_diff']:.1f}")
    put("RerunRmsObsPt", f"{100 * rn['rms_diff_observed']:.1f}"); put("RerunRmsIndepPt", f"{100 * rn['rms_diff_independent_flips']:.1f}")
    put("RerunExcessSdPt", f"{100 * rn['sd_excess_per_run']:.1f}")
    put("RerunChiSq", f"{rn['chi2']:.1f}"); put("RerunDf", rn["chi2_df"]); put("RerunP", f"{rn['chi2_p']:.3f}")
    put("RerunTaskSePt", f"{100 * rn['median_task_se_single_run']:.1f}")
    put("RerunMaxAbsDiffPt", f"{100 * max(abs(q['acc_diff']) for q in rn['pairs']):.1f}")
    put("RerunCostRatioMedian", f"{rn['median_cost_ratio']:.2f}"); put("RerunCostRatioMax", f"{rn['max_cost_ratio']:.2f}")
    agree = [q["task_agreement"] for q in rn["pairs"] if q["task_agreement"] is not None]
    if agree:
        put("RerunAgreeMin", pct(min(agree))); put("RerunAgreeMax", pct(max(agree)))
    bsum = rn["boards"]
    put("RerunSetBefore", sum(v["set"] for v in bsum.values())); put("RerunSetAfter", sum(v["set_with_run_level_noise"] for v in bsum.values()))
cc = load("cluster_check.json")
if cc:
    for k, v in cc.items():
        tag = {"hal_swebench_verified_mini": "Swebench", "hal_usaco": "Usaco"}.get(k, k)
        put(f"Cluster{tag}Groups", v["n_clusters"]); put(f"Cluster{tag}Largest", v["largest_cluster"])
        put(f"Cluster{tag}SetTasks", v["set_tasks"]); put(f"Cluster{tag}SetClusters", v["set_clusters"])
        put(f"Cluster{tag}MarginTasks", f"{v['margin_tasks']:.2f}"); put(f"Cluster{tag}MarginClusters", f"{v['margin_clusters']:.2f}")
        put(f"Cluster{tag}BestTasks", v["best_set_tasks"]); put(f"Cluster{tag}BestClusters", v["best_set_clusters"])

# ---- the HAL data pipeline: what was read, kept and dropped ----
summ_path = os.path.join(ROOT, "data", "boards", "hal_summary.json")
idx_path = os.path.join(ROOT, "data", "hal", "index.json")
if os.path.exists(summ_path):
    summ = json.load(open(summ_path, encoding="utf-8"))
    put("HalBenchmarks", len(summ))
    put("HalRunsRead", sum(v["runs"] for v in summ.values()))
    put("HalRunsKept", sum(v.get("kept", 0) for v in summ.values()))
    drops = [d["reason"] for v in summ.values() for d in v.get("dropped", [])]
    put("HalDropped", len(drops))
    put("HalDroppedNoCalls", sum(1 for r in drops if "no logged" in r))
    put("HalDroppedCoverage", sum(1 for r in drops if r.startswith("outcome coverage")))
    put("HalDroppedOther", sum(1 for r in drops if "no logged" not in r and not r.startswith("outcome coverage")))
    put("HalBoardsRunTotalKept", sum(v.get("kept", 0) for v in summ.values() if v.get("cost_source") == "run_total_even"))
if os.path.exists(idx_path):
    idx = json.load(open(idx_path, encoding="utf-8"))
    put("HalUploadsIndexed", len(idx))
    put("HalUploadsSkipped", sum(1 for v in idx.values() if v.get("skip")))
    put("HalUploadsFailed", sum(1 for v in idx.values() if v.get("error") and not v.get("skip")))
    put("HalUploadsRead", sum(1 for v in idx.values() if not v.get("error") and not v.get("skip")))
    v2 = [e for e in idx.values() if e.get("v") == 2 and e.get("tokens_counted")]
    ratios = [e["tokens_raw"] / e["tokens_counted"] for e in v2]
    put("DupRunsTotal", len(v2))
    put("DupRunsExactlyTwo", sum(1 for r in ratios if abs(r - 2) < 1e-3))
    put("DupRunsBetween", sum(1 for r in ratios if 1.001 < r < 2 - 1e-3))
    put("DupRunsAboveTwo", sum(1 for r in ratios if r > 2 + 1e-3))
    put("DupRunsOne", sum(1 for r in ratios if r <= 1.001))
    put("DupRunsAffected", sum(1 for r in ratios if r > 1.001))
    put("DupOverstatementPct", pct(sum(e["tokens_raw"] for e in v2) / max(1.0, sum(e["tokens_counted"] for e in v2)) - 1))
    pairs = [(e["cost_reported"], e["cost_recomputed"]) for e in v2 if e.get("cost_reported") and e.get("cost_recomputed") and e["cost_recomputed"] >= 1.0]
    if pairs:
        rat = sorted(a / b for a, b in pairs)
        put("CostRatioMax", f"{rat[-1]:.0f}"); put("CostRatioMedian", f"{rat[len(rat) // 2]:.2f}")
        put("CostRatioRuns", len(pairs)); put("CostRatioAboveOneAndHalf", sum(1 for r in rat if r > 1.5))
        put("CostReportedTotal", f"{sum(a for a, _ in pairs):,.0f}"); put("CostRecountedTotal", f"{sum(b for _, b in pairs):,.0f}")

# ---- certification under other batching rules ----
variants = {"Week": load("certify_hal_week.json"), "Worst": load("certify_hal_worst_first.json")}
base_cert = load("certify_hal.json")
if base_cert and all(variants.values()):
    def _tot(d, key):
        return sum(v.get(key, 0) for v in d.values() if "error" not in v)
    for tag, d in variants.items():
        put(f"Cert{tag}Entrants", _tot(d, "n_entrants")); put(f"Cert{tag}Spending", _tot(d, "n_certified_spending")); put(f"Cert{tag}IUSpending", _tot(d, "n_certified_iu_spending"))
    sp = [_tot(d, "n_certified_spending") for d in (base_cert, *variants.values())]
    iu = [_tot(d, "n_certified_iu_spending") for d in (base_cert, *variants.values())]
    put("CertVariantSpendingMin", min(sp)); put("CertVariantSpendingMax", max(sp)); put("CertVariantIUMin", min(iu)); put("CertVariantIUMax", max(iu))

# ---- sensitivity of the audit to the costing rules ----
sens = load("hal_sensitivity.json")
if sens:
    put("SensBoards", len(sens))
    fc = [v["variants"]["full_cache"] for v in sens.values()]; ni = [v["variants"]["no_imputation"] for v in sens.values()]
    put("SensFullCacheFrontierChanged", sum(x["frontier_changed"] for x in fc)); put("SensFullCacheFrontierBoards", sum(1 for x in fc if x["frontier_changed"] > 0))
    put("SensFullCacheSetChanged", sum(x["set_changed"] for x in fc))
    put("SensNoImpFrontierChanged", sum(x["frontier_changed"] for x in ni)); put("SensNoImpSetChanged", sum(x["set_changed"] for x in ni))
    put("SensNoImpFrontierBoards", sum(1 for x in ni if x["frontier_changed"] > 0))
    worst = max(sens.items(), key=lambda kv: kv[1]["variants"]["full_cache"]["mean_cost_ratio"])
    put("SensCostRatioMax", f"{worst[1]['variants']['full_cache']['mean_cost_ratio']:.1f}")
    put("SensCostRatioMaxName", {"hal_swebench_verified_mini": "SWE-bench Verified mini", "hal_corebench_hard": "CORE-Bench hard", "hal_gaia": "GAIA",
                                 "hal_colbench_backend_programming": "ColBench backend", "hal_taubench_airline": "tau-bench airline"}.get(worst[0], worst[0]))

# ---- how many system pairs can be told apart on cost ----
cs = load("cost_separation.json")
if cs:
    put("CostSepHalMin", f"{min(v['separated_share'] for v in cs['hal'].values()):.2f}"); put("CostSepHalMax", f"{max(v['separated_share'] for v in cs['hal'].values()):.2f}")
    put("CostSepSimBase", f"{cs['sim']['base']:.2f}"); put("CostSepSimTied", f"{cs['sim']['near-tied costs']:.2f}")

# ---- the package version the paper was produced with ----
try:
    import re as _re2
    _pp = os.path.join(os.path.dirname(ROOT), "Phase B - Statistics Paper 1", "honestboard", "pyproject.toml")
    put("HonestboardVersion", _re2.search(r'^version = "([^"]+)"', open(_pp, encoding="utf-8").read(), _re2.M).group(1))
except Exception:
    put("HonestboardVersion", "0.5.1")

with open(os.path.join(PAP, "numbers.tex"), "w") as f:
    for k, v in NUM.items():
        f.write(f"\\newcommand{{\\Num{k}}}{{{v}}}\n")
print(f"wrote {len(NUM)} macros to paper/numbers.tex")

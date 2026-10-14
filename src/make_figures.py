"""Figures for Paper 2.

fig1_frontiers.pdf : one panel per HAL benchmark, cost per task (log scale) against accuracy, with the plain frontier
                     drawn as a step line, systems in the frontier confidence set filled and the rest hollow, and the
                     cheapest plausible best marked. A dagger marks a board whose cost is a run total treated as fixed.
fig2_setsizes.pdf  : observed frontier size against frontier-set size per benchmark, with the number of tasks.
Reads results/frontier_hal.json. Writes figures/*.pdf and *.png.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, GRAY = "#1f5fbf", "#e07b00", "#8c8c8c"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})

res = json.loads((ROOT / "results" / "frontier_hal.json").read_text(encoding="utf-8"))
names = {"hal_assistantbench": "AssistantBench", "hal_colbench_backend_programming": "ColBench backend", "hal_colbench_frontend_design": "ColBench frontend",
         "hal_corebench_hard": "CORE-Bench hard", "hal_gaia": "GAIA", "hal_online_mind2web": "Online-Mind2Web", "hal_scicode": "SciCode",
         "hal_scienceagentbench": "ScienceAgentBench", "hal_swebench_verified_mini": "SWE-bench Verified mini", "hal_taubench_airline": "tau-bench airline",
         "hal_usaco": "USACO"}

keys = [k for k in names if k in res]
n = len(keys)
cols = 3
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(9.5, 2.75 * rows))
axes = axes.ravel()
for ax, k in zip(axes, keys):
    r = res[k]
    tab = sorted(r["table"], key=lambda x: x["cost"])
    xs = [max(row["cost"], 1e-4) for row in tab]
    ys = [100 * row["accuracy"] for row in tab]
    inset = [row["in_frontier_set"] for row in tab]
    onf = [row["on_observed_frontier"] for row in tab]
    ax.scatter([x for x, s in zip(xs, inset) if s], [y for y, s in zip(ys, inset) if s], s=18, color=BLUE, zorder=3, label="in the frontier set")
    ax.scatter([x for x, s in zip(xs, inset) if not s], [y for y, s in zip(ys, inset) if not s], s=18, facecolors="none", edgecolors=GRAY, zorder=3, label="excluded from it")
    fx = [x for x, o in zip(xs, onf) if o]                                 # the plain frontier as a staircase
    fy = [y for y, o in zip(ys, onf) if o]
    if fx:
        ax.step(fx + [max(xs)], fy + [fy[-1]], where="post", color=ORANGE, lw=1.2, zorder=2, label="plain frontier")
    if r.get("band"):
        bd = r["band"]
        gx = [max(g, 1e-4) for g in bd["grid"]] + [max(xs)]
        ax.fill_between(gx, [100 * v for v in bd["lower"]] + [100 * bd["lower"][-1]], [100 * v for v in bd["upper"]] + [100 * bd["upper"][-1]],
                        step="post", color=ORANGE, alpha=0.15, lw=0, zorder=1, label="95% band for the frontier")
    cb = r["cheapest_plausible_best"]
    ax.scatter([max(cb["cost"], 1e-4)], [100 * cb["accuracy"]], s=80, marker="*", color=ORANGE, zorder=4, label="cheapest plausible best")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_formatter(NullFormatter())
    dag = r"$^{\dagger}$" if r.get("cost_source") == "run_total_even" else ""
    ax.set_title(f"{names[k]}{dag}\n{r['systems']} systems, {r['tasks']} tasks: plain frontier {len(r['observed_frontier'])}, set {len(r['frontier_set'])}", fontsize=8)
    ax.set_xlabel("cost per task, dollars")
    ax.set_ylabel("accuracy, %")
for ax in axes[n:]:
    ax.axis("off")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(FIG / "fig1_frontiers.pdf")
fig.savefig(FIG / "fig1_frontiers.png", dpi=160)
print("wrote fig1_frontiers")

order = sorted(keys, key=lambda k: (len(res[k]["frontier_set"]) / res[k]["systems"], res[k]["systems"]))   # by share that cannot be excluded
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for y, k in enumerate(order):
    r = res[k]
    o, s, K = len(r["observed_frontier"]), len(r["frontier_set"]), r["systems"]
    ax.plot([o, s], [y, y], color=GRAY, lw=1.2, zorder=1)
    ax.scatter([K], [y], s=26, facecolors="none", edgecolors=GRAY, zorder=2, label="systems on the board" if y == 0 else None)
    ax.scatter([o], [y], s=26, color=ORANGE, zorder=3, label="on the plain frontier" if y == 0 else None)
    ax.scatter([s], [y], s=26, color=BLUE, zorder=3, label="cannot be excluded from it" if y == 0 else None)
    ax.annotate(f"{r['tasks']} tasks", (K, y), fontsize=6.5, xytext=(6, -2.5), textcoords="offset points", color=GRAY)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([names[k] + (r"$^{\dagger}$" if res[k].get("cost_source") == "run_total_even" else "") for k in order], fontsize=8)
ax.set_xlabel("number of systems")
ax.set_xlim(0, max(res[k]["systems"] for k in keys) + 16)
ax.legend(loc="upper right", frameon=False, fontsize=7.5)
ax.set_title("Benchmarks ordered by the share of systems that cannot be excluded", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig2_setsizes.pdf")
fig.savefig(FIG / "fig2_setsizes.png", dpi=160)
print("wrote fig2_setsizes")

# ---- fig3: HAL's published frontier against the once-counted, repriced one, on the two boards that change most ----
flips_path = ROOT / "results" / "frontier_flips.json"
if flips_path.exists():
    fl = json.loads(flips_path.read_text(encoding="utf-8"))
    pick = sorted(fl, key=lambda k: -fl[k]["changed_total"])[:2]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, k in zip(axes, pick):
        rows = sorted(fl[k]["table"], key=lambda r: r["cost_reported"])
        for key, color, label, marker in (("cost_reported", GRAY, "HAL's reported cost", "o"), ("cost_ours", BLUE, "once-counted, repriced", "o")):
            xs = [max(r[key], 1e-4) for r in rows]
            ys = [100 * r["accuracy"] for r in rows]
            onf = [r["on_reported"] if key == "cost_reported" else r["on_ours"] for r in rows]
            ax.scatter(xs, ys, s=16, facecolors=color if key == "cost_ours" else "none", edgecolors=color, zorder=3, label=label)
            fx = [x for x, o in zip(xs, onf) if o]
            fy = [y for y, o in zip(ys, onf) if o]
            if fx:
                ax.step(fx + [max(xs)], fy + [fy[-1]], where="post", color=color, lw=1.2, ls="--" if key == "cost_reported" else "-", zorder=2)
        changed = [r for r in rows if r["on_reported"] != r["on_ours"]]
        for r in changed:
            ax.annotate("", xy=(max(r["cost_ours"], 1e-4), 100 * r["accuracy"]), xytext=(max(r["cost_reported"], 1e-4), 100 * r["accuracy"]),
                        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9), zorder=4)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{names[k]}: leaving the frontier {fl[k]['lost_total']}, joining it {fl[k]['gained_total']}", fontsize=8)
        ax.set_xlabel("cost per task, dollars")
        ax.set_ylabel("accuracy, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig3_flips.pdf")
    fig.savefig(FIG / "fig3_flips.png", dpi=160)
    print("wrote fig3_flips")

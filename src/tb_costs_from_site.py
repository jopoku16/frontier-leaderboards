"""Cost per submission for Terminal-Bench 2, from the leaderboard page.

The leaderboard dataset's job files carry a cost in only 3 of 244 jobs, but the public leaderboard page
(https://www.tbench.ai/leaderboard/terminal-bench/2.0) embeds, for every listed submission, total_cost_usd, the
token counts and the accuracy. This script reads the saved page (data/tbench_leaderboard_page.html, fetched with
curl), pulls those rows out of the embedded Next.js payload, and matches them to the submission names of the board
built for Paper 1 (`Agent__Model`), writing data/costs_terminalbench.csv with the matched names and
data/tbench_site_rows.json with every row as found.

usage: python tb_costs_from_site.py
"""
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
P1 = HERE.parent / "Phase B - Statistics Paper 1"

text = (HERE / "data" / "tbench_leaderboard_page.html").read_text(encoding="utf-8")
u = text.replace('\\"', '"').replace("\\\\", "\\")

# every metrics object, and the names near it: the page nests {"metrics": {...}} inside a row with display names
rows = []
for m in re.finditer(r'"total_cost_usd"\s*:\s*([0-9.]+)', u):
    start = u.rfind("{", 0, m.start() - 2000) if False else None
    rows.append(m.start())

# a more robust pass: find each row object by its metrics block and look backwards for the display names
records = []
for m in re.finditer(r'"metrics"\s*:\s*(\{[^{}]*\})', u):
    try:
        metrics = json.loads(m.group(1))
    except json.JSONDecodeError:
        continue
    if "total_cost_usd" not in metrics:
        continue
    window = u[max(0, m.start() - 1500): m.start()]
    def grab(key):
        hits = re.findall(r'"%s"\s*:\s*"([^"]*)"' % key, window)
        return hits[-1] if hits else None
    records.append({
        "agent_display": grab("agent_display"), "agent_org": grab("agent_org"), "model_display": grab("model_display"),
        "model_org": grab("model_org"), "name": grab("name"), "date": grab("date"),
        "accuracy": metrics.get("accuracy"), "total_cost_usd": metrics.get("total_cost_usd"), "n_trials": metrics.get("n_trials"),
        "total_tokens": metrics.get("total_tokens"), "output_tokens": metrics.get("output_tokens"),
        "cached_input_tokens": metrics.get("cached_input_tokens"), "uncached_input_tokens": metrics.get("uncached_input_tokens"),
    })
(HERE / "data" / "tbench_site_rows.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
print(f"site rows with cost: {len(records)}")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


board = json.load(open(P1 / "data" / "boards" / "terminalbench.json", encoding="utf-8"))
names = [s["name"] for s in board["systems"]]
by_norm = {norm(n): n for n in names}
matched, unmatched = [], []
for r in records:
    cands = [norm((r["agent_display"] or "") + (r["model_display"] or "")), norm(r["name"] or "")]
    hit = next((by_norm[c] for c in cands if c in by_norm), None)
    if hit is None:                                      # loose match: agent and model fragments both present
        a, mdl = norm(r["agent_display"]), norm(r["model_display"])
        loose = [n for n in names if a and mdl and a in norm(n) and mdl in norm(n)]
        hit = loose[0] if len(loose) == 1 else None
    (matched if hit else unmatched).append((hit, r))
print(f"matched to board names: {len(matched)}; unmatched: {len(unmatched)}")
for h, r in unmatched[:10]:
    print("  unmatched:", r["agent_display"], "|", r["model_display"], "|", r["name"], "| cost", r["total_cost_usd"])
with open(HERE / "data" / "costs_terminalbench.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["system", "cost", "cost_per_trial", "n_trials", "site_accuracy", "agent_display", "model_display"])
    for h, r in matched:
        c = r["total_cost_usd"]
        w.writerow([h, c, round(c / r["n_trials"], 6) if r.get("n_trials") else "", r.get("n_trials"), r["accuracy"], r["agent_display"], r["model_display"]])
print("wrote data/costs_terminalbench.csv")

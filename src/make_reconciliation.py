"""One row per HAL archive: what HAL reports, what we recount, and what became of the run.

Columns: benchmark, archive (the zip's name), agent, models, HAL's reported total cost, the ratio of HAL's token total to
the once-counted token total (the double count), our once-counted and repriced total, the number of model calls dropped as
duplicates, outcome coverage, tasks priced by imputation, and the status (kept in the audit, dropped with the reason, or
skipped because the archive's results member could not be decrypted in memory).

Writes results/hal_reconciliation.csv and paper/supp_reconciliation.tex (a long table for the supplement).
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from build_hal_boards import system_name  # noqa: E402


def tex(s):
    return re.sub(r"([_&%#$])", r"\\\1", str(s))


def main():
    idx = json.loads((ROOT / "data" / "hal" / "index.json").read_text(encoding="utf-8"))
    summ = json.loads((ROOT / "data" / "boards" / "hal_summary.json").read_text(encoding="utf-8"))
    kept = {}
    for p in (ROOT / "data" / "boards").glob("hal_*.json"):
        if p.name.endswith("_costs.json") or p.name == "hal_summary.json":
            continue
        for s in json.loads(p.read_text(encoding="utf-8"))["systems"]:
            kept[s["source"]] = s
    dropped = {}
    for bench, v in summ.items():
        for d in v.get("dropped", []):
            dropped.setdefault(d["system"].rstrip("_"), []).append(d["reason"])
    rows = []
    for archive, e in sorted(idx.items(), key=lambda kv: kv[0]):
        stem = archive[:-4]
        rec_path = ROOT / "data" / "hal" / "runs" / f"{stem}.json"
        row = {"archive": archive, "benchmark": e.get("benchmark") or "", "agent": e.get("agent") or "", "reported_cost": e.get("cost_reported"),
               "once_counted_cost": e.get("cost_recomputed"), "duplicate_calls": e.get("duplicate_calls"),
               "token_ratio": (e["tokens_raw"] / e["tokens_counted"]) if e.get("tokens_counted") else None}
        if archive in kept:
            s = kept[archive]
            row.update({"status": "kept", "outcome_coverage": s.get("outcome_coverage"), "tasks_imputed": s.get("tasks_without_calls")})
        elif rec_path.exists():
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            name = system_name(rec).rstrip("_")
            reasons = dropped.get(name, ["dropped (see the board summary)"])
            row.update({"status": "dropped: " + reasons[0], "outcome_coverage": None, "tasks_imputed": None})
        else:
            row.update({"status": "skipped: " + (e.get("error") or "not read"), "outcome_coverage": None, "tasks_imputed": None})
        rows.append(row)
    fields = ["benchmark", "archive", "agent", "reported_cost", "token_ratio", "once_counted_cost", "duplicate_calls", "outcome_coverage", "tasks_imputed", "status"]
    with open(ROOT / "results" / "hal_reconciliation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])) for k in fields})
    lines = []
    for r in sorted(rows, key=lambda r: (r["benchmark"], r["archive"])):
        arch = re.sub(r"_UPLOAD(_updated)?$", "", r["archive"][:-4])
        arch = (arch[:32] + "~") if len(arch) > 33 else arch
        rep = "" if r["reported_cost"] is None else f"{r['reported_cost']:,.2f}"
        cnt = "" if r["once_counted_cost"] is None else f"{r['once_counted_cost']:,.2f}"
        ratio = "" if r["token_ratio"] is None else f"{r['token_ratio']:.2f}"
        status = r["status"].replace("dropped: ", "dropped: ").replace("skipped: skipped: ", "skipped: ")
        status = re.sub(r"\(cost unknown\)", "", status)
        status = (status[:30] + "~") if len(status) > 31 else status
        lines.append(f"{tex(r['benchmark'].replace('hal_', '')[:14])} & \\texttt{{\\scriptsize {tex(arch)}}} & {rep} & {ratio} & {cnt} & {tex(status)} \\\\")
    body = ("\\begin{longtable}{llrrrl}\n\\caption{Every HAL archive: the benchmark, the archive's name, HAL's reported total cost, the ratio of HAL's token total to the "
            "once-counted token total, our once-counted and repriced total cost, and the run's status. The same table, with agent names and the number of duplicate "
            "calls, is in \\texttt{results/hal\\_reconciliation.csv}.}\\label{tab:supp-reconciliation}\\\\\n\\toprule\n"
            "Benchmark & Archive & Reported & Ratio & Recounted & Status \\\\\n\\midrule\n\\endfirsthead\n\\toprule\n"
            "Benchmark & Archive & Reported & Ratio & Recounted & Status \\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n" + "\n".join(lines) + "\n\\end{longtable}\n")
    (ROOT / "paper" / "supp_reconciliation.tex").write_text("{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n" + body + "}\n", encoding="utf-8")
    n_kept = sum(1 for r in rows if r["status"] == "kept")
    print(f"{len(rows)} archives: {n_kept} kept, {sum(1 for r in rows if r['status'].startswith('dropped'))} dropped, {sum(1 for r in rows if r['status'].startswith('skipped'))} skipped")


if __name__ == "__main__":
    main()

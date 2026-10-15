"""Fetch the per-instance records of the HELM Capabilities leaderboard and reduce them to compact per-run files.

HELM Capabilities (Liang et al., Stanford CRFM) scores language models on five fixed benchmarks (GPQA, IFEval, MMLU-Pro,
Omni-MATH, WildBench) and publishes, for every model and benchmark, per-instance outcomes together with the number of prompt
and completion tokens of every request. Models arrive release by release (v1.0.0 to v1.15.0), which gives a natural arrival
order. Cost is not published: we price the tokens with one price table (build_helm_boards.py).

For each run this script keeps, per instance, the prompt tokens, the completion tokens and every correctness or score
metric, and writes data/helm/runs/<run>.json. Already fetched runs are skipped, so the script can be rerun.

usage: python helm_fetch.py [--workers 8]
Writes data/helm/releases.json, data/helm/run_index.json and data/helm/runs/*.json
"""
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "helm"
RUNS = OUT / "runs"
BASE = "https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output"
METRIC_WORDS = ("correct", "accuracy", "score", "quasi", "exact_match")


def get(url, tries=5):
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "frontier-leaderboards (research; jopoku16@users.noreply.github.com)"}), timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def run_file(run):
    return RUNS / (re.sub(r"[^A-Za-z0-9_.=-]", "_", run)[:150] + "_" + hashlib.md5(run.encode()).hexdigest()[:8] + ".json")


def reduce_run(run, suite):
    url = f"{BASE}/runs/{suite}/{urllib.parse.quote(run, safe='')}/per_instance_stats.json"
    data = json.loads(get(url))
    inst = {}
    for e in data:
        rec = inst.setdefault(e["instance_id"], {"p": None, "c": None, "y": {}})
        for s in e["stats"]:
            n = s["name"]["name"]
            if s["name"].get("perturbation"):
                continue
            if n == "num_prompt_tokens":
                rec["p"] = s["sum"]
            elif n == "num_completion_tokens":
                rec["c"] = s["sum"]
            elif any(w in n for w in METRIC_WORDS) and s.get("count", 0) > 0:
                rec["y"][n] = s["mean"]
    scenario = run.split(":", 1)[0]
    model = re.search(r"model=([^,]+)", run).group(1)
    rec_out = {"run": run, "suite": suite, "scenario": scenario, "model": model, "n_instances": len(inst), "instances": inst}
    run_file(run).write_text(json.dumps(rec_out), encoding="utf-8")
    return run


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    RUNS.mkdir(parents=True, exist_ok=True)
    runs = json.loads(get(f"{BASE}/releases/v1.15.0/runs_to_run_suites.json"))
    (OUT / "run_index.json").write_text(json.dumps(runs, indent=0), encoding="utf-8")
    suites = sorted({s for s in runs.values()}, key=lambda s: [int(x) for x in s.lstrip("v").split(".")])
    rel = {}
    for s in suites:
        try:
            rel[s] = json.loads(get(f"{BASE}/releases/{s}/summary.json")).get("date")
        except Exception as e:  # noqa: BLE001
            rel[s] = None
            print("no date for", s, e)
    (OUT / "releases.json").write_text(json.dumps(rel, indent=1), encoding="utf-8")
    todo = [(r, s) for r, s in runs.items() if not run_file(r).exists()]
    print(f"{len(runs)} runs, {len(todo)} to fetch", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(reduce_run, r, s): r for r, s in todo}
        for f, r in futs.items():
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print("FAILED", r, e, flush=True)
                continue
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

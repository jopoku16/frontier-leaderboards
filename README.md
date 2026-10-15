# Who is really on the frontier?

Code, manuscript and results for

> David Banahene and Jeffery Opoku. *Who Is Really on the Frontier? Confidence Sets for Cost-Accuracy Pareto Leaderboards.* 2026.

A leaderboard that reports cost beside accuracy is read through its Pareto frontier, the systems no cheaper system beats. Every published frontier is drawn from point estimates on a sample of tasks. This paper gives a confidence set for the frontier, an undominated frequency per system, the cheapest system that cannot be told apart from the best, two certificates for new frontier entrants with chain-wise error control, an exact finite-sample frontier set, the task count each board would need to resolve its frontier, and selection-adjusted accuracy for systems on the frontier, and audits the Holistic Agent Leaderboard (HAL) with per-task costs rebuilt from its public call logs.

The methods live in the open-source package [honestboard](https://github.com/jopoku16/honestboard) (version 0.6.1 or later): `pareto_frontier`, `frontier_band`, `exact_frontier_set`, `entrant_pvalues`, `entrant_pvalues_exact`, `frontier_curse`, and the command `python -m honestboard frontier BOARD.json COSTS.json --curse`.

## Layout

| Path | What |
|---|---|
| `paper/main.tex`, `paper/main.pdf` | the manuscript; `numbers.tex` and `tab_*.tex` are generated |
| `paper/supplement.tex`, `paper/supp_*.tex` | the supplement (every system, every certification step, every frontier member) |
| `src/hal_fetch.py`, `hal_decrypt.py`, `hal_extract.py` | read only the results member of each HAL trace archive by HTTP range requests, decrypt it with the harness's public key, and reduce it to per-task outcomes, tokens and costs, counting each model call once |
| `src/build_hal_boards.py` | one board per benchmark (honestboard format) with per-task costs |
| `src/frontier_hal.py` | the frontier audit (`results/frontier_hal.json`) |
| `src/certify_entrants.py` | certification of frontier entrants in arrival order (`results/certify_hal.json`) |
| `src/frontier_curse_hal.py` | selection-adjusted accuracy of frontier members (`results/curse_hal.json`) |
| `src/validate_frontier.py`, `src/validate_curse.py` | known-truth simulations (`results/validation_*.json`) |
| `src/validate_heldout.py`, `src/rerun_noise.py`, `src/cluster_check.py`, `src/cost_separation.py` | held-out halves of the real boards, run-to-run noise from rerun pairs, task clusters, and how many system pairs can be told apart on cost |
| `src/inflation_sensitivity.py` | frontier sets if every standard error were f times larger, read from the stored leads (`results/inflation_sensitivity.json`) |
| `data/groups/` | task-group labels for ScienceAgentBench (github_name and domain columns of the public osunlp/ScienceAgentBench task dataset), used by `cluster_check.py` |
| `src/helm_fetch.py`, `src/build_helm_boards.py` | the second leaderboard: HELM Capabilities per-instance results and tokens from the public bucket, priced into boards (`data/boards/helm_*.json`, not redistributed) |
| `src/replicate_certificates.py` | do certified entrants keep their lead on fresh tasks (`results/replication_certificates_*.json`) |
| `src/tie_robust_experiment.py`, `src/tie_robust_stress.py` | prototypes and least-favourable checks of the tie-robust set |
| `src/hal_sensitivity.py`, `src/make_reconciliation.py` | the audit under other costing rules, and one row per HAL archive (`results/hal_reconciliation.csv`) |
| `src/make_numbers.py`, `make_figures.py`, `make_supplement.py` | every number, table and figure in the paper |
| `data/hal/model_prices.json`, `cache_prices.json` | the one price table used for every run |
| `data/hal/index.json`, `data/boards/hal_summary.json` | what was read from HAL, what was kept, what was dropped and why |

## Reproduce

```
pip install -r requirements.txt
python src/hal_fetch.py                 # hours: streams the results member of 380 archives (about 113 GB of zips, a few GB read)
python src/build_hal_boards.py
python src/frontier_hal.py --B 4000
python src/certify_entrants.py
python src/frontier_curse_hal.py --B 4000
python src/validate_frontier.py 300      # or six parallel parts: --part K 6 for K in 0..5, then --merge
python src/validate_heldout.py 200
python src/hal_sensitivity.py
python src/validate_curse.py 300
python src/make_numbers.py && python src/make_figures.py && python src/make_supplement.py
cd paper && latexmk -pdf main.tex && latexmk -pdf supplement.tex
```

## Data

The per-task tables derived from HAL's archives (`data/hal/runs/`, `data/boards/hal_*.json`) are not redistributed here: the dataset card states no licence, and a question to the HAL team is pending. The pipeline rebuilds them from the public archives. Raw archives are never stored.

Two facts about HAL's data that the pipeline handles and the paper reports: HAL's cost totals double count nested trace spans on runs whose agent calls the provider through a wrapper library (each call is counted once here), and HAL's totals embed the prices in force when each run was made (everything is repriced under one table).

## Licence

Code: MIT. Manuscript text and figures: CC BY 4.0.

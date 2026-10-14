# Who is really on the frontier?

Code, manuscript and results for

> David Banahene and Jeffery Opoku. *Who Is Really on the Frontier? Confidence Sets for Cost-Accuracy Pareto Leaderboards.* 2026.

A leaderboard that reports cost beside accuracy is read through its Pareto frontier, the systems no cheaper system beats. Every published frontier is drawn from point estimates on a sample of tasks. This paper gives a confidence set for the frontier, an undominated frequency per system, the cheapest system that cannot be told apart from the best, two certificates for new frontier entrants with chain-wise error control, an exact finite-sample frontier set, the task count each board would need to resolve its frontier, and selection-adjusted accuracy for systems on the frontier, and audits the Holistic Agent Leaderboard (HAL) with per-task costs rebuilt from its public call logs.

The methods live in the open-source package [honestboard](https://github.com/jopoku16/honestboard) (version 0.2.1 or later): `pareto_frontier`, `exact_frontier_set`, `entrant_pvalues`, `frontier_curse`, and the command `python -m honestboard frontier BOARD.json COSTS.json --curse`.

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
python src/validate_frontier.py 300
python src/validate_curse.py 300
python src/make_numbers.py && python src/make_figures.py && python src/make_supplement.py
cd paper && latexmk -pdf main.tex && latexmk -pdf supplement.tex
```

## Data

The per-task tables derived from HAL's archives (`data/hal/runs/`, `data/boards/hal_*.json`) are not redistributed here: the dataset card states no licence, and a question to the HAL team is pending. The pipeline rebuilds them from the public archives. Raw archives are never stored.

Two facts about HAL's data that the pipeline handles and the paper reports: HAL's cost totals double count nested trace spans on runs whose agent calls the provider through a wrapper library (each call is counted once here), and HAL's totals embed the prices in force when each run was made (everything is repriced under one table).

## Licence

Code: MIT. Manuscript text and figures: CC BY 4.0.

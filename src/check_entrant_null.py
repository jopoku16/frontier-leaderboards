"""What the frontier-entrant certificate does and does not control (the remark after Proposition 2).

The test is pairwise: entrant b against j*, the observed most accurate cheaper system. Two worlds, both with b truly
dominated by a cheaper system k of the same true accuracy (independent outcomes):

  independent   the other cheaper systems are independent of b (weaker, the same, or a little better);
  near-copy     one cheaper system is a degraded copy of b (each of b's passes flipped to a fail with probability
                delta, so it is correlated with b and usually a little worse), which is what a leaderboard holds when
                the same agent is run with a smaller model, a lower effort setting, or simply rerun.

On each board we condition on the frontier event (b beats every cheaper system on the sample), take the observed
best cheaper system as the holder, and record whether the selective record test certifies b at the 5% level. Under
the composite null "some cheaper system truly dominates b" the certification rate should be at most 5% for a valid
certificate of Pareto membership; it is not in the near-copy world, which is why the paper states the certificate as
a record claim restricted to the cheaper pool.

usage: python check_entrant_null.py [R]
Writes results/entrant_null.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Phase B - Statistics Paper 1" / "honestboard" / "src"))
from honestboard.chain import selective_sign_pvalue  # noqa: E402


def one(rng, T, mu, delta, n_other, world, alpha=0.05):
    """Returns (frontier event happened, certified) for one simulated arrival."""
    diff = rng.normal(0, 1.0, T)
    base = np.log(mu / (1 - mu))
    pb = 1 / (1 + np.exp(-(base + diff)))
    yb = (rng.random(T) < pb).astype(int)
    yk = (rng.random(T) < pb).astype(int)                                   # k: same true accuracy, independent, cheaper
    others = [yk]
    for _ in range(n_other):                                                 # weaker independent systems, cheaper
        po = 1 / (1 + np.exp(-(base - 0.4 + diff)))
        others.append((rng.random(T) < po).astype(int))
    if world == "near-copy":
        yc = yb.copy()
        flip = (yc == 1) & (rng.random(T) < delta)
        yc[flip] = 0
        others.append(yc)
    accs = [o.mean() for o in others]
    if yb.mean() <= max(accs) + 1e-12:
        return False, False                                                  # not on the frontier: never tested
    j = int(np.argmax(accs))
    ya = others[j]
    n_b = int(((yb == 1) & (ya == 0)).sum())
    n_a = int(((ya == 1) & (yb == 0)).sum())
    return True, selective_sign_pvalue(n_b, n_a) <= alpha


def main(R):
    rng = np.random.default_rng(5)
    out = {"R": R, "alpha": 0.05, "cases": []}
    for world in ("independent", "near-copy"):
        for T, mu, delta, n_other in ((50, 0.5, 0.2, 2), (100, 0.5, 0.2, 5), (150, 0.6, 0.2, 2), (300, 0.5, 0.05, 2)):
            sel = cert = 0
            for _ in range(R):
                s, c = one(rng, T, mu, delta, n_other, world)
                sel += s
                cert += c
            rate = cert / max(1, sel)
            out["cases"].append({"world": world, "T": T, "mu": mu, "delta": delta, "n_other": n_other, "selected": sel, "certified": cert, "rate": rate})
            print(f"{world:12s} T={T:3d} mu={mu} delta={delta} others={n_other}: selected {sel:5d} of {R}, certified {rate:.3f}", flush=True)
    (ROOT / "results" / "entrant_null.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)

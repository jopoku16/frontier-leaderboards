"""What the two frontier-entrant certificates control (Section 3.5).

Two tests of an entrant b that landed on the frontier:
  pairwise   the selective exact sign test of b against the observed most accurate cheaper system j* (Prop. 2):
             a record claim within the cheaper pool;
  composite  the intersection-union test of polyhedral selective p-values, one per cheaper system, all conditional
             on the whole frontier event (Prop. 3): a certificate of Pareto membership under the normal approximation.

Three worlds. In the two null worlds a cheaper system k with the same true accuracy as b (independent outcomes)
truly dominates b:
  independent   the other cheaper systems are independent of b and a little weaker;
  near-copy     one cheaper system is a degraded copy of b (each of b's passes flipped with probability delta), the
                shape of a leaderboard holding the same agent with a smaller model, a lower effort, or a rerun;
and in the alternative world b is truly better than every cheaper system by a margin, to compare power:
  better        b's true accuracy is lifted by `lift` on the logit scale.
Rates are over the boards on which b landed on the frontier (the only ones a maintainer tests).

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
from honestboard.frontier import entrant_pvalues_arrays, entrant_pvalues_exact_arrays  # noqa: E402

R_EXACT = 20000     # closed form for pass/fail outcomes: as fast as the other two tests
DRAWS_EXACT = 1000


def one(rng, T, mu, delta, n_other, world, alpha=0.05, lift=0.5, exact=False):
    """Returns (landed on the frontier, pairwise certified, composite certified, exact composite certified)."""
    diff = rng.normal(0, 1.0, T)
    base = np.log(mu / (1 - mu))
    pb = 1 / (1 + np.exp(-(base + (lift if world == "better" else 0.0) + diff)))
    yb = (rng.random(T) < pb).astype(float)
    pk = 1 / (1 + np.exp(-(base + diff)))
    others = [(rng.random(T) < pk).astype(float)]                              # k: same true accuracy as the null b, cheaper
    for _ in range(n_other):
        po = 1 / (1 + np.exp(-(base - 0.4 + diff)))
        others.append((rng.random(T) < po).astype(float))
    if world == "near-copy":
        yc = yb.copy()
        yc[(yc == 1) & (rng.random(T) < delta)] = 0
        others.append(yc)
    accs = np.array([o.mean() for o in others])
    if yb.mean() <= accs.max() + 1e-12:
        return False, False, False, False
    j = int(np.argmax(accs))
    ya = others[j]
    n_b = int(((yb == 1) & (ya == 0)).sum()); n_a = int(((ya == 1) & (yb == 0)).sum())
    pair = selective_sign_pvalue(n_b, n_a) <= alpha
    Y = np.vstack([yb] + others)                                               # b is row 0; the pool is every other row
    acc = Y.mean(axis=1)
    Sigma = np.cov(Y) / T
    p_iu, _ = entrant_pvalues_arrays(acc, Sigma, 0, list(range(1, Y.shape[0])), T)
    comp = np.isfinite(p_iu) and p_iu <= alpha
    ex = False
    if exact:
        p_ex, _, _ = entrant_pvalues_exact_arrays(Y, 0, list(range(1, Y.shape[0])), draws=DRAWS_EXACT, seed=int(rng.integers(1 << 30)))
        ex = p_ex <= alpha
    return True, bool(pair), bool(comp), bool(ex)


def main(R):
    rng = np.random.default_rng(5)
    out = {"R": R, "alpha": 0.05, "cases": []}
    cases = ((50, 0.5, 0.2, 2), (100, 0.5, 0.2, 5), (150, 0.6, 0.2, 2), (300, 0.5, 0.05, 2))
    for world in ("independent", "near-copy", "better"):
        for T, mu, delta, n_other in cases:
            sel = cp = cc = sel_e = ce = 0
            for r_i in range(R):
                s, a, b, e = one(rng, T, mu, delta, n_other, world, exact=(r_i < R_EXACT))
                sel += s; cp += a; cc += b
                if r_i < R_EXACT:
                    sel_e += s; ce += e
            rp, rc, re_ = cp / max(1, sel), cc / max(1, sel), ce / max(1, sel_e)
            out["cases"].append({"world": world, "T": T, "mu": mu, "delta": delta, "n_other": n_other, "selected": sel,
                                 "rate_pairwise": rp, "rate_composite": rc, "rate_exact": re_, "R_exact": R_EXACT, "selected_exact": sel_e})
            print(f"{world:12s} T={T:3d} mu={mu} delta={delta} others={n_other}: selected {sel:5d} of {R}; certified pairwise {rp:.3f}, composite {rc:.3f}, exact {re_:.3f} (on {sel_e} boards)", flush=True)
    (ROOT / "results" / "entrant_null.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)

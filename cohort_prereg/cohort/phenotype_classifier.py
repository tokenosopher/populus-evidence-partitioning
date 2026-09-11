"""Cohort phenotype classifier — implements PREREG_ROLE_MARKED_COHORT.md Appendix A exactly.

Classification unit: individual trained trajectory. Inputs per trajectory:
  A2, A3 : untouched primary final-bank accuracies at depths two, three
  T2     : exact full depth-two training-bank accuracy
  T3     : exact accuracy on the frozen first-150-program depth-three training subset
  S1     : held-out-phrasing single-operation accuracy (descriptive only; never classifies)
T = max(T2, T3). All-cut never enters classification. Classes evaluated in fixed
order; first match assigned. Thresholds are frozen by the preregistration.
"""
import json
import math
import sys


class InputError(Exception):
    pass


def _finite01(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x) and 0.0 <= x <= 1.0

CLASSES = [
    "full_generalizer",
    "deep_only_generalization_inversion",
    "shallow_only_partial",
    "memorizer",
    "non_acquirer",
    "intermediate_mixed",
]


def classify(A2, A3, T2, T3, S1=None):
    for name, v in (("A2", A2), ("A3", A3), ("T2", T2), ("T3", T3)):
        if not _finite01(v):
            raise InputError(f"non-finite/out-of-range {name}={v!r}")
    T = max(T2, T3)
    if A2 >= 0.70 and A3 >= 0.70:
        return "full_generalizer"
    if A3 >= 0.40 and A2 < 0.40 and (A3 - A2) >= 0.20:
        return "deep_only_generalization_inversion"
    if A2 >= 0.40 and A3 < 0.40:
        return "shallow_only_partial"
    if T >= 0.90 and A2 < 0.20 and A3 < 0.20:
        return "memorizer"
    if T < 0.90 and A2 < 0.20 and A3 < 0.20:
        return "non_acquirer"
    return "intermediate_mixed"


def initialization_report(trajectories):
    """trajectories: list of dicts with keys init, order, A2, A3, T2, T3, S1.
    Returns per-initialization ordered class pairs, 6x6 concordance counts,
    and concordant/discordant totals. Never collapses discordant pairs."""
    by_init = {}
    for t in trajectories:
        key = (t["init"], t["order"])
        if t["order"] in by_init.get(t["init"], {}):
            raise InputError(f"duplicate (init, order) record: {key}")
        by_init.setdefault(t["init"], {})[t["order"]] = classify(
            t["A2"], t["A3"], t["T2"], t["T3"], t.get("S1"))
    conc = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    pairs, n_conc, n_disc = {}, 0, 0
    for init, orders in sorted(by_init.items()):
        ks = sorted(orders)
        if len(ks) != 2:
            pairs[init] = {"status": "incomplete", "classes": [orders[k] for k in ks]}
            continue
        a, b = orders[ks[0]], orders[ks[1]]
        conc[a][b] += 1
        omap = {ks[0]: a, ks[1]: b}
        if a == b:
            pairs[init] = {"status": "concordant", "phenotype": a, "orders": omap}
            n_conc += 1
        else:
            pairs[init] = {"status": "discordant", "orders": omap}
            n_disc += 1
    return {"per_initialization": pairs, "concordance_6x6": conc,
            "n_concordant": n_conc, "n_discordant": n_disc}


if __name__ == "__main__":
    rows = json.load(open(sys.argv[1]))
    out = {"per_trajectory": [
        dict(r, phenotype=classify(r["A2"], r["A3"], r["T2"], r["T3"], r.get("S1")))
        for r in rows]}
    out["initialization_report"] = initialization_report(rows)
    print(json.dumps(out, indent=1))

"""Cohort statistics engine — PREREG section 8 exactly.

Cluster bootstrap: resample complete initialization blocks with replacement,
preserving both data orders and all conditions jointly. 100,000 draws, seed
271828, percentile 2.5/97.5 interval. Exact two-sided sign-flip test over
initialization-level mean differences. Descriptive uncertainty summaries only;
gates are computed elsewhere by the verdict script.
"""
import itertools
import json
import math
import random
import statistics
import sys


class InputError(Exception):
    pass


def linear_quantile(xs, q):
    h = (len(xs) - 1) * q
    lo = math.floor(h); hi = math.ceil(h)
    if lo == hi:
        return xs[lo]
    w = h - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def _validate(diffs_by_init, expected_k=None):
    for init, block in diffs_by_init.items():
        if len(block) != 2:
            raise InputError(f"init {init} has {len(block)} orders, expected 2")
        for d in block:
            if not (isinstance(d, (int, float)) and not isinstance(d, bool) and math.isfinite(d)):
                raise InputError(f"non-finite difference in {init}: {d!r}")
    if expected_k is not None and len(diffs_by_init) != expected_k:
        raise InputError(f"{len(diffs_by_init)} clusters, expected {expected_k}")

BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 271828


def _init_means(diffs_by_init):
    return {k: sum(v) / len(v) for k, v in diffs_by_init.items()}


def cluster_bootstrap(diffs_by_init, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """diffs_by_init: {init_id: [order-a diff, order-b diff]} for one contrast+depth."""
    rng = random.Random(seed)
    inits = sorted(diffs_by_init)
    k = len(inits)
    means = []
    for _ in range(draws):
        sample = [diffs_by_init[inits[rng.randrange(k)]] for _ in range(k)]
        flat = [d for block in sample for d in block]
        means.append(sum(flat) / len(flat))
    means.sort()
    lo = linear_quantile(means, 0.025)
    hi = linear_quantile(means, 0.975)
    flat_all = [d for v in diffs_by_init.values() for d in v]
    return {"mean_paired_difference": sum(flat_all) / len(flat_all),
            "cluster_bootstrap_2p5": lo, "cluster_bootstrap_97p5": hi,
            "draws": draws, "seed": seed, "n_clusters": k,
            "percentile_method": "linear_(N-1)q"}


def sign_flip_exact(diffs_by_init):
    """Exact two-sided sign-flip over initialization-level means."""
    m = sorted(_init_means(diffs_by_init).values())
    k = len(m)
    observed = abs(sum(m))
    count = 0
    total = 2 ** k
    for signs in itertools.product((1, -1), repeat=k):
        if abs(sum(s * x for s, x in zip(signs, m))) >= observed - 1e-12:
            count += 1
    return {"p_two_sided": count / total, "n_clusters": k,
            "nominal_min_p_nonzero": 2 / total, "initialization_means": m}


def analyze_contrast(diffs_by_init, expected_k=None):
    _validate(diffs_by_init, expected_k)
    per_order = {k: list(v) for k, v in sorted(diffs_by_init.items())}
    flat = [d for v in diffs_by_init.values() for d in v]
    return {
        "order_specific_differences": per_order,
        "trajectory_median": statistics.median(flat),
        "initialization_means": _init_means(diffs_by_init),
        "bootstrap": cluster_bootstrap(diffs_by_init),
        "sign_flip": sign_flip_exact(diffs_by_init),
    }


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))  # {contrast: {depth: {init: [d1, d2]}}}
    out = {c: {d: analyze_contrast(v) for d, v in depths.items()}
           for c, depths in data.items()}
    print(json.dumps(out, indent=1))

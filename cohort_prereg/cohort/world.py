"""Fresh-world construction for the cohort — PREREG section 4.

Wraps the parent's frozen op/split machinery (populus.bridge_tasks exposes
seed-parameterized make_ops/_build_split via env seeds) with:
  - the cohort's NEUTRAL grammar (cohort.neutral_grammar), replacing the
    parent's second-person templates entirely;
  - the collision-free primary evaluation bank constructor;
  - the G+ marker-permutation audit bank constructor;
  - the disjoint monitoring (liveness) bank constructor.

Structural validity checks (the ONLY permitted world-rejection reasons,
frozen per §4): duplicate primitive maps; insufficient collision-free
held-out programs; bank overlap. Tokenizer failure is checked on the
admission box. Every rejected candidate seed is ledgered by the caller.

Local testing uses EXPLICIT FIXTURE SEEDS, never the frozen seed rule; the
frozen rule's outputs are consumed only after the freeze ceremony.
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Development-exposure denylist — derived at import time from the
# machine-readable exposure manifest (REVIEW round-3 blocker 1: single
# source of truth, no hand-maintained second list). The frozen candidate
# stream must reject any candidate whose seed appears in the exposed set
# for its component role, BEFORE structural checks; rejections are
# ledgered as dev-exposed.
import json as _json
_EXP = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "manifests",
                                    "development_seed_exposures.json")))
DEV_EXPOSED = {k: frozenset(v) for k, v in _EXP["aggregate"].items()}
DEV_EXPOSED_OP_SEEDS = DEV_EXPOSED["exposed_operator_seeds"]


def candidate_is_dev_exposed(op_seed, split_seed):
    return (op_seed in DEV_EXPOSED["exposed_operator_seeds"]
            or split_seed in DEV_EXPOSED["exposed_split_seeds"])


def seed_bundle_is_dev_exposed(bundle):
    """ROLE-AWARE complete-bundle denial (REVIEW round-4, blocker 1
    remainder). bundle: {"operator": int, "split": int, "grammar": int,
    "filler": int, "orders": [int...], "initializations": [int...],
    "world_id": str}. The whole bundle is rejected if ANY component is
    development-exposed in its corresponding role; comparison is
    role-specific (numeric equality across roles does not reject).
    The frozen orchestration rule: derive the complete bundle for
    candidate counter c; if exposed, ledger and increment c; never
    accept a bundle and improvise per-field replacements."""
    role_map = (("operator", "exposed_operator_seeds"),
                ("split", "exposed_split_seeds"),
                ("grammar", "exposed_grammar_seeds"),
                ("filler", "exposed_filler_seeds"))
    for role, key in role_map:
        if bundle[role] in DEV_EXPOSED[key]:
            return f"{role}:{bundle[role]}"
    for s in bundle.get("orders", ()):
        if s in DEV_EXPOSED["exposed_order_seeds"]:
            return f"order:{s}"
    for s in bundle.get("initializations", ()):
        if s in DEV_EXPOSED["exposed_initialization_seeds"]:
            return f"initialization:{s}"
    synth = {w for e in _EXP["entries"]
             for w in e.get("synthetic_world_ids", ())}
    if bundle.get("world_id") in synth:
        return f"world_id:{bundle['world_id']}"
    return None


def build_world(op_seed, split_seed):
    """Construct ops + split under explicit seeds. Returns dict with ops,
    split banks, and composite-map tables for collision analysis."""
    os.environ["BRIDGE_OP_SEED"] = str(op_seed)
    os.environ["BRIDGE_SPLIT_SEED"] = str(split_seed)
    # force re-import under the new seeds. NOTE: deleting the sys.modules
    # entry alone is NOT enough — the parent package keeps a stale
    # attribute, and `from populus import bridge_tasks` would silently
    # return the OLD module (same world for every candidate seed).
    import importlib
    for m in list(sys.modules):
        if m.startswith("populus.bridge_tasks"):
            del sys.modules[m]
    import populus
    if hasattr(populus, "bridge_tasks"):
        delattr(populus, "bridge_tasks")
    T = importlib.import_module("populus.bridge_tasks")
    return T


def composite_fn(T, seq):
    return tuple(T.apply_chain(list(seq), x) for x in range(17))


def structural_checks(T, min_collision_free_d3=20, min_collision_free_d2=8):
    """The frozen mechanically-testable validity checks.

    Collision-free programs are drawn from the HELD-OUT pools (ho2 +
    ho_fn pairs at depth two; ho3_familiar + ho3_h2_only + ho_fn triples
    at depth three), NOT from the diag banks: the diag banks are the
    in-run qualification/monitoring diagnostics consumed by trainer
    probes, and their 60/12 sizes make >=20/>=8 collision-free counts
    mechanically unattainable (parent map-novel rate ~13/60). ho_fn
    programs are map-novel by construction (F_HO excluded from all
    training); the rest are filtered against the full training function
    set. Deterministic sorted order; first-seen dedup."""
    ops_maps = [composite_fn(T, (i,)) for i in range(len(T.OPS))]
    ok_dup = len(set(ops_maps)) == len(ops_maps)
    # REVIEW round-2 D2: training also contains identity episodes and
    # every single-operation primitive; a held-out program whose complete
    # map equals identity or a primitive is map-redundant.
    identity_map = tuple(range(17))
    no_identity_primitive = identity_map not in set(ops_maps)
    train_fns = {identity_map}
    train_fns.update(ops_maps)
    train_fns.update(composite_fn(T, tuple(s)) for s in T.SPLIT["train2"])
    train_fns.update(composite_fn(T, tuple(s)) for s in T.SPLIT["train3"])
    d2_pool = sorted(dict.fromkeys(
        [tuple(s) for s in T.SPLIT["ho2"]] +
        [tuple(s) for s in T.SPLIT["ho_fn"]["pairs"]]))
    d3_pool = sorted(dict.fromkeys(
        [tuple(s) for s in T.SPLIT["ho3_familiar"]] +
        [tuple(s) for s in T.SPLIT["ho3_h2_only"]] +
        [tuple(s) for s in T.SPLIT["ho_fn"]["triples"]]))
    # REVIEW round-2: disjointness checks are explicit VALID components
    # (never bare assert), written to the candidate ledger on failure.
    train_programs = {tuple(s) for k in ("train2", "train3")
                      for s in T.SPLIT[k]}
    diag_programs = {tuple(s) for k in ("diag2", "diag3")
                     for s in T.SPLIT[k]}
    pool_programs = set(d2_pool) | set(d3_pool)
    primary_vs_train_disjoint = pool_programs.isdisjoint(train_programs)
    primary_vs_diag_disjoint = pool_programs.isdisjoint(diag_programs)
    cf_d3 = [s for s in d3_pool if composite_fn(T, s) not in train_fns]
    cf_d2 = [s for s in d2_pool if composite_fn(T, s) not in train_fns]
    # REVIEW round-3: every operator in the primary pools must belong to
    # the single-operation curriculum support (no unseen-primitive mixing)
    primary_ops = {op for s in (cf_d2 + cf_d3) for op in s}
    all_primary_ops_singleton_trained = primary_ops <= set(range(len(T.OPS)))
    # REVIEW round-3: tag-audit viability is part of the frozen
    # first-valid-world criterion — every non-identity permutation must
    # have a nonzero discriminating denominator over the capped program
    # set (otherwise the tag verdict would be INCOMPLETE by construction)
    _perms = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]
    _capped = cf_d3[:min(24, len(cf_d3))]
    _pp = {p: 0 for p in _perms}
    for _seq in _capped:
        for _x in range(17):
            _intact = T.apply_chain(list(_seq), _x)
            for _p in _perms:
                _pseq = [_seq[_p.index(_i)] for _i in range(3)]
                if T.apply_chain(_pseq, _x) != _intact:
                    _pp[_p] += 1
    tag_audit_all_permutations_nonzero = all(v > 0 for v in _pp.values())
    banks = [tuple(sorted(map(tuple, T.SPLIT[k]))) for k in ("train2", "train3", "diag2", "diag3")]
    flat = [s for b in banks for s in b]
    ok_overlap = len(flat) == len(set(flat))
    return {"no_duplicate_primitive_maps": ok_dup,
            "no_identity_primitive": no_identity_primitive,
            "collision_free_d3": len(cf_d3), "collision_free_d2": len(cf_d2),
            "enough_collision_free_d3": len(cf_d3) >= min_collision_free_d3,
            "enough_collision_free_d2": len(cf_d2) >= min_collision_free_d2,
            "banks_disjoint": ok_overlap,
            "primary_vs_train_disjoint": primary_vs_train_disjoint,
            "primary_vs_diag_disjoint": primary_vs_diag_disjoint,
            "all_primary_ops_singleton_trained": all_primary_ops_singleton_trained,
            "tag_audit_all_permutations_nonzero": tag_audit_all_permutations_nonzero,
            "tag_audit_per_perm_discriminating": {str(k): v for k, v in _pp.items()},
            "VALID": (ok_dup and no_identity_primitive
                      and (len(cf_d3) >= min_collision_free_d3)
                      and (len(cf_d2) >= min_collision_free_d2)
                      and ok_overlap and primary_vs_train_disjoint
                      and primary_vs_diag_disjoint
                      and all_primary_ops_singleton_trained
                      and tag_audit_all_permutations_nonzero),
            "cf_d3_programs": [list(s) for s in cf_d3],
            "cf_d2_programs": [list(s) for s in cf_d2]}


def primary_bank(T, checks, values=range(17), rotations=(0, 1, 2, 3)):
    """Collision-free primary evaluation bank: every collision-free held-out
    program x 17 values x diagnostic rotations. Deterministic order."""
    bank = []
    for depth, progs in (("d2", checks["cf_d2_programs"]), ("d3", checks["cf_d3_programs"])):
        for seq in progs:
            for x in values:
                for rot in rotations:
                    bank.append({"depth": depth, "seq": list(seq), "x": x, "rot": rot,
                                 "answer": T.apply_chain(list(seq), x)})
    return bank


def permutation_audit_bank(T, checks, values=range(17),
                           rotations=(0, 1, 2, 3)):
    """G+ marker-permutation audit bank — PREREG §5.4, amended per
    REVIEW round-2 D3: cap = first min(24, N_cf3) collision-free d3
    programs in the frozen deterministic order; ALL five non-identity
    permutations x 17 values x FOUR sealed rotations; non-discriminating
    (intact == permuted answer) cases removed mechanically and counted
    per (program, permutation). Scoring is permutation-stratified: the
    gate value is the unweighted mean of the five permutation-specific
    fidelities; every permutation must have a nonzero denominator."""
    perms = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]
    progs = checks["cf_d3_programs"][:min(24, len(checks["cf_d3_programs"]))]
    bank, excluded = [], {}
    for seq in progs:
        for x in values:
            intact = T.apply_chain(list(seq), x)
            for p in perms:
                pseq = [seq[p.index(i)] for i in range(3)]
                permuted = T.apply_chain(pseq, x)
                if permuted == intact:
                    k = f"{list(seq)}|{list(p)}"
                    excluded[k] = excluded.get(k, 0) + 1
                    continue
                for rot in rotations:
                    bank.append({"seq": list(seq), "x": x, "rot": rot,
                                 "perm": list(p),
                                 "intact_answer": intact,
                                 "counterfactual_answer": permuted})
    per_perm_den = {}
    for c in bank:
        k = str(tuple(c["perm"]))
        per_perm_den[k] = per_perm_den.get(k, 0) + 1
    if any(v == 0 for v in per_perm_den.values()) or len(per_perm_den) != len(perms):
        raise ValueError("a permutation has zero frozen denominator")
    return {"bank": bank,
            "excluded_collisions_by_program_perm": excluded,
            "excluded_collisions": sum(excluded.values()),
            "per_perm_denominators": per_perm_den,
            "n_programs": len(progs), "n_perms": len(perms)}


def monitoring_bank(T, primary, per_depth=(6, 12), values=range(0, 17, 4)):
    """Liveness/monitoring bank — FROZEN RULE (REVIEW round-2):
    monitoring PROGRAMS come from diag2/diag3 ONLY, rendered ONLY with
    QUAL phrasings (matching the in-run probe convention); primary and
    tag-audit programs come from held-out non-diagnostic sets rendered
    with SEALED phrasings. Deterministic: first per_depth programs of
    each diag bank in frozen order, stride-4 start values.
    Disjointness from the primary bank holds by construction
    (structural_checks: primary_vs_diag_disjoint in VALID)."""
    bank = []
    for depth, key, k in (("d2", "diag2", per_depth[0]),
                          ("d3", "diag3", per_depth[1])):
        for seq in [list(s) for s in T.SPLIT[key][:k]]:
            for x in values:
                bank.append({"depth": depth, "seq": seq, "x": x,
                             "phrasing": "qual",
                             "answer": T.apply_chain(seq, x)})
    prim_keys = {(tuple(c["seq"]), c["x"]) for c in primary}
    if {(tuple(c["seq"]), c["x"]) for c in bank} & prim_keys:
        raise ValueError("monitoring bank overlaps primary bank")
    return bank

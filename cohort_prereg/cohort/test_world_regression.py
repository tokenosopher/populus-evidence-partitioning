"""Permanent regression tests — REVIEW round-2 §3.6.

1. stale-module bug: consecutive build_world calls with different seeds
   MUST produce different worlds (the original bug silently rebuilt the
   same world for every candidate seed).
2. seed determinism: A, B, A -> A reproduces exactly, B differs.
3. VALID contains every disjointness component explicitly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cohort.world import build_world, structural_checks  # noqa


def main():
    Ta = build_world(910000, 920000)
    ops_a1 = list(Ta.OPS)
    Tb = build_world(910001, 920001)
    ops_b = list(Tb.OPS)
    assert ops_b != ops_a1, "STALE MODULE: different seeds, same world"
    Ta2 = build_world(910000, 920000)
    assert list(Ta2.OPS) == ops_a1, "seed A not reproducible"
    c = structural_checks(Ta2)
    COMPONENTS = ("no_duplicate_primitive_maps", "no_identity_primitive",
                  "banks_disjoint", "primary_vs_train_disjoint",
                  "primary_vs_diag_disjoint", "enough_collision_free_d3",
                  "enough_collision_free_d2",
                  "all_primary_ops_singleton_trained",
                  "tag_audit_all_permutations_nonzero")
    for k in COMPONENTS:
        assert k in c, f"missing VALID component {k}"
    assert c["VALID"] == all(c[k] for k in COMPONENTS), \
        "VALID not the AND of its components"
    # REVIEW round-3: the code-consumed denylist must equal the exposure
    # manifest aggregate (no independently hand-written second list).
    import json
    from cohort.world import DEV_EXPOSED, candidate_is_dev_exposed
    man = json.load(open(Path(__file__).resolve().parent.parent /
                         "manifests" / "development_seed_exposures.json"))
    for k, v in man["aggregate"].items():
        assert DEV_EXPOSED[k] == frozenset(v), f"denylist drift in {k}"
    ent_ops = {s for e in man["entries"] for s in e["operator_seeds"]}
    assert ent_ops == set(man["aggregate"]["exposed_operator_seeds"]), \
        "aggregate does not cover entries"
    from cohort.world import seed_bundle_is_dev_exposed
    B = {"operator": 1, "split": 2, "grammar": 3, "filler": 4,
         "orders": [5], "initializations": [6], "world_id": "x"}
    assert seed_bundle_is_dev_exposed(B) is None
    assert seed_bundle_is_dev_exposed({**B, "operator": 910000})
    assert seed_bundle_is_dev_exposed({**B, "grammar": 636363})
    assert seed_bundle_is_dev_exposed({**B, "filler": 424242})
    assert seed_bundle_is_dev_exposed({**B, "orders": [950]})
    assert seed_bundle_is_dev_exposed({**B, "initializations": [0]})
    assert seed_bundle_is_dev_exposed({**B, "world_id": "opX_splitY"})
    # role-specificity: 636363 as an OPERATOR seed must NOT reject
    assert seed_bundle_is_dev_exposed({**B, "operator": 636363}) is None
    assert candidate_is_dev_exposed(910000, 1)
    assert candidate_is_dev_exposed(1, 920000)
    assert candidate_is_dev_exposed(424242, 2)
    assert not candidate_is_dev_exposed(123456789, 987654321)
    print("WORLD REGRESSION TESTS: ALL PASS (incl. denylist == manifest)")


if __name__ == "__main__":
    main()

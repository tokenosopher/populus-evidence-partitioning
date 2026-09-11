"""Pre-run manifest validator — PREREG v0.3 §4 completeness gate.

Run BEFORE any training begins. Verifies the populated pre-run manifest
is complete and internally consistent; fails closed on any missing key,
malformed hash, roster/tier inconsistency, or denominator mismatch.
Purely mechanical — consumes no model output and no trained artifact.
"""
import json
import re
import sys

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

TIER_LAYOUT = {"GOLD": (6, 2), "STANDARD": (3, 2), "MINIMUM_DECISIVE": (5, 2)}
CONDITIONS = ("R-", "R+", "G-", "G+", "N+")

REQUIRED_TOP = (
    "world_id", "tier", "seeds", "operator_table",
    "split_sha256", "grammar_sha256", "condition_config_sha256",
    "filler_bank_sha256", "common_semantic_stream_hashes",
    "condition_serialized_stream_hashes",
    "bank_hashes", "bank_sizes", "expected_denominators",
    "evaluator_hashes", "verdict_script_sha256",
    "init_order_map", "designated_trajectory_ids",
    "mechanism_audit_trajectory_ids",
    "freeze_commit", "content_deposit_timestamp",
)
REQUIRED_SEEDS = ("op_seed", "split_seed", "grammar_seed", "filler_seed",
                  "init_seeds", "order_seeds")
REQUIRED_BANKS = ("primary_d2", "primary_d3", "T2", "T3", "S1",
                  "tag_audit", "monitoring")
REQUIRED_EVALS = ("cohort_eval", "cohort_tag_audit", "cohort_society",
                  "result_compiler", "phenotype_classifier", "stats_engine")


class ManifestError(ValueError):
    pass


def _need(cond, msg):
    if not cond:
        raise ManifestError(msg)


def _hex(h, name):
    _need(isinstance(h, str) and _HEX64.match(h), f"bad sha256 for {name}: {h!r}")


def validate(man):
    for k in REQUIRED_TOP:
        _need(k in man, f"missing key {k}")
    tier = man["tier"]
    _need(tier in TIER_LAYOUT, f"unknown tier {tier}")
    n_init, n_order = TIER_LAYOUT[tier]

    seeds = man["seeds"]
    for k in REQUIRED_SEEDS:
        _need(k in seeds, f"missing seed {k}")
    for k in ("op_seed", "split_seed", "grammar_seed", "filler_seed"):
        _need(isinstance(seeds[k], int), f"seed {k} must be int")
    _need(len(seeds["init_seeds"]) == n_init
          and all(isinstance(s, int) for s in seeds["init_seeds"]),
          f"init_seeds must be {n_init} ints")
    _need(len(seeds["order_seeds"]) == n_init
          and all(len(p) == n_order and all(isinstance(s, int) for s in p)
                  for p in seeds["order_seeds"]),
          f"order_seeds must be {n_init} pairs")
    flat_orders = [s for p in seeds["order_seeds"] for s in p]
    _need(len(set(flat_orders)) == len(flat_orders), "duplicate order seeds")
    _need(len(set(seeds["init_seeds"])) == n_init, "duplicate init seeds")

    _need(man["world_id"] ==
          f"op{seeds['op_seed']}_split{seeds['split_seed']}",
          "world_id does not match seeds")

    ops = man["operator_table"]
    _need(len(ops) == 12 and all(len(o) == 2 for o in ops),
          "operator_table must be 12 (a,b) pairs")
    _need(len({tuple(o) for o in ops}) == 12, "duplicate operators")

    for name in ("split_sha256", "grammar_sha256", "condition_config_sha256",
                 "filler_bank_sha256", "verdict_script_sha256"):
        _hex(man[name], name)
    for k in REQUIRED_BANKS:
        _need(k in man["bank_hashes"], f"missing bank hash {k}")
        _hex(man["bank_hashes"][k], f"bank_hashes.{k}")
        _need(k in man["bank_sizes"], f"missing bank size {k}")
        _need(isinstance(man["bank_sizes"][k], int) and man["bank_sizes"][k] > 0,
              f"bad bank size {k}")
    for k in REQUIRED_EVALS:
        _need(k in man["evaluator_hashes"], f"missing evaluator hash {k}")
        _hex(man["evaluator_hashes"][k], f"evaluator_hashes.{k}")

    exp = man["expected_denominators"]
    for k in ("primary_d2", "primary_d3", "T2", "T3", "S1", "tag"):
        _need(isinstance(exp.get(k), int) and exp[k] > 0,
              f"bad expected denominator {k}")
    for k in ("primary_d2", "primary_d3", "T2", "T3", "S1"):
        _need(exp[k] == man["bank_sizes"][k],
              f"expected_denominators.{k} != bank_sizes.{k}")
    _need(exp["tag"] == man["bank_sizes"]["tag_audit"],
          "expected_denominators.tag != bank_sizes.tag_audit")
    _need(exp["T3"] == 150 * 17, "T3 denominator must be 2550")

    iom = man["init_order_map"]
    _need(len(iom) == n_init, f"init_order_map must have {n_init} inits")
    roster = set()
    for entry in iom:
        _need(set(entry) >= {"init_seed", "order_seeds"}, "bad init_order_map entry")
        _need(entry["init_seed"] in seeds["init_seeds"], "unknown init seed in map")
        for o in entry["order_seeds"]:
            roster.add(f"m{entry['init_seed']}_o{o}")
    _need(roster == set(man["designated_trajectory_ids"]),
          "designated_trajectory_ids != init_order_map product")
    _need(len(man["designated_trajectory_ids"]) == n_init * n_order,
          "roster size != tier layout")

    # §6.1 mechanism-audit sample: for R-, G-, N+ the LOWER order seed per
    # init; R+/G+ audit ALL trajectories (dialect matrix) — ids listed too.
    aud = man["mechanism_audit_trajectory_ids"]
    for cond in ("R-", "G-", "N+"):
        _need(cond in aud, f"missing audit sample for {cond}")
        expect = {f"m{e['init_seed']}_o{min(e['order_seeds'])}" for e in iom}
        _need(set(aud[cond]) == expect,
              f"audit sample for {cond} must be the lower order seed per init")
    for cond in ("R+", "G+"):
        _need(cond in aud and set(aud[cond]) == roster,
              f"audit sample for {cond} must be ALL trajectories")

    sem = man["common_semantic_stream_hashes"]
    _need(set(sem) == set(map(str, flat_orders)),
          "semantic stream hashes must cover every order seed")
    for k, v in sem.items():
        _hex(v, f"semantic stream {k}")
    ser = man["condition_serialized_stream_hashes"]
    _need(set(ser) == set(CONDITIONS), "serialized hashes must cover 5 conditions")
    for c, d in ser.items():
        _need(set(d) == set(map(str, flat_orders)),
              f"serialized hashes for {c} must cover every order seed")
        for k, v in d.items():
            _hex(v, f"serialized stream {c}/{k}")
    sems = set(sem.values())
    _need(len(sems) == len(flat_orders), "semantic hashes must differ per order")
    all_ser = [v for d in ser.values() for v in d.values()]
    _need(len(set(all_ser)) == len(all_ser),
          "serialized hashes must be pairwise distinct")
    _need(not (set(all_ser) & sems), "serialized hash equals a semantic hash")

    _hex(man["freeze_commit"], "freeze_commit") if len(man["freeze_commit"]) == 64 \
        else _need(re.match(r"^[0-9a-f]{7,40}$", man["freeze_commit"]),
                   "freeze_commit must be a git sha")
    _need(isinstance(man["content_deposit_timestamp"], str)
          and man["content_deposit_timestamp"],
          "missing content deposit timestamp")
    return {"ok": True, "tier": tier,
            "n_trajectories": len(man["designated_trajectory_ids"]),
            "world_id": man["world_id"]}


if __name__ == "__main__":
    print(json.dumps(validate(json.load(open(sys.argv[1]))), indent=1))

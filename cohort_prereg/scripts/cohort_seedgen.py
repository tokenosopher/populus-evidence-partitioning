"""Freeze ceremony PHASE B — post-freeze seed generation + pre-run manifest.

MUST run only after the phase-A freeze commit exists and its immutable
archive timestamp is recorded (PREREG §4: no seed generation before that
timestamp). Takes the freeze commit sha + archive timestamp as explicit
arguments — refusing to run without them enforces the ordering.

Steps (all deterministic given the frozen seed rule):
  1. derive all seeds (cohort/seed_rule.py);
  2. world candidate search: first candidate passing structural checks is
     BINDING; every rejected candidate + failing check is ledgered;
  3. build + hash grammar JSON (pinned tokenizer), filler bank, all banks;
  4. compute 3-batch admission fingerprints per order seed;
  5. assemble PRE_RUN_MANIFEST.json (stream hashes merged from
     scripts/cohort_stream_hashes.py output), validate with
     cohort/manifest_validator, write.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cohort import seed_rule                                        # noqa
from cohort.manifest_validator import validate                      # noqa

P = argparse.ArgumentParser()
P.add_argument("--freeze-commit", required=True)
P.add_argument("--deposit-timestamp", required=True,
               help="immutable archive timestamp of the phase-A deposit")
P.add_argument("--tokenizer-json", required=True)
P.add_argument("--stream-hashes", required=True,
               help="output of cohort_stream_hashes.py for all 12 order "
                    "seeds at 20000 steps")
P.add_argument("--out-dir", default="results/cohort_frozen")
A = P.parse_args()

OUT = ROOT / A.out_dir
OUT.mkdir(parents=True, exist_ok=True)

from tokenizers import Tokenizer                                    # noqa
CFG = json.load(open(ROOT / "cohort" / "condition_config.json"))
tok = Tokenizer.from_file(A.tokenizer_json)
assert hashlib.sha256(open(A.tokenizer_json, "rb").read()).hexdigest() \
    == CFG["tokenizer"]["tokenizer_json_sha256"], "tokenizer != frozen pin"
tlen = lambda t: len(tok.encode(t, add_special_tokens=False).ids)  # noqa

seeds = seed_rule.all_seeds()
init_seeds = [seeds[f"init_{i}"] for i in range(6)]
order_seeds = [[seeds[f"order_{i}_0"], seeds[f"order_{i}_1"]]
               for i in range(6)]
split_seed = seeds["split"]

# --- world candidate search (§4: first VALID candidate is binding) ---
from cohort.world import (build_world, monitoring_bank,             # noqa
                          permutation_audit_bank, primary_bank,
                          seed_bundle_is_dev_exposed, structural_checks)
ledger = []
world = None
for i in range(200):
    cand = seed_rule.world_candidate_seed(i)
    bundle = {"operator": cand, "split": split_seed,
              "grammar": seeds["grammar"], "filler": seeds["filler"],
              "orders": [o for p in order_seeds for o in p],
              "initializations": init_seeds,
              "world_id": f"op{cand}_split{split_seed}"}
    exposed = seed_bundle_is_dev_exposed(bundle)
    if exposed:
        ledger.append({"candidate_index": i, "op_seed": cand,
                       "rejected": "dev-exposed", "component": exposed})
        continue
    T = build_world(cand, split_seed)
    checks = structural_checks(T)
    entry = {"candidate_index": i, "op_seed": cand,
             "checks": {k: v for k, v in checks.items()
                        if not k.startswith("cf_")}}
    ledger.append(entry)
    if checks["VALID"]:
        world = (cand, T, checks)
        break
(OUT / "world_candidate_ledger.json").write_text(
    json.dumps(ledger, indent=1))
assert world, "no valid world in 200 candidates — STOP, consult referee"
op_seed, T, checks = world
print(f"world: candidate {len(ledger)-1}, op_seed {op_seed}, "
      f"cf_d2 {checks['collision_free_d2']} cf_d3 {checks['collision_free_d3']}")

# --- grammar ---
from cohort.grammar_build import write_grammar                      # noqa
gpath = OUT / "neutral_grammar_FROZEN.json"
gsha = write_grammar(gpath, T.OPS, seeds["grammar"], tlen)
print(f"grammar sha {gsha[:16]}")

# --- filler bank ---
from cohort.filler_constructor import build_bank                    # noqa


class _T:
    def encode(self, text, add_special_tokens=False):
        class R: pass
        r = R(); r.ids = tok.encode(text,
                                    add_special_tokens=add_special_tokens).ids
        return r
    def decode(self, ids): return tok.decode(ids)


fbank, frep = build_bank(_T(), width=CFG["span_widths"]["T_SPAN"])
assert frep["all_exact_width"] and frep["all_leak_free"]
fsha = hashlib.sha256(json.dumps(fbank).encode()).hexdigest()
(OUT / "filler_bank_FROZEN.json").write_text(
    json.dumps({"bank": fbank, "report": frep, "sha256": fsha}, indent=1))

# --- banks ---
pb = primary_bank(T, checks)
ab = permutation_audit_bank(T, checks)
mb = monitoring_bank(T, pb)
p2 = [c for c in pb if c["depth"] == "d2"]
p3 = [c for c in pb if c["depth"] == "d3"]
bh = lambda x: hashlib.sha256(                                      # noqa
    json.dumps(x, sort_keys=True).encode()).hexdigest()
bank_hashes = {"primary_d2": bh(p2), "primary_d3": bh(p3),
               "T2": bh(sorted(map(tuple, T.SPLIT["train2"]))),
               "T3": bh([tuple(s) for s in T.SPLIT["train3"][:150]]),
               "S1": bh({"ops": T.OPS, "positions": [1, 2, 3],
                         "rotations": [0, 1]}),
               "tag_audit": bh(ab["bank"]), "monitoring": bh(mb)}
bank_sizes = {"primary_d2": len(p2), "primary_d3": len(p3),
              "T2": len(T.SPLIT["train2"]) * 17, "T3": 150 * 17,
              "S1": len(T.OPS) * 3 * 17 * 2,
              "tag_audit": len(ab["bank"]), "monitoring": len(mb)}
(OUT / "banks_FROZEN.json").write_text(json.dumps(
    {"primary_d2": p2, "primary_d3": p3, "tag_audit": ab,
     "monitoring": mb, "hashes": bank_hashes, "sizes": bank_sizes},
    indent=1))
print(f"banks: {bank_sizes}")

# --- stream hashes (precomputed) ---
sh = json.load(open(A.stream_hashes))
assert sh["steps"] == 20000 and int(sh["op_seed"]) == op_seed \
    and int(sh["split_seed"]) == split_seed \
    and int(sh["filler_seed"]) == seeds["filler"] \
    and sh["grammar_sha"] == gsha, "stream hashes computed under wrong world"
flat_orders = [o for p in order_seeds for o in p]
assert set(sh["semantic"]) == set(map(str, flat_orders))

# --- evaluator hashes ---
fh = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()  # noqa
manifest = {
    "world_id": f"op{op_seed}_split{split_seed}",
    "tier": "GOLD",
    "seeds": {"op_seed": op_seed, "split_seed": split_seed,
              "grammar_seed": seeds["grammar"],
              "filler_seed": seeds["filler"],
              "init_seeds": init_seeds, "order_seeds": order_seeds},
    "operator_table": [list(o) for o in T.OPS],
    "split_sha256": bh({k: sorted(map(tuple, v)) if isinstance(v, list)
                        else "x" for k, v in T.SPLIT.items()
                        if k in ("train2", "train3", "diag2", "diag3",
                                 "ho2", "ho3_familiar")}),
    "grammar_sha256": gsha,
    "condition_config_sha256": fh("cohort/condition_config.json"),
    "filler_bank_sha256": fsha,
    "common_semantic_stream_hashes": sh["semantic"],
    "condition_serialized_stream_hashes": sh["serialized"],
    "bank_hashes": bank_hashes, "bank_sizes": bank_sizes,
    "tag_audit_per_perm_denominators": ab["per_perm_denominators"],
    "expected_denominators": {"primary_d2": len(p2), "primary_d3": len(p3),
                              "T2": bank_sizes["T2"], "T3": 2550,
                              "S1": bank_sizes["S1"],
                              "tag": len(ab["bank"])},
    "evaluator_hashes": {"cohort_eval": fh("scripts/cohort_eval.py"),
                         "cohort_tag_audit": fh("scripts/cohort_tag_audit.py"),
                         "cohort_society": fh("scripts/cohort_society.py"),
                         "result_compiler": fh("cohort/result_compiler.py"),
                         "phenotype_classifier": fh("cohort/phenotype_classifier.py"),
                         "stats_engine": fh("cohort/stats_engine.py")},
    "verdict_script_sha256": fh("cohort/verdict.py"),
    "init_order_map": [{"init_seed": i, "order_seeds": o}
                       for i, o in zip(init_seeds, order_seeds)],
    "designated_trajectory_ids": [f"m{i}_o{o}" for i, p in
                                  zip(init_seeds, order_seeds) for o in p],
    "mechanism_audit_trajectory_ids": {
        **{c: [f"m{i}_o{min(p)}" for i, p in zip(init_seeds, order_seeds)]
           for c in ("R-", "G-", "N+")},
        **{c: [f"m{i}_o{o}" for i, p in zip(init_seeds, order_seeds)
               for o in p] for c in ("R+", "G+")}},
    "freeze_commit": A.freeze_commit,
    "content_deposit_timestamp": A.deposit_timestamp,
}
print("validator:", validate(manifest))
mpath = OUT / "PRE_RUN_MANIFEST.json"
mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))
print(f"WROTE {mpath}")
print(f"manifest sha256: {hashlib.sha256(mpath.read_bytes()).hexdigest()}")
print("NEXT: commit + push manifest to evidence repo, archive, THEN train.")

"""Freeze ceremony PHASE A — scientific-content hash inventory.

Produces cohort/FREEZE_HASHES.json: SHA-256 of every frozen scientific
artifact. After this, the operator (or agent, with user authorization):
  1. commits this repo (the freeze commit),
  2. copies the frozen set + prereg into the public evidence repo
     (~/code/populus-release), commits, pushes,
  3. triggers Software Heritage save-code-now on the evidence repo and
     records the request timestamp + eventual SWHID in a pointer commit,
  4. only THEN runs phase B (scripts/cohort_seedgen.py <freeze-commit>).
No seed is generated in phase A. Rerunning is safe (pure hashing).
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FROZEN = [
    "PREREG_ROLE_MARKED_COHORT.md",
    "TIER_DECLARATION.json",
    "cohort/condition_config.json",
    "cohort/condition_layout.py",
    "cohort/torch_package.py",
    "cohort/sampler.py",
    "cohort/filler_constructor.py",
    "cohort/neutral_grammar.py",
    "cohort/grammar_build.py",
    "cohort/marker_filler.py",
    "cohort/world.py",
    "cohort/seed_rule.py",
    "cohort/phenotype_classifier.py",
    "cohort/verdict.py",
    "cohort/test_verdict.py",
    "cohort/stats_engine.py",
    "cohort/result_compiler.py",
    "cohort/manifest_validator.py",
    "scripts/cohort_society.py",
    "scripts/cohort_eval.py",
    "scripts/cohort_tag_audit.py",
    "scripts/cohort_admission_fixture.py",
    "scripts/cohort_stream_hashes.py",
    "scripts/cohort_mechanism_audit.py",
    "scripts/cohort_dialect_matrix.py",
    "scripts/cohort_dialect_aggregate.py",
    "manifests/development_seed_exposures.json",
    "scripts/cohort_mask_admission.py",
    "scripts/cohort_verdict_lock.py",
    "cohort/test_world_regression.py",
    "populus/bridge.py",
    "populus/bridge_qwen.py",
    "populus/bridge_train.py",
    "populus/bridge_tasks.py",
    "results/bridge/serialization_contract.json",
    "results/bridge/label_context.json",
    "results/bridge/genome_pin.json",
]


def main():
    inv = {}
    for rel in FROZEN:
        p = ROOT / rel
        if not p.exists():
            sys.exit(f"MISSING FROZEN FILE: {rel}")
        inv[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "head_at_inventory": git,
           "worktree_dirty_at_inventory": bool(dirty),
           "files": inv,
           "inventory_sha256": hashlib.sha256(json.dumps(
               inv, sort_keys=True).encode()).hexdigest()}
    (ROOT / "cohort" / "FREEZE_HASHES.json").write_text(
        json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in out.items() if k != "files"},
                     indent=1))
    print(f"{len(inv)} files hashed -> cohort/FREEZE_HASHES.json")
    if dirty:
        print("WARNING: worktree dirty — commit before treating this "
              "as the freeze inventory.")


if __name__ == "__main__":
    main()

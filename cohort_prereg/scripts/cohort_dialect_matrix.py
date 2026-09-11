"""§6.2 dialect-matrix COORDINATOR — REVIEW round-3 blocker 2.

For one matched R+/G+ twin pair, guarantees the complete 2x2 matrix:
  (donor=R+, recipient=R+)  self
  (donor=G+, recipient=G+)  self
  (donor=R+, recipient=G+)  cross
  (donor=G+, recipient=R+)  cross
by invoking the frozen cell evaluator (scripts/cohort_mechanism_audit.py)
four times, then asserting the COMMON-CASE discipline: for each
recipient, the self cell and the cross cell must report IDENTICAL
eligible-case-id hashes per interface (they share the recipient's
intact-correct set; donor availability differences would surface here
and abort rather than silently changing denominators). Emits a matrix
bundle consumed by scripts/cohort_dialect_aggregate.py.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

P = argparse.ArgumentParser()
P.add_argument("--ckpt-rplus", required=True)
P.add_argument("--ckpt-gplus", required=True)
P.add_argument("--device", default="cuda")
P.add_argument("--out-dir", required=True)
A = P.parse_args()

outdir = Path(A.out_dir)
outdir.mkdir(parents=True, exist_ok=True)
# PRE-PASS (round-3 option A): harvest both donors' value coverage,
# intersect per interface, freeze the intersection for every cell so
# each recipient's self and cross cells share an IDENTICAL case set.
cov_paths = {}
for tag, ck in (("R", A.ckpt_rplus), ("G", A.ckpt_gplus)):
    cp = outdir / f"coverage_{tag}.json"
    cov_paths[tag] = cp
    if not cp.exists():
        r = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts/cohort_mechanism_audit.py"),
             "--donor-ckpt", ck, "--recipient-ckpt", ck,
             "--device", A.device, "--emit-coverage",
             "--out", str(cp)])
        if r.returncode != 0:
            sys.exit(f"coverage harvest {tag} FAILED")
covs = {t: json.load(open(p))["coverage"] for t, p in cov_paths.items()}
inter = {k: sorted(set(covs["R"][k]) & set(covs["G"][k]))
         for k in ("0", "1", "2")}
inter_path = outdir / "coverage_intersection.json"
inter_path.write_text(json.dumps({"coverage": inter,
                                  "sources": {t: str(p) for t, p in
                                              cov_paths.items()}},
                                 indent=1))
print("coverage intersection per iface:",
      {k: len(v) for k, v in inter.items()})
if any(len(v) < 2 for v in inter.values()):
    # round-4: mark, never abort the primary verdict or trigger any
    # replacement world/model/donor rule
    (outdir / "MATRIX_NOT_VIABLE").write_text(json.dumps(
        {"reason": "coverage intersection too small",
         "intersection_sizes": {k: len(v) for k, v in inter.items()}}))
    print("MATRIX_NOT_VIABLE — marker written; secondary matrix "
          "reported as not viable; primary cohort verdict unaffected")
    sys.exit(0)

CELLS = [("self_R", A.ckpt_rplus, A.ckpt_rplus),
         ("self_G", A.ckpt_gplus, A.ckpt_gplus),
         ("cross_RtoG", A.ckpt_rplus, A.ckpt_gplus),
         ("cross_GtoR", A.ckpt_gplus, A.ckpt_rplus)]
paths = {}
for name, donor, recipient in CELLS:
    out = outdir / f"cell_{name}.json"
    paths[name] = out
    if out.exists():
        print(f"cell {name}: exists, skipping")
        continue
    cmd = [sys.executable, str(ROOT / "scripts/cohort_mechanism_audit.py"),
           "--donor-ckpt", donor, "--recipient-ckpt", recipient,
           "--device", A.device, "--coverage-file", str(inter_path),
           "--out", str(out)]
    print(f"cell {name}: running", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"cell {name} FAILED")

cells = {n: json.load(open(p)) for n, p in paths.items()}
# common-case discipline: per recipient, self and cross eligible sets
# must be identical per interface
pairs = [("self_R", "cross_GtoR"), ("self_G", "cross_RtoG")]
for self_name, cross_name in pairs:
    a, b = cells[self_name], cells[cross_name]
    assert a["recipient"] == b["recipient"], (self_name, cross_name)
    for k in ("0", "1", "2"):
        ha = a["eligible_case_ids_sha256"][k]
        hb = b["eligible_case_ids_sha256"][k]
        if ha != hb:
            sys.exit(f"COMMON-CASE VIOLATION recipient="
                     f"{a['recipient']} iface {k}: {ha[:12]} != {hb[:12]}"
                     " — donor availability altered the case set; the"
                     " matrix is invalid under the frozen discipline")
print("common-case discipline: PASS (both recipients, all interfaces)")
import hashlib
def fsha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
bundle = {"pair": {"R+": cells["self_R"]["recipient"],
                   "G+": cells["self_G"]["recipient"]},
          "cells": {n: str(p) for n, p in paths.items()},
          "bank_sha256": cells["self_R"]["bank_sha256"],
          "coverage_provenance": {
              "pre_intersection_maps": {t: {"path": str(p),
                                            "sha256": fsha(p)}
                                        for t, p in cov_paths.items()},
              "intersection": {"path": str(inter_path),
                               "sha256": fsha(inter_path)}}}
(outdir / "matrix_bundle.json").write_text(json.dumps(bundle, indent=1))
print(f"WROTE {outdir}/matrix_bundle.json")

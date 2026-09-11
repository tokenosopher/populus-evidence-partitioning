"""Canonical result compiler — the single trusted path from raw per-case
evaluation outcomes to the JSON the verdict/stats/classifier consume.
Build-review requirement: without this, the decision scripts judge whatever
numbers they are handed but do not establish those numbers are the frozen
experiment's outcomes.

Responsibilities (all before any verdict):
  - read the hashed pre-run manifest's exact designated roster;
  - verify every raw record's condition/init/order/world/step identity and
    its bank hash against the manifest;
  - compute A2, A3, allcut3, T2, T3, S1 and intact/cf tag rates from INTEGER
    numerator/denominator pairs (never float inputs);
  - verify expected denominators; reject duplicates, extras, missing, NaN;
  - derive C1..C6 via the frozen sign map and verify the per-trajectory
    identity A_d(R+)-A_d(G+) == [A_d(R+)-A_d(N+)] + [A_d(N+)-A_d(G+)];
  - emit a canonical bundle + its SHA-256.
"""
import hashlib
import json
import sys

CONDITIONS = ["R-", "R+", "G-", "G+", "N+"]


class CompileError(Exception):
    pass


def _rate(num, den):
    if not (isinstance(num, int) and isinstance(den, int)):
        raise CompileError(f"non-integer count {num!r}/{den!r}")
    if den <= 0 or not (0 <= num <= den):
        raise CompileError(f"bad count {num}/{den}")
    return num / den


def compile_bundle(manifest, raw):
    tier = manifest["tier"]
    roster = list(manifest["designated_trajectory_ids"])
    exp = manifest["expected_denominators"]      # {"primary_d2":N,"primary_d3":N,"T2":N,"T3":150,"S1":N,"tag":N}
    # allcut_d3 runs on the SAME primary d3 bank -> same denominator
    world_id = manifest["world_id"]
    arms = {c: {} for c in CONDITIONS}
    tag = {}
    seen = set()
    for rec in raw["records"]:
        cond, tid = rec["condition"], rec["trajectory_id"]
        if cond not in CONDITIONS:
            raise CompileError(f"unknown condition {cond}")
        if tid not in roster:
            raise CompileError(f"undesignated trajectory {tid}")
        if (cond, tid) in seen:
            raise CompileError(f"duplicate record {(cond, tid)}")
        seen.add((cond, tid))
        if rec["world_id"] != world_id or rec["step"] != 20000:
            raise CompileError(f"identity mismatch {(cond, tid)}")
        for field, den_key in (("primary_d2", "primary_d2"), ("primary_d3", "primary_d3"),
                               ("allcut_d3", "primary_d3"), ("T2", "T2"), ("T3", "T3"), ("S1", "S1")):
            if rec[field]["den"] != exp[den_key]:
                raise CompileError(f"denominator mismatch {field} {(cond,tid)}: "
                                   f"{rec[field]['den']} != {exp[den_key]}")
        arms[cond][tid] = {
            "A2": _rate(rec["primary_d2"]["num"], rec["primary_d2"]["den"]),
            "A3": _rate(rec["primary_d3"]["num"], rec["primary_d3"]["den"]),
            "allcut3": _rate(rec["allcut_d3"]["num"], rec["allcut_d3"]["den"]),
            "T2": _rate(rec["T2"]["num"], rec["T2"]["den"]),
            "T3": _rate(rec["T3"]["num"], rec["T3"]["den"]),
            "S1": _rate(rec["S1"]["num"], rec["S1"]["den"]),
        }
    for rec in raw.get("tag_records", []):
        tid = rec["trajectory_id"]
        if tid not in roster:
            raise CompileError(f"tag rec undesignated {tid}")
        if rec["bank_sha256"] != manifest["bank_hashes"]["tag_audit"]:
            raise CompileError(f"tag bank hash mismatch for {tid}")
        per = rec.get("per_perm")
        if not per or len(per) != 5:
            raise CompileError(f"tag record missing 5-permutation strata {tid}")
        tot = 0
        for k, v in per.items():
            if not (isinstance(v.get("den"), int) and v["den"] > 0):
                raise CompileError(f"zero/invalid perm denominator {tid} {k}")
            _rate(v["intact_num"], v["den"]); _rate(v["cf_num"], v["den"])
            tot += v["den"]
        if tot != exp["tag"]:
            raise CompileError(f"tag denominators sum {tot} != {exp['tag']} {tid}")
        mpp = manifest.get("tag_audit_per_perm_denominators")
        if mpp is not None and {k: v["den"] for k, v in per.items()} != mpp:
            raise CompileError(f"per-perm denominators differ from manifest {tid}")
        # stratified means recomputed HERE from integer counts
        # (REVIEW round-2 D3: unweighted mean of permutation fidelities)
        tag[tid] = {"intact": sum(_rate(v["intact_num"], v["den"])
                                  for v in per.values()) / len(per),
                    "cf_track": sum(_rate(v["cf_num"], v["den"])
                                    for v in per.values()) / len(per),
                    "n_total": tot,
                    "per_perm": {k: {"intact": _rate(v["intact_num"], v["den"]),
                                     "cf": _rate(v["cf_num"], v["den"]),
                                     "den": v["den"]} for k, v in per.items()}}

    # identity check C1 == C4 + C5 per trajectory (where all three arms present)
    for tid in roster:
        rp, gp, npl = arms["R+"].get(tid), arms["G+"].get(tid), arms["N+"].get(tid)
        if rp and gp and npl:
            for d in ("A2", "A3"):
                c1 = rp[d] - gp[d]
                dec = (rp[d] - npl[d]) + (npl[d] - gp[d])
                if abs(c1 - dec) > 1e-9:
                    raise CompileError(f"identity break {tid} {d}: {c1} != {dec}")

    bundle = {"tier": tier, "world_id": world_id, "arms": arms, "gplus_tag_audit": tag,
              "roster": roster}
    blob = json.dumps(bundle, sort_keys=True, allow_nan=False).encode()
    bundle["bundle_sha256"] = hashlib.sha256(blob).hexdigest()
    return bundle


if __name__ == "__main__":
    manifest = json.load(open(sys.argv[1]))
    raw = json.load(open(sys.argv[2]))
    print(json.dumps(compile_bundle(manifest, raw), indent=1, allow_nan=False))

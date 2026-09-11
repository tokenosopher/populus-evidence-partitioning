"""Frozen cohort verdict script — PREREG sections 5.2-5.5, hardened per build review.

Consumes (1) the hashed pre-run manifest (fixes the exact designated trajectory
roster) and (2) a canonical results bundle. Iterates ONLY over the designated
roster: missing/null -> INCOMPLETE; extra/duplicate/wrong-identity -> INPUT_ERROR;
non-finite or out-of-range -> INPUT_ERROR. INCOMPLETE is a true third state
(pass value null), never collapsed to False. No omnibus pass. Runs once.
"""
import json
import math
import statistics
import sys

TIER_N = {"GOLD": 12, "STANDARD": 6, "MINIMUM_DECISIVE": 10}
DIRECTIONAL = {"C1": ("R+", "G+"), "C2": ("G+", "G-"),
               "C4": ("R+", "N+"), "C5": ("N+", "G+"), "C6": ("R-", "G-")}


class InputError(Exception):
    pass


def _finite01(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x) and 0.0 <= x <= 1.0


def _cell(arm_map, tid, fields):
    """Fetch one designated trajectory's record; None if absent (INCOMPLETE),
    raise on malformed values (INPUT_ERROR)."""
    rec = arm_map.get(tid)
    if rec is None:
        return None
    for f in fields:
        if f not in rec:
            raise InputError(f"missing field {f} for {tid}")
        if not _finite01(rec[f]):
            raise InputError(f"non-finite/out-of-range {f}={rec[f]!r} for {tid}")
    return rec


def _roster_check(arm_map, roster, name):
    extra = set(arm_map) - set(roster)
    if extra:
        raise InputError(f"{name}: undesignated trajectories present: {sorted(extra)}")


def directional_pass(adv_name, oth_name, arms, roster, n):
    adv, oth = arms.get(adv_name, {}), arms.get(oth_name, {})
    _roster_check(adv, roster, adv_name); _roster_check(oth, roster, oth_name)
    d2, d3, aA2, aA3, aCut = [], [], [], [], []
    present = 0
    for tid in roster:
        a = _cell(adv, tid, ("A2", "A3", "allcut3"))
        o = _cell(oth, tid, ("A2", "A3"))
        if a is None or o is None:
            continue
        present += 1
        d2.append(a["A2"] - o["A2"]); d3.append(a["A3"] - o["A3"])
        aA2.append(a["A2"]); aA3.append(a["A3"]); aCut.append(a["allcut3"])
    if present < n:
        return {"status": "INCOMPLETE", "pass": None, "n_present": present, "n_required": n}
    c_margin = sum(1 for x, y in zip(d2, d3) if x >= 0.20 and y >= 0.20)
    reverse = sum(1 for x, y in zip(d2, d3) if x <= -0.20 and y <= -0.20)
    c_median = statistics.median(d2) >= 0.25 and statistics.median(d3) >= 0.25
    c_floor = statistics.median(aA3) >= 0.70
    c_norun = all(a2 >= 0.40 and a3 >= 0.40 for a2, a3 in zip(aA2, aA3))
    c_comm = sum(1 for c in aCut if c <= 0.11) >= (n - 1)
    ok = (c_margin >= n - 1) and c_median and c_floor and c_norun and c_comm
    return {"status": "PASS" if ok else "FAIL", "pass": bool(ok),
            "components": {"margin_ge_0.20_both": c_margin, "n_required": n - 1,
                           "reverse_direction_count": reverse,
                           "median_d2": statistics.median(d2), "median_d3": statistics.median(d3),
                           "median_ge_0.25_both": c_median,
                           "adv_median_d3": statistics.median(aA3), "adv_median_d3_ge_0.70": c_floor,
                           "no_adv_run_below_0.40_either": c_norun,
                           "commdep_allcut_le_0.11_count": c_comm},
            "signed_differences": {tid: {"d2": a["A2"]-o["A2"], "d3": a["A3"]-o["A3"]}
                                   for tid in roster
                                   for a in [adv.get(tid)] for o in [oth.get(tid)] if a and o}}


def practical_null(a_name, b_name, arms, roster, n, label):
    a, b = arms.get(a_name, {}), arms.get(b_name, {})
    _roster_check(a, roster, a_name); _roster_check(b, roster, b_name)
    rows = []
    for tid in roster:
        ra = _cell(a, tid, ("A2", "A3")); rb = _cell(b, tid, ("A2", "A3"))
        if ra is None or rb is None:
            continue
        rows.append((ra, rb))
    if len(rows) < n:
        return {"status": "INCOMPLETE", "label": label, "n_present": len(rows), "n_required": n}
    competent = (statistics.median([r[0]["A3"] for r in rows]) >= 0.70 and
                 statistics.median([r[1]["A3"] for r in rows]) >= 0.70)
    no_low = all(r[s]["A2"] >= 0.40 and r[s]["A3"] >= 0.40 for r in rows for s in (0, 1))
    ad2 = [abs(r[0]["A2"] - r[1]["A2"]) for r in rows]
    ad3 = [abs(r[0]["A3"] - r[1]["A3"]) for r in rows]
    signed_med = {"d2": statistics.median([r[0]["A2"] - r[1]["A2"] for r in rows]),
                  "d3": statistics.median([r[0]["A3"] - r[1]["A3"] for r in rows])}
    within = sum(1 for x, y in zip(ad2, ad3) if x <= 0.20 and y <= 0.20)
    med_ok = statistics.median(ad2) <= 0.10 and statistics.median(ad3) <= 0.10
    if not (competent and no_low):
        return {"status": "UNINFORMATIVE_incompetent", "label": label,
                "signed_median": signed_med, "median_abs_d2": statistics.median(ad2),
                "median_abs_d3": statistics.median(ad3)}
    ok = (within >= n - 1) and med_ok
    return {"status": "PRACTICAL_NULL" if ok else "SIGNED_EFFECT_NO_EQUIVALENCE",
            "label": label, "within_bound_count": within, "n_required": n - 1,
            "signed_median": signed_med,
            "median_abs_d2": statistics.median(ad2), "median_abs_d3": statistics.median(ad3)}


def gplus_tag_use(audit, roster, n):
    if audit is None:
        return {"status": "INCOMPLETE", "pass": None, "reason": "no audit block"}
    _roster_check(audit, roster, "gplus_tag_audit")
    good, present = 0, 0
    for tid in roster:
        rec = audit.get(tid)
        if rec is None:
            continue
        for f in ("intact", "cf_track", "n_total"):
            if f not in rec:
                raise InputError(f"tag audit missing {f} for {tid}")
        if not (_finite01(rec["intact"]) and _finite01(rec["cf_track"])):
            raise InputError(f"tag audit non-finite for {tid}")
        present += 1
        if rec["intact"] >= 0.90 and rec["cf_track"] >= 0.90:
            good += 1
    if present < n:
        return {"status": "INCOMPLETE", "pass": None, "n_present": present, "n_required": n}
    ok = good >= (n - 1)
    return {"status": "PASS" if ok else "FAIL", "pass": bool(ok),
            "n_marker_following": good, "n_required": n - 1}


def _combine(*statuses):
    if any(s == "INCOMPLETE" for s in statuses):
        return {"status": "INCOMPLETE", "pass": None}
    if all(s == "PASS" for s in statuses):
        return {"status": "PASS", "pass": True}
    return {"status": "FAIL", "pass": False}


def run(manifest, results):
    tier = manifest["tier"]
    n = TIER_N[tier]
    roster = tuple(manifest["designated_trajectory_ids"])
    if len(roster) != n or len(set(roster)) != n:
        raise InputError(f"roster size {len(roster)} != tier n {n} or has duplicates")
    if results.get("tier") != tier:
        raise InputError("results tier disagrees with manifest tier")
    arms = results["arms"]
    out = {"tier": tier, "n": n, "roster_size": len(roster)}
    contrasts = ["C1"] if tier == "MINIMUM_DECISIVE" else list(DIRECTIONAL)
    dec = {c: directional_pass(*DIRECTIONAL[c], arms, roster, n) for c in contrasts}
    out["directional"] = dec
    c1 = dec["C1"]
    tag = gplus_tag_use(results.get("gplus_tag_audit"), roster, n)
    out["gplus_tag_audit"] = tag
    out["C1_BEHAVIORAL"] = {"status": c1["status"], "pass": c1.get("pass")}
    out["GPLUS_TAG_USE"] = {"status": tag["status"], "pass": tag.get("pass")}
    out["PRIMARY_ROLE_EQUALIZED"] = _combine(c1["status"], tag["status"])
    if tier != "MINIMUM_DECISIVE":
        out["C3_PRACTICAL_REDUNDANCY"] = practical_null("R+", "R-", arms, roster, n, "C3")
        out["C4_PRACTICAL_NULL"] = practical_null("R+", "N+", arms, roster, n, "C4")
        out["C5_PRACTICAL_NULL"] = practical_null("N+", "G+", arms, roster, n, "C5")
    return out


if __name__ == "__main__":
    manifest = json.load(open(sys.argv[1]))
    results = json.load(open(sys.argv[2]), parse_constant=lambda c: (_ for _ in ()).throw(
        InputError(f"non-finite JSON constant {c}")))
    print(json.dumps(run(manifest, results), indent=1, allow_nan=False))

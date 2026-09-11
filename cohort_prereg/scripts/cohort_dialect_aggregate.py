"""§6.2 dialect-matrix AGGREGATOR — REVIEW round-4 amended indicators.

Recipient-normalized ratios rho_k(D->R,i) = p_k(D->R,i)/p_k(R->R,i) for
preservation (same) and counterfactual following (cf_follow) separately;
raw numerator + recipient-self denominator always shown; zero self-
denominator -> "undefined"; never clipped.

FROZEN indicators (round-4 §1.3, per recipient-interface):
  strong_within_regime_canonicality : self same >= 0.90 AND
                                      self cf_follow >= 0.90
  partial_within_regime_canonicality: both >= 0.60 AND NOT strong
  operational_context_entanglement  : self same < 0.60 AND
                                      deletion <= 0.70 on the success-
                                      conditioned eligible set (whose
                                      intact baseline is 1.000 by
                                      construction, so this is a >=0.30
                                      deletion loss); does NOT uniquely
                                      identify context entanglement as
                                      the only possible mechanism
  cross_regime_interoperability     : in BOTH donor directions ALL of:
                                      self same >= 0.60; self cf_follow
                                      >= 0.60; raw cross same >= 0.50;
                                      raw cross cf_follow >= 0.50;
                                      rho_preservation defined >= 0.50;
                                      rho_cf_follow defined >= 0.50
Strata with n_eligible < 30 emit status LOW_DENOMINATOR with raw
metrics only — NO positive or negative indicator assignment.
"""
import argparse
import json
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument("bundle")
P.add_argument("--out", required=True)
A = P.parse_args()

bundle = json.load(open(A.bundle))
cells = {n: json.load(open(p)) for n, p in bundle["cells"].items()}
EX, PA, XI = 0.90, 0.60, 0.50
CE, DEL_CEIL = 0.60, 0.70
LOW = 30


def rate(cell, k, cond):
    s = cell["summary"][k][cond]
    return s["rate"], s["num"], s["den"]


def ratio(num_rate, den_rate):
    if den_rate in (None, 0):
        return "undefined"
    return round(num_rate / den_rate, 6)


out = {"pair": bundle["pair"], "bank_sha256": bundle["bank_sha256"],
       "thresholds": {"strong": EX, "partial": PA,
                      "entangle_self_same_lt": CE,
                      "entangle_deletion_le": DEL_CEIL,
                      "interop": XI, "low_denominator": LOW},
       "recipients": {}}
for rec_label, self_name, cross_name in (
        ("R+", "self_R", "cross_GtoR"), ("G+", "self_G", "cross_RtoG")):
    selfc, crossc = cells[self_name], cells[cross_name]
    r = {"trajectory": selfc["recipient"], "interfaces": {}}
    for k in ("0", "1", "2"):
        s_same, s_same_n, s_same_d = rate(selfc, k, "same")
        s_cf, s_cf_n, s_cf_d = rate(selfc, k, "cf_follow")
        x_same, x_same_n, x_same_d = rate(crossc, k, "same")
        x_cf, x_cf_n, x_cf_d = rate(crossc, k, "cf_follow")
        dele, _, _ = rate(selfc, k, "deletion")
        entry = {
            "self_same": {"rate": s_same, "num": s_same_n,
                          "den": s_same_d},
            "self_cf_follow": {"rate": s_cf, "num": s_cf_n,
                               "den": s_cf_d},
            "cross_same": {"rate": x_same, "num": x_same_n,
                           "den": x_same_d},
            "cross_cf_follow": {"rate": x_cf, "num": x_cf_n,
                                "den": x_cf_d},
            "self_deletion": {"rate": dele},
            "rho_preservation": ratio(x_same, s_same if s_same_d else 0),
            "rho_cf_follow": ratio(x_cf, s_cf if s_cf_d else 0)}
        if s_same_d < LOW or x_same_d < LOW:
            entry["status"] = "LOW_DENOMINATOR"
            entry["indicators"] = None
        else:
            strong = (s_same is not None and s_cf is not None
                      and s_same >= EX and s_cf >= EX)
            partial = (not strong and s_same is not None
                       and s_cf is not None
                       and s_same >= PA and s_cf >= PA)
            entangled = (s_same is not None and s_same < CE
                         and dele is not None and dele <= DEL_CEIL)
            entry["status"] = "OK"
            entry["indicators"] = {
                "strong_within_regime_canonicality": strong,
                "partial_within_regime_canonicality": partial,
                "operational_context_entanglement": entangled}
        r["interfaces"][k] = entry
    out["recipients"][rec_label] = r

# interoperability: full six-condition conjunction, BOTH directions,
# only where both strata are status OK
for k in ("0", "1", "2"):
    entries = [out["recipients"][lab]["interfaces"][k]
               for lab in ("R+", "G+")]
    if any(e["status"] != "OK" for e in entries):
        interop = None
    else:
        interop = True
        for e in entries:
            interop = interop and (
                e["self_same"]["rate"] is not None
                and e["self_same"]["rate"] >= PA
                and e["self_cf_follow"]["rate"] is not None
                and e["self_cf_follow"]["rate"] >= PA
                and e["cross_same"]["rate"] is not None
                and e["cross_same"]["rate"] >= XI
                and e["cross_cf_follow"]["rate"] is not None
                and e["cross_cf_follow"]["rate"] >= XI
                and isinstance(e["rho_preservation"], float)
                and e["rho_preservation"] >= XI
                and isinstance(e["rho_cf_follow"], float)
                and e["rho_cf_follow"] >= XI)
    for e in entries:
        if e["indicators"] is not None:
            e["indicators"]["cross_regime_interoperability"] = interop

Path(A.out).write_text(json.dumps(out, indent=1))
print(f"WROTE {A.out}")

"""Cohort mechanism/dialect evaluator — PREREG v0.4 §6.1 + §6.2.

One script, two modes:
  --donor-ckpt == --recipient-ckpt : §6.1 SELF mechanism audit
  --donor-ckpt != --recipient-ckpt : one §6.2 cross-dialect matrix cell
Battery per (recipient, interface, base-correct episode) — the COMPLETE
preregistered §6.1 menu (round-3 blocker 2):
  same          donor packet carrying the episode's running value v
  cf_follow     donor packet carrying v'=(v+3) mod 17, scored against
                the suffix-implied counterfactual target
  shift_retain  the same v' packet scored against the original answer
  deranged      donor packet with random value w != v (retention)
  deletion      zero packet (donor-independent)
  noise         approximately norm-matched Gaussian packet: direction
                from a counter-based PRF stream (SHA-256 of
                "noise:<world>:<bank_idx>:<iface>" seeding a torch
                Generator), scaled PER PACKET to the intact packet's
                Frobenius norm; scored against the original answer
  span_rewrite  TEXT-level control MATCHED to the packet counterfactual
                (REVIEW round-4 §1.2): with F_k the affine prefix map
                through the sender at interface k, the start-value span
                is rewritten to x_cf = F_k^{-1}((F_k(x)+3) mod 17) —
                operators, phrasings, masks, and native packet
                generation unchanged. The evaluator ASSERTS
                F_k(x_cf) == v_cf and that the rewritten episode's
                final target equals the suffix-implied packet-
                counterfactual target; any failure aborts the record.
                At interface 0 this reduces to x+3; later it does not.
Common-case discipline (round-3): the per-interface eligible case set is
  intact-correct  AND  same/cf/deranged donors available
computed ONCE; every condition scores exactly this set; its case-id
hash is emitted so §6.2 self and cross cells can assert identical
denominators before compatibility ratios are formed.
Fixed sealed episode bank (PREREG §6.1: never adaptively enlarged): the
tag-audit program set (first min(24, N_cf3) collision-free depth-three
programs, frozen order), all 17 values, both sealed rotations {0, 1}.
Donor packets harvested under the donor's native regime; recipients
evaluated under their native regime (packaging = frozen cohort
torch_package with the trajectory's own condition; N+ fillers via the
frozen PRF schedule with ordinal = episode index in bank order).
Success-conditioning and denominators are reported per §6.1; estimates
with fewer than 30 eligible recipient cases are labelled low-denominator
descriptive. Per-case predictions persisted (referee mandate)."""
import argparse
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from cohort.torch_package import select_fillers, torch_package     # noqa
from cohort.filler_constructor import build_length_banks           # noqa
from cohort.neutral_grammar import (FORWARD_SPAN_NEUTRAL,          # noqa
                                    VALUE_SPAN_NEUTRAL)

P = argparse.ArgumentParser()
P.add_argument("--donor-ckpt", required=True)
P.add_argument("--recipient-ckpt", required=True)
P.add_argument("--device", default="cuda")
P.add_argument("--out", required=True)
P.add_argument("--emit-coverage", action="store_true",
               help="harvest the donor only; write its per-interface "
                    "value-coverage map to --out and exit")
P.add_argument("--coverage-file", default=None,
               help="frozen per-interface value coverage (intersection "
                    "of all donors in the matrix); restricts "
                    "eligibility so every matrix cell for a recipient "
                    "uses the IDENTICAL case set (round-3 option A)")
A = P.parse_args()

for _v in ("BRIDGE_OP_SEED", "BRIDGE_SPLIT_SEED",
           "COHORT_GRAMMAR_PATH", "COHORT_FILLER_SEED"):
    assert os.environ.get(_v), f"missing env {_v}"

from populus.bridge import BridgeA1                                # noqa
from populus.bridge_qwen import QwenGenome                         # noqa
from cohort.world import build_world, structural_checks            # noqa

DEV = A.device
C = json.load(open(ROOT / "results/bridge/serialization_contract.json"))
Tq, Ts = C["T_QUESTION"], C["T_SPAN"]
CFG = json.load(open(ROOT / "cohort" / "condition_config.json"))
HDR_IDS = {k: list(map(int, CFG["header_token_blocks"][k]))
           for k in ("mine", "other", "slot")}
GRAMMAR = json.loads(open(os.environ["COHORT_GRAMMAR_PATH"], "rb").read())
SEALED_PHR = {int(k): v["sealed"] for k, v in GRAMMAR["ops"].items()}
FILLER_SEED = int(os.environ["COHORT_FILLER_SEED"])
WORLD_ID = (f"op{os.environ['BRIDGE_OP_SEED']}"
            f"_split{os.environ['BRIDGE_SPLIT_SEED']}")
CF_OFFSET = 3
MIN_PER = 4
LOW_DENOM = 30

T = build_world(int(os.environ["BRIDGE_OP_SEED"]),
                int(os.environ["BRIDGE_SPLIT_SEED"]))
CHECKS = structural_checks(T)
assert CHECKS["VALID"]
PROGS = [tuple(s) for s in
         CHECKS["cf_d3_programs"][:min(24, len(CHECKS["cf_d3_programs"]))]]
# fixed sealed episode bank in frozen order
BANK = [(seq, x, rot) for seq in PROGS for x in range(17)
        for rot in (0, 1)]
BANK_SHA = hashlib.sha256(json.dumps(
    [[list(s), x, r] for s, x, r in BANK]).encode()).hexdigest()


def a_running(seq, x, k):
    return x if k == 0 else T.apply_chain(list(seq[:k]), x)


def prefix_affine(seq, k):
    """(a, b) with F_k(x) = (a*x + b) mod 17 — the prefix map through
    the sender at interface k."""
    a, b = 1, 0
    for o in seq[:k]:
        ao, bo = T.OPS[o]
        a, b = (ao * a) % 17, (ao * b + bo) % 17
    return a, b


def prefix_inverse(seq, k, v):
    """x with F_k(x) == v (every primitive is a bijection over Z17)."""
    a, b = prefix_affine(seq, k)
    return (pow(a, -1, 17) * (v - b)) % 17


def spans_for(seq, x, rot):
    spans = [VALUE_SPAN_NEUTRAL.format(x=x)]
    spans += [SEALED_PHR[o][(rot + i) % 4] for i, o in enumerate(seq)]
    while len(spans) < 4:
        spans.append(FORWARD_SPAN_NEUTRAL)
    return spans


class Stack:
    def __init__(self, ckpt_path):
        self.lm = QwenGenome().attach_lora().to(DEV)
        ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
        assert ck["step"] == 20000
        self.bridge = BridgeA1(self.lm).to(DEV).eval()
        self.bridge.load_state_dict(ck["bridge_trainable"], strict=False)
        self.lm.core.load_state_dict(ck["lora"], strict=False)
        self.bridge.beta.fill_(float(ck["beta"]))
        self.cond = ck["condition"]
        self.tag = f"{self.cond}_m{ck['model_seed']}_o{ck['order_seed']}"
        self.label_ids = torch.tensor(
            [r["token_id"] for r in json.load(open(
                ROOT / "results/bridge/label_context.json"))["labels"]],
            device=DEV)
        self.q_ids, _ = self.lm.encode(
            ["Private notes were distributed to the cells. After every "
             "instruction has been applied in cell order, what is the "
             "final number? Answer with a single capital letter using "
             "the code A=0, B=1, C=2, ... Q=16.\nThe answer is"],
            fixed_length=Tq, device=DEV)
        if self.cond == "N+":
            class _T:
                def __init__(s, tok): s.tok = tok
                def encode(s, t, add_special_tokens=False):
                    class R: pass
                    r = R()
                    r.ids = s.tok(t, add_special_tokens=False)["input_ids"]
                    return r
                def decode(s, ids): return s.tok.decode(ids)
            self.banks, _ = build_length_banks(_T(self.lm.tok),
                                               range(5, Ts + 1))
            self.pad = int(self.lm.tok.pad_token_id)

    @torch.no_grad()
    def run_case(self, bank_idx, sub=None, cut_mail=False):
        seq, x, rot = BANK[bank_idx]
        s_ids, s_mask = self.lm.encode(spans_for(seq, x, rot),
                                       fixed_length=Ts, device=DEV)
        fill = None
        if self.cond == "N+":
            fill = select_fillers(s_mask.cpu(), 1, Ts, self.banks,
                                  FILLER_SEED, WORLD_ID, bank_idx,
                                  self.pad, stream_id="mechanism")
        oi, om = torch_package(self.cond, s_ids, s_mask, 1, Ts,
                               HDR_IDS, fill)
        out = self.bridge.forward_episode(self.q_ids, oi, om,
                                          substitute_traj=sub,
                                          cut_mail=cut_mail)
        corr = (out["logits"] - out["base_logits"])[:, self.label_ids]
        return int(corr.argmax(-1)), out["packets"]

    @torch.no_grad()
    def run_case_spans(self, spans, ordinal):
        """Arbitrary span text under this trajectory's native regime;
        ordinal feeds the N+ filler PRF (mechanism stream)."""
        s_ids, s_mask = self.lm.encode(spans, fixed_length=Ts,
                                       device=DEV)
        fill = None
        if self.cond == "N+":
            fill = select_fillers(s_mask.cpu(), 1, Ts, self.banks,
                                  FILLER_SEED, WORLD_ID, ordinal,
                                  self.pad, stream_id="mechanism")
        oi, om = torch_package(self.cond, s_ids, s_mask, 1, Ts,
                               HDR_IDS, fill)
        out = self.bridge.forward_episode(self.q_ids, oi, om)
        corr = (out["logits"] - out["base_logits"])[:, self.label_ids]
        return int(corr.argmax(-1))

    def free(self):
        del self.lm, self.bridge
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()


# ---- harvest donor packets on the SAME fixed bank ----
donor = Stack(A.donor_ckpt)
donor_tag = donor.tag
donor_bank = {k: {v: [] for v in range(17)} for k in (0, 1, 2)}
donor_success = 0
for bi in range(len(BANK)):
    seq, x, rot = BANK[bi]
    pred, packets = donor.run_case(bi)
    if pred != T.apply_chain(list(seq), x):
        continue
    donor_success += 1
    for k in (0, 1, 2):
        v = a_running(seq, x, k)
        if len(donor_bank[k][v]) < MIN_PER + 2:
            donor_bank[k][v].append(
                (packets[0][k].detach().cpu().clone(),
                 {"seq": list(seq), "x": x, "rot": rot}))
coverage = {k: sorted(v for v in donor_bank[k] if donor_bank[k][v])
            for k in (0, 1, 2)}
print(f"donor {donor_tag}: success {donor_success}/{len(BANK)}, "
      f"values covered per iface "
      f"{ {k: len(v) for k, v in coverage.items()} }", flush=True)
if A.emit_coverage:
    Path(A.out).write_text(json.dumps(
        {"donor": donor_tag, "coverage": coverage,
         "donor_success": donor_success}, indent=1))
    print(f"WROTE coverage {A.out}")
    sys.exit(0)
COV = coverage
if A.coverage_file:
    COV = {int(k): sorted(v) for k, v in
           json.load(open(A.coverage_file))["coverage"].items()}
    for k in (0, 1, 2):
        missing = set(COV[k]) - set(coverage[k])
        if missing:
            sys.exit(f"donor {donor_tag} lacks values {sorted(missing)} "
                     f"at iface {k} demanded by the coverage file")
donor.free()

# ---- evaluate recipient ----
rec_stack = Stack(A.recipient_ckpt)
rng = random.Random(8500)
conds = ("intact", "allcut_bank", "same", "cf_follow", "shift_retain",
         "deranged", "deletion", "noise", "span_rewrite")
res = {k: {c: [0, 0] for c in conds} for k in (0, 1, 2)}
excl = {k: {"base_wrong": 0, "no_donor": 0} for k in (0, 1, 2)}
eligible_ids = {0: [], 1: [], 2: []}
full_bank_intact = 0
zero_norm_intact_packets = {0: 0, 1: 0, 2: 0}
records = []
for bi in range(len(BANK)):
    seq, x, rot = BANK[bi]
    ans = T.apply_chain(list(seq), x)
    base_pred, base_packets = rec_stack.run_case(bi)
    if base_pred == ans:
        full_bank_intact += 1
        allcut_pred, _ = rec_stack.run_case(bi, cut_mail=True)
    if base_pred != ans:
        for k in (0, 1, 2):
            excl[k]["base_wrong"] += 1
        continue
    for k in (0, 1, 2):
        v = a_running(seq, x, k)
        vcf = (v + CF_OFFSET) % 17
        cov_k = COV[k]
        wder_pool = [w for w in cov_k if w != v]
        if v not in cov_k or vcf not in cov_k or not wder_pool:
            excl[k]["no_donor"] += 1
            continue
        wder = rng.choice(wder_pool)
        pool_v = donor_bank[k][v]
        pool_cf = donor_bank[k][vcf]
        pool_w = donor_bank[k][wder]
        if not (pool_v and pool_cf and pool_w):
            excl[k]["no_donor"] += 1
            continue
        target_cf = T.apply_chain(list(seq[k:]), vcf)
        dv = pool_v[rng.randrange(len(pool_v))]
        dcf = pool_cf[rng.randrange(len(pool_cf))]
        dw = pool_w[rng.randrange(len(pool_w))]
        pk_v, pk_cf, pk_w = (dv[0].to(DEV), dcf[0].to(DEV), dw[0].to(DEV))
        zero = torch.zeros_like(pk_v)
        rec = {"bank_idx": bi, "iface": k, "seq": list(seq), "x": x,
               "rot": rot, "v": v, "vcf": vcf, "wder": wder,
               "ans": ans, "target_cf": target_cf,
               "donor_meta": {"same": dv[1], "cf": dcf[1],
                              "deranged": dw[1]},
               "same_exact_overlap": int(dv[1]["seq"] == list(seq)
                                         and dv[1]["x"] == x
                                         and dv[1]["rot"] == rot),
               "same_prog_overlap": int(dv[1]["seq"] == list(seq)),
               "preds": {}}
        # approximately norm-matched noise: PRF-seeded direction,
        # per-packet Frobenius-norm matching to the recipient's own
        # intact packet at this interface
        # noise:v1 spec (round-4 §1.1): key = noise:v1:world:bank:iface
        # (recipient identity deliberately omitted -> the same direction
        # is reused across recipients for comparability); hash-to-seed =
        # first 8 big-endian bytes of SHA-256 mod 2^63; distribution =
        # torch.randn (standard normal) on CPU float32, then scaled to
        # the intact packet's Frobenius norm and cast to device.
        base_pk = base_packets[0][k]
        gseed = int.from_bytes(hashlib.sha256(
            f"noise:v1:{WORLD_ID}:{bi}:{k}".encode()).digest()[:8],
            "big") % (2**63)
        gg = torch.Generator(device="cpu").manual_seed(gseed)
        nz = torch.randn(base_pk.shape, generator=gg,
                         dtype=torch.float32)
        zn = float(nz.norm())
        if not (zn > 0 and torch.isfinite(nz).all()):
            raise AssertionError("noise direction zero/non-finite norm")
        pk_norm = float(base_pk.norm())
        if pk_norm > 0:
            nz = nz / zn * pk_norm
        else:
            nz = torch.zeros_like(nz)
            zero_norm_intact_packets[k] += 1
        nz = nz.to(DEV)
        for cname, pk, tgt in (("same", pk_v, ans),
                               ("cf_follow", pk_cf, target_cf),
                               ("shift_retain", pk_cf, ans),
                               ("deranged", pk_w, ans),
                               ("deletion", zero, ans),
                               ("noise", nz, ans)):
            pred, _ = rec_stack.run_case(bi, sub={(k, k): pk.unsqueeze(0)})
            res[k][cname][0] += int(pred == tgt)
            res[k][cname][1] += 1
            rec["preds"][cname] = pred
        # matched span-rewrite (round-4 §1.2): x_cf = F_k^{-1}(v_cf)
        xr = prefix_inverse(seq, k, vcf)
        if a_running(seq, xr, k) != vcf:
            raise AssertionError(
                f"span-rewrite inversion failed: F_{k}({xr}) != {vcf}")
        tgt_rw = T.apply_chain(list(seq), xr)
        if tgt_rw != target_cf:
            raise AssertionError(
                f"rewritten final target {tgt_rw} != suffix-implied "
                f"packet-counterfactual target {target_cf}")
        pred_rw = rec_stack.run_case_spans(
            spans_for(seq, xr, rot), bi)
        res[k]["span_rewrite"][0] += int(pred_rw == tgt_rw)
        res[k]["span_rewrite"][1] += 1
        rec["preds"]["span_rewrite"] = pred_rw
        rec["span_rewrite_x_cf"] = xr
        rec["span_rewrite_target"] = tgt_rw
        # intact baseline (=1 by success-conditioning; reported for the
        # frozen record) and all-packets-cut on this audit case
        res[k]["intact"][0] += 1
        res[k]["intact"][1] += 1
        res[k]["allcut_bank"][0] += int(allcut_pred == ans)
        res[k]["allcut_bank"][1] += 1
        rec["preds"]["allcut_bank"] = allcut_pred
        eligible_ids[k].append(bi)
        records.append(rec)
rec_tag = rec_stack.tag
rec_stack.free()

summary = {}
for k in (0, 1, 2):
    summary[k] = {}
    for cname in conds:
        num, den = res[k][cname]
        summary[k][cname] = {
            "num": num, "den": den,
            "rate": round(num / den, 4) if den else None,
            "low_denominator_descriptive": den < LOW_DENOM}
eligible_hashes = {k: hashlib.sha256(
    json.dumps(eligible_ids[k]).encode()).hexdigest() for k in (0, 1, 2)}
out = {"donor": donor_tag, "recipient": rec_tag,
       "recipient_intact_full_bank": {
           "num": full_bank_intact, "den": len(BANK)},
       "eligible_case_ids": eligible_ids,
       "eligible_case_ids_sha256": eligible_hashes,
       "n_eligible": {k: len(v) for k, v in eligible_ids.items()},
       "mode": "self" if A.donor_ckpt == A.recipient_ckpt else "cross",
       "world_id": WORLD_ID, "bank_sha256": BANK_SHA,
       "bank_size": len(BANK), "cf_offset": CF_OFFSET,
       "donor_success": donor_success,
       "coverage_file": A.coverage_file,
       "zero_norm_intact_packets": zero_norm_intact_packets,
       "exclusions": excl, "summary": summary,
       "per_case_records": records}
Path(A.out).write_text(json.dumps(out, indent=1))
print(f"WROTE {A.out} ({len(records)} recipient cases)", flush=True)

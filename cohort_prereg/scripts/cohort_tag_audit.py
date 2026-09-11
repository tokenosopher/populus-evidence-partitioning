"""G+ marker-permutation audit evaluator — PREREG v0.3 §5.4/§5.5.

For every case in the frozen discriminating audit bank: score the INTACT
serialization against the intact target and the PERMUTED-marker
serialization against the permutation-implied target. Both fractions use
the complete sealed bank denominator; nothing is conditioned on intact
correctness and no model-specific filtering exists. Evidence text, token
positions, and masks are identical between the pair (torch-verified per
batch); only operator-cell mine:/other: header IDs differ. G+ only.
Rendering: sealed pool, spans[1+i] = SEALED[op][(rot + i) % 4], four
rotations. Scoring is permutation-stratified (REVIEW round-2 D3): intact
and counterfactual fidelities are computed separately per non-identity
permutation; the gate values are the unweighted means of the five
permutation-specific fidelities. Raw per-permutation integer counts are
emitted alongside.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from cohort.torch_package import (torch_package,                   # noqa
                                  torch_package_permuted)
from cohort.neutral_grammar import (FORWARD_SPAN_NEUTRAL,          # noqa
                                    VALUE_SPAN_NEUTRAL)

P = argparse.ArgumentParser()
P.add_argument("--ckpt", required=True)
P.add_argument("--device", default="cuda")
P.add_argument("--batch", type=int, default=64)
P.add_argument("--out", required=True)
A = P.parse_args()

for _v in ("BRIDGE_OP_SEED", "BRIDGE_SPLIT_SEED",
           "COHORT_GRAMMAR_PATH", "COHORT_FILLER_SEED"):
    assert os.environ.get(_v), f"missing env {_v}"

from populus.bridge_train import Trainer                           # noqa
from populus.bridge_qwen import QwenGenome                         # noqa
from cohort.world import (build_world, permutation_audit_bank,     # noqa
                          structural_checks)

CFG = json.load(open(ROOT / "cohort" / "condition_config.json"))
HDR_IDS = {k: list(map(int, CFG["header_token_blocks"][k]))
           for k in ("mine", "other", "slot")}
GRAMMAR_RAW = open(os.environ["COHORT_GRAMMAR_PATH"], "rb").read()
GRAMMAR_SHA = hashlib.sha256(GRAMMAR_RAW).hexdigest()
GRAMMAR = json.loads(GRAMMAR_RAW)
SEALED_PHR = {int(k): v["sealed"] for k, v in GRAMMAR["ops"].items()}

DEV = A.device
CK = torch.load(A.ckpt, map_location=DEV, weights_only=False)
assert CK["step"] == 20000
assert CK["condition"] == "G+", \
    f"gating tag audit is G+ only (PREREG §5.4), got {CK['condition']}"
assert CK["grammar_sha"] == GRAMMAR_SHA

torch.manual_seed(0)
lm = QwenGenome().attach_lora().to(DEV)
tr = Trainer(lm, data_seed=0, device=DEV)
tr.bridge.load_state_dict(CK["bridge_trainable"], strict=False)
lm.core.load_state_dict(CK["lora"], strict=False)
tr.bridge.beta.fill_(float(CK["beta"]))
tr.bridge.eval()
for name, surface in CFG["header_token_blocks"]["surface_forms"].items():
    ids = lm.tok(surface, add_special_tokens=False)["input_ids"]
    assert ids == HDR_IDS[name], (name, ids)

T = build_world(int(os.environ["BRIDGE_OP_SEED"]),
                int(os.environ["BRIDGE_SPLIT_SEED"]))
CHECKS = structural_checks(T)
assert CHECKS["VALID"]
AB = permutation_audit_bank(T, CHECKS)
BANK = AB["bank"]
BANK_SHA = hashlib.sha256(json.dumps(BANK, sort_keys=True).encode()).hexdigest()
print(f"audit bank: {len(BANK)} cases ({AB['n_programs']} programs, "
      f"{AB['excluded_collisions']} collisions excluded)", flush=True)


def spans_of(case):
    spans = [VALUE_SPAN_NEUTRAL.format(x=case["x"])]
    spans += [SEALED_PHR[o][(case["rot"] + i) % 4]
              for i, o in enumerate(case["seq"])]
    return spans


@torch.no_grad()
def run():
    per = {str(tuple(p)): {"intact_num": 0, "cf_num": 0, "den": 0}
           for p in ({tuple(c["perm"]) for c in BANK})}
    for i in range(0, len(BANK), A.batch):
        chunk = BANK[i:i + A.batch]
        n = len(chunk)
        s_ids, s_mask = lm.encode(
            [s for c in chunk for s in spans_of(c)],
            fixed_length=tr.Ts, device=DEV)
        ii, im = torch_package("G+", s_ids, s_mask, n, tr.Ts, HDR_IDS)
        q = tr.q_ids.expand(n, -1)
        out = tr.bridge.forward_episode(q, ii, im)
        pred_i = ((out["logits"] - out["base_logits"])
                  [:, tr.label_ids]).argmax(-1).tolist()
        # permuted pass: group rows by perm (identical evidence tensors)
        pred_p = [None] * n
        for perm in {tuple(c["perm"]) for c in chunk}:
            rows = [r for r, c in enumerate(chunk)
                    if tuple(c["perm"]) == perm]
            sel = torch.tensor(
                [[r * 4 + j for j in range(4)] for r in rows],
                device=DEV).flatten()
            pi, pm = torch_package_permuted(
                s_ids[sel], s_mask[sel], len(rows), tr.Ts, HDR_IDS, perm)
            # §5.5 pair invariant: masks identical, only header ids move
            assert torch.equal(pm, im[rows])
            out_p = tr.bridge.forward_episode(
                q[:len(rows)], pi, pm)
            pp = ((out_p["logits"] - out_p["base_logits"])
                  [:, tr.label_ids]).argmax(-1).tolist()
            for r, p in zip(rows, pp):
                pred_p[r] = p
        for c, a, b in zip(chunk, pred_i, pred_p):
            k = str(tuple(c["perm"]))
            per[k]["den"] += 1
            per[k]["intact_num"] += int(a == c["intact_answer"])
            per[k]["cf_num"] += int(b == c["counterfactual_answer"])
    return per


PER = run()
assert all(v["den"] > 0 for v in PER.values())
assert {k: v["den"] for k, v in PER.items()} == AB["per_perm_denominators"]
intact_strat = sum(v["intact_num"] / v["den"] for v in PER.values()) / len(PER)
cf_strat = sum(v["cf_num"] / v["den"] for v in PER.values()) / len(PER)
den = len(BANK)
ih = sum(v["intact_num"] for v in PER.values())
ch = sum(v["cf_num"] for v in PER.values())
print(f"stratified intact {intact_strat:.4f} cf_track {cf_strat:.4f} "
      f"(raw {ih}/{den}, {ch}/{den})", flush=True)
rec = {"trajectory_id": f"m{CK['model_seed']}_o{CK['order_seed']}",
       "condition": "G+",
       "world_id": f"op{os.environ['BRIDGE_OP_SEED']}"
                   f"_split{os.environ['BRIDGE_SPLIT_SEED']}",
       "step": int(CK["step"]),
       "bank_sha256": BANK_SHA,
       "intact": {"num": ih, "den": den},
       "cf": {"num": ch, "den": den},
       "per_perm": PER,
       "intact_stratified": round(intact_strat, 6),
       "cf_track_stratified": round(cf_strat, 6),
       "meta": {"ckpt": A.ckpt,
                "ckpt_sha256": hashlib.sha256(
                    open(A.ckpt, "rb").read()).hexdigest(),
                "bank_sha": BANK_SHA,
                "n_programs": AB["n_programs"],
                "excluded_collisions": AB["excluded_collisions"],
                "grammar_sha": GRAMMAR_SHA}}
Path(A.out).write_text(json.dumps(rec, indent=1, allow_nan=False))
print(f"WROTE {A.out}", flush=True)

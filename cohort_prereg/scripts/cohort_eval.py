"""Cohort final-checkpoint evaluator — PREREG v0.3 §5.2/§7 scoring.

Emits RAW INTEGER counts only (num/den per component); rates are computed
downstream by cohort/result_compiler.py. Frozen deterministic renderings:
  primary/A-banks : sealed pool,  spans[1+i] = SEALED[op][(rot + i) % 4],
                    rot in {0,1,2,3} (four rotations; REVIEW round-2 D4-A)
  T2/T3 training  : train pool,   spans[1+i] = TRAIN[op][(x + i) % 24]
  S1 single-op    : sealed pool,  spans[pos] = SEALED[op][(rot + pos) % 4]
All under the trajectory's NATIVE condition regime (torch_package); the
communication-severed rerun (cut_mail) covers the d3 primary bank only.
N+ filler at eval: bank case index (deterministic, episode-independent)
as ep_base, same frozen filler bank/offset as training.
Primary evaluation is the step-20,000 checkpoint ONLY (asserted).
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

from cohort.condition_layout import CONDITIONS                     # noqa
from cohort.filler_constructor import build_length_banks           # noqa
from cohort.torch_package import select_fillers, torch_package     # noqa
from cohort.neutral_grammar import (FORWARD_SPAN_NEUTRAL,          # noqa
                                    VALUE_SPAN_NEUTRAL)

P = argparse.ArgumentParser()
P.add_argument("--ckpt", required=True)
P.add_argument("--condition", choices=CONDITIONS, required=True)
P.add_argument("--device", default="cuda")
P.add_argument("--batch", type=int, default=64)
P.add_argument("--out", required=True)
A = P.parse_args()

for _v in ("BRIDGE_OP_SEED", "BRIDGE_SPLIT_SEED",
           "COHORT_GRAMMAR_PATH", "COHORT_FILLER_SEED"):
    assert os.environ.get(_v), f"missing env {_v}"

from populus.bridge_tasks import SPLIT, apply_chain                # noqa
from populus.bridge_train import Trainer                           # noqa
from populus.bridge_qwen import QwenGenome                         # noqa
from cohort.world import build_world, structural_checks            # noqa

CFG = json.load(open(ROOT / "cohort" / "condition_config.json"))
HDR_IDS = {k: list(map(int, CFG["header_token_blocks"][k]))
           for k in ("mine", "other", "slot")}
GRAMMAR_RAW = open(os.environ["COHORT_GRAMMAR_PATH"], "rb").read()
GRAMMAR_SHA = hashlib.sha256(GRAMMAR_RAW).hexdigest()
GRAMMAR = json.loads(GRAMMAR_RAW)
TRAIN_PHR = {int(k): v["train"] for k, v in GRAMMAR["ops"].items()}
SEALED_PHR = {int(k): v["sealed"] for k, v in GRAMMAR["ops"].items()}
FILLER_SEED = int(os.environ["COHORT_FILLER_SEED"])
CFG_SHA = hashlib.sha256(
    open(ROOT / "cohort" / "condition_config.json", "rb").read()).hexdigest()

DEV = A.device
CK = torch.load(A.ckpt, map_location=DEV, weights_only=False)
assert CK["step"] == 20000, f"primary eval is step-20000 only, got {CK['step']}"
assert CK["condition"] == A.condition, (CK["condition"], A.condition)
assert CK["grammar_sha"] == GRAMMAR_SHA
assert CK["config_sha"] == CFG_SHA
assert str(CK["op_seed"]) == os.environ["BRIDGE_OP_SEED"]
assert str(CK["split_seed"]) == os.environ["BRIDGE_SPLIT_SEED"]
assert int(CK["filler_seed"]) == FILLER_SEED

torch.manual_seed(0)
lm = QwenGenome().attach_lora().to(DEV)
tr = Trainer(lm, data_seed=0, device=DEV)
tr.bridge.load_state_dict(CK["bridge_trainable"], strict=False)
lm.core.load_state_dict(CK["lora"], strict=False)
tr.bridge.beta.fill_(float(CK["beta"]))
tr.bridge.eval()


class _T:
    def __init__(s, tok): s.tok = tok
    def encode(s, text, add_special_tokens=False):
        class R: pass
        r = R(); r.ids = s.tok(text, add_special_tokens=False)["input_ids"]
        return r
    def decode(s, ids): return s.tok.decode(ids)


FILL_BANKS, FILL_REP = build_length_banks(_T(lm.tok), range(5, tr.Ts + 1))
WORLD_ID = (f"op{os.environ['BRIDGE_OP_SEED']}"
            f"_split{os.environ['BRIDGE_SPLIT_SEED']}")
PAD = int(lm.tok.pad_token_id)
for name, surface in CFG["header_token_blocks"]["surface_forms"].items():
    ids = lm.tok(surface, add_special_tokens=False)["input_ids"]
    assert ids == HDR_IDS[name], (name, ids)

# world (env-seeded) for bank construction
T = build_world(int(os.environ["BRIDGE_OP_SEED"]),
                int(os.environ["BRIDGE_SPLIT_SEED"]))
CHECKS = structural_checks(T)
assert CHECKS["VALID"], "world failed structural checks at eval time"


def spans_for(seq, x, renderer):
    spans = [VALUE_SPAN_NEUTRAL.format(x=x)]
    spans += [renderer(o, i) for i, o in enumerate(seq)]
    while len(spans) < 4:
        spans.append(FORWARD_SPAN_NEUTRAL)
    return spans


def bank_primary():
    cases = []
    for depth, progs in (("d2", CHECKS["cf_d2_programs"]),
                         ("d3", CHECKS["cf_d3_programs"])):
        for seq in progs:
            for x in range(17):
                for rot in (0, 1, 2, 3):
                    cases.append({
                        "depth": depth,
                        "spans": spans_for(seq, x, lambda o, i:
                                           SEALED_PHR[o][(rot + i) % 4]),
                        "answer": apply_chain(list(seq), x)})
    return cases


def bank_T(progs, n_prog):
    cases = []
    for seq in list(progs)[:n_prog]:
        for x in range(17):
            cases.append({
                "spans": spans_for(seq, x, lambda o, i:
                                   TRAIN_PHR[o][(x + i) % 24]),
                "answer": apply_chain(list(seq), x)})
    return cases


def bank_S1():
    cases = []
    for op in range(len(T.OPS)):
        for pos in (1, 2, 3):
            for x in range(17):
                for rot in (0, 1, 2, 3):
                    spans = [VALUE_SPAN_NEUTRAL.format(x=x)] \
                        + [FORWARD_SPAN_NEUTRAL] * 3
                    spans[pos] = SEALED_PHR[op][(rot + pos) % 4]
                    cases.append({"spans": spans,
                                  "answer": apply_chain([op], x)})
    return cases


@torch.no_grad()
def score(cases, cut_mail=False):
    hits = 0
    for i in range(0, len(cases), A.batch):
        chunk = cases[i:i + A.batch]
        n = len(chunk)
        s_ids, s_mask = lm.encode(
            [s for c in chunk for s in c["spans"]],
            fixed_length=tr.Ts, device=DEV)
        fill = None
        if A.condition == "N+":
            fill = select_fillers(s_mask.cpu(), n, tr.Ts, FILL_BANKS,
                                  FILLER_SEED, WORLD_ID, i, PAD,
                                  stream_id="primary")
        s_ids, s_mask = torch_package(
            A.condition, s_ids, s_mask, n, tr.Ts, HDR_IDS, fill)
        out = tr.bridge.forward_episode(
            tr.q_ids.expand(n, -1), s_ids, s_mask, cut_mail=cut_mail)
        corr = (out["logits"] - out["base_logits"])[:, tr.label_ids]
        pred = corr.argmax(-1).tolist()
        hits += sum(int(p == c["answer"])
                    for p, c in zip(pred, chunk))
    return hits


def bank_hash(cases):
    return hashlib.sha256(json.dumps(
        [{k: v for k, v in c.items()} for c in cases],
        sort_keys=True).encode()).hexdigest()


primary = bank_primary()
p2 = [c for c in primary if c["depth"] == "d2"]
p3 = [c for c in primary if c["depth"] == "d3"]
t2 = bank_T(T.SPLIT["train2"], len(T.SPLIT["train2"]))
t3 = bank_T(T.SPLIT["train3"], 150)
s1 = bank_S1()
print(f"banks: d2 {len(p2)} d3 {len(p3)} T2 {len(t2)} T3 {len(t3)} "
      f"S1 {len(s1)}", flush=True)

n2 = score(p2)
print(f"A2 raw {n2}/{len(p2)}", flush=True)
n3 = score(p3)
print(f"A3 raw {n3}/{len(p3)}", flush=True)
nc3 = score(p3, cut_mail=True)
print(f"allcut3 raw {nc3}/{len(p3)}", flush=True)
nT2 = score(t2)
nT3 = score(t3)
nS1 = score(s1)
print(f"T2 {nT2}/{len(t2)} T3 {nT3}/{len(t3)} S1 {nS1}/{len(s1)}",
      flush=True)

rec = {
    "condition": A.condition,
    "trajectory_id": f"m{CK['model_seed']}_o{CK['order_seed']}",
    "world_id": f"op{os.environ['BRIDGE_OP_SEED']}"
                f"_split{os.environ['BRIDGE_SPLIT_SEED']}",
    "step": int(CK["step"]),
    "primary_d2": {"num": n2, "den": len(p2)},
    "primary_d3": {"num": n3, "den": len(p3)},
    "allcut_d3": {"num": nc3, "den": len(p3)},
    "T2": {"num": nT2, "den": len(t2)},
    "T3": {"num": nT3, "den": len(t3)},
    "S1": {"num": nS1, "den": len(s1)},
    "meta": {
        "ckpt": A.ckpt,
        "ckpt_sha256": hashlib.sha256(
            open(A.ckpt, "rb").read()).hexdigest(),
        "grammar_sha": GRAMMAR_SHA, "config_sha": CFG_SHA,
        "filler_seed": FILLER_SEED, "filler_report": {str(k): v for k, v in FILL_REP.items()},
        "bank_hashes": {"primary_d2": bank_hash(p2),
                        "primary_d3": bank_hash(p3),
                        "T2": bank_hash(t2), "T3": bank_hash(t3),
                        "S1": bank_hash(s1)},
        "serialized_stream_hash": CK.get("serialized_stream_hash"),
        "semantic_stream_hash": CK.get("semantic_stream_hash"),
    }}
Path(A.out).write_text(json.dumps(rec, indent=1, allow_nan=False))
print(f"WROTE {A.out}", flush=True)

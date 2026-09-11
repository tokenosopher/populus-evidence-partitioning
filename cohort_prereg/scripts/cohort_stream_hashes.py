"""Full-run stream-hash precomputation — PREREG §4 manifest fields.

Computes, WITHOUT the model, the complete 20,000-batch semantic and
serialized rolling hashes for every (order seed, condition) pair, byte-
identical to what the trainer will report (proven by the 3-batch
admission cross-check; this script at --steps 3 must reproduce the
admission fixture exactly, which is asserted when --check-fixture is
given). Numpy-vectorized; unique span texts tokenized once.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cohort.condition_layout import CONDITIONS                     # noqa
from cohort.filler_constructor import (build_length_banks,         # noqa
                                       filler_id)
from cohort.sampler import sample_spans                            # noqa
from cohort.neutral_grammar import (FORWARD_SPAN_NEUTRAL,          # noqa
                                    VALUE_SPAN_NEUTRAL)

PAD_ID = 151643
BATCH = 32

P = argparse.ArgumentParser()
P.add_argument("--tokenizer-json", required=True)
P.add_argument("--order-seeds", required=True,
               help="comma-separated data-order seeds")
P.add_argument("--steps", type=int, default=20000)
P.add_argument("--out", required=True)
P.add_argument("--check-fixture", default=None,
               help="admission fixture JSON; assert equality at --steps 3")
A = P.parse_args()

for _v in ("BRIDGE_OP_SEED", "BRIDGE_SPLIT_SEED",
           "COHORT_GRAMMAR_PATH", "COHORT_FILLER_SEED"):
    assert os.environ.get(_v), f"missing env {_v}"

from cohort.world import build_world                               # noqa

T = build_world(int(os.environ["BRIDGE_OP_SEED"]),
                int(os.environ["BRIDGE_SPLIT_SEED"]))
GRAMMAR = json.load(open(os.environ["COHORT_GRAMMAR_PATH"]))
TRAIN_PHR = {int(k): v["train"] for k, v in GRAMMAR["ops"].items()}
CFG = json.load(open(ROOT / "cohort" / "condition_config.json"))
HDR = {k: np.array(CFG["header_token_blocks"][k], dtype=np.int64)
       for k in ("mine", "other", "slot")}
W = CFG["header_token_blocks"]["width"]
Ts = CFG["span_widths"]["T_SPAN"]
FILLER_SEED = int(os.environ["COHORT_FILLER_SEED"])

from tokenizers import Tokenizer                                   # noqa
tok = Tokenizer.from_file(A.tokenizer_json)
assert hashlib.sha256(open(A.tokenizer_json, "rb").read()).hexdigest() \
    == CFG["tokenizer"]["tokenizer_json_sha256"]


class _T:
    def encode(self, text, add_special_tokens=False):
        class R: pass
        r = R(); r.ids = tok.encode(text,
                                    add_special_tokens=add_special_tokens).ids
        return r
    def decode(self, ids): return tok.decode(ids)


FBANKS, FREP = build_length_banks(_T(), range(5, Ts + 1))
WORLD_ID = (f"op{os.environ['BRIDGE_OP_SEED']}"
            f"_split{os.environ['BRIDGE_SPLIT_SEED']}")

_enc_cache = {}


def enc(text):
    if text not in _enc_cache:
        ids = tok.encode(text, add_special_tokens=False).ids
        assert len(ids) <= Ts
        i = np.full(Ts, PAD_ID, dtype=np.int64)
        m = np.zeros(Ts, dtype=np.int64)
        i[:len(ids)] = ids; m[:len(ids)] = 1
        _enc_cache[text] = (i, m)
    return _enc_cache[text]


def run(order_seed):
    """One pass over the episode stream; all five conditions hashed
    simultaneously from the same sampled batches."""
    rng = random.Random(order_seed)
    sem = hashlib.sha256()
    ser = {c: hashlib.sha256() for c in CONDITIONS}
    L = W + Ts
    for step in range(A.steps):
        eps = [sample_spans(rng, T.SPLIT, TRAIN_PHR, T.apply_chain,
                            VALUE_SPAN_NEUTRAL, FORWARD_SPAN_NEUTRAL)
               for _ in range(BATCH)]
        sem.update(json.dumps([m for _, _, m in eps]).encode())
        si = np.empty((BATCH, 4, Ts), dtype=np.int64)
        sm = np.empty((BATCH, 4, Ts), dtype=np.int64)
        for r, (sp, _, _) in enumerate(eps):
            for j, s in enumerate(sp):
                si[r, j], sm[r, j] = enc(s)
        ep_base = step * BATCH
        for cond in CONDITIONS:
            oi = np.empty((BATCH, 4, 4 * L), dtype=np.int64)
            om = np.empty((BATCH, 4, 4 * L), dtype=np.int64)
            if cond == "N+":
                fill = np.full((BATCH, 4, Ts), PAD_ID, dtype=np.int64)
                lens = sm.sum(axis=2)
                for r in range(BATCH):
                    for j in range(4):
                        aln = int(lens[r, j])
                        bank = FBANKS[aln]
                        fid = filler_id(FILLER_SEED, WORLD_ID,
                                        ep_base + r, j, len(bank), "train")
                        fill[r, j, :aln] = bank[fid]
            for c in range(4):
                for j in range(4):
                    if cond in ("R+", "G+", "N+"):
                        h = HDR["mine"] if j == c else HDR["other"]
                    else:
                        h = HDR["slot"]
                    a, b = j * L, j * L + W
                    oi[:, c, a:b] = h
                    content, cmask = si[:, j], sm[:, j]
                    if cond == "N+" and j != c:
                        content = fill[:, j]
                        # mask UNCHANGED (length-matched filler)
                    if cond in ("R-", "R+") and j != c:
                        om[:, c, a:b] = 0
                        om[:, c, b:b + Ts] = 0
                    else:
                        om[:, c, a:b] = 1
                        om[:, c, b:b + Ts] = cmask
                    oi[:, c, b:b + Ts] = content
            ser[cond].update(oi.tobytes())
            ser[cond].update(om.tobytes())
    return sem.hexdigest(), {c: ser[c].hexdigest() for c in CONDITIONS}


out = {"steps": A.steps, "batch": BATCH,
       "op_seed": os.environ["BRIDGE_OP_SEED"],
       "split_seed": os.environ["BRIDGE_SPLIT_SEED"],
       "filler_seed": FILLER_SEED,
       "grammar_sha": hashlib.sha256(open(
           os.environ["COHORT_GRAMMAR_PATH"], "rb").read()).hexdigest(),
       "semantic": {}, "serialized": {c: {} for c in CONDITIONS}}
for o in [int(x) for x in A.order_seeds.split(",")]:
    s, d = run(o)
    out["semantic"][str(o)] = s
    for c in CONDITIONS:
        out["serialized"][c][str(o)] = d[c]
    print(f"order {o}: sem={s[:16]} " +
          " ".join(f"{c}={d[c][:12]}" for c in CONDITIONS), flush=True)

if A.check_fixture:
    fx = json.load(open(A.check_fixture))
    assert A.steps == fx["n_steps"]
    o = str(fx["order_seed"])
    for c in CONDITIONS:
        assert out["semantic"][o] == fx["fingerprints"][c]["sem"], c
        assert out["serialized"][c][o] == fx["fingerprints"][c]["ser"], c
    print("FIXTURE CROSS-CHECK: ALL MATCH")

Path(A.out).write_text(json.dumps(out, indent=1))
print(f"WROTE {A.out}")

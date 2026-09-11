"""Machine-admission fixture — per-condition bit-exact prefix fingerprints.

Computes, WITHOUT the model (tokenizer + shared sampler + torch-free
layout only), the semantic and serialized rolling-hash values after the
first 3 batches for each of the five conditions at given seeds. The box
trainer logs `ADMISSION_PREFIX3 sem=... ser=...` at step 3; admission
REQUIRES equality with this fixture's output for every condition. Any
divergence (tokenizer revision, sampler drift, layout drift, filler
drift, endianness) fails closed.

The serialized hash reproduces the trainer's exact bytes: HF right-pads
spans to Ts with pad id 151643 / mask 0; packaged tensors are int64
C-order (rows, cells, positions); hash order per batch = ids then mask.
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

from cohort.condition_layout import CONDITIONS, build_cell_input   # noqa
from cohort.filler_constructor import (build_length_banks,         # noqa
                                       filler_id)
from cohort.sampler import sample_spans                            # noqa
from cohort.neutral_grammar import (FORWARD_SPAN_NEUTRAL,          # noqa
                                    VALUE_SPAN_NEUTRAL)

PAD_ID = 151643
BATCH = 32
N_STEPS = 3

P = argparse.ArgumentParser()
P.add_argument("--tokenizer-json", required=True)
P.add_argument("--order-seed", type=int, required=True)
P.add_argument("--out", required=True)
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
HDR_IDS = {k: list(map(int, CFG["header_token_blocks"][k]))
           for k in ("mine", "other", "slot")}
Ts = CFG["span_widths"]["T_SPAN"]
FILLER_SEED = int(os.environ["COHORT_FILLER_SEED"])

from tokenizers import Tokenizer                                   # noqa
tok = Tokenizer.from_file(A.tokenizer_json)
tok_sha = hashlib.sha256(open(A.tokenizer_json, "rb").read()).hexdigest()
assert tok_sha == CFG["tokenizer"]["tokenizer_json_sha256"], \
    "tokenizer file differs from the frozen pin"


class _T:
    def encode(self, text, add_special_tokens=False):
        class R: pass
        r = R(); r.ids = tok.encode(text,
                                    add_special_tokens=add_special_tokens).ids
        return r
    def decode(self, ids): return tok.decode(ids)


BANKS, REP = build_length_banks(_T(), range(5, Ts + 1))
WORLD_ID = (f"op{os.environ['BRIDGE_OP_SEED']}"
            f"_split{os.environ['BRIDGE_SPLIT_SEED']}")
for name, surface in CFG["header_token_blocks"]["surface_forms"].items():
    ids = tok.encode(surface, add_special_tokens=False).ids
    assert ids == HDR_IDS[name], (name, ids)


def encode_fixed(text):
    ids = tok.encode(text, add_special_tokens=False).ids
    assert len(ids) <= Ts, f"over-length span: {text!r}"
    mask = [1] * len(ids) + [0] * (Ts - len(ids))
    return ids + [PAD_ID] * (Ts - len(ids)), mask


out = {}
for cond in CONDITIONS:
    rng = random.Random(A.order_seed)
    sem = hashlib.sha256()
    ser = hashlib.sha256()
    for step in range(N_STEPS):
        eps = [sample_spans(rng, T.SPLIT, TRAIN_PHR, T.apply_chain,
                            VALUE_SPAN_NEUTRAL, FORWARD_SPAN_NEUTRAL)
               for _ in range(BATCH)]
        rows_i, rows_m = [], []
        for sp, _, _ in eps:
            for s in sp:
                i, m = encode_fixed(s)
                rows_i.append(i); rows_m.append(m)
        ep_base = step * BATCH
        b_i, b_m = [], []
        for r in range(BATCH):
            si = rows_i[r * 4:(r + 1) * 4]
            sm = rows_m[r * 4:(r + 1) * 4]
            frows = None
            if cond == "N+":
                frows = []
                for j in range(4):
                    L = sum(sm[j])
                    bank = BANKS[L]
                    fid = filler_id(FILLER_SEED, WORLD_ID,
                                    ep_base + r, j, len(bank), "train")
                    frows.append(bank[fid] + [PAD_ID] * (Ts - L))
            cells_i, cells_m = [], []
            for c in range(4):
                ci, cm = build_cell_input(cond, c, si, sm, HDR_IDS, frows)
                cells_i.append(ci); cells_m.append(cm)
            b_i.append(cells_i); b_m.append(cells_m)
        sem.update(json.dumps([m for _, _, m in eps]).encode())
        ser.update(np.array(b_i, dtype=np.int64).tobytes())
        ser.update(np.array(b_m, dtype=np.int64).tobytes())
    out[cond] = {"sem": sem.hexdigest(), "ser": ser.hexdigest()}
    print(f"{cond}: sem={out[cond]['sem'][:16]} ser={out[cond]['ser'][:16]}",
          flush=True)

rec = {"order_seed": A.order_seed, "n_steps": N_STEPS, "batch": BATCH,
       "op_seed": os.environ["BRIDGE_OP_SEED"],
       "split_seed": os.environ["BRIDGE_SPLIT_SEED"],
       "filler_seed": FILLER_SEED,
       "grammar_sha": hashlib.sha256(
           open(os.environ["COHORT_GRAMMAR_PATH"], "rb").read()).hexdigest(),
       "tokenizer_sha": tok_sha,
       "fingerprints": out}
Path(A.out).write_text(json.dumps(rec, indent=1))
print(f"WROTE {A.out}")

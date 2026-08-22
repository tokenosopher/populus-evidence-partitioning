"""Universality Scout S1+S2 (PLAN_SCOUT_UNIVERSALITY.md, FROZEN):
strict zero-shot family-D evaluation of frozen Run-3V checkpoints
(S1), with unconditional cut-mail control (S2). Masked layout
mirrors the Run-3V evaluator exactly; arm read from the checkpoint.

Usage (per ckpt, on an admitted machine):
  RUN3V_WORLD=F BRIDGE_OP_SEED=6011 BRIDGE_SPLIT_SEED=2203 \
  BRIDGE_GRAMMAR_SEED=7717 python scripts/bridge_scout_s1s2.py \
      --ckpt results/run3v/run3v_F_P_m201_o951_ckpt_final.pt
Writes results/scout/scout_s1s2_<tag>.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
assert os.environ.get("RUN3V_WORLD") == "F"
assert os.environ.get("BRIDGE_OP_SEED") == "6011"
assert os.environ.get("BRIDGE_SPLIT_SEED") == "2203"
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from populus.bridge import BridgeA1                             # noqa
from populus.bridge_qwen import QwenGenome                      # noqa
from populus.bridge_tasks import ANSWER_PREFIX, QUESTION        # noqa
from populus import bridge_tasks as A                           # noqa
from populus import bridge_tasks_d as D                         # noqa

EXPECT_HASH = "2bd0b301a5b079b1"
assert D.bank_hash() == EXPECT_HASH, \
    f"family-D bank hash mismatch: {D.bank_hash()} != {EXPECT_HASH}"

P_ARGS = argparse.ArgumentParser()
P_ARGS.add_argument("--ckpt", required=True)
ARGS = P_ARGS.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = json.load(open("results/bridge/serialization_contract.json"))
Tq, Ts = C["T_QUESTION"], C["T_SPAN"]

lm = QwenGenome().attach_lora().to(DEV)
bridge = BridgeA1(lm).to(DEV).eval()
ck = torch.load(ARGS.ckpt, map_location=DEV, weights_only=False)
def load_subset_exact(module, payload, label):
    result = module.load_state_dict(payload, strict=False)
    assert not result.unexpected_keys, (label,
                                        result.unexpected_keys)
    state = module.state_dict()
    absent = set(payload) - set(state)
    assert not absent, (label, "absent keys", sorted(absent))
    for key, expected in payload.items():
        assert torch.equal(state[key].detach().cpu(),
                           expected.detach().cpu()), \
            (label, "load mismatch", key)


load_subset_exact(bridge, ck["bridge_trainable"], "bridge")
load_subset_exact(lm.core, ck["lora"], "lora")
ARM = ck["arm"]
AUDIT_ALLOWED = {("P", 200, 950), ("P", 201, 951),
                 ("P", 203, 903), ("P", 203, 953),
                 ("P", 204, 904), ("P", 204, 954),
                 ("G", 204, 954)}
_cid = (ck["arm"], ck["model_seed"], ck["order_seed"])
assert _cid in AUDIT_ALLOWED, _cid
assert int(ck["step"]) == 20_000, (
    "universality scout requires the final Run-3V checkpoint",
    ck["step"])
label_ids = torch.tensor([r["token_id"] for r in json.load(
    open("results/bridge/label_context.json"))["labels"]],
    device=DEV)
q_ids, _ = lm.encode([QUESTION + ANSWER_PREFIX], fixed_length=Tq,
                     device=DEV)
EYE4 = torch.eye(4, device=DEV)


def package(s_ids, s_mask, n):
    s_ids = s_ids.reshape(n, 4, Ts)
    s_mask = s_mask.reshape(n, 4, Ts)
    flat_i = s_ids.reshape(n, 1, 4 * Ts).expand(n, 4, -1).contiguous()
    if ARM == "G":
        return flat_i, s_mask.reshape(n, 1, 4 * Ts).expand(
            n, 4, -1).contiguous()
    per_cell = s_mask.unsqueeze(1) * \
        EYE4.to(s_mask.dtype).unsqueeze(0).unsqueeze(-1)
    return flat_i, per_cell.reshape(n, 4, 4 * Ts)


@torch.no_grad()
def score(cases, cut=False, batch=64):
    hits = 0
    for i in range(0, len(cases), batch):
        chunk = cases[i:i + batch]
        texts = [s for sp, _ in chunk for s in sp]
        s_ids, s_mask = lm.encode(texts, fixed_length=Ts, device=DEV)
        B = len(chunk)
        s_ids, s_mask = package(s_ids, s_mask, B)
        out = bridge.forward_episode(q_ids.expand(B, -1), s_ids,
                                     s_mask, cut_mail=cut)
        corr = (out["logits"] - out["base_logits"])[:, label_ids]
        tgt = torch.tensor([a for _, a in chunk], device=DEV)
        hits += int((corr.argmax(-1) == tgt).sum())
    return round(hits / len(cases), 4) if cases else None


def cases_for(seqs, tpl_pool):
    cases = []
    for seq in seqs:
        for x in range(17):
            for j in range(len(tpl_pool)):
                spans = [D.VALUE_SPAN.format(x=x)]
                spans += [D.render_op(o, tpl_pool[(j + i)
                                                  % len(tpl_pool)])
                          for i, o in enumerate(seq)]
                while len(spans) < 4:
                    spans.append(D.FORWARD_SPAN)
                cases.append((spans, D.apply_chain(list(seq), x)))
    return cases


A_train_maps = {(1, 0)}
A_train_maps |= {A.comp_map((op_id,)) for op_id in range(A.N_OPS)}
A_train_maps |= {A.comp_map(tuple(s)) for s in A.SPLIT["train2"]}
A_train_maps |= {A.comp_map(tuple(s)) for s in A.SPLIT["train3"]}
A_primitive_maps = {A.comp_map((o,)) for o in range(A.N_OPS)}
D_primitive_maps = {D.comp_map((o,)) for o in range(D.N_OPS)}
primitive_overlap = D_primitive_maps & A_primitive_maps
assert not primitive_overlap, primitive_overlap

rep = {"ckpt": ARGS.ckpt, "arm": ARM, "d_bank_hash": D.bank_hash(),
       "model_seed": ck.get("model_seed"),
       "order_seed": ck.get("order_seed"),
       "D_primitive_overlap_with_A": 0,
       "A_train_map_count": len(A_train_maps)}
for name in ("diag2", "diag3"):
    seqs = D.SPLIT[name]
    novel = [s for s in seqs if D.comp_map(s) not in A_train_maps]
    over = [s for s in seqs if D.comp_map(s) in A_train_maps]
    rep[f"S1_{name}"] = score(cases_for(seqs, D.TPL_DIAG))
    rep[f"S1_{name}_Amap_novel"] = score(cases_for(novel, D.TPL_DIAG))
    rep[f"S1_{name}_Amap_overlap"] = score(cases_for(over, D.TPL_DIAG))
    rep[f"S2_{name}_allcut"] = score(cases_for(seqs, D.TPL_DIAG),
                                     cut=True)
    rep[f"{name}_n_novel"] = len(novel)
    rep[f"{name}_n_overlap"] = len(over)
    print(name, "S1", rep[f"S1_{name}"], "novel",
          rep[f"S1_{name}_Amap_novel"], "S2",
          rep[f"S2_{name}_allcut"], flush=True)

tag = f"{ARM}_m{rep['model_seed']}_o{rep['order_seed']}"
Path("results/scout").mkdir(parents=True, exist_ok=True)
Path(f"results/scout/scout_s1s2_{tag}.json").write_text(
    json.dumps(rep, indent=1))
print("S1S2_DONE", tag, flush=True)

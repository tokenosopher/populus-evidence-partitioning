"""Collision-stratified rescoring (referee mandate,
REVIEW_run3v_verdict_ruling.md §4): score every Run-3V final
checkpoint separately on map-novel vs map-redundant held-out
programmes. A held-out chain is MAP-REDUNDANT iff its composite
17-value map equals the composite map of at least one train2/train3
chain; MAP-NOVEL otherwise. Evaluation mirrors the run3v evaluator's
masked layout exactly (arm read from the checkpoint).

Usage (per ckpt):
  RUN3V_WORLD=F BRIDGE_OP_SEED=6011 BRIDGE_SPLIT_SEED=2203 \
  BRIDGE_GRAMMAR_SEED=7717 python scripts/bridge_run3v_stratified.py \
      --ckpt results/run3v_F_P_m200_o900_ckpt_final.pt
Writes results/run3v/run3v_strat_<tag>.json.
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
assert os.environ.get("BRIDGE_GRAMMAR_SEED") == "7717"
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from populus.bridge import BridgeA1                            # noqa
from populus.bridge_qwen import QwenGenome                     # noqa
from populus.bridge_tasks import (ANSWER_PREFIX, FORWARD_SPAN,  # noqa
                                  QUESTION, SPLIT, VALUE_SPAN,
                                  apply_chain)

P_ARGS = argparse.ArgumentParser()
P_ARGS.add_argument("--ckpt", required=True)
A = P_ARGS.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = json.load(open("results/bridge/serialization_contract.json"))
Tq, Ts = C["T_QUESTION"], C["T_SPAN"]
GRAMMAR = json.load(open("results/run3v/template_grammar_run3v.json"))
QUAL_PHR = {int(k): [r["text"] for r in v["qual"]]
            for k, v in GRAMMAR["ops"].items()}

lm = QwenGenome().attach_lora().to(DEV)
bridge = BridgeA1(lm).to(DEV).eval()
ck = torch.load(A.ckpt, map_location=DEV, weights_only=False)
bridge.load_state_dict(ck["bridge_trainable"], strict=False)
lm.core.load_state_dict(ck["lora"], strict=False)
ARM = ck["arm"]
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
def score(cases, batch=64):
    hits = 0
    for i in range(0, len(cases), batch):
        chunk = cases[i:i + batch]
        texts = [s for sp, _ in chunk for s in sp]
        s_ids, s_mask = lm.encode(texts, fixed_length=Ts, device=DEV)
        B = len(chunk)
        s_ids, s_mask = package(s_ids, s_mask, B)
        out = bridge.forward_episode(q_ids.expand(B, -1), s_ids,
                                     s_mask)
        corr = (out["logits"] - out["base_logits"])[:, label_ids]
        tgt = torch.tensor([a for _, a in chunk], device=DEV)
        hits += int((corr.argmax(-1) == tgt).sum())
    return round(hits / len(cases), 4) if cases else None


def fn(seq):
    return tuple(apply_chain(list(seq), x) for x in range(17))


train_fns = {fn(s) for s in SPLIT["train2"]} | \
            {fn(s) for s in SPLIT["train3"]}


def cases_for(seqs):
    cases = []
    for seq in seqs:
        for x in range(17):
            for j in range(4):
                spans = [VALUE_SPAN.format(x=x)]
                spans += [QUAL_PHR[o][(j + i) % 4]
                          for i, o in enumerate(seq)]
                while len(spans) < 4:
                    spans.append(FORWARD_SPAN)
                cases.append((spans, apply_chain(list(seq), x)))
    return cases


rep = {"ckpt": A.ckpt, "arm": ARM,
       "model_seed": ck.get("model_seed"),
       "order_seed": ck.get("order_seed")}
for name in ("diag2", "diag3"):
    novel = [s for s in SPLIT[name] if fn(s) not in train_fns]
    redun = [s for s in SPLIT[name] if fn(s) in train_fns]
    rep[f"{name}_n_novel_seqs"] = len(novel)
    rep[f"{name}_n_redundant_seqs"] = len(redun)
    rep[f"{name}_map_novel"] = score(cases_for(novel))
    rep[f"{name}_map_redundant"] = score(cases_for(redun))
    print(name, "novel", rep[f"{name}_map_novel"],
          f"({len(novel)} seqs)", "redundant",
          rep[f"{name}_map_redundant"], f"({len(redun)} seqs)",
          flush=True)

tag = f"F_{ARM}_m{rep['model_seed']}_o{rep['order_seed']}"
Path("results/run3v").mkdir(parents=True, exist_ok=True)
Path(f"results/run3v/run3v_strat_{tag}.json").write_text(
    json.dumps(rep, indent=1))
print("STRAT_DONE", tag, flush=True)

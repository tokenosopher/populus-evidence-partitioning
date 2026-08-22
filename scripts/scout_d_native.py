"""Family-D native-base full-information reporter (scout plan
amendment 4: REPORTED, not gated).

Frozen base Qwen2.5-0.5B-Instruct, adapter OFF, ALL evidence in one
context. Full enumeration of the D diagnostic banks (diag2, diag3)
with the diag template pool — the same case convention as S1 —
serialized raw and via the chat template. Diagnostic only: shows how
much of D is solvable by the base model without any society.

Usage:
  RUN3V_WORLD=F BRIDGE_OP_SEED=6011 BRIDGE_SPLIT_SEED=2203 \
  BRIDGE_GRAMMAR_SEED=7717 python scripts/scout_d_native.py
Writes results/scout/scout_d_native.json.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert os.environ.get("RUN3V_WORLD") == "F"
assert os.environ.get("BRIDGE_OP_SEED") == "6011"
assert os.environ.get("BRIDGE_SPLIT_SEED") == "2203"

from populus import bridge_tasks_d as D                       # noqa
from populus.bridge_qwen import QwenGenome                    # noqa
from populus.bridge_tasks import ANSWER_PREFIX, QUESTION      # noqa

assert D.bank_hash() == "2bd0b301a5b079b1"

DEV = "cuda"
lm = QwenGenome().to(DEV)
label_ids = torch.tensor([r["token_id"] for r in json.load(
    open("results/bridge/label_context.json"))["labels"]],
    device=DEV)


def cases_for(seqs, tpl_pool):
    cases = []
    for seq in seqs:
        for x in range(17):
            for j in range(len(tpl_pool)):
                spans = [D.VALUE_SPAN.format(x=x)]
                spans += [D.render_op(o, tpl_pool[(j + i)
                                                  % len(tpl_pool)])
                          for i, o in enumerate(seq)]
                cases.append((spans, D.apply_chain(list(seq), x)))
    return cases


@torch.no_grad()
def score(cases, chat: bool):
    hits = 0
    for spans, answer in cases:
        body = "\n".join(spans) + "\n" + QUESTION + ANSWER_PREFIX
        if chat:
            body = lm.tok.apply_chat_template(
                [{"role": "user",
                  "content": "\n".join(spans) + "\n" + QUESTION}],
                tokenize=False, add_generation_prompt=True) \
                + "The answer is"
        ids = torch.tensor([lm.tok.encode(
            body, add_special_tokens=False)], device=DEV)
        h = lm.forward_embeds(lm.embed(ids), torch.ones_like(ids),
                              adapter_mode="base")[:, -1]
        pred = lm.lm_head(h)[:, label_ids].argmax(-1).item()
        hits += int(pred == answer)
    return round(hits / len(cases), 4)


rep = {"d_bank_hash": D.bank_hash(),
       "gpu": torch.cuda.get_device_name(0)}
for name in ("diag2", "diag3"):
    cases = cases_for(D.SPLIT[name], D.TPL_DIAG)
    rep[f"native_fullinfo_{name}_raw"] = score(cases, chat=False)
    rep[f"native_fullinfo_{name}_chat"] = score(cases, chat=True)
    rep[f"{name}_n_cases"] = len(cases)
    print(name, "raw", rep[f"native_fullinfo_{name}_raw"],
          "chat", rep[f"native_fullinfo_{name}_chat"], flush=True)

Path("results/scout").mkdir(parents=True, exist_ok=True)
Path("results/scout/scout_d_native.json").write_text(
    json.dumps(rep, indent=1))
print("D_NATIVE_DONE", flush=True)

"""Family-D preflight artifact (REVIEW_scout_code.md fix 12).
Pure-math checks run anywhere; the T_SPAN tokenizer check runs where
the pinned tokenizer is available (GPU box) and is marked pending
otherwise. Writes results/scout/family_d_preflight.json.
"""
import itertools
import json
import os
import sys
from pathlib import Path

# CRITICAL: family-A seeds must be pinned BEFORE any populus import
# (bridge_tasks_d transitively imports bridge_tasks at module load).
os.environ.setdefault("BRIDGE_OP_SEED", "6011")
os.environ.setdefault("BRIDGE_SPLIT_SEED", "2203")
assert os.environ["BRIDGE_OP_SEED"] == "6011"
assert os.environ["BRIDGE_SPLIT_SEED"] == "2203"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from populus import bridge_tasks_d as D                          # noqa

rep = {}
prims = [D.comp_map((o,)) for o in range(12)]
rep["distinct_primitive_maps"] = len(set(prims)) == 12
rep["all_bijections"] = all(
    sorted(D.apply_op(o, x) for x in range(17)) == list(range(17))
    for o in range(12))
ok = True
for depth in (1, 2, 3):
    for seq in itertools.product(range(12), repeat=depth):
        a, b = D.comp_map(seq)
        for x in range(17):
            if D.apply_chain(list(seq), x) != (a * x + b) % 17:
                ok = False
rep["chain_matches_comp_map_exhaustive"] = ok

from populus import bridge_tasks as A                            # noqa
assert A.OP_SEED == 6011 and A.SPLIT_SEED == 2203
rep["primitive_overlap_with_A"] = len(
    set(prims) & {A.comp_map((o,)) for o in range(12)})

s = D.SPLIT
forbidden = []
f_ho = set(map(tuple, s["f_ho"]))
for name in ("diag2", "diag3", "ho2"):
    forbidden.append(all(D.comp_map(x) not in f_ho for x in s[name]))
excl = set(map(tuple, s["excluded_pairs"]))
forbidden.append(all(
    (t[0], t[1]) not in excl and (t[1], t[2]) not in excl
    for t in s["train3"]))
rep["forbidden_intersections_empty"] = all(forbidden)
train2 = set(map(tuple, s["train2"]))
rep["termination_balance"] = all(
    (t[0], t[1]) in train2 and (t[1], t[2]) in train2
    for t in s["train3"])
rep["bank_hash"] = D.bank_hash()
rep["bank_hash_expected"] = "2bd0b301a5b079b1"
rep["hash_match"] = rep["bank_hash"] == rep["bank_hash_expected"]

# Tokenizer fit check. On an official machine this must pass or abort.
try:
    from populus.bridge_qwen import QwenGenome

    lm = QwenGenome()
    C = json.load(open("results/bridge/serialization_contract.json"))
    Ts = C["T_SPAN"]

    texts = [D.VALUE_SPAN.format(x=x) for x in range(17)]
    texts += [D.FORWARD_SPAN, A.FORWARD_SPAN]
    texts += [
        D.render_op(op_id, tpl)
        for op_id in range(12)
        for tpl in range(14)
    ]

    encoded = lm.tok(
        texts,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]
    lengths = [len(ids) for ids in encoded]
    overflow = [
        {"text": text, "n_tokens": n_tokens}
        for text, n_tokens in zip(texts, lengths)
        if n_tokens > Ts
    ]

    rep["t_span_fit"] = {
        "checked": len(texts),
        "max_tokens": max(lengths),
        "limit": Ts,
        "overflow": overflow,
    }
    assert not overflow, (
        f"Family-D span overflow at T_SPAN={Ts}: {overflow}"
    )

except ModuleNotFoundError as e:
    rep["t_span_fit"] = (
        f"PENDING (tokenizer dependencies unavailable: {e})"
    )

Path("results/scout").mkdir(parents=True, exist_ok=True)
Path("results/scout/family_d_preflight.json").write_text(
    json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items()
                  if k != "t_span_fit"}, indent=1))
print("t_span_fit:", str(rep["t_span_fit"])[:100])

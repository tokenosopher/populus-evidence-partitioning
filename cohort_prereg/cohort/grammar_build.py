"""Frozen neutral-grammar JSON builder — PREREG section 4.

Deterministic: (op table, grammar_seed, pinned tokenizer) -> grammar JSON.
The tokenizer's only role is the frozen T_SPAN length filter applied IN
SHUFFLE ORDER before taking n_take — a pure filter, never re-authoring
(cohort/neutral_grammar.build_split contract). Split 24 train / 4 dev /
4 qual / 4 sealed per op. All texts re-asserted neutral at build time.
"""
import hashlib
import json

from . import neutral_grammar as ng


def build_grammar(ops, grammar_seed, token_len, t_span=33):
    """ops: list of (a, b); token_len: text -> token count at the pinned
    tokenizer WITHOUT leading space (lm.encode convention)."""
    out = {"ops": {}, "meta": {"grammar_seed": grammar_seed, "t_span": t_span,
                               "n_ops": len(ops),
                               "counts": {"train": 24, "dev": 4, "qual": 4, "sealed": 4},
                               "value_span": ng.VALUE_SPAN_NEUTRAL,
                               "forward_span": ng.FORWARD_SPAN_NEUTRAL}}
    ng.assert_neutral(ng.FORWARD_SPAN_NEUTRAL)
    for x in range(17):
        ng.assert_neutral(ng.VALUE_SPAN_NEUTRAL.format(x=x))
        assert token_len(ng.VALUE_SPAN_NEUTRAL.format(x=x)) <= t_span
    assert token_len(ng.FORWARD_SPAN_NEUTRAL) <= t_span
    for op_id, (a, b) in enumerate(ops):
        sp = ng.build_split(a, b, grammar_seed)
        fit = [(k, t) for k, t in sp["candidates_in_shuffle_order"]
               if token_len(t) <= t_span]
        assert len(fit) >= sp["n_take"], \
            f"op {op_id}: only {len(fit)} renderings fit T_SPAN={t_span}"
        take = fit[:sp["n_take"]]
        for _, t in take:
            ng.assert_neutral(t)
        texts = [t for _, t in take]
        cs = sp["split_counts"]
        i1 = cs["train"]; i2 = i1 + cs["dev"]; i3 = i2 + cs["qual"]
        out["ops"][str(op_id)] = {
            "a": a, "b": b,
            "train": texts[:i1], "dev": texts[i1:i2],
            "qual": texts[i2:i3], "sealed": texts[i3:sp["n_take"]],
            "keys": [list(k) for k, _ in take],
            "n_candidates_fit": len(fit)}
    body = json.dumps(out, sort_keys=True, indent=1)
    return out, hashlib.sha256(body.encode()).hexdigest()


def write_grammar(path, ops, grammar_seed, token_len, t_span=33):
    out, sha = build_grammar(ops, grammar_seed, token_len, t_span)
    with open(path, "w") as f:
        json.dump(out, f, sort_keys=True, indent=1)
    return sha

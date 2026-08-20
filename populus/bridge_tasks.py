"""Bridge-A1: Linguistic Operator Chain — programme banks and splits.

Spec: PLAN_RUN3.md as amended by REVIEW_consolidated_ruling.md DP-3 and
REVIEW_r2n_ruling.md §C (hierarchical split non-negotiable) and
REVIEW_autopsy_ruling.md §4b (balanced terminated/continuing exposure).

Domain: x in Z17; 12 fixed bijective affine primitives from a fresh
secret table (BRIDGE_OP_SEED, never used by any earlier stage). All
primitives are seen in training; ordered COMPOSITIONS are split.

Hierarchical split (machine-checked; scripts/bridge_bank_audit.py):
  EXCLUDED_PAIRS = H2 (held-out pairs) ∪ pairs whose composite map is
  in the held-out FUNCTION set F_HO.
  - train2: all pairs not excluded (terminated 2-op programmes).
  - train3: triples containing NO excluded pair as a contiguous
    subchain and whose composite map is not in F_HO and not selected
    for ho3_familiar.
  - ho2: H2 (standalone; never trained at any depth, never inside any
    training triple — the Run-2 containment failure cannot recur).
  - ho3_familiar: unseen whole triples assembled from familiar
    subchains (both contiguous pairs trained).
  - ho3_h2: triples containing a held-out pair as contiguous subchain
    (structurally untrainable; sealed diagnostic).
  - ho_fn: programmes whose COMPOSITE FUNCTION is held out (F_HO maps
    never produced by any training programme of any depth).
  Termination balance holds by construction: every pair occurring as
  a contiguous subchain of a training triple is itself a train2
  terminated programme.

Training mixture (frozen): 50/50 two-op / three-op programme level.
2-op episodes use the SAME chain and terminal interface (ops at cells
1,2; cell 3 holds a forwarder span; mouth reads cell 3 after round 3).

Surface language: per primitive, 14 description templates — indices
0-7 train, 8-9 diagnostic, 10-13 sealed held-out.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random

K = 17
N_OPS = 12
OP_SEED = int(os.environ.get("BRIDGE_OP_SEED", 4021))
SPLIT_SEED = int(os.environ.get("BRIDGE_SPLIT_SEED", 977))
N_H2 = 29            # held-out pairs (of 144)
N_FN = 18            # held-out composite functions (of 272 affine maps)
N_HO3_FAMILIAR = 200  # unseen whole triples from familiar subchains
N_DIAG2 = 12         # qualification diagnostic pairs (programme-disjoint)
N_DIAG3 = 60         # qualification diagnostic triples


def make_ops(seed=OP_SEED):
    rng = random.Random(seed)
    ops = []
    while len(ops) < N_OPS:
        a = rng.randrange(1, K)
        b = rng.randrange(0, K)
        if (a, b) not in ops and (a, b) != (1, 0):
            ops.append((a, b))
    return ops


OPS = make_ops()


def apply_op(op_id, x):
    a, b = OPS[op_id]
    return (a * x + b) % K


def apply_chain(seq, x):
    for o in seq:
        x = apply_op(o, x)
    return x


def comp_map(seq):
    a, b = 1, 0
    for o in seq:
        ao, bo = OPS[o]
        a, b = (ao * a) % K, (ao * b + bo) % K
    return (a, b)


def _build_split():
    """v2.1.1 construction order (REVIEW_run3_v21_ruling.md §1): the
    held-out FUNCTION set F_HO is chosen FIRST; every subsequently
    drawn bank is constrained away from it, so no diagnostic or H2
    programme can preview a sealed composite function.
    Invariants enforced: maps(diag2 ∪ diag3) ∩ F_HO = ∅;
    maps(H2) ∩ F_HO = ∅; H3-H2 is split into H3-H2-only (familiar
    composite function) and H3-H2xFHO (joint-hardness, reported
    separately)."""
    rng = random.Random(SPLIT_SEED)
    pairs = list(itertools.product(range(N_OPS), repeat=2))
    triples = list(itertools.product(range(N_OPS), repeat=3))
    # 1. held-out composite functions first (never primitive/identity)
    triple_maps = sorted({comp_map(t) for t in triples})
    protected = {comp_map((o,)) for o in range(N_OPS)} | {(1, 0)}
    f_ho = set(rng.sample([m for m in triple_maps
                           if m not in protected], N_FN))
    fn_pairs = [p for p in pairs if comp_map(p) in f_ho]
    # 2. H2 from pairs whose composite map is NOT held out
    h2_pool = [p for p in pairs if comp_map(p) not in f_ho]
    h2 = set(rng.sample(h2_pool, N_H2))
    # 3. diagnostic pairs: disjoint from H2, maps outside F_HO
    diag2 = set(rng.sample([p for p in h2_pool if p not in h2],
                           N_DIAG2))
    excluded_pairs = h2 | diag2 | set(fn_pairs)

    def has_excluded_subchain(t):
        return (t[0], t[1]) in excluded_pairs \
            or (t[1], t[2]) in excluded_pairs

    train2 = [p for p in pairs if p not in excluded_pairs]
    clean3 = [t for t in triples
              if not has_excluded_subchain(t)
              and comp_map(t) not in f_ho]
    h2_sub = [t for t in triples
              if (t[0], t[1]) in h2 or (t[1], t[2]) in h2]
    ho3_h2_only = [t for t in h2_sub if comp_map(t) not in f_ho]
    ho3_h2_fho = [t for t in h2_sub if comp_map(t) in f_ho]
    ho_fn_triples = [t for t in triples if comp_map(t) in f_ho]
    ho3_familiar = set(rng.sample(clean3, N_HO3_FAMILIAR))
    rest3 = [t for t in clean3 if t not in ho3_familiar]
    diag3 = set(rng.sample(rest3, N_DIAG3))
    train3 = [t for t in rest3 if t not in diag3]
    return {
        "train2": train2, "train3": train3,
        "diag2": sorted(diag2), "diag3": sorted(diag3),
        "ho2": sorted(h2), "ho3_familiar": sorted(ho3_familiar),
        "ho3_h2_only": ho3_h2_only, "ho3_h2_fho": ho3_h2_fho,
        "ho_fn": {"pairs": fn_pairs, "triples": ho_fn_triples},
        "f_ho": sorted(f_ho), "excluded_pairs": sorted(excluded_pairs),
    }


SPLIT = _build_split()


def bank_hash():
    """Full-coverage immutable hash (v2.1.1 ruling §1): commits to the
    operator table, EVERY programme bank, F_HO, all templates, the
    exact question and answer-prefix bytes, the label mapping, and the
    generation seeds."""
    h = hashlib.sha256()
    payload = {"ops": OPS, "op_seed": OP_SEED,
               "split_seed": SPLIT_SEED,
               "banks": {k: sorted(v) for k, v in SPLIT.items()
                         if isinstance(v, list)},
               "ho_fn": {"pairs": sorted(SPLIT["ho_fn"]["pairs"]),
                         "triples": sorted(SPLIT["ho_fn"]["triples"])},
               "templates": OP_TEMPLATES,
               "template_split": {"train": TPL_TRAIN, "diag": TPL_DIAG,
                                  "sealed": TPL_SEALED},
               "value_span": VALUE_SPAN, "forward_span": FORWARD_SPAN,
               "question": QUESTION, "answer_prefix": ANSWER_PREFIX,
               "labels": LABELS}
    h.update(json.dumps(payload, sort_keys=True,
                        default=list).encode())
    return h.hexdigest()[:16]


# ------------------------- surface language --------------------------
# 14 templates per primitive; {a} multiplier, {b} addend. Indices:
# 0-7 train, 8-9 diagnostic, 10-13 sealed held-out.
OP_TEMPLATES = [
    "Multiply the number by {a}, then add {b}. Work modulo 17.",
    "Take the value you receive, scale it by a factor of {a}, and "
    "then increase it by {b} (everything wraps around at 17).",
    "Your rule: new value = {a} times the old value plus {b}, "
    "reduced mod 17.",
    "Apply this transformation: first times {a}, afterwards plus {b}; "
    "results stay on the 0-16 clock.",
    "The operation you must perform is multiplication by {a} followed "
    "by addition of {b}, in arithmetic modulo 17.",
    "Whatever number reaches you, compute {a}*n + {b} and keep only "
    "the remainder after dividing by 17.",
    "Stretch the incoming number {a}-fold, then shift it up by {b}, "
    "wrapping on a 17-hour clock.",
    "Use the rule n -> ({a}n + {b}) mod 17 on the number you are "
    "given.",
    # ---- diagnostic (8-9) ----
    "Transform the received value: multiply it by {a} and add {b}; "
    "all arithmetic is mod 17.",
    "Your secret instruction is to compute {a} times the number, "
    "plus {b}, modulo seventeen.",
    # ---- sealed held-out (10-13) ----
    "When a number arrives, produce {a} copies of it summed together, "
    "then count {b} further, wrapping past 16 back to 0.",
    "The assigned rule takes n to {a}n+{b} on the mod-17 number "
    "circle.",
    "Enlarge the value by a multiple of {a}; follow that with a step "
    "of +{b}; remain within modulo-17 arithmetic.",
    "You must send onward the remainder of ({a} times n plus {b}) "
    "divided by 17.",
]
TPL_TRAIN = list(range(0, 8))
TPL_DIAG = [8, 9]
TPL_SEALED = list(range(10, 14))

FORWARD_SPAN = ("You hold no operation. Pass the number you receive "
                "onward unchanged.")
VALUE_SPAN = "Your private note: the starting number is {x}."
# Canonical answer labels: letters A-Q (A=0 ... Q=16). Numeric labels
# 10-16 tokenize as TWO tokens in the pinned Qwen vocabulary, which
# violates the single-token-label requirement (REVIEW_consolidated_
# ruling.md DP-3); letters A-Q are tokenizer-verified single tokens.
LABELS = "ABCDEFGHIJKLMNOPQ"
QUESTION = ("Private notes were distributed to the cells. After every "
            "instruction has been applied in cell order, what is the "
            "final number? Answer with a single capital letter using "
            "the code A=0, B=1, C=2, ... Q=16.")
# Frozen answer boundary (v2.1.1 ruling §4): the mouth reads the next-
# token position after QUESTION + ANSWER_PREFIX. Chosen empirically at
# the pinned tokenizer revision: appending each label letter to this
# exact context adds EXACTLY ONE new token (17 unique ids), verified
# and archived by scripts/bridge_label_context.py.
ANSWER_PREFIX = "\nThe answer is"


def render_op(op_id, tpl_idx):
    a, b = OPS[op_id]
    return OP_TEMPLATES[tpl_idx].format(a=a, b=b)


def episode(seq, x, tpl_ids=None, tpl_pool=TPL_TRAIN, rng=None):
    """Build one episode dict. seq is a 2- or 3-op programme; spans
    always fill cells 0-3 with the SAME chain and terminal interface
    (2-op: forwarder span at cell 3)."""
    if tpl_ids is None:
        rng = rng or random
        tpl_ids = [rng.choice(tpl_pool) for _ in seq]
    spans = [VALUE_SPAN.format(x=x)]
    for o, t in zip(seq, tpl_ids):
        spans.append(render_op(o, t))
    if len(seq) == 2:
        spans.append(FORWARD_SPAN)
    return {"spans": spans, "question": QUESTION,
            "answer": apply_chain(seq, x), "ops": list(seq), "x": x,
            "depth": len(seq), "templates": list(tpl_ids)}


def split_report():
    """Machine-checked overlap report (REVIEW_r2n_ruling.md §C): every
    forbidden intersection must be exactly empty."""
    s = SPLIT
    tr3_sub = {(t[0], t[1]) for t in s["train3"]} \
        | {(t[1], t[2]) for t in s["train3"]}
    tr_maps = {comp_map(p) for p in s["train2"]} \
        | {comp_map(t) for t in s["train3"]}
    diag_maps = {comp_map(p) for p in s["diag2"]} \
        | {comp_map(t) for t in s["diag3"]}
    fho = set(s["f_ho"])
    report = {
        "bank_hash": bank_hash(), "op_seed": OP_SEED,
        "split_seed": SPLIT_SEED,
        "sizes": {k: len(s[k]) for k in
                  ("train2", "train3", "diag2", "diag3", "ho2",
                   "ho3_familiar", "ho3_h2_only", "ho3_h2_fho")}
        | {"ho_fn_pairs": len(s["ho_fn"]["pairs"]),
           "ho_fn_triples": len(s["ho_fn"]["triples"]),
           "f_ho": len(s["f_ho"])},
        "forbidden_intersections": {
            "ho2_in_train2": len(set(s["ho2"]) & set(s["train2"])),
            "ho2_as_train3_subchain": len(set(s["ho2"]) & tr3_sub),
            "diag2_in_train2": len(set(s["diag2"]) & set(s["train2"])),
            "diag2_as_train3_subchain":
                len(set(s["diag2"]) & tr3_sub),
            "diag2_overlap_ho2": len(set(s["diag2"]) & set(s["ho2"])),
            "diag3_in_train3": len(set(s["diag3"]) & set(s["train3"])),
            "diag3_overlap_ho3_familiar":
                len(set(s["diag3"]) & set(s["ho3_familiar"])),
            "ho3_familiar_in_train3":
                len(set(s["ho3_familiar"]) & set(s["train3"])),
            "f_ho_maps_reachable_by_training":
                len(fho & tr_maps),
            "train3_subchains_not_terminated_in_train2":
                len(tr3_sub - set(s["train2"])),
            "diag_maps_in_f_ho": len(diag_maps & fho),
            "h2_maps_in_f_ho":
                len({comp_map(p) for p in s["ho2"]} & fho),
            "ho3_h2_only_maps_in_f_ho":
                len({comp_map(t) for t in s["ho3_h2_only"]} & fho),
        },
        "cross_bank_programme_intersections": {
            f"{a}&{b}": len(set(map(tuple, s[a]))
                            & set(map(tuple, s[b])))
            for a, b in (("train3", "ho3_familiar"),
                         ("train3", "diag3"),
                         ("train3", "ho3_h2_only"),
                         ("train3", "ho3_h2_fho"),
                         ("diag3", "ho3_h2_only"),
                         ("ho3_familiar", "ho3_h2_only"))},
        "termination_balance_ok":
            (set(t[:2] for t in s["train3"])
             | set(t[1:] for t in s["train3"])) <= set(s["train2"]),
        "template_split": {"train": TPL_TRAIN, "diag": TPL_DIAG,
                          "sealed": TPL_SEALED},
    }
    report["all_clear"] = (
        all(v == 0 for v in report["forbidden_intersections"].values())
        and all(v == 0 for v in
                report["cross_bank_programme_intersections"].values())
        and report["termination_balance_ok"])
    return report

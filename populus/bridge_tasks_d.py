"""Family D — ring-pointer world (universality scout, PLAN_SCOUT_
UNIVERSALITY.md). Shares the frozen interface with family A exactly:
K=17 state space, LABELS A-Q, QUESTION + ANSWER_PREFIX bytes, span
budget, four-slot episode shape. Differs in surface language (ring
navigation vocabulary; the affine coefficients are NEVER stated) and
operator identities. All D ops are affine bijections on Z17 by design
(the scout isolates linguistic-frame novelty from algebraic novelty;
referee ruling scout-design-ratification governs any amendment).

Ops (12, fixed semantics; indexing shuffled by D_OP_SEED):
  forward k  (k in 2,3,7,8):  x -> x+k  (fwd 5 replaced by fwd 8 pre-freeze: (1,5) collided with an A primitive; ruling map-control 2)
  back k     (k in 1,4,6):    x -> x-k
  reflect c  (c in 0,3,8):    x -> 2c-x (reflection through c)
  stretch m  (m in 2,3):      x -> m*x
Seeds: D_OP_SEED=8021, D_SPLIT_SEED=3407. The template grammar is a
fixed literal list (not procedurally generated); D_GRAMMAR_SEED=9313
is recorded as inert manifest metadata only.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random

from populus.bridge_tasks import (ANSWER_PREFIX, LABELS, QUESTION,
                                  FORWARD_SPAN as _A_FORWARD)  # noqa

K = 17
D_OP_SEED = int(os.environ.get("BRIDGE_D_OP_SEED", 8021))
D_SPLIT_SEED = int(os.environ.get("BRIDGE_D_SPLIT_SEED", 3407))
N_OPS = 12
N_FN = 18
N_H2 = 29
N_HO3_FAMILIAR = 200
N_DIAG2 = 12
N_DIAG3 = 60

_SEMANTIC = ([("fwd", k) for k in (2, 3, 7, 8)]
             + [("back", k) for k in (1, 4, 6)]
             + [("refl", c) for c in (0, 3, 8)]
             + [("stretch", m) for m in (2, 3)])


def make_ops(seed=D_OP_SEED):
    rng = random.Random(seed)
    sem = list(_SEMANTIC)
    rng.shuffle(sem)
    return sem


D_OPS = make_ops()


def _affine(kind, p):
    if kind == "fwd":
        return (1, p % K)
    if kind == "back":
        return (1, (-p) % K)
    if kind == "refl":
        # reflection THROUGH slot c: x -> 2c - x (matches the
        # template language "mirror/reflect through slot {p}";
        # pre-launch erratum: earlier draft implemented c - x)
        return (K - 1, (2 * p) % K)
    return (p % K, 0)


def apply_op(op_id, x):
    a, b = _affine(*D_OPS[op_id])
    return (a * x + b) % K


def apply_chain(seq, x):
    for o in seq:
        x = apply_op(o, x)
    return x


def comp_map(seq):
    a, b = 1, 0
    for o in seq:
        ao, bo = _affine(*D_OPS[o])
        a, b = (ao * a) % K, (ao * b + bo) % K
    return (a, b)


# --------------------- surface language (ring) -----------------------
# 14 templates PER KIND; coefficients never appear — only navigation
# vocabulary. Indices 0-7 train, 8-9 diagnostic, 10-13 sealed.
D_TEMPLATES = {
 "fwd": [
  "Move the pointer forward {p} slots around the ring.",
  "Advance {p} positions clockwise; the ring has 17 slots and wraps.",
  "Step the marker ahead by {p} places (past slot 16 comes slot 0).",
  "Slide {p} slots onward along the circle of seventeen.",
  "Push the pointer {p} steps in the increasing direction, wrapping.",
  "Walk {p} slots forward; after the last slot the ring restarts.",
  "Rotate your position {p} notches ahead on the seventeen-slot dial.",
  "Carry the marker {p} places clockwise round the ring.",
  "Nudge the pointer onward through {p} slots of the wrapping ring.",
  "Progress {p} stops around the circle; it loops at seventeen.",
  "Shift ahead {p} spaces on the ring road of seventeen stops.",
  "Send the marker {p} slots further along, wrapping at the end.",
  "Count {p} slots onward from where you stand and stop there.",
  "Advance the dial {p} clicks in the forward sense.",
 ],
 "back": [
  "Move the pointer back {p} slots around the ring.",
  "Retreat {p} positions counter-clockwise; the ring wraps at 17.",
  "Step the marker {p} places backwards (before slot 0 comes 16).",
  "Slide {p} slots in reverse along the circle of seventeen.",
  "Pull the pointer {p} steps in the decreasing direction, wrapping.",
  "Walk {p} slots backward; before the first slot the ring restarts.",
  "Rotate your position {p} notches back on the seventeen-slot dial.",
  "Carry the marker {p} places anticlockwise round the ring.",
  "Nudge the pointer backward through {p} slots of the wrapping ring.",
  "Regress {p} stops around the circle; it loops at seventeen.",
  "Shift back {p} spaces on the ring road of seventeen stops.",
  "Send the marker {p} slots earlier along, wrapping at the start.",
  "Count {p} slots backward from where you stand and stop there.",
  "Turn the dial {p} clicks in the reverse sense.",
 ],
 "refl": [
  "Jump to the mirror slot: your position reflected through slot {p}.",
  "Flip across slot {p}: land as far past it as you now sit before "
  "it, on the wrapping ring.",
  "Reflect the pointer in slot {p}; distances are preserved, side "
  "is swapped, the ring wraps.",
  "Mirror your position about slot {p} on the seventeen-slot circle.",
  "Swap to the slot symmetric to yours with respect to slot {p}.",
  "Fold the ring at slot {p} and move to where your slot lands.",
  "Take the reflection of your position through the axis at slot "
  "{p}, wrapping past the ends.",
  "Leap to the opposite side of slot {p}, matching your current "
  "distance from it.",
  "Invert your position around slot {p} on the wrapping dial.",
  "Cross to the mirror-image slot determined by the axis {p}.",
  "Pivot about slot {p}: your new slot is its reflection of yours.",
  "Exchange your slot for its mirror twin across slot {p}.",
  "Land symmetric to your slot, the symmetry centred on slot {p}.",
  "Reflect through the marker fixed at slot {p} and stay wrapped.",
 ],
 "stretch": [
  "Jump to the slot whose index is {p} times your current one, "
  "wrapping on the seventeen-slot ring.",
  "Multiply your slot number by {p} and go there (the ring wraps).",
  "Leap to position {p}-fold of the one you occupy, modulo the ring.",
  "Move to the slot at {p} times your index around the circle.",
  "Scale your position by {p} on the wrapping dial and land there.",
  "Take {p} copies of your slot index, summed around the ring, and "
  "go to that slot.",
  "Relocate to the {p}-times-farther slot measured from slot 0, "
  "wrapping past the end.",
  "Your next slot is your current index repeated {p} times around "
  "the ring.",
  "Go to the product slot: {p} times where you now stand, wrapped.",
  "Spring to the slot indexed {p} times your own, on the ring.",
  "Stretch your distance from slot 0 by the factor {p}, wrapping.",
  "Bound ahead to {p}-fold your index on the circle of seventeen.",
  "Set your position to {p} multiples of the current slot number.",
  "Occupy the slot standing at {p} times your index, ring-wrapped.",
 ],
}
TPL_TRAIN = list(range(0, 8))
TPL_DIAG = [8, 9]
TPL_SEALED = list(range(10, 14))

FORWARD_SPAN = ("You hold no move. Pass the slot you receive onward "
                "unchanged.")
VALUE_SPAN = "Your private note: the pointer starts at slot {x}."


def render_op(op_id, tpl_idx):
    kind, p = D_OPS[op_id]
    return D_TEMPLATES[kind][tpl_idx].format(p=p)


def _build_split():
    rng = random.Random(D_SPLIT_SEED)
    pairs = list(itertools.product(range(N_OPS), repeat=2))
    triples = list(itertools.product(range(N_OPS), repeat=3))
    triple_maps = sorted({comp_map(t) for t in triples})
    protected = {comp_map((o,)) for o in range(N_OPS)} | {(1, 0)}
    f_ho = set(rng.sample([m for m in triple_maps
                           if m not in protected], N_FN))
    fn_pairs = [p for p in pairs if comp_map(p) in f_ho]
    h2_pool = [p for p in pairs if comp_map(p) not in f_ho]
    h2 = set(rng.sample(h2_pool, N_H2))
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
    return {"train2": train2, "train3": train3,
            "diag2": sorted(diag2), "diag3": sorted(diag3),
            "ho2": sorted(h2), "ho3_familiar": sorted(ho3_familiar),
            "ho3_h2_only": ho3_h2_only, "ho3_h2_fho": ho3_h2_fho,
            "ho_fn": {"pairs": fn_pairs, "triples": ho_fn_triples},
            "f_ho": sorted(f_ho),
            "excluded_pairs": sorted(excluded_pairs)}


SPLIT = _build_split()


def bank_hash():
    h = hashlib.sha256()
    payload = {"ops": D_OPS, "op_seed": D_OP_SEED,
               "split_seed": D_SPLIT_SEED,
               "banks": {k: sorted(v) for k, v in SPLIT.items()
                         if isinstance(v, list)},
               "ho_fn": {"pairs": sorted(SPLIT["ho_fn"]["pairs"]),
                         "triples": sorted(SPLIT["ho_fn"]["triples"])},
               "templates": D_TEMPLATES,
               "template_split": {"train": TPL_TRAIN, "diag": TPL_DIAG,
                                  "sealed": TPL_SEALED},
               "value_span": VALUE_SPAN, "forward_span": FORWARD_SPAN,
               "question": QUESTION, "answer_prefix": ANSWER_PREFIX,
               "labels": LABELS}
    h.update(json.dumps(payload, sort_keys=True,
                        default=list).encode())
    return h.hexdigest()[:16]


def episode(seq, x, tpl_ids=None, tpl_pool=TPL_TRAIN, rng=None):
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

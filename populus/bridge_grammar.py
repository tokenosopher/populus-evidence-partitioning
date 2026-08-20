"""D2 procedural template grammar (REVIEW_solvability_ruling §3a).

Deterministic compositional generator for operation renderings:
independent variation in frame (imperative/declarative/functional),
operation order statement (forward / add-stated-after), variable
naming, modulo wording, and optional nonsemantic filler. Exact
coefficient semantics unchanged. Per primitive: the first 36 DISTINCT
renderings (in fixed enumeration order, seeded shuffle) that fit the
frozen T_SPAN token budget are split 24 train / 4 development / 4
qualification / 4 sealed — generated and hashed BEFORE any remedy
run (scripts/bridge_grammar_build.py verifies token lengths at the
pinned tokenizer and emits the hashed artifact).

Conditional on D0 identifying a language-axis failure; pre-built so
the ladder never waits on authoring.
"""
from __future__ import annotations

import hashlib
import json
import os
import random

GRAMMAR_SEED = int(os.environ.get("BRIDGE_GRAMMAR_SEED", 5150))

FRAMES = [
    "{core}",
    "Your rule: {core}",
    "Your instruction is this: {core}",
    "You must do the following: {core}",
    "The operation to perform: {core}",
    "Apply this rule: {core}",
]
VARIABLES = ["the number", "the value you receive", "n",
             "the incoming value"]
ORDERS = [
    "multiply {var} by {a}, then add {b}",
    "add {b} after multiplying {var} by {a}",
    "take {var} times {a} and increase the result by {b}",
    "compute {a} times {var} plus {b}",
]
MODS = [
    ", all modulo 17.",
    ", keeping only the remainder after dividing by 17.",
    "; results wrap around at 17.",
    ", staying on the 0-16 clock.",
    " in mod-17 arithmetic.",
]
FILLERS = ["", " Do this exactly.", " Work carefully."]


def all_renderings(a: int, b: int):
    """Deterministic enumeration of the full grammar product for one
    (a, b) primitive, seeded-shuffled so set splits are not ordered by
    grammar axis."""
    outs = []
    for fi, frame in enumerate(FRAMES):
        for vi, var in enumerate(VARIABLES):
            for oi, order in enumerate(ORDERS):
                for mi, mod in enumerate(MODS):
                    for gi, filler in enumerate(FILLERS):
                        core = order.format(var=var, a=a, b=b)
                        text = frame.format(core=core)
                        if not text[0].isupper():
                            text = text[0].upper() + text[1:]
                        text = text.rstrip(".") if mod.startswith(",") \
                            or mod.startswith(";") or mod.startswith(" i") \
                            else text
                        text = text + mod + filler
                        outs.append(((fi, vi, oi, mi, gi), text))
    rng = random.Random(GRAMMAR_SEED + a * 31 + b)
    rng.shuffle(outs)
    return outs


def grammar_hash(artifact: dict) -> str:
    return hashlib.sha256(json.dumps(artifact, sort_keys=True)
                          .encode()).hexdigest()[:16]

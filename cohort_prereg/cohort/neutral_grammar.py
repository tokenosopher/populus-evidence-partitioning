"""Fresh-world NEUTRAL grammar generator — PREREG section 4.

Third-person operator templates only. No template addresses the reader as
"you"/"your" (removes the linguistic-addressing aggravator flagged in the
parent audit). Deterministic enumeration + seeded shuffle; the token-length
fit check against the pinned tokenizer runs on the admission box (marked
TOKENIZER-PREFLIGHT below) and is NOT a design choice.

This module is pure string construction: fully testable for determinism,
neutrality (no second person), and split disjointness without a GPU.
"""
import hashlib
import json
import re

# strictly third-person; "you"/"your" forbidden by construction and asserted
FRAMES = [
    "The operation specified for this slot is to {core}.",
    "The rule for this slot is to {core}.",
    "This slot's operation is to {core}.",
    "For this position, the assigned transformation is to {core}.",
    "The assigned instruction at this position is to {core}.",
    "At this slot, the specified rule is to {core}.",
]
VARIABLES = ["the number", "the incoming value", "the received number", "the input"]
ORDERS = [
    "multiply {var} by {a} and then add {b}",
    "add {b} to {a} times {var}",
    "take {var} times {a}, then increase it by {b}",
    "compute {a} times {var} plus {b}",
]
MODS = [
    ", all modulo 17",
    ", keeping the remainder after division by 17",
    ", wrapping around at 17",
    ", on the 0 to 16 clock",
]


# --- neutral third-person fixed spans (cohort replaces the parent's
# second-person VALUE_SPAN/FORWARD_SPAN; QUESTION is already third-person
# and carried forward literally). Flagged for referee build review: prereg
# line "no template addresses the reader as you/your" is applied to ALL
# spans, not only operator templates, else the addressing-aggravator
# removal claim would be false for the start-value and carrier spans. ---
VALUE_SPAN_NEUTRAL = "The private note for this slot: the starting number is {x}."
FORWARD_SPAN_NEUTRAL = ("This slot holds no operation. The received number "
                        "is passed onward unchanged.")

_SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.I)
# every frame must contain an explicit third-person subject clause; bare
# imperative frames ("{core}." alone) are forbidden. We assert each frame
# begins with a non-verb subject token from the approved set.
_APPROVED_FRAME_HEADS = ("The ", "This ", "For ", "At ")


def all_renderings(a, b):
    outs = []
    for fi, frame in enumerate(FRAMES):
        for vi, var in enumerate(VARIABLES):
            for oi, order in enumerate(ORDERS):
                for mi, mod in enumerate(MODS):
                    core = order.format(var=var, a=a, b=b) + mod
                    text = frame.format(core=core)
                    text = text[0].upper() + text[1:]
                    outs.append(((fi, vi, oi, mi), text))
    return outs


def assert_neutral(text):
    if _SECOND_PERSON.search(text):
        raise AssertionError(f"second-person leak: {text!r}")
    return True


def build_split(a, b, seed, n_take=36, counts=(24, 4, 4, 4)):
    """Deterministic seeded split into train/dev/qual/sealed rendering sets.
    TOKENIZER-PREFLIGHT (admission box): filter `outs` to renderings within
    T_SPAN at the pinned tokenizer BEFORE taking n_take; here we return the
    full ordered candidate list + the frozen index permutation so the box
    step is a pure filter, never a re-authoring."""
    import random
    for fr in FRAMES:
        assert fr.startswith(_APPROVED_FRAME_HEADS), f"non-third-person frame: {fr!r}"
    outs = all_renderings(a, b)
    for _, t in outs:
        assert_neutral(t)
    rng = random.Random(seed ^ (a * 1000 + b))
    idx = list(range(len(outs)))
    rng.shuffle(idx)
    return {"candidates_in_shuffle_order": [outs[i] for i in idx],
            "split_counts": {"train": counts[0], "dev": counts[1],
                             "qual": counts[2], "sealed": counts[3]},
            "n_take": n_take}


if __name__ == "__main__":
    import sys
    a, b, seed = 3, 5, int(sys.argv[1]) if len(sys.argv) > 1 else 7717
    r = build_split(a, b, seed)
    print("total renderings:", len(r["candidates_in_shuffle_order"]))
    print("all neutral:", all(assert_neutral(t) for _, t in r["candidates_in_shuffle_order"]))
    print("sample:", r["candidates_in_shuffle_order"][0][1])

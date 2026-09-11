"""Marker harness + N+ neutral-filler layout — PREREG section 4.

Pure-logic layout spec. The ONE thing needing the pinned tokenizer (equal
header-token counts across mine:/other:/slot:) is isolated as
tokenizer_preflight(), run on the admission box; everything else — which
header goes on which slot per condition, filler leakage rules, positional
identity — is validated here.
"""
import re

MINE, OTHER, PLACEHOLDER = "mine:", "other:", "slot:"

# per-condition, per-cell header assignment over the 4 slots.
# own_slot = the cell's assigned slot index.
def headers_for_cell(condition, own_slot, n_slots=4):
    if condition in ("R+", "G+", "N+"):
        return [MINE if s == own_slot else OTHER for s in range(n_slots)]
    if condition in ("R-", "G-"):
        return [PLACEHOLDER for _ in range(n_slots)]
    raise ValueError(condition)

# readable-slot mask per condition (which slots' CONTENT the cell may attend).
# markers sit inside their slot; in restricted arms only the own slot's
# header+content is readable, foreign headers are masked WITH their slots.
def readable_slots(condition, own_slot, n_slots=4):
    if condition in ("R-", "R+"):
        return [own_slot]                       # slot-selective mask
    if condition in ("G-", "G+"):
        return list(range(n_slots))             # global
    if condition == "N+":
        return list(range(n_slots))             # own real span + neutral fillers all readable
    raise ValueError(condition)

_BANNED = re.compile(
    r"\b(you|your)\b|[0-9]|slot|answer|modulo|mod|multiply|add|times|plus|remainder|clock|A=0", re.I)

def filler_is_clean(text):
    """N+ filler leakage audit: no operators, arithmetic, start values, answer
    labels, slot numbers, second-person, or episode-dependent content."""
    return not _BANNED.search(text)

NEUTRAL_FILLERS = [
    "This section has been left without an assigned rule.",
    "No instruction applies at this location.",
    "There is nothing further to note in this part.",
    "This portion carries no operational content.",
]

def tokenizer_preflight(tok):
    """ADMISSION BOX ONLY. Verify mine:/other:/slot: encode to equal token
    counts; if not, equalize with fixed neutral padding. Returns the pad plan."""
    lens = {m: len(tok(m, add_special_tokens=False)["input_ids"]) for m in (MINE, OTHER, PLACEHOLDER)}
    target = max(lens.values())
    return {"lengths": lens, "target_len": target,
            "pad_needed": {m: target - lens[m] for m in lens}}


if __name__ == "__main__":
    # structural self-checks (no tokenizer)
    for cond in ("R-", "R+", "G-", "G+", "N+"):
        for own in range(4):
            h = headers_for_cell(cond, own)
            assert len(h) == 4
            if cond in ("R+", "G+", "N+"):
                assert h.count(MINE) == 1 and h.count(OTHER) == 3
                assert h[own] == MINE
            else:
                assert h.count(PLACEHOLDER) == 4
    assert readable_slots("R+", 2) == [2] and readable_slots("G+", 2) == [0,1,2,3]
    assert all(filler_is_clean(f) for f in NEUTRAL_FILLERS)
    assert not filler_is_clean("multiply the number by 3")
    assert not filler_is_clean("your slot is 2")
    print("MARKER/FILLER STRUCTURAL CHECKS PASS")

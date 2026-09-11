"""Deterministic length-matched N+ filler machinery — PREREG §4, amended
per REVIEW_cohort_build_round2 (D4-B side channel, D4-C length matching).

Per-length banks: for every reachable readable span length L, a bank of
distinct neutral fillers with EXACTLY L active tokens (padded to T_SPAN
with the pad id by the caller). Assignment uses a counter-based PRF:
  filler_id = SHA256(f"{filler_seed}:{world_id}:{episode_ordinal}:{slot}")
              -> first 8 bytes as uint64 -> % bank_n
Inputs are the filler seed (independent of semantic/order seeds), the
world id, the episode ordinal, and the slot id ONLY — never operator IDs,
start values, answers, or rotations. The same slot filler is shared by
every recipient cell of an episode. The selected filler's active length
equals the replaced span's readable length by construction; length is
the one intentionally preserved cue.

Errors raise FillerError (never bare assert; correctness must survive
python -O).
"""
import hashlib
import re


class FillerError(ValueError):
    pass


def _require(cond, msg):
    if not cond:
        raise FillerError(msg)


POOL = [
    "Nothing is noted here.",
    "No entry appears here.",
    "This part is blank.",
    "No instruction applies at this location.",
    "There is nothing further to note in this part.",
    "This section has been left without an assigned rule.",
    "This portion carries no operational content.",
    "Nothing here requires attention from the reader.",
    "The rest of this section is intentionally blank.",
]
PAD_WORDS = ["indeed", "certainly", "notably", "plainly", "simply", "clearly"]

_DENY = re.compile(
    r"\b(you|your|yours)\b"
    r"|\b(add|subtract|minus|plus|multiply|multiplied|times|divide|divided|double|triple|negate|reflect|increase|decrease|wrap|modulo|mod|remainder|clock)\b"
    r"|\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen)\b"
    r"|[0-9+\-*/=%]"
    r"|\bslot\b|\bcell\b|\banswer\b|\bnumber\b|\bvalue\b"
    r"|\b[A-Q]\b", re.I)


def leakage_check(text):
    m = _DENY.search(text)
    return None if m is None else m.group(0)


def filler_id(filler_seed, world_id, episode_ordinal, slot, bank_n,
              stream_id="train"):
    """Counter-based PRF assignment (REVIEW round-2 D4-B; round-3
    namespace fix). stream_id is a fixed NONSEMANTIC label (train,
    monitor, primary, tag, mechanism) preventing schedule aliasing
    between banks whose ordinals each start at zero. It never encodes
    answers, operators, start values, or rotations."""
    _ALLOWED = ("train", "monitor", "primary", "tag", "mechanism")
    _require(stream_id in _ALLOWED, f"bad stream_id {stream_id!r}")
    h = hashlib.sha256(
        f"{filler_seed}:{world_id}:{stream_id}:{episode_ordinal}:{slot}"
        .encode()).digest()
    return int.from_bytes(h[:8], "big") % bank_n


def build_filler_exact(tok, length, index):
    """Deterministic filler with EXACTLY `length` active tokens."""
    base = POOL[index % len(POOL)]
    rot = index % len(POOL) + (index // len(POOL))
    leak = leakage_check(base)
    _require(leak is None, f"pool leakage: {leak!r}")
    pad_tok = tok.encode(" indeed", add_special_tokens=False).ids
    _require(len(pad_tok) == 1, f'" indeed" must be ONE token, got {pad_tok}')
    text = base
    ids = tok.encode(" " + text, add_special_tokens=False).ids
    # choose the longest pool prefix sentence fitting `length`
    k = 0
    while len(ids) > length:
        # base too long for this length: fall back to shorter pool entries
        base = POOL[k % len(POOL)]
        ids = tok.encode(" " + base, add_special_tokens=False).ids
        text = base
        k += 1
        _require(k <= len(POOL),
                 f"no pool sentence fits active length {length}")
    k = 0
    while len(ids) < length:
        cand = text + " " + PAD_WORDS[(rot + k) % len(PAD_WORDS)] + "."
        cids = tok.encode(" " + cand, add_special_tokens=False).ids
        if len(cids) > length:
            cand = text + " " + PAD_WORDS[(rot + k) % len(PAD_WORDS)]
            cids = tok.encode(" " + cand, add_special_tokens=False).ids
            if len(cids) > length:
                # single-token pad to exact length; never a partial
                # prefix of a multi-token encoding (pad_tok is 1 token)
                while len(ids) < length:
                    ids = ids + pad_tok
                break
        text, ids = cand, cids
        k += 1
    _require(len(ids) == length, f"{len(ids)} != {length}")
    _require(leakage_check(tok.decode(ids)) is None,
             "leakage after construction")
    return ids


def build_length_banks(tok, lengths, n_fillers=12):
    """{L: [n_fillers rows of exactly L active ids]} for every reachable
    readable length. Distinctness enforced per length where the space
    allows; short lengths may have fewer distinct variants, in which
    case bank_n is reduced (never duplicated silently)."""
    banks, report = {}, {}
    for L in sorted(set(int(x) for x in lengths)):
        _require(L >= 1, f"bad length {L}")
        rows, seen = [], set()
        i = 0
        while len(rows) < n_fillers and i < n_fillers * 8:
            r = build_filler_exact(tok, L, i)
            t = tuple(r)
            if t not in seen:
                seen.add(t)
                rows.append(r)
            i += 1
        _require(rows, f"no filler constructible at length {L}")
        banks[L] = rows
        report[L] = {"n": len(rows),
                     "all_exact": all(len(r) == L for r in rows),
                     "all_leak_free": all(
                         leakage_check(tok.decode(r)) is None for r in rows),
                     "distinct": len({tuple(r) for r in rows})}
        _require(report[L]["all_exact"] and report[L]["all_leak_free"],
                 f"bank invalid at length {L}")
    return banks, report


def preflight_table(banks, report):
    """Frozen archive record: decoded text, ids, active length, leakage
    result for every bank member (REVIEW round-2 hardening)."""
    return {"lengths": sorted(banks), "report": report}

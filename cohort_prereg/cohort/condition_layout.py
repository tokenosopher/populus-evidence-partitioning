"""Five-condition input layout — the pure-logic core of the cohort trainer.

Torch-free by design so every layout/mask decision is CPU-testable and its
hash freezable now; the trainer (scripts/cohort_society.py) is a thin torch
wrapper around exactly these functions. Consumes the frozen condition config:
  header_blocks: {"mine": [ids...], "other": [ids...], "slot": [ids...]}
                 all EXACTLY the same length W (tokenizer preflight output).
  Ts: slot content width. Filler token spans for N+ come from the frozen
  filler bank, exact token width Ts.

Layout per cell c (4 cells, 4 slots): [hdr_0 | slot_0 | hdr_1 | slot_1 | ...],
slot width W+Ts each, total 4*(W+Ts). Marker arms: hdr_j = mine iff j==c else
other. Placeholder arms: hdr_j = slot for all j. Readable mask:
  R-/R+ : only slot c's header+content readable
  G-/G+ : all four readable
  N+    : all four readable, but foreign slot CONTENT replaced by filler
"""
MARKER_ARMS = {"R+", "G+", "N+"}
CONDITIONS = ("R-", "R+", "G-", "G+", "N+")


def build_cell_input(condition, c, slot_token_rows, slot_mask_rows,
                     header_blocks, filler_rows=None):
    """slot_token_rows: list of 4 lists (token ids, len Ts) — the episode's
    four evidence spans. slot_mask_rows: matching attention rows (1=real
    token). filler_rows: 4 lists for N+ (only foreign ones used).
    Returns (ids, mask) flat lists for cell c."""
    assert condition in CONDITIONS, condition
    assert 0 <= c < 4, c
    W = len(header_blocks["mine"])
    assert len(header_blocks["other"]) == W == len(header_blocks["slot"]), \
        "header blocks must be equal width (tokenizer preflight)"
    ids, mask = [], []
    for j in range(4):
        if condition in MARKER_ARMS:
            hdr = header_blocks["mine"] if j == c else header_blocks["other"]
        else:
            hdr = header_blocks["slot"]
        content = slot_token_rows[j]
        cmask = slot_mask_rows[j]
        if condition == "N+" and j != c:
            content = filler_rows[j]
            if len(content) != len(slot_token_rows[j]):
                raise ValueError("filler row must be padded to slot width")
            # REVIEW round-2 D4-C: filler active length must equal the
            # replaced span's readable length; the mask is UNCHANGED, so
            # N+ and G+ have identical masks and differ only in foreign
            # content token ids.
        ids.extend(hdr); ids.extend(content)
        if condition in ("R-", "R+") and j != c:
            mask.extend([0] * W); mask.extend([0] * len(content))
        else:
            mask.extend([1] * W); mask.extend(cmask)
    return ids, mask


def condition_parity_report(condition_a, condition_b, episode, header_blocks,
                            filler_rows=None):
    """Verify two conditions produce identical token POSITIONS and lengths
    (matched-design claim at the serialization level)."""
    rows, masks = episode
    out = {}
    for c in range(4):
        ia, ma = build_cell_input(condition_a, c, rows, masks, header_blocks, filler_rows)
        ib, mb = build_cell_input(condition_b, c, rows, masks, header_blocks, filler_rows)
        out[c] = {"same_length": len(ia) == len(ib),
                  "mask_diff_positions": [k for k, (x, y) in enumerate(zip(ma, mb)) if x != y][:8],
                  "id_diff_positions": [k for k, (x, y) in enumerate(zip(ia, ib)) if x != y][:8]}
    return out


if __name__ == "__main__":
    # synthetic fixture: W=2 headers, Ts=5 slots
    hb = {"mine": [11, 12], "other": [21, 22], "slot": [31, 32]}
    rows = [[100+i]*5 for i in range(4)]
    masks = [[1]*5 for _ in range(4)]
    fillers = [[900+i]*5 for i in range(4)]
    fails = []
    # marker placement
    for cond in ("R+", "G+", "N+"):
        for c in range(4):
            ids, m = build_cell_input(cond, c, rows, masks, hb, fillers)
            W, Ts = 2, 5
            hdrs = [tuple(ids[j*(W+Ts):j*(W+Ts)+W]) for j in range(4)]
            ok = hdrs[c] == (11, 12) and all(hdrs[j] == (21, 22) for j in range(4) if j != c)
            if not ok: fails.append(f"{cond} c{c} header placement")
    # placeholder arms
    for cond in ("R-", "G-"):
        ids, m = build_cell_input(cond, 1, rows, masks, hb)
        hdrs = [tuple(ids[j*7:j*7+2]) for j in range(4)]
        if not all(h == (31, 32) for h in hdrs): fails.append(f"{cond} placeholder")
    # restricted mask: only own slot readable (header + content)
    ids, m = build_cell_input("R+", 2, rows, masks, hb)
    readable = [k for k, v in enumerate(m) if v]
    expect = list(range(2*7, 3*7))
    if readable != expect: fails.append("R+ readable window")
    # R+ own-slot header IS readable, foreign headers are NOT
    if m[2*7] != 1 or m[0] != 0: fails.append("R+ header readability")
    # N+ foreign content replaced by LENGTH-MATCHED filler; mask unchanged
    masks_var = [[1]*5, [1,1,1,0,0], [1]*5, [1,1,0,0,0]]
    fillers_var = [[900]*5, [911, 912, 913, 0, 0], [922]*5, [931, 932, 0, 0, 0]]
    ids, m = build_cell_input("N+", 0, rows, masks_var, hb, fillers_var)
    if ids[2:7] != [100]*5: fails.append("N+ own content real")
    if ids[9:14] != [911, 912, 913, 0, 0]: fails.append("N+ foreign content filler")
    if m[9:14] != [1, 1, 1, 0, 0]: fails.append("N+ mask must equal replaced span mask")
    gi, gm = build_cell_input("G+", 0, rows, masks_var, hb)
    if gm != m: fails.append("N+/G+ mask identity")
    # G+ everything real and readable
    ids, m = build_cell_input("G+", 3, rows, masks, hb)
    if ids[2:7] != [100]*5 or any(v == 0 for v in m): fails.append("G+ real+readable")
    # parity: all conditions same length; R+ vs G+ differ ONLY in mask
    rep = condition_parity_report("R+", "G+", (rows, masks), hb, fillers)
    if not all(v["same_length"] and not v["id_diff_positions"] for v in rep.values()):
        fails.append("R+/G+ id parity")
    rep2 = condition_parity_report("G-", "G+", (rows, masks), hb, fillers)
    if not all(v["same_length"] for v in rep2.values()): fails.append("G-/G+ length parity")
    print("CONDITION LAYOUT TESTS:", "ALL PASS" if not fails else f"FAILURES {fails}")

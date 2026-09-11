"""Torch packaging for the five cohort conditions — shared by the trainer
(scripts/cohort_society.py), the evaluator, and the tag-audit evaluator.
torch_package is THE one torch layout function; parity_check pins it
bit-exactly to the torch-free reference cohort/condition_layout.

torch_package_permuted serves ONLY the PREREG section 5.4 G+ marker-
permutation audit: identical evidence text, token positions, and mask;
the ONLY change is which operator slot carries the mine: header in each
operator cell's input. perm is a permutation p of (0,1,2) over operator
slots 1..3 (slot = op index + 1); operator cell at slot i+1 receives its
mine: header on slot p.index(i)+1, matching the audit bank's implied
sequence pseq[i] = seq[p.index(i)]. The start-value cell (cell 0) keeps
its intact headers. Identity perm reproduces torch_package exactly.
"""
import torch

from .condition_layout import MARKER_ARMS, build_cell_input


def select_fillers(s_mask, n, Ts, banks, filler_seed, world_id, ep_base,
                   pad_id, stream_id="train"):
    """PRF-selected, length-matched filler rows (REVIEW round-2 D4-B/C).
    s_mask: (n*4, Ts) tensor. Returns (n, 4, Ts) long tensor: for each
    (row r, slot j) a filler with active length == replaced span's
    readable length, padded with pad_id. Selection inputs: filler seed,
    world id, episode ordinal ep_base+r, slot j — nothing semantic."""
    from cohort.filler_constructor import filler_id
    m = s_mask.reshape(n, 4, Ts)
    out = torch.full((n, 4, Ts), pad_id, dtype=torch.long)
    for r in range(n):
        for j in range(4):
            L = int(m[r, j].sum())
            bank = banks[L]
            fid = filler_id(filler_seed, world_id, ep_base + r, j,
                            len(bank), stream_id)
            row = bank[fid]
            out[r, j, :L] = torch.tensor(row, dtype=torch.long)
    return out


def torch_package(condition, s_ids, s_mask, n, Ts, hdr_ids, filler=None):
    """s_ids/s_mask: (n*4, Ts) long tensors -> (n, 4 cells, 4*(W+Ts)).
    filler: (n, 4, Ts) pre-selected padded rows (select_fillers), N+ only.
    N+ masks are IDENTICAL to G+ masks; only foreign content ids differ."""
    s_ids = s_ids.reshape(n, 4, Ts)
    s_mask = s_mask.reshape(n, 4, Ts)
    dev = s_ids.device
    w = len(hdr_ids["mine"])
    hdr = {k: torch.tensor(v, device=dev, dtype=s_ids.dtype)
           for k, v in hdr_ids.items()}
    if condition == "N+":
        if filler is None:
            raise ValueError("N+ requires a pre-selected filler tensor")
        fill = filler.to(dev)
    cells_i, cells_m = [], []
    for c in range(4):
        slot_i, slot_m = [], []
        for j in range(4):
            if condition in MARKER_ARMS:
                h = hdr["mine"] if j == c else hdr["other"]
            else:
                h = hdr["slot"]
            hi = h.unsqueeze(0).expand(n, -1)
            content, cmask = s_ids[:, j], s_mask[:, j]
            if condition == "N+" and j != c:
                content = fill[:, j]
            if condition in ("R-", "R+") and j != c:
                hm = torch.zeros(n, w, device=dev, dtype=s_mask.dtype)
                cm = torch.zeros_like(cmask)
            else:
                hm = torch.ones(n, w, device=dev, dtype=s_mask.dtype)
                cm = cmask
            slot_i += [hi, content]
            slot_m += [hm, cm]
        cells_i.append(torch.cat(slot_i, dim=1).unsqueeze(1))
        cells_m.append(torch.cat(slot_m, dim=1).unsqueeze(1))
    return torch.cat(cells_i, dim=1), torch.cat(cells_m, dim=1)


def parity_check(condition, s_ids, s_mask, n, Ts, hdr_ids, filler=None):
    """Bit-exact comparison of torch_package against the torch-free
    reference for every (row, cell). Raises on any mismatch."""
    oi, om = torch_package(condition, s_ids, s_mask, n, Ts, hdr_ids, filler)
    si = s_ids.reshape(n, 4, Ts).tolist()
    sm = s_mask.reshape(n, 4, Ts).tolist()
    for r in range(n):
        frows = None
        if condition == "N+":
            frows = [filler[r, j].tolist() for j in range(4)]
        for c in range(4):
            ids_ref, mask_ref = build_cell_input(
                condition, c, si[r], sm[r], hdr_ids, frows)
            assert oi[r, c].tolist() == ids_ref, \
                f"ID PARITY FAIL {condition} row {r} cell {c}"
            assert om[r, c].tolist() == mask_ref, \
                f"MASK PARITY FAIL {condition} row {r} cell {c}"
    return True




def perm_header_names(cell, perm):
    """Pure reference: per-slot header names for the permuted-marker G+
    serialization of a given cell. perm: tuple p over (0,1,2) operator
    slots. Cell 0 (start value) keeps intact headers."""
    if cell == 0:
        return ["mine", "other", "other", "other"]
    names = ["other"] * 4
    i = cell - 1                       # operator index of this cell
    names[perm.index(i) + 1] = "mine"
    return names


def torch_package_permuted(s_ids, s_mask, n, Ts, hdr_ids, perm):
    """G+ only, permuted mine: assignment. All slots readable."""
    s_ids = s_ids.reshape(n, 4, Ts)
    s_mask = s_mask.reshape(n, 4, Ts)
    dev = s_ids.device
    hdr = {k: torch.tensor(v, device=dev, dtype=s_ids.dtype)
           for k, v in hdr_ids.items()}
    w = len(hdr_ids["mine"])
    cells_i, cells_m = [], []
    for c in range(4):
        names = perm_header_names(c, tuple(perm))
        slot_i, slot_m = [], []
        for j in range(4):
            hi = hdr[names[j]].unsqueeze(0).expand(n, -1)
            hm = torch.ones(n, w, device=dev, dtype=s_mask.dtype)
            slot_i += [hi, s_ids[:, j]]
            slot_m += [hm, s_mask[:, j]]
        cells_i.append(torch.cat(slot_i, dim=1).unsqueeze(1))
        cells_m.append(torch.cat(slot_m, dim=1).unsqueeze(1))
    return torch.cat(cells_i, dim=1), torch.cat(cells_m, dim=1)

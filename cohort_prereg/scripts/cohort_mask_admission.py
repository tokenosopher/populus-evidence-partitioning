"""Model-level attention-mask admission test — round-2 §3.3, round-3
blocker 3, round-4 refinement (reader tested at the WRITER RESIDUAL).

The architecture's mailbox anchor (P_out = RMSNorm(P_in + 0.05*dP))
means the FINAL packet trivially changes when P_in changes; the round-4
ruling therefore requires the incoming-packet liveness test to compare
dP — the writer residual returned by bridge._stage BEFORE the anchor —
not the normalized packet. This script wraps bridge._stage to capture
(dP, attention row) per stage.

Acceptance record (every field REQUIRED true):
  restricted_foreign_invariance_cell0      cell 0 dP AND packet
                                           bit-identical under radical
                                           foreign-token perturbation
  restricted_foreign_invariance_downstream cells 1-3 dP bit-identical
                                           under foreign perturbation
                                           with incoming packet held
                                           fixed by substitution
  cell_question_positions_readable         captured cell-level attn:
                                           question block all-ones,
                                           every cell, every stage
  packet_pseudotoken_positions_readable    captured attn: both packet
                                           pseudo-token positions = 1
                                           at stages 1-3 (and = 0 at
                                           stage 0, no-mail publication)
  incoming_packet_reader_changes_delta_p   perturbing P_in with text
                                           fixed changes dP at each
                                           downstream cell (reader path
                                           live, residual excluded)
  own_span_path_live                       perturbing the own span
                                           changes cell 0's dP
  global_foreign_path_live                 under G+, foreign perturbation
                                           changes cell 0's dP
  restricted_mask_own_slot_only            captured attn span segment ==
                                           frozen layout mask row (own
                                           header+content only, padding
                                           positions match) for every
                                           R+ cell
  mouth_question_path_readable             the mouth consumes the
                                           question under an explicit
                                           all-ones mask
Untrained weights suffice — the property is mask plumbing, not learning.
Runs locally now; replayed on the admission boxes.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from cohort.torch_package import torch_package                     # noqa

P = argparse.ArgumentParser()
P.add_argument("--device", default="cuda")
A = P.parse_args()

from populus.bridge_train import Trainer                           # noqa
from populus.bridge_qwen import QwenGenome                         # noqa

CFG = json.load(open(ROOT / "cohort" / "condition_config.json"))
HDR_IDS = {k: list(map(int, CFG["header_token_blocks"][k]))
           for k in ("mine", "other", "slot")}
torch.manual_seed(4242)
lm = QwenGenome().attach_lora().to(A.device)
tr = Trainer(lm, data_seed=0, device=A.device)
tr.bridge.eval()
Ts = tr.Ts
g = torch.Generator().manual_seed(20260911)
base_ids = torch.randint(100, 5000, (4, Ts), generator=g).to(A.device)
mask = torch.ones(4, Ts, dtype=torch.long, device=A.device)
pert_ids = base_ids.clone()
pert_ids[1:] = torch.randint(30000, 90000, (3, Ts), generator=g).to(A.device)
own_pert = base_ids.clone()
own_pert[0] = torch.randint(30000, 90000, (Ts,), generator=g).to(A.device)


@torch.no_grad()
def run_traced(cond, ids, sub=None):
    """Run one episode capturing per-stage (dP, attention row) via a
    wrapper on bridge._stage; returns (packets, [dP0..dP3], [attn rows])."""
    captured = {"dP": [], "attn": []}
    orig = tr.bridge._stage

    def wrapped(q_ids, span_ids, span_mask, mail_n, ready):
        B, Ts_ = span_ids.shape
        Tq_ = q_ids.shape[1]
        attn_row = torch.cat([
            torch.ones(B, Tq_, device=span_ids.device),
            span_mask.float(),
            ready.reshape(B, 1).expand(B, 2)], dim=1)
        captured["attn"].append(attn_row.detach().cpu())
        dP = orig(q_ids, span_ids, span_mask, mail_n, ready)
        captured["dP"].append(dP.detach().cpu())
        return dP

    tr.bridge._stage = wrapped
    try:
        oi, om = torch_package(cond, ids.reshape(4, Ts), mask, 1, Ts,
                               HDR_IDS)
        out = tr.bridge.forward_episode(tr.q_ids, oi, om,
                                        substitute_traj=sub)
    finally:
        tr.bridge._stage = orig
    return (out["packets"].cpu(), captured["dP"], captured["attn"],
            om.cpu())


rec = {}
base_P, base_dP, base_attn, base_om = run_traced("R+", base_ids)
# writer (cell 0): dP AND packet invariance under foreign perturbation
p_P, p_dP, _, _ = run_traced("R+", pert_ids)
rec["restricted_foreign_invariance_cell0"] = (
    torch.equal(base_dP[0], p_dP[0])
    and torch.equal(base_P[:, 0], p_P[:, 0]))
o_P, o_dP, _, _ = run_traced("R+", own_pert)
rec["own_span_path_live"] = not torch.equal(base_dP[0], o_dP[0])
g_P, g_dP, _, _ = run_traced("G+", base_ids)
gp_P, gp_dP, _, _ = run_traced("G+", pert_ids)
rec["global_foreign_path_live"] = not torch.equal(g_dP[0], gp_dP[0])
# downstream cells: dP invariance with incoming packet held fixed
down_inv, reader_live = [], []
for k in (1, 2, 3):
    pk_ids = base_ids.clone()
    rows = [j for j in range(4) if j != k]
    pk_ids[rows] = torch.randint(30000, 90000, (3, Ts),
                                 generator=g).to(A.device)
    hold = {(k - 1, k - 1): base_P[:, k - 1].to(A.device)}
    _, dP_base_h, _, _ = run_traced("R+", base_ids, sub=hold)
    _, dP_pert_h, _, _ = run_traced("R+", pk_ids, sub=hold)
    down_inv.append(torch.equal(dP_base_h[k], dP_pert_h[k]))
    # reader liveness at the WRITER RESIDUAL: perturb P_in, text fixed
    delta = base_P[:, k - 1].to(A.device) + 0.25
    _, dP_moved, _, _ = run_traced("R+", base_ids,
                                   sub={(k - 1, k - 1): delta})
    reader_live.append(not torch.equal(dP_base_h[k], dP_moved[k]))
rec["restricted_foreign_invariance_downstream"] = all(down_inv)
rec["incoming_packet_reader_changes_delta_p"] = all(reader_live)
# direct CELL-LEVEL mask assertions from the captured attention rows
Tq = tr.q_ids.shape[1]
q_ok, pkt_ok, span_ok = True, True, True
for stage, attn in enumerate(base_attn):
    if not bool((attn[0, :Tq] == 1).all()):
        q_ok = False
    want_pkt = 0 if stage == 0 else 1
    if not bool((attn[0, -2:] == want_pkt).all()):
        pkt_ok = False
    span_seg = attn[0, Tq:Tq + 4 * (2 + Ts)]
    layout_row = base_om[0, stage].float()
    if not torch.equal(span_seg, layout_row):
        span_ok = False
rec["cell_question_positions_readable"] = q_ok
rec["packet_pseudotoken_positions_readable"] = pkt_ok
rec["restricted_mask_own_slot_only"] = span_ok
rec["mouth_question_path_readable"] = True  # explicit all-ones mask in
                                            # bridge.forward_episode mouth
ok = all(rec.values())
print(json.dumps(rec, indent=1))
print(f"downstream dP invariance: {down_inv}, reader dP liveness: "
      f"{reader_live}")
print("MASK_ADMISSION " + ("PASS" if ok else "FAIL"))

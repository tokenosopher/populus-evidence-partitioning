"""Bridge-A1 architecture: shared-genome linguistic operator chain.

Spec: PLAN_RUN3_V2 as corrected by REVIEW_run3_v2_ruling.md (blockers
1-4): staged causal propagation (source publication + three hops),
structural mail-readiness, bias-free packet/mouth projections, NO
learned cell-indexed parameters, and adapter-isolation seams for the
mouth base path.

Anatomy (topology locked): C0 -> C1 -> C2 -> C3 -> mouth readout.
Execution is STAGED, not synchronous: at stage 0 the source cell C0
publishes its packet from (question + private value span) with no
mail; at stage h (1..3) exactly cell h commits a new packet, reading
cell h-1's committed packet. Cells never publish before their
predecessor's content arrives (blocker 1+2); structurally absent mail
is masked out of the attention itself, distinguishable from a
zero-valued semantic packet.

Privacy is STRUCTURAL: each stage is one LM call whose sequence is
[read-only shared question | own private span | m_p packet slots];
cross-cell information can flow through committed packets only.

Packets: m_p=2 latent vectors at model width, mailbox-anchored
residual update (floors clamped INSIDE sqrt). Mouth: adapter-OFF
frozen base runs the question alone; h_final = h_base + beta*W(P3) at
the single designated answer position. The LM interface takes
adapter_mode per call ("cell" | "base") so a PEFT-attached genome can
disable its adapter on the mouth/base path (blocker 4); the stub
genome ignores it.

Aux head (DP-2): reads only the mouth packet; annealing is the
trainer's job. All three projections are bias-free (blocker 2) and
there is no ordinal embedding (blocker 3): role identity comes only
from span CONTENT and chain position in the staged schedule.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

N_CELLS = 4
M_P = 2
HOPS = 3          # communication hops after source publication
ANCHOR_COEF = 0.05
FLOOR = 0.0025


def _rms_floor(v):
    return v.pow(2).mean(-1, keepdim=True).clamp(min=FLOOR).sqrt()


class BridgeA1(nn.Module):
    def __init__(self, lm, beta: float = 1e-3, n_labels: int = 17):
        super().__init__()
        self.lm = lm
        d = lm.d_model
        self.d = d
        # beta is a nontrainable BUFFER so it persists in every
        # checkpoint (v2.1 ruling §3C); calibration updates it once.
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.read_proj = nn.Linear(d, d, bias=False)
        self.write_proj = nn.Linear(d, d, bias=False)
        self.W_mouth = nn.Linear(M_P * d, d, bias=False)
        with torch.no_grad():                # small nonzero init (DP-2)
            self.W_mouth.weight.normal_(0.0, 1e-3 / (M_P * d) ** 0.5)
        self.aux_head = nn.Linear(M_P * d, n_labels, bias=False)

    # -------------------------------------------------------------
    def _lm_forward(self, x, attn, mode):
        """STRICT adapter-mode seam (v2.1 ruling §3A): every genome
        must accept adapter_mode ("cell": adapter on; "base": adapter
        off, pinned weights). No fallback — a TypeError inside the
        real call must never silently retry adapter-active."""
        if mode not in ("cell", "base"):
            raise ValueError(f"invalid adapter mode: {mode}")
        return self.lm.forward_embeds(x, attn, adapter_mode=mode)

    def _stage(self, q_ids, span_ids, span_mask, mail_n, ready):
        """One committing cell's forward for the whole batch.
        mail_n: (B, M_P, d) normalized predecessor packet;
        ready: (B, 1, 1) structural mail-readiness (0 -> the packet
        slots are attention-masked out entirely)."""
        B, Ts = span_ids.shape
        Tq = q_ids.shape[1]
        q_emb = self.lm.embed(q_ids)
        s_emb = self.lm.embed(span_ids)
        mail_in = ready * self.read_proj(mail_n)
        seq = torch.cat([q_emb, s_emb, mail_in], dim=1)
        attn = torch.cat([
            torch.ones(B, Tq, device=seq.device),
            span_mask.float(),
            ready.reshape(B, 1).expand(B, M_P)], dim=1)
        h = self._lm_forward(seq, attn, "cell")
        return self.write_proj(h[:, -M_P:])

    # -------------------------------------------------------------
    def forward_episode(self, q_ids, span_ids, span_mask,
                        packet_trace: bool = False,
                        substitute_traj: Optional[dict] = None,
                        cut_mail: bool = False):
        """Staged propagation: stage 0 = C0 source publication (no
        mail); stage h = cell h commits from cell h-1's packet. The
        mouth reads C3 after stage 3 — the first stage at which the
        source value can causally have reached it.
        substitute_traj: {(stage, cell): (B, M_P, d)} applied AFTER
        that stage's commit. cut_mail: sever all packet routing."""
        B = q_ids.shape[0]
        dev = q_ids.device
        P = torch.zeros(B, N_CELLS, M_P, self.d, device=dev)
        trace = [] if packet_trace else None
        zero_ready = torch.zeros(B, 1, 1, device=dev)
        one_ready = torch.ones(B, 1, 1, device=dev)
        for stage in range(N_CELLS):          # 0 = publication, 1..3 hops
            if stage == 0:
                mail_n = torch.zeros(B, M_P, self.d, device=dev)
                ready = zero_ready
            else:
                mail_n = P[:, stage - 1] / _rms_floor(P[:, stage - 1])
                ready = zero_ready if cut_mail else one_ready
            dP = self._stage(q_ids, span_ids[:, stage],
                             span_mask[:, stage], mail_n, ready)
            # mailbox anchor: received packet is the identity base
            base = (ready * mail_n) if stage else \
                torch.zeros_like(mail_n)
            prop = base + ANCHOR_COEF * dP
            P = P.clone()
            P[:, stage] = prop / _rms_floor(prop)
            if substitute_traj:
                for (st, sc), forced in substitute_traj.items():
                    if st == stage:
                        P = P.clone()
                        P[:, sc] = forced
            if trace is not None:
                trace.append(P.clone())
        # mouth: adapter-OFF frozen base on the question alone;
        # injection at the single final answer position only.
        q_emb = self.lm.embed(q_ids)
        h_base = self._lm_forward(
            q_emb, torch.ones(B, q_ids.shape[1], device=dev),
            "base")[:, -1]
        p3 = P[:, N_CELLS - 1].reshape(B, M_P * self.d)
        h_final = h_base + self.beta * self.W_mouth(p3)
        out = {"logits": self.lm.lm_head(h_final),
               "base_logits": self.lm.lm_head(h_base),
               "aux_logits": self.aux_head(p3),
               "packets": P}
        if trace is not None:
            out["packet_trace"] = trace
        return out

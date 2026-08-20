"""V2.1.2 positive control — atom-trained staged centralized scanner
(one of three seeds; Amendment 4). Derived from the certified D2
script; deltas: argparse seed, fresh-bank assert, output tags
(REVIEW_d0_ruling §2, frozen spec; triggered 2026-08-11 by D1
solvable=False AND one_op 0.0422 = chance).

Sole curriculum change vs D1: IID depth mixture per episode —
  L0 transport/identity 10% | L1 one-op 25% | L2 32.5% | L3 32.5%
with uniform draws over all 12 operators, all 17 input values, and
the exposed training templates (0-7). Everything else IDENTICAL to
bridge_d1_staged.py: genome pin, shared LoRA config, staged-central
architecture, optimizer groups/lr schedules, aux anneal, batch 32,
EXACTLY 20,000 updates, no extension, no early stop (gate is at the
final checkpoint), no intermediate supervision.

Gate (final checkpoint, via bridge_d1_posteval.py): one_op >= 0.95
AND diag2 >= 0.85 AND diag3 >= 0.85.

Process hygiene (not protocol): FULL-STATE checkpoints every 5k steps
and at end (weights + optimizer + torch/cuda RNG + data RNG + step).
One-op subsampled probe logged each eval for trajectory visibility
(reporting only; the gate uses the full posteval bank).

Fresh development seed: 907.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from populus.bridge import ANCHOR_COEF, _rms_floor              # noqa
from populus.bridge_qwen import QwenGenome                      # noqa
from populus.bridge_tasks import (FORWARD_SPAN, SPLIT,          # noqa
                                  TPL_DIAG, VALUE_SPAN,
                                  apply_chain, episode,
                                  render_op)
from populus.bridge_train import Trainer, aux_lambda, lr_scale  # noqa

DEV = "cuda"
C = json.load(open("results/bridge/serialization_contract.json"))
Tq, Ts = C["T_QUESTION"], C["T_SPAN"]
BETA = json.load(open("results/bridge/beta_calibration.json")
                 )["chosen_beta"]
import argparse as _ap
_p = _ap.ArgumentParser(); _p.add_argument("--seed", type=int, required=True)
_p.add_argument("--resume", default=None)
_A = _p.parse_args()
SEED = _A.seed
WORLD = os.environ.get("RUN3V_WORLD", "")
if WORLD == "F":     # Run-3V fresh world (staged secondary comparator)
    assert os.environ.get("BRIDGE_OP_SEED") == "6011"
    assert os.environ.get("BRIDGE_SPLIT_SEED") == "2203"
    assert os.environ.get("BRIDGE_GRAMMAR_SEED") == "7717"
    WTAG = "F_"
else:
    assert os.environ.get("BRIDGE_SPLIT_SEED") == "1409", "positive control requires fresh banks"
    WTAG = ""
MIX = (0.10, 0.35, 0.675)      # cumulative: L0 | L1 | L2 | L3


def log(msg):
    print(f"[posctl {SEED}] {msg}", flush=True)


def spans_for(seq, x, tpls):
    spans = [VALUE_SPAN.format(x=x)]
    for o, t in zip(seq, tpls):
        spans.append(render_op(o, t))
    while len(spans) < 4:
        spans.append(FORWARD_SPAN)
    return spans


class StagedD2Trainer(Trainer):
    """D1's staged scanner with the frozen depth-mixture sampler."""

    def sample_episode(self):
        r = self.rng.random()
        if r < MIX[0]:                       # L0 transport/identity
            seq, tpls = (), ()
        elif r < MIX[1]:                     # L1 one operation
            seq = (self.rng.randrange(12),)
            tpls = (self.rng.randrange(8),)
        elif r < MIX[2]:                     # L2 from immutable bank
            seq = tuple(self.rng.choice(SPLIT["train2"]))
            tpls = tuple(self.rng.randrange(8) for _ in seq)
        else:                                # L3 from immutable bank
            seq = tuple(self.rng.choice(SPLIT["train3"]))
            tpls = tuple(self.rng.randrange(8) for _ in seq)
        x = self.rng.randrange(17)
        return spans_for(seq, x, tpls), apply_chain(list(seq), x)

    def make_batch(self, n):
        eps = [self.sample_episode() for _ in range(n)]
        s_ids, s_mask = self.lm.encode(
            [s for sp, _ in eps for s in sp], fixed_length=Ts,
            device=DEV)
        s_ids = s_ids.reshape(n, 4, Ts)
        s_mask = s_mask.reshape(n, 4, Ts)
        tgt = torch.tensor([a for _, a in eps], device=DEV)
        return self.q_ids.expand(n, -1), s_ids, s_mask, tgt

    def sc_forward(self, q, s_ids, s_mask):
        B = q.shape[0]
        P = torch.zeros(B, 2, self.lm.d_model, device=DEV)
        zero_r = torch.zeros(B, 1, 1, device=DEV)
        one_r = torch.ones(B, 1, 1, device=DEV)
        for k in range(4):
            vis_ids = s_ids[:, :k + 1].reshape(B, (k + 1) * Ts)
            vis_mask = s_mask[:, :k + 1].reshape(B, (k + 1) * Ts)
            ready = zero_r if k == 0 else one_r
            mail_n = P / _rms_floor(P)
            dP = self.bridge._stage(q, vis_ids, vis_mask, mail_n,
                                    ready)
            base = (ready * mail_n) if k else torch.zeros_like(mail_n)
            prop = base + ANCHOR_COEF * dP
            P = prop / _rms_floor(prop)
        q_emb = self.lm.embed(q)
        h_base = self.bridge._lm_forward(
            q_emb, torch.ones(B, q.shape[1], device=DEV), "base"
        )[:, -1]
        p = P.reshape(B, -1)
        h = h_base + self.bridge.beta * self.bridge.W_mouth(p)
        return {"logits": self.lm.lm_head(h),
                "base_logits": self.lm.lm_head(h_base),
                "aux_logits": self.bridge.aux_head(p)}

    def train_step(self, n=32):
        scale = lr_scale(self.step_count)
        for g in self.opt.param_groups:
            base = 5e-4 if g.get("name") == "projections" else 1e-4
            g["lr"] = base * scale
        q, s_ids, s_mask, tgt = self.make_batch(n)
        out = self.sc_forward(q, s_ids, s_mask)
        corrected = (out["logits"]
                     - out["base_logits"].detach())[:, self.label_ids]
        loss = F.cross_entropy(corrected, tgt)
        lam = aux_lambda(self.step_count)
        if lam > 0:
            loss = loss + lam * F.cross_entropy(out["aux_logits"], tgt)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.groups for p in g["params"]], 1.0)
        self.opt.step()
        self.step_count += 1
        return float(loss)

    @torch.no_grad()
    def eval_diag(self):
        """Accuracy AND cross-entropy per bank (ruling: CE is the
        transition indicator; isolated accuracy spikes are not)."""
        r = {}
        for name in ("diag2", "diag3"):
            hits = tot = 0
            ce_sum = 0.0
            for seq in SPLIT[name]:
                for x in range(0, 17, 2):
                    ep = episode(tuple(seq), x,
                                 tpl_ids=[TPL_DIAG[i % 2] for i in
                                          range(len(seq))])
                    s_ids, s_mask = self.lm.encode(
                        ep["spans"], fixed_length=Ts, device=DEV)
                    out = self.sc_forward(
                        self.q_ids, s_ids.unsqueeze(0),
                        s_mask.unsqueeze(0))
                    corr = (out["logits"] - out["base_logits"]
                            )[:, self.label_ids]
                    hits += int(corr.argmax(-1).item() == ep["answer"])
                    ce_sum += float(F.cross_entropy(
                        corr, torch.tensor([ep["answer"]],
                                           device=DEV)))
                    tot += 1
            r[name] = round(hits / tot, 4)
            r[name + "_ce"] = round(ce_sum / tot, 4)
        return r

    @torch.no_grad()
    def eval_one_op_probe(self):
        """Subsampled one-op probe for trajectory visibility only
        (12 ops x 5 values x 2 templates = 120 cases); the D2 gate
        uses the full posteval bank."""
        hits = tot = 0
        ce_sum = 0.0
        for op in range(12):
            for x in range(0, 17, 4):
                for tpl in (0, 4):
                    spans = spans_for((op,), x, (tpl,))
                    s_ids, s_mask = self.lm.encode(
                        spans, fixed_length=Ts, device=DEV)
                    out = self.sc_forward(
                        self.q_ids, s_ids.unsqueeze(0),
                        s_mask.unsqueeze(0))
                    corr = (out["logits"] - out["base_logits"]
                            )[:, self.label_ids]
                    a = apply_chain([op], x)
                    hits += int(corr.argmax(-1).item() == a)
                    ce_sum += float(F.cross_entropy(
                        corr, torch.tensor([a], device=DEV)))
                    tot += 1
        return round(hits / tot, 4), round(ce_sum / tot, 4)

    def full_ckpt(self, path):
        torch.save({
            "bridge_trainable": {n: t for n, t in
                                 self.bridge.state_dict().items()
                                 if not n.startswith("lm.")},
            "lora": {n: t for n, t in
                     self.lm.core.state_dict().items()
                     if "lora_" in n},
            "opt": self.opt.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "data_rng": self.rng.getstate(),
            "step": self.step_count}, path)


torch.manual_seed(SEED)
lm = QwenGenome().attach_lora().to(DEV)
tr = StagedD2Trainer(lm, data_seed=SEED, device=DEV)
tr.bridge.beta.fill_(BETA)
audit = tr.optimizer_audit()
assert audit["all_clear"], audit
RESUMED_FROM = None
if _A.resume:
    rck = torch.load(_A.resume, map_location=DEV, weights_only=False)
    tr.bridge.load_state_dict(rck["bridge_trainable"], strict=False)
    lm.core.load_state_dict(rck["lora"], strict=False)
    tr.opt.load_state_dict(rck["opt"])
    torch.set_rng_state(rck["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([t.cpu() for t in rck["cuda_rng"]])
    tr.rng.setstate(rck["data_rng"])
    tr.step_count = rck["step"]
    RESUMED_FROM = {"ckpt": _A.resume, "step": rck["step"]}
    log(f"RESUMED bit-exact from step {rck['step']}")
log(f"start: {torch.cuda.get_device_name(0)} beta={BETA} seed={SEED} "
    f"mix=L0:10/L1:25/L2:32.5/L3:32.5")

traj = []
t0 = time.time()
Path("results").mkdir(exist_ok=True)
for step in range(20_000 - tr.step_count):
    loss = tr.train_step()
    s = step + 1
    if s % 250 == 0:
        d = tr.eval_diag()
        oo, oo_ce = tr.eval_one_op_probe()
        traj.append({"step": s, "loss": round(loss, 3),
                     "one_op_probe": oo, "one_op_ce": oo_ce, **d})
        log(f"step {s}: loss {loss:.3f} one_op~{oo}/ce{oo_ce} "
            f"diag {d}")
    if s % 5000 == 0:
        tr.full_ckpt(f"results/posctl_{WTAG}s{SEED}_ckpt_{s}.pt")
        Path("results/d2_traj_partial.json").write_text(
            json.dumps(traj))

tr.full_ckpt(f"results/posctl_{WTAG}s{SEED}_ckpt_final.pt")
rep = {"seed": SEED, "resumed_from": RESUMED_FROM, "mixture": "L0:0.10/L1:0.25/L2:0.325/L3:0.325",
       "trajectory": traj, "wall_s": round(time.time() - t0),
       "gpu": torch.cuda.get_device_name(0)}
Path(f"results/posctl_{WTAG}s{SEED}.json").write_text(json.dumps(rep, indent=1))
Path(f"DONE_POSCTL_{WTAG}{SEED}").touch()
log(f"D2 COMPLETE: final one_op~{traj[-1]['one_op_probe']} "
    f"diag2 {traj[-1]['diag2']} diag3 {traj[-1]['diag3']}")

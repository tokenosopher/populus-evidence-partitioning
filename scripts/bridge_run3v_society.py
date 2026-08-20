"""Run-3V paired-arm society trainer (cohort ruling §B, Run-3V spec).

ONE trajectory of one arm. Arms share EVERYTHING (init, batch order,
sampler stream, packets, topology, budget) except evidence
visibility:

  --arm P : private relay — each cell sees ONLY its own span
            (the qualification society, unchanged).
  --arm G : globally visible relay — every cell sees EVERY span
            (concatenated, fixed order), same packets/stages/params.

Usage (official):
  BRIDGE_OP_SEED=6011 BRIDGE_SPLIT_SEED=2203 \
  python scripts/bridge_run3v_society.py --arm P \
      --model-seed 200 --order-seed 900

Fresh world: op seed 6011, split seed 2203 (hash 94e506d408b6def1),
grammar results/run3v/template_grammar_run3v.json (def14eb4182e2949).
Everything else frozen per V2.1.2 mechanics: four-depth mixture with
L1 position balancing, batch 32, EXACTLY 20,000 updates, full-state
checkpoints every 5k, realized-count logging every 1k, batch-stream
hashes. Probes informational only; verdicts via the Run-3V evaluator.
"""
import argparse
import hashlib
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

WORLD = os.environ.get("RUN3V_WORLD", "R")
if WORLD == "R":     # Run-3V-R: existing qualification world (ruling)
    assert os.environ.get("BRIDGE_SPLIT_SEED") == "1409"
    GRAMMAR_PATH = "results/bridge/template_grammar.json"
else:                # sealed fresh world (confirmatory reserve)
    assert os.environ.get("BRIDGE_OP_SEED") == "6011"
    assert os.environ.get("BRIDGE_SPLIT_SEED") == "2203"
    assert os.environ.get("BRIDGE_GRAMMAR_SEED") == "7717"
    GRAMMAR_PATH = "results/run3v/template_grammar_run3v.json"

from populus.bridge_tasks import (FORWARD_SPAN, SPLIT,        # noqa
                                  VALUE_SPAN, apply_chain,
                                  render_op)
from populus.bridge_train import (Trainer, aux_lambda,        # noqa
                                  lr_scale)

P_ARGS = argparse.ArgumentParser()
P_ARGS.add_argument("--arm", choices=("P", "G"), required=True)
P_ARGS.add_argument("--model-seed", type=int, required=True)
P_ARGS.add_argument("--order-seed", type=int, required=True)
P_ARGS.add_argument("--steps", type=int, default=20_000)
P_ARGS.add_argument("--resume", default=None,
                    help="full-state ckpt: bit-exact continuation "
                         "(weights+opt+torch/cuda/data RNG+step)")
A = P_ARGS.parse_args()
DEV = "cuda"
BETA = 0.003
MIX = (0.10, 0.35, 0.675)
TAG = f"{WORLD}_{A.arm}_m{A.model_seed}_o{A.order_seed}"

GRAMMAR = json.load(open(GRAMMAR_PATH))
QUAL_PHR = {int(k): [r["text"] for r in v["qual"]]
            for k, v in GRAMMAR["ops"].items()}


def log(msg):
    print(f"[run3v {TAG}] {msg}", flush=True)


class Run3VTrainer(Trainer):
    """V2.1.2 sampler; arm-dependent evidence packaging."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.counts = {"depth": {}, "op": {}, "value": {},
                       "l1_pos": {}, "template": {}}
        self.rolling_hash = hashlib.sha256()

    def _bump(self, key, val):
        d = self.counts[key]
        d[str(val)] = d.get(str(val), 0) + 1

    def sample_spans(self):
        r = self.rng.random()
        x = self.rng.randrange(17)
        if r < MIX[0]:
            spans = [VALUE_SPAN.format(x=x)] + [FORWARD_SPAN] * 3
            meta = ((), x, (), None)
            self._bump("depth", 0)
        elif r < MIX[1]:
            op = self.rng.randrange(12)
            tpl = self.rng.randrange(8)
            pos = self.rng.randrange(1, 4)
            spans = [VALUE_SPAN.format(x=x)] + [FORWARD_SPAN] * 3
            spans[pos] = render_op(op, tpl)
            meta = ((op,), x, (tpl,), pos)
            self._bump("depth", 1); self._bump("op", op)
            self._bump("l1_pos", pos); self._bump("template", tpl)
        else:
            bank = SPLIT["train2"] if r < MIX[2] else SPLIT["train3"]
            seq = tuple(bank[self.rng.randrange(len(bank))])
            tpls = tuple(self.rng.randrange(8) for _ in seq)
            spans = [VALUE_SPAN.format(x=x)]
            spans += [render_op(o, t) for o, t in zip(seq, tpls)]
            while len(spans) < 4:
                spans.append(FORWARD_SPAN)
            meta = (seq, x, tpls, None)
            self._bump("depth", len(seq))
            for o in seq: self._bump("op", o)
            for t in tpls: self._bump("template", t)
        self._bump("value", x)
        return spans, apply_chain(list(meta[0]), x), meta

    def package(self, s_ids, s_mask, n):
        """MASKED-LAYOUT design (referee's preferred implementation):
        BOTH arms give every cell the identical fixed four-slot
        concatenation (same tokens, same RoPE positions, same
        compute). The ONLY arm difference is the attention mask:
          Arm G: all four span slots readable;
          Arm P: only the cell's OWN slot readable (foreign slots
                 have exactly zero attention influence)."""
        s_ids = s_ids.reshape(n, 4, self.Ts)
        s_mask = s_mask.reshape(n, 4, self.Ts)
        flat_i = s_ids.reshape(n, 1, 4 * self.Ts) \
                      .expand(n, 4, -1).contiguous()
        if A.arm == "G":
            flat_m = s_mask.reshape(n, 1, 4 * self.Ts) \
                           .expand(n, 4, -1).contiguous()
            return flat_i, flat_m
        eye = torch.eye(4, device=s_mask.device,
                        dtype=s_mask.dtype)          # (cell, slot)
        per_cell = s_mask.unsqueeze(1) * \
            eye.unsqueeze(0).unsqueeze(-1)           # (n,4,4,Ts)
        return flat_i, per_cell.reshape(n, 4, 4 * self.Ts)

    # Masked layout gives every cell the full 4-slot context (the
    # referee's comparability requirement), ~4x the qual token load;
    # batch 32 no longer fits a 32GB card in one pass. The frozen
    # batch-32 update is realized as fixed 2x16 gradient accumulation
    # (chunk-mean scaled by chunk/n = exact full-batch mean gradient
    # up to summation order; chunking is constant, identical in BOTH
    # arms and on every machine, so cross-machine bit-exactness and
    # the batch-stream hash spec are unchanged).
    ACC_CHUNK = 16

    def train_step(self, n=32):
        scale = lr_scale(self.step_count)
        for g in self.opt.param_groups:
            base = 5e-4 if g.get("name") == "projections" else 1e-4
            g["lr"] = base * scale
        q, s_ids, s_mask, tgt = self.make_batch(n)
        lam = aux_lambda(self.step_count)
        self.opt.zero_grad()
        total = 0.0
        for i in range(0, n, self.ACC_CHUNK):
            sl = slice(i, min(n, i + self.ACC_CHUNK))
            w = (sl.stop - sl.start) / n
            out = self.bridge.forward_episode(q[sl], s_ids[sl],
                                              s_mask[sl])
            corrected = (out["logits"]
                         - out["base_logits"].detach()
                         )[:, self.label_ids]
            loss = F.cross_entropy(corrected, tgt[sl])
            if lam > 0:
                loss = loss + lam * F.cross_entropy(
                    out["aux_logits"], tgt[sl])
            (loss * w).backward()
            total += float(loss) * w
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.groups for p in g["params"]], 1.0)
        self.opt.step()
        self.step_count += 1
        return total

    def make_batch(self, n=32, **_):
        eps = [self.sample_spans() for _ in range(n)]
        s_ids, s_mask = self.lm.encode(
            [s for sp, _, _ in eps for s in sp],
            fixed_length=self.Ts, device=self.dev)
        s_ids, s_mask = self.package(s_ids, s_mask, n)
        q = self.q_ids.expand(n, -1)
        tgt = torch.tensor([a for _, a, _ in eps], device=self.dev)
        h = hashlib.sha256(json.dumps(
            [m for _, _, m in eps]).encode()).hexdigest()[:12]
        self.batch_hashes.append(h)
        self.rolling_hash.update(h.encode())
        return q, s_ids, s_mask, tgt

    @torch.no_grad()
    def probe(self):
        def run_case(spans, answer):
            s_ids, s_mask = self.lm.encode(
                spans, fixed_length=self.Ts, device=self.dev)
            s_ids, s_mask = self.package(s_ids, s_mask, 1)
            out = self.bridge.forward_episode(self.q_ids, s_ids,
                                              s_mask)
            corr = (out["logits"] - out["base_logits"]
                    )[:, self.label_ids]
            return int(corr.argmax(-1).item() == answer)

        r = {}
        hits = tot = 0
        for x in range(0, 17, 2):
            hits += run_case([VALUE_SPAN.format(x=x)]
                             + [FORWARD_SPAN] * 3, x)
            tot += 1
        r["l0"] = round(hits / tot, 4)
        hits = tot = 0
        for op in range(0, 12, 2):
            for pos in (1, 2, 3):
                x = (op * 5 + pos) % 17
                spans = [VALUE_SPAN.format(x=x)] + [FORWARD_SPAN] * 3
                spans[pos] = QUAL_PHR[op][pos % 4]
                hits += run_case(spans, apply_chain([op], x))
                tot += 1
        r["l1_qual"] = round(hits / tot, 4)
        for name in ("diag2", "diag3"):
            hits = tot = 0
            for seq in SPLIT[name][:12]:
                for x in range(0, 17, 4):
                    spans = [VALUE_SPAN.format(x=x)]
                    spans += [QUAL_PHR[o][(i + x) % 4]
                              for i, o in enumerate(seq)]
                    while len(spans) < 4:
                        spans.append(FORWARD_SPAN)
                    hits += run_case(spans, apply_chain(list(seq), x))
                    tot += 1
            r[name] = round(hits / tot, 4)
        return r

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
            "step": self.step_count,
            "arm": A.arm,
            "model_seed": A.model_seed, "order_seed": A.order_seed,
            "beta": BETA,
            "batch_stream_hash": self.rolling_hash.hexdigest(),
        }, path)


from populus.bridge_qwen import QwenGenome                    # noqa
torch.manual_seed(A.model_seed)
lm = QwenGenome().attach_lora().to(DEV)
tr = Run3VTrainer(lm, data_seed=A.order_seed, device=DEV)
tr.bridge.beta.fill_(BETA)
assert tr.optimizer_audit()["all_clear"]
RESUMED_FROM = None
if A.resume:
    rck = torch.load(A.resume, map_location=DEV, weights_only=False)
    assert rck["arm"] == A.arm
    assert rck["model_seed"] == A.model_seed
    assert rck["order_seed"] == A.order_seed
    tr.bridge.load_state_dict(rck["bridge_trainable"], strict=False)
    lm.core.load_state_dict(rck["lora"], strict=False)
    tr.opt.load_state_dict(rck["opt"])
    torch.set_rng_state(rck["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([t.cpu() for t in rck["cuda_rng"]])
    tr.rng.setstate(rck["data_rng"])
    tr.step_count = rck["step"]
    RESUMED_FROM = {"ckpt": A.resume, "step": rck["step"],
                    "pre_resume_stream_hash": rck["batch_stream_hash"]}
    log(f"RESUMED bit-exact from step {rck['step']} "
        f"(interrupted-run continuation; RNG/opt state restored)")
log(f"start: {torch.cuda.get_device_name(0)} arm={A.arm} "
    f"op=6011 split=2203")

traj = []
t0 = time.time()
Path("results").mkdir(exist_ok=True)
for step in range(A.steps - tr.step_count):
    loss = tr.train_step()
    s = tr.step_count
    if s % 500 == 0:
        p = tr.probe()
        traj.append({"step": s, "loss": round(loss, 3), **p})
        log(f"step {s}: loss {loss:.3f} probe {p}")
    if s % 1000 == 0:
        log(f"realized counts @ {s}: "
            f"{json.dumps(tr.counts, sort_keys=True)}")
    if s % 5000 == 0:
        tr.full_ckpt(f"results/run3v_{TAG}_ckpt_{s}.pt")

tr.full_ckpt(f"results/run3v_{TAG}_ckpt_final.pt")
rep = {"arm": A.arm, "model_seed": A.model_seed,
       "order_seed": A.order_seed, "trajectory": traj,
       "wall_s": round(time.time() - t0),
       "gpu": torch.cuda.get_device_name(0),
       "realized_counts": tr.counts,
       "resumed_from": RESUMED_FROM,
       "batch_stream_hash": tr.rolling_hash.hexdigest()}
Path(f"results/run3v_{TAG}.json").write_text(json.dumps(rep, indent=1))
Path(f"DONE_RUN3V_{TAG}").touch()
log(f"RUN3V TRAJECTORY COMPLETE: {traj[-1] if traj else None}")

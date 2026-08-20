"""V2.1.2 society qualification run (ONE trajectory).

Usage (official):
  BRIDGE_SPLIT_SEED=1409 python scripts/bridge_qual_society.py \
      --model-seed 100 --order-seed 700

Frozen per PLAN_RUN3_V212_CANONICAL.md:
- society (BridgeA1.forward_episode), frozen optimizer protocol,
  batch 32, EXACTLY 20,000 completed optimizer steps, no early stop;
- four-depth IID sampler: L0 0.10 / L1 0.25 / L2 0.325 / L3 0.325
  over the FRESH banks (split seed 1409, hash b555bb4887b375f4);
- L1 operator position balanced across C1/C2/C3 (forwarders at the
  other cell positions); L0 = value span + three forwarders;
- training phrasings: exposed templates 0-7 (unchanged);
- realized depth/operator/value/position/template counts logged every
  1,000 updates; batch-stream hashes retained (rolling + per-batch);
- FULL-state checkpoints (trainables + optimizer + torch/cuda/data
  RNG) every 5,000 steps and at 20,000; beta 0.003 in the buffer;
- in-run trajectory evals every 500 steps are INFORMATIONAL ONLY
  (subsampled; fresh grammar 'qual' phrasings for L1/diag); the
  verdict comes solely from the final-checkpoint evaluator
  (bridge_qual_evaluator.py).

Writes results/qual_m{M}_o{O}.json, checkpoints
results/qual_m{M}_o{O}_ckpt_{step}.pt, sentinel DONE_QUAL_m{M}_o{O}.
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

assert os.environ.get("BRIDGE_SPLIT_SEED") == "1409", \
    "official qualification requires BRIDGE_SPLIT_SEED=1409"

from populus.bridge_tasks import (FORWARD_SPAN, SPLIT,        # noqa
                                  VALUE_SPAN, apply_chain,
                                  render_op)
from populus.bridge_train import (Trainer, aux_lambda,        # noqa
                                  lr_scale)

P_ARGS = argparse.ArgumentParser()
P_ARGS.add_argument("--model-seed", type=int, required=True)
P_ARGS.add_argument("--order-seed", type=int, required=True)
P_ARGS.add_argument("--steps", type=int, default=20_000)
A = P_ARGS.parse_args()
DEV = "cuda"
BETA = 0.003                       # frozen common beta (ruling §4)
MIX = (0.10, 0.35, 0.675)
TAG = f"m{A.model_seed}_o{A.order_seed}"

GRAMMAR = json.load(open("results/bridge/template_grammar.json"))
QUAL_PHR = {int(k): [r["text"] for r in v["qual"]]
            for k, v in GRAMMAR["ops"].items()}


def log(msg):
    print(f"[qual {TAG}] {msg}", flush=True)


class QualSocietyTrainer(Trainer):
    """Frozen society trainer + the V2.1.2 four-depth sampler."""

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
        if r < MIX[0]:                      # L0 identity/transport
            spans = [VALUE_SPAN.format(x=x)] + [FORWARD_SPAN] * 3
            meta = ((), x, (), None)
            self._bump("depth", 0)
        elif r < MIX[1]:                    # L1, position-balanced
            op = self.rng.randrange(12)
            tpl = self.rng.randrange(8)
            pos = self.rng.randrange(1, 4)  # C1..C3
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
        answer = apply_chain(list(meta[0]), x)
        return spans, answer, meta

    def make_batch(self, n=32, **_):
        eps = [self.sample_spans() for _ in range(n)]
        s_ids, s_mask = self.lm.encode(
            [s for sp, _, _ in eps for s in sp],
            fixed_length=self.Ts, device=self.dev)
        q = self.q_ids.expand(n, -1)
        tgt = torch.tensor([a for _, a, _ in eps], device=self.dev)
        h = hashlib.sha256(json.dumps(
            [m for _, _, m in eps]).encode()).hexdigest()[:12]
        self.batch_hashes.append(h)
        self.rolling_hash.update(h.encode())
        return (q, s_ids.reshape(n, 4, self.Ts),
                s_mask.reshape(n, 4, self.Ts), tgt)

    # ---- informational trajectory probes (subsampled) ------------
    @torch.no_grad()
    def probe(self):
        def run_case(spans, answer):
            s_ids, s_mask = self.lm.encode(
                spans, fixed_length=self.Ts, device=self.dev)
            out = self.bridge.forward_episode(
                self.q_ids, s_ids.unsqueeze(0), s_mask.unsqueeze(0))
            corr = (out["logits"] - out["base_logits"]
                    )[:, self.label_ids]
            return int(corr.argmax(-1).item() == answer)

        r = {}
        hits = tot = 0                       # L0
        for x in range(0, 17, 2):
            hits += run_case([VALUE_SPAN.format(x=x)]
                             + [FORWARD_SPAN] * 3, x)
            tot += 1
        r["l0"] = round(hits / tot, 4)
        hits = tot = 0                       # L1, qual phrasings
        for op in range(0, 12, 2):
            for pos in (1, 2, 3):
                x = (op * 5 + pos) % 17
                spans = [VALUE_SPAN.format(x=x)] + [FORWARD_SPAN] * 3
                spans[pos] = QUAL_PHR[op][pos % 4]
                hits += run_case(spans, apply_chain([op], x))
                tot += 1
        r["l1_qual"] = round(hits / tot, 4)
        for name in ("diag2", "diag3"):      # subsampled diag, qual phr
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
            "model_seed": A.model_seed, "order_seed": A.order_seed,
            "beta": BETA,
            "batch_stream_hash": self.rolling_hash.hexdigest(),
        }, path)


from populus.bridge_qwen import QwenGenome                    # noqa
torch.manual_seed(A.model_seed)
lm = QwenGenome().attach_lora().to(DEV)
tr = QualSocietyTrainer(lm, data_seed=A.order_seed, device=DEV)
tr.bridge.beta.fill_(BETA)
audit = tr.optimizer_audit()
assert audit["all_clear"], audit
log(f"start: {torch.cuda.get_device_name(0)} beta={BETA} "
    f"mix=10/25/32.5/32.5 split=1409")

traj = []
t0 = time.time()
Path("results").mkdir(exist_ok=True)
for step in range(A.steps):
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
        tr.full_ckpt(f"results/qual_{TAG}_ckpt_{s}.pt")

tr.full_ckpt(f"results/qual_{TAG}_ckpt_final.pt")
rep = {"model_seed": A.model_seed, "order_seed": A.order_seed,
       "trajectory": traj, "wall_s": round(time.time() - t0),
       "gpu": torch.cuda.get_device_name(0),
       "realized_counts": tr.counts,
       "batch_stream_hash": tr.rolling_hash.hexdigest(),
       "batch_hash_tail": tr.batch_hashes[-20:]}
Path(f"results/qual_{TAG}.json").write_text(json.dumps(rep, indent=1))
Path(f"DONE_QUAL_{TAG}").touch()
log(f"QUAL RUN COMPLETE: final probe {traj[-1] if traj else None}")

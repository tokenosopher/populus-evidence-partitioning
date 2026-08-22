"""Universality Scout S7 + family-D learnability driver
(PLAN_SCOUT_UNIVERSALITY.md, FROZEN AS AMENDED).

Modes:
  learnability : fresh society, family D, full 20,000 updates
                 (gate: L1>=0.95, diag2>=0.70, diag3>=0.60,
                  all-cut<=0.11 both depths, finite packets).
  s7a : interface-frozen / LoRA-adapted (reader, writer, mouth, beta
        frozen; LoRA + reinitialized aux head train) from A ckpt.
  s7b : full adaptation from complete A checkpoint.
  s7c : A-LoRA init + REINITIALIZED reader/writer/mouth (+aux);
        everything trains. S7b - S7c isolates the A-born interface.
S7 modes: exactly 5,000 updates, probe every 250 (fixed instrument,
identical across arms), fresh optimizer, Run-3 schedules over their
first 5,000 updates, full diag eval + all-cut at final.

Usage:
  RUN3V_WORLD=F BRIDGE_OP_SEED=6011 BRIDGE_SPLIT_SEED=2203 \
  BRIDGE_GRAMMAR_SEED=7717 python scripts/bridge_scout_s7.py \
    --mode s7b --from-ckpt results/run3v/..._m201_o951_ckpt_final.pt \
    --model-seed 201 --order-seed 951 --data-seed 640
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
assert os.environ.get("RUN3V_WORLD") == "F"
assert os.environ.get("BRIDGE_OP_SEED") == "6011"
assert os.environ.get("BRIDGE_SPLIT_SEED") == "2203"
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import torch.nn.functional as F                                  # noqa
from populus.bridge_train import Trainer, aux_lambda, lr_scale   # noqa
from populus import bridge_tasks_d as D                          # noqa

assert D.bank_hash() == "2bd0b301a5b079b1"

P = argparse.ArgumentParser()
P.add_argument("--mode", required=True,
               choices=("learnability", "s7a", "s7b", "s7c"))
P.add_argument("--from-ckpt", default=None)
P.add_argument("--model-seed", type=int, required=True)
P.add_argument("--order-seed", type=int, required=True)
P.add_argument("--data-seed", type=int, required=True,
               help="D training-stream seed; IDENTICAL across the "
                    "three S7 arms of one checkpoint (frozen spec)")
A = P.parse_args()
STEPS = 20_000 if A.mode == "learnability" else 5_000
PROBE_EVERY = 500 if A.mode == "learnability" else 250
AUX_REINIT_SEED = 424242
FRESH_PROJ_SEED = 434343
BETA = 0.003
MIX = (0.10, 0.35, 0.675)      # cumulative: L0 .10, L1 .25, L2 .325
DEV = "cuda"
TAG = f"{A.mode}_m{A.model_seed}_o{A.order_seed}_d{A.data_seed}"


def log(m):
    print(f"[scout {TAG}] {m}", flush=True)


class ScoutDTrainer(Trainer):
    """Family-D sampler; restricted (P) masked packaging ONLY."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rolling_hash = hashlib.sha256()
        self.packets_finite = True

    def sample_spans(self):
        r = self.rng.random()
        x = self.rng.randrange(17)
        if r < MIX[0]:
            spans = [D.VALUE_SPAN.format(x=x)] + [D.FORWARD_SPAN] * 3
            seq, tpls = (), ()
        elif r < MIX[1]:
            op = self.rng.randrange(12)
            tpl = self.rng.randrange(8)
            pos = self.rng.randrange(1, 4)
            spans = [D.VALUE_SPAN.format(x=x)] + [D.FORWARD_SPAN] * 3
            spans[pos] = D.render_op(op, tpl)
            seq, tpls = (op,), (tpl,)
        else:
            bank = D.SPLIT["train2"] if r < MIX[2] else \
                D.SPLIT["train3"]
            seq = tuple(bank[self.rng.randrange(len(bank))])
            tpls = tuple(self.rng.randrange(8) for _ in seq)
            spans = [D.VALUE_SPAN.format(x=x)]
            spans += [D.render_op(o, t) for o, t in zip(seq, tpls)]
            while len(spans) < 4:
                spans.append(D.FORWARD_SPAN)
        return spans, D.apply_chain(list(seq), x), (seq, x, tpls,
                                                    None)

    def package(self, s_ids, s_mask, n):
        s_ids = s_ids.reshape(n, 4, self.Ts)
        s_mask = s_mask.reshape(n, 4, self.Ts)
        flat_i = s_ids.reshape(n, 1, 4 * self.Ts) \
                      .expand(n, 4, -1).contiguous()
        eye = torch.eye(4, device=s_mask.device, dtype=s_mask.dtype)
        per_cell = s_mask.unsqueeze(1) * \
            eye.unsqueeze(0).unsqueeze(-1)
        return flat_i, per_cell.reshape(n, 4, 4 * self.Ts)

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
            if not torch.isfinite(out["packets"]).all():
                self.packets_finite = False
                raise FloatingPointError(
                    f"non-finite packet at update {self.step_count}")
            corrected = (out["logits"]
                         - out["base_logits"].detach()
                         )[:, self.label_ids]
            loss = F.cross_entropy(corrected, tgt[sl])
            if lam > 0:
                loss = loss + lam * F.cross_entropy(
                    out["aux_logits"], tgt[sl])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at update {self.step_count}")
            (loss * w).backward()
            total += float(loss) * w
        params = [p for g in self.groups for p in g["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite gradient norm at update "
                f"{self.step_count}")
        for p_ in params:
            if p_.grad is not None and \
                    not torch.isfinite(p_.grad).all():
                raise FloatingPointError(
                    f"non-finite gradient at update "
                    f"{self.step_count}")
        self.opt.step()
        for p_ in params:
            if not torch.isfinite(p_).all():
                raise FloatingPointError(
                    f"non-finite parameter at update "
                    f"{self.step_count}")
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
    def probe(self, full=False, cut=False):
        def run_case(spans, answer):
            s_ids, s_mask = self.lm.encode(
                spans, fixed_length=self.Ts, device=self.dev)
            s_ids, s_mask = self.package(s_ids, s_mask, 1)
            out = self.bridge.forward_episode(self.q_ids, s_ids,
                                              s_mask, cut_mail=cut)
            corr = (out["logits"] - out["base_logits"]
                    )[:, self.label_ids]
            return int(corr.argmax(-1).item() == answer)

        r = {}

        # L0
        hits = tot = 0
        l0_xs = range(17) if full else range(0, 17, 2)
        for x in l0_xs:
            hits += run_case(
                [D.VALUE_SPAN.format(x=x)] + [D.FORWARD_SPAN] * 3,
                x,
            )
            tot += 1
        r["l0"] = round(hits / tot, 4)

        # L1
        hits = tot = 0
        step_ops = range(12) if full else range(0, 12, 2)
        tpl_pool = D.TPL_DIAG if full else D.TPL_DIAG[:1]
        for op in step_ops:
            for pos in (1, 2, 3):
                xs = range(17) if full else ((op * 5 + pos) % 17,)
                for tpl in tpl_pool:
                    for x in xs:
                        spans = [D.VALUE_SPAN.format(x=x)] \
                            + [D.FORWARD_SPAN] * 3
                        spans[pos] = D.render_op(op, tpl)
                        hits += run_case(spans, D.apply_op(op, x))
                        tot += 1
        r["l1"] = round(hits / tot, 4)

        # L2 / L3
        for name in ("diag2", "diag3"):
            hits = tot = 0
            bank = D.SPLIT[name] if full else D.SPLIT[name][:12]
            xs = range(17) if full else range(0, 17, 4)
            rotations = range(len(D.TPL_DIAG)) if full else range(1)
            for seq in bank:
                for x in xs:
                    for rotation in rotations:
                        spans = [D.VALUE_SPAN.format(x=x)]
                        spans += [
                            D.render_op(
                                op,
                                D.TPL_DIAG[
                                    (rotation + position)
                                    % len(D.TPL_DIAG)
                                ],
                            )
                            for position, op in enumerate(seq)
                        ]
                        while len(spans) < 4:
                            spans.append(D.FORWARD_SPAN)
                        hits += run_case(
                            spans,
                            D.apply_chain(list(seq), x),
                        )
                        tot += 1
            r[name] = round(hits / tot, 4)

        return r


from populus.bridge_qwen import QwenGenome                    # noqa
torch.manual_seed(A.model_seed)
lm = QwenGenome().attach_lora().to(DEV)
tr = ScoutDTrainer(lm, data_seed=A.data_seed, device=DEV)
tr.bridge.beta.fill_(BETA)
if A.mode == "learnability":
    assert tr.optimizer_audit()["all_clear"]

def load_subset_exact(module, payload, label):
    result = module.load_state_dict(payload, strict=False)
    assert not result.unexpected_keys, (label,
                                        result.unexpected_keys)
    state = module.state_dict()
    absent = set(payload) - set(state)
    assert not absent, (label, "absent keys", sorted(absent))
    for key, expected in payload.items():
        assert torch.equal(state[key].detach().cpu(),
                           expected.detach().cpu()), \
            (label, "load mismatch", key)


S7_ALLOWED = {("P", 201, 951), ("P", 203, 953)}
S7_DATA_SEEDS = {(201, 951): 640, (203, 953): 641}
INIT = {"mode": A.mode, "from_ckpt": A.from_ckpt}
if A.mode != "learnability":
    assert A.from_ckpt, "--from-ckpt required for S7 arms"
    ck = torch.load(A.from_ckpt, map_location=DEV,
                    weights_only=False)
    checkpoint_id = (ck["arm"], ck["model_seed"], ck["order_seed"])
    assert checkpoint_id in S7_ALLOWED, checkpoint_id
    assert ck["model_seed"] == A.model_seed
    assert ck["order_seed"] == A.order_seed
    assert int(ck["step"]) == 20_000, (
        "S7 requires the final Run-3V checkpoint", ck["step"])
    expected_data_seed = S7_DATA_SEEDS[
        (ck["model_seed"], ck["order_seed"])]
    assert A.data_seed == expected_data_seed, (
        A.data_seed, expected_data_seed)
    if A.mode in ("s7a", "s7b"):
        load_subset_exact(tr.bridge, ck["bridge_trainable"],
                          "bridge")
    load_subset_exact(lm.core, ck["lora"], "lora")
    if A.mode == "s7c":
        # fresh reader/writer/mouth: architecture-native init from a
        # seeded fresh BridgeA1, identical across checkpoints
        from populus.bridge import BridgeA1 as _B
        torch.manual_seed(FRESH_PROJ_SEED)
        fresh = _B(lm)
        fsd = fresh.state_dict()
        for n_, p_ in tr.bridge.named_parameters():
            if not n_.startswith("lm.") and "aux" not in n_:
                p_.data.copy_(fsd[n_])
        del fresh
    # aux head: reinitialize identically in EVERY arm
    g = torch.Generator(device="cpu").manual_seed(AUX_REINIT_SEED)
    for p_ in tr.bridge.aux_head.parameters():
        p_.data.copy_(torch.randn(p_.shape, generator=g) * 0.02)
    if A.mode == "s7a":
        frozen = []
        for n_, p_ in tr.bridge.named_parameters():
            if (not n_.startswith("lm.")) and ("aux" not in n_):
                p_.requires_grad_(False)
                frozen.append(n_)
        aux = list(tr.bridge.aux_head.parameters())
        lora = lm.lora_parameters()
        tr.groups = [
            {"params": aux, "lr": 5e-4, "weight_decay": 0.01,
             "name": "projections"},
            {"params": lora, "lr": 1e-4, "weight_decay": 0.0,
             "name": "lora"}]
        tr.opt = torch.optim.AdamW(tr.groups)
        INIT["frozen_params"] = frozen
    audit = tr.optimizer_audit()
    assert audit["all_clear"], audit
    INIT["ckpt_step"] = ck["step"]

trainable = {n for n, p in tr.bridge.named_parameters()
             if p.requires_grad and not n.startswith("lm.")}
log(f"trainable bridge params: {sorted(trainable)}")

initial_probe = tr.probe()
traj = [{"step": 0, "loss": None, **initial_probe}]
log(f"step 0: probe {initial_probe}")
t0 = time.time()
Path("results/scout").mkdir(parents=True, exist_ok=True)
for _ in range(STEPS):
    loss = tr.train_step()
    s = tr.step_count
    if s % PROBE_EVERY == 0:
        p = tr.probe()
        traj.append({"step": s, "loss": round(loss, 3), **p})
        log(f"step {s}: loss {loss:.3f} probe {p}")

final = tr.probe(full=True)
final_cut = tr.probe(full=True, cut=True)
auc = 0.0
for left, right in zip(traj[:-1], traj[1:]):
    width = right["step"] - left["step"]
    auc += 0.5 * (left["diag3"] + right["diag3"]) * width
auc /= STEPS  # normalized to [0, 1]
steps_to = {th: next((t["step"] for t in traj if t["diag3"] >= th),
                     None) for th in (0.40, 0.60, 0.80)}
rep = {"tag": TAG, "mode": A.mode, "init": INIT, "steps": STEPS,
       "d_bank_hash": D.bank_hash(),
       "data_seed": A.data_seed, "trajectory": traj,
       "final_full": final, "final_full_allcut": final_cut,
       "auc_diag3_normalized": round(auc, 6), "steps_to": steps_to,
       "packets_finite": tr.packets_finite,
       "batch_stream_hash": tr.rolling_hash.hexdigest(),
       "wall_s": round(time.time() - t0),
       "gpu": torch.cuda.get_device_name(0)}
torch.save({"bridge_trainable": {n: t for n, t in
                                 tr.bridge.state_dict().items()
                                 if not n.startswith("lm.")},
            "lora": {n: t for n, t in lm.core.state_dict().items()
                     if "lora_" in n},
            "step": tr.step_count, "arm": "P", "mode": A.mode,
            "model_seed": A.model_seed, "order_seed": A.order_seed},
           f"results/scout/scout_{TAG}_ckpt_final.pt")
Path(f"results/scout/scout_{TAG}.json").write_text(
    json.dumps(rep, indent=1))
print(f"S7_DONE {TAG} final {final} cut {final_cut}", flush=True)

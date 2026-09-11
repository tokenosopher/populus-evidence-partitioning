"""Bridge-A1 trainer (PLAN_RUN3 v2.1.1 §B operational protocol).

Frozen protocol: AdamW with two disjoint parameter groups
(projections+aux lr 5e-4 wd 0.01; LoRA lr 1e-4 wd 0.0); linear warmup
1k -> constant to 5k -> cosine to 10% by 20k; exactly 20,000 completed
optimizer.step() calls; grad clip 1.0; fp32; batch 32 episodes; 50/50
depth sampling at programme level, uniform banks/x/train-templates;
aux-head anneal 0.5 (0-4k) -> linear to 0 (4k-10k) -> 0; DP-7
residualized loss; diagnostic eval every 500 updates; full-state
checkpoints every 500 through 5k then every 2500; the update-20,000
checkpoint IS the run's checkpoint (no selection).

optimizer_audit() emits the v2.1.1 §5 proof:
  projection_group ∩ lora_group = ∅;
  projection_group ∪ lora_group = all trainables (each exactly once);
  no frozen parameter in any group; counts match the manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .bridge import BridgeA1
from .bridge_tasks import (ANSWER_PREFIX, LABELS, QUESTION, SPLIT,
                           TPL_DIAG, TPL_TRAIN, episode)

TOTAL_UPDATES = 20_000
BATCH = 32
CLIP = 1.0


def lr_scale(step):
    if step < 1_000:
        return step / 1_000
    if step < 5_000:
        return 1.0
    frac = (step - 5_000) / (TOTAL_UPDATES - 5_000)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac))


def aux_lambda(step):
    if step < 4_000:
        return 0.5
    if step < 10_000:
        return 0.5 * (1 - (step - 4_000) / 6_000)
    return 0.0


class Trainer:
    def __init__(self, lm, data_seed: int, device="cpu"):
        self.lm = lm
        self.bridge = BridgeA1(lm).to(device)
        self.dev = torch.device(device)
        self.rng = random.Random(data_seed)
        self.data_seed = data_seed
        self.label_ids = torch.tensor(
            [r["token_id"] for r in json.load(
                open("results/bridge/label_context.json"))["labels"]],
            device=self.dev)
        c = json.load(open("results/bridge/serialization_contract.json"))
        self.Tq, self.Ts = c["T_QUESTION"], c["T_SPAN"]
        self.q_ids, _ = lm.encode([QUESTION + ANSWER_PREFIX],
                                  fixed_length=self.Tq,
                                  device=self.dev)
        # disjoint optimizer groups (audited)
        proj = [p for n, p in self.bridge.named_parameters()
                if p.requires_grad and not n.startswith("lm.")]
        lora = lm.lora_parameters()
        self.groups = [
            {"params": proj, "lr": 5e-4, "weight_decay": 0.01,
             "name": "projections"},
            {"params": lora, "lr": 1e-4, "weight_decay": 0.0,
             "name": "lora"}]
        self.opt = torch.optim.AdamW(self.groups)
        self.step_count = 0
        self.batch_hashes = []

    # ------------------------------------------------------------
    def optimizer_audit(self):
        proj = {id(p) for p in self.groups[0]["params"]}
        lora = {id(p) for p in self.groups[1]["params"]}
        all_train = {id(p) for p in self.bridge.parameters()
                     if p.requires_grad}
        frozen_in_groups = sum(
            1 for g in self.groups for p in g["params"]
            if not p.requires_grad)
        listed = [id(p) for g in self.groups for p in g["params"]]
        audit = {
            "intersection_empty": len(proj & lora) == 0,
            "union_covers_all_trainables":
                (proj | lora) == all_train,
            "each_exactly_once": len(listed) == len(set(listed)),
            "frozen_params_in_groups": frozen_in_groups,
            "n_projection_params": sum(
                p.numel() for p in self.groups[0]["params"]),
            "n_lora_params": sum(
                p.numel() for p in self.groups[1]["params"]),
        }
        audit["all_clear"] = (audit["intersection_empty"]
                              and audit["union_covers_all_trainables"]
                              and audit["each_exactly_once"]
                              and frozen_in_groups == 0)
        return audit

    # ------------------------------------------------------------
    def make_batch(self, n=BATCH, banks=("train2", "train3"),
                   tpl_pool=TPL_TRAIN):
        eps = []
        for _ in range(n):
            bank = SPLIT[banks[self.rng.random() < 0.5]] \
                if len(banks) == 2 else SPLIT[banks[0]]
            seq = bank[self.rng.randrange(len(bank))]
            x = self.rng.randrange(17)
            eps.append(episode(tuple(seq), x, tpl_pool=tpl_pool,
                               rng=self.rng))
        s_ids, s_mask = self.lm.encode(
            [s for e in eps for s in e["spans"]], fixed_length=self.Ts,
            device=self.dev)
        q = self.q_ids.expand(n, -1)
        tgt = torch.tensor([e["answer"] for e in eps],
                           device=self.dev)
        h = hashlib.sha256(json.dumps(
            [(e["ops"], e["x"], e["templates"]) for e in eps]
        ).encode()).hexdigest()[:12]
        self.batch_hashes.append(h)
        return (q, s_ids.reshape(n, 4, self.Ts),
                s_mask.reshape(n, 4, self.Ts), tgt)

    # ------------------------------------------------------------
    def train_step(self, n=BATCH):
        # n != BATCH is permitted ONLY for local preflight sanity
        # checks; official runs always use the frozen BATCH=32.
        scale = lr_scale(self.step_count)
        for g in self.opt.param_groups:
            base = 5e-4 if g.get("name") == "projections" else 1e-4
            g["lr"] = base * scale
        q, s_ids, s_mask, tgt = self.make_batch(n)
        out = self.bridge.forward_episode(q, s_ids, s_mask)
        corrected = (out["logits"]
                     - out["base_logits"].detach())[:, self.label_ids]
        loss = F.cross_entropy(corrected, tgt)
        lam = aux_lambda(self.step_count)
        if lam > 0:
            loss = loss + lam * F.cross_entropy(out["aux_logits"], tgt)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.groups for p in g["params"]], CLIP)
        self.opt.step()
        self.step_count += 1        # counts completed optimizer.step()
        return float(loss)

    # ------------------------------------------------------------
    @torch.no_grad()
    def evaluate_diagnostic(self):
        r = {}
        for name, bank in (("diag2", SPLIT["diag2"]),
                           ("diag3", SPLIT["diag3"])):
            hits = tot = 0
            for seq in bank:
                for x in range(17):
                    ep = episode(tuple(seq), x,
                                 tpl_ids=[TPL_DIAG[i % 2] for i in
                                          range(len(seq))])
                    s_ids, s_mask = self.lm.encode(
                        ep["spans"], fixed_length=self.Ts,
                        device=self.dev)
                    out = self.bridge.forward_episode(
                        self.q_ids, s_ids.unsqueeze(0),
                        s_mask.unsqueeze(0))
                    corr = (out["logits"] - out["base_logits"]
                            )[:, self.label_ids]
                    hits += int(corr.argmax(-1).item() == ep["answer"])
                    tot += 1
            r[name] = round(hits / tot, 4)
        return r

    # ------------------------------------------------------------
    def save_checkpoint(self, path):
        torch.save({
            "bridge": self.bridge.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.step_count,
            "data_seed": self.data_seed,
            "rng_state": self.rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "batch_hash_tail": self.batch_hashes[-20:],
        }, path)

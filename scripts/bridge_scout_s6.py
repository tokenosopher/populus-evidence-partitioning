"""Universality Scout S6 (FROZEN): cross-program ridge probe for the
D running value in packets captured during frozen S1-style runs.
Train/test disjoint in operator sequences AND phrasings AND start
values. Diagnostic only (ruling: probe success is not causal
transfer). Reports program-held-out accuracy, permuted-label baseline, per-
interface accuracy, norm-matched-random control.

Usage: RUN3V_WORLD=F BRIDGE_OP_SEED=6011 BRIDGE_SPLIT_SEED=2203 \
  BRIDGE_GRAMMAR_SEED=7717 python scripts/bridge_scout_s6.py \
    --ckpt <ckpt>
"""
import argparse, json, os, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
assert os.environ.get("RUN3V_WORLD") == "F"
import torch
torch.use_deterministic_algorithms(True)
torch.manual_seed(1717)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
control_gen = torch.Generator(device="cpu").manual_seed(1717)
from populus.bridge import BridgeA1                             # noqa
from populus.bridge_qwen import QwenGenome                      # noqa
from populus.bridge_tasks import ANSWER_PREFIX, QUESTION        # noqa
from populus import bridge_tasks_d as D                         # noqa
assert D.bank_hash() == "2bd0b301a5b079b1"

P = argparse.ArgumentParser()
P.add_argument("--ckpt", required=True)
A = P.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = json.load(open("results/bridge/serialization_contract.json"))
Tq, Ts = C["T_QUESTION"], C["T_SPAN"]
rng = random.Random(1717)

lm = QwenGenome().attach_lora().to(DEV)
bridge = BridgeA1(lm).to(DEV).eval()
ck = torch.load(A.ckpt, map_location=DEV, weights_only=False)
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


load_subset_exact(bridge, ck["bridge_trainable"], "bridge")
load_subset_exact(lm.core, ck["lora"], "lora")
ARM = ck["arm"]
assert ARM in {"P", "G"}
AUDIT_ALLOWED = {("P", 200, 950), ("P", 201, 951),
                 ("P", 203, 903), ("P", 203, 953),
                 ("P", 204, 904), ("P", 204, 954),
                 ("G", 204, 954)}
_cid = (ck["arm"], ck["model_seed"], ck["order_seed"])
assert _cid in AUDIT_ALLOWED, _cid
assert int(ck["step"]) == 20_000, (
    "universality scout requires the final Run-3V checkpoint",
    ck["step"])
q_ids, _ = lm.encode([QUESTION + ANSWER_PREFIX], fixed_length=Tq,
                     device=DEV)
EYE4 = torch.eye(4, device=DEV)


def package(s_ids, s_mask, n):
    s_ids = s_ids.reshape(n, 4, Ts); s_mask = s_mask.reshape(n, 4, Ts)
    flat_i = s_ids.reshape(n, 1, 4 * Ts).expand(n, 4, -1).contiguous()
    if ARM == "G":
        return flat_i, s_mask.reshape(n, 1, 4 * Ts).expand(
            n, 4, -1).contiguous()
    per_cell = s_mask.unsqueeze(1) * \
        EYE4.to(s_mask.dtype).unsqueeze(0).unsqueeze(-1)
    return flat_i, per_cell.reshape(n, 4, 4 * Ts)


@torch.no_grad()
def packets_for(seq, x, tpl):
    spans = [D.VALUE_SPAN.format(x=x)]
    spans += [D.render_op(o, tpl[(i) % len(tpl)])
              for i, o in enumerate(seq)]
    while len(spans) < 4:
        spans.append(D.FORWARD_SPAN)
    s_ids, s_mask = lm.encode(spans, fixed_length=Ts, device=DEV)
    s_ids, s_mask = package(s_ids, s_mask, 1)
    out = bridge.forward_episode(q_ids, s_ids, s_mask)
    return out["packets"][0]          # (4, M_P, d)


def running(seq, x, k):
    return x if k == 0 else D.apply_chain(list(seq[:k]), x)


seqs = list(D.SPLIT["diag3"])
rng.shuffle(seqs)
tr_seqs, te_seqs = seqs[:40], seqs[40:]
TR_TPL, TE_TPL = D.TPL_TRAIN[:4], D.TPL_DIAG
TR_X = [x for x in range(17) if x % 3 != 0]
TE_X = [x for x in range(17) if x % 3 == 0]


def harvest(seqs_, tpls, xs, n_target):
    X, Y, K = [], [], []
    while len(Y) < n_target:
        seq = seqs_[rng.randrange(len(seqs_))]
        x = xs[rng.randrange(len(xs))]
        pk = packets_for(seq, x, tpls)
        for k in (0, 1, 2):
            X.append(pk[k].flatten().float().cpu())
            Y.append(running(seq, x, k)); K.append(k)
    return torch.stack(X), torch.tensor(Y), torch.tensor(K)


Xtr, Ytr, Ktr = harvest(tr_seqs, TR_TPL, TR_X, 900)
Xte, Yte, Kte = harvest(te_seqs, TE_TPL, TE_X, 300)


def ridge_fit(X, Y, lam=10.0):
    Xd = X.double()
    Yoh = torch.zeros(len(Y), 17, dtype=torch.float64)
    Yoh[range(len(Y)), Y] = 1.0
    gram = Xd @ Xd.T + lam * torch.eye(len(Xd),
                                       dtype=torch.float64)
    return Xd.T @ torch.linalg.solve(gram, Yoh)


def acc(W, X, Y):
    pred = (X.double() @ W).argmax(1)
    return round(float((pred == Y).float().mean()), 4)


W = ridge_fit(Xtr, Ytr)
rep = {"ckpt": A.ckpt, "arm": ARM,
       "S6_program_heldout_acc": acc(W, Xte, Yte), "chance": 1 / 17}
perm = Ytr.clone()
for k in (0, 1, 2):
    idx = torch.where(Ktr == k)[0]
    order = torch.randperm(len(idx), generator=control_gen)
    perm[idx] = perm[idx][order]
rep["S6_permuted_baseline"] = acc(ridge_fit(Xtr, perm), Xte, Yte)
for k in (0, 1, 2):
    m = Kte == k
    rep[f"S6_iface_{k}"] = acc(W, Xte[m], Yte[m])
Xrand = torch.randn(Xte.shape, generator=control_gen,
                    dtype=Xte.dtype)
Xrand = Xrand / Xrand.norm(dim=1, keepdim=True).clamp_min(1e-12) \
    * Xte.norm(dim=1, keepdim=True)
rep["S6_norm_matched_random"] = acc(W, Xrand, Yte)
tag = f"{ck['arm']}_m{ck.get('model_seed')}_o{ck.get('order_seed')}"
Path("results/scout").mkdir(parents=True, exist_ok=True)
Path(f"results/scout/scout_s6_{tag}.json").write_text(
    json.dumps(rep, indent=1))
print("S6_DONE", tag, json.dumps(rep), flush=True)

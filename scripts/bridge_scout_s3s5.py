"""Universality Scout S3-S5 (PLAN_SCOUT_UNIVERSALITY.md, FROZEN):
packet/mouth/local-primitive portability hierarchy on frozen Run-3V
checkpoints, using A-born natural donor packets injected into
family-D contexts.

Per ruling REVIEW_scout_ratification.md §5:
  S3: A-born donor traverses a D-context relay of ALREADY-LEARNED
      structural forwarders (family A's forwarder span; the only D
      elements are the value span and question layout). Same-value
      preservation and counterfactual following, threshold 0.90.
  S4: mouth portability: donor with the correct final value injected
      at the last interface, forwarder cell 3, D context vs A context
      positive control; threshold 0.95.
  S5: local D-primitive competence: one D operator phrase at one cell,
      A forwarders elsewhere, A-born donor incoming; scored against
      the mathematically correct transformed value; 0.90 descriptive.

Usage:
  RUN3V_WORLD=F BRIDGE_OP_SEED=6011 BRIDGE_SPLIT_SEED=2203 \
  BRIDGE_GRAMMAR_SEED=7717 python scripts/bridge_scout_s3s5.py \
      --ckpt <ckpt> [--n-donors 240]
Writes results/scout/scout_s3s5_<tag>.json.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
assert os.environ.get("RUN3V_WORLD") == "F"
assert os.environ.get("BRIDGE_OP_SEED") == "6011"
assert os.environ.get("BRIDGE_SPLIT_SEED") == "2203"
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from populus.bridge import BridgeA1                             # noqa
from populus.bridge_qwen import QwenGenome                      # noqa
from populus import bridge_tasks as A                           # noqa
from populus import bridge_tasks_d as D                         # noqa

assert D.bank_hash() == "2bd0b301a5b079b1"

P = argparse.ArgumentParser()
P.add_argument("--ckpt", required=True)
P.add_argument("--n-donors", type=int, default=240)
ARGS = P.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = json.load(open("results/bridge/serialization_contract.json"))
Tq, Ts = C["T_QUESTION"], C["T_SPAN"]
rng = random.Random(90210)

lm = QwenGenome().attach_lora().to(DEV)
bridge = BridgeA1(lm).to(DEV).eval()
ck = torch.load(ARGS.ckpt, map_location=DEV, weights_only=False)
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
ARM = ck["arm"]   # P societies + the G mechanistic comparator
AUDIT_ALLOWED = {("P", 200, 950), ("P", 201, 951),
                 ("P", 203, 903), ("P", 203, 953),
                 ("P", 204, 904), ("P", 204, 954),
                 ("G", 204, 954)}
_cid = (ck["arm"], ck["model_seed"], ck["order_seed"])
assert _cid in AUDIT_ALLOWED, _cid
assert int(ck["step"]) == 20_000, (
    "universality scout requires the final Run-3V checkpoint",
    ck["step"])
label_ids = torch.tensor([r["token_id"] for r in json.load(
    open("results/bridge/label_context.json"))["labels"]],
    device=DEV)
q_ids, _ = lm.encode([A.QUESTION + A.ANSWER_PREFIX], fixed_length=Tq,
                     device=DEV)
EYE4 = torch.eye(4, device=DEV)


def package(s_ids, s_mask, n):
    s_ids = s_ids.reshape(n, 4, Ts)
    s_mask = s_mask.reshape(n, 4, Ts)
    flat_i = s_ids.reshape(n, 1, 4 * Ts).expand(n, 4, -1).contiguous()
    if ARM == "G":
        return flat_i, s_mask.reshape(n, 1, 4 * Ts).expand(
            n, 4, -1).contiguous()
    per_cell = s_mask.unsqueeze(1) * \
        EYE4.to(s_mask.dtype).unsqueeze(0).unsqueeze(-1)
    return flat_i, per_cell.reshape(n, 4, 4 * Ts)


@torch.no_grad()
def run(spans, substitute=None):
    """substitute: {(stage, cell): (1, M_P, d)} per bridge API; the
    audit convention {(k, k): donor} replaces cell k's committed
    packet, i.e. the interface k -> k+1."""
    s_ids, s_mask = lm.encode(spans, fixed_length=Ts, device=DEV)
    s_ids, s_mask = package(s_ids, s_mask, 1)
    out = bridge.forward_episode(q_ids, s_ids, s_mask,
                                 substitute_traj=substitute)
    corr = (out["logits"] - out["base_logits"])[:, label_ids]
    return int(corr.argmax(-1)), out["packets"]


def a_episode(seq, x):
    spans = [A.VALUE_SPAN.format(x=x)]
    spans += [A.render_op(o, rng.choice(A.TPL_DIAG)) for o in seq]
    while len(spans) < 4:
        spans.append(A.FORWARD_SPAN)
    return spans


def a_running(seq, x, k):
    """Running value carried by cell k's packet: cell j applies
    seq[j-1]; cell 0 publishes x (audit convention)."""
    return x if k == 0 else A.apply_chain(list(seq[:k]), x)


# ---- donor harvest: A-born packets indexed by (interface, value) ---
MIN_DONORS_PER_VALUE = 5
MAX_HARVEST_ATTEMPTS = 200_000
bank = A.SPLIT["diag3"]
donors = {k: {v: [] for v in range(17)} for k in (0, 1, 2, 3)}
harvested = 0
attempts = 0


def minimum_donor_count():
    return min(len(donors[k][v]) for k in donors for v in range(17))


while (harvested < ARGS.n_donors
       or minimum_donor_count() < MIN_DONORS_PER_VALUE):
    attempts += 1
    if attempts > MAX_HARVEST_ATTEMPTS:
        raise RuntimeError(
            "insufficient natural-donor coverage: "
            f"harvested={harvested}, min={minimum_donor_count()}")
    seq = tuple(bank[rng.randrange(len(bank))])
    x = rng.randrange(17)
    pred, packets = run(a_episode(seq, x))
    if pred != A.apply_chain(list(seq), x):
        continue
    for k in (0, 1, 2, 3):
        v = a_running(seq, x, k)
        donors[k][v].append(packets[0][k].clone())
    harvested += 1
coverage_counts = {k: {v: len(donors[k][v]) for v in range(17)}
                   for k in donors}
assert minimum_donor_count() >= MIN_DONORS_PER_VALUE
print(f"[s3s5] min donor coverage: {minimum_donor_count()}",
      flush=True)


def d_context(op_span_at=None, op_span=None, x=0):
    """D value span + A structural forwarders; optionally one D
    operator phrase at cell position op_span_at (1-3)."""
    spans = [D.VALUE_SPAN.format(x=x)]
    for pos in (1, 2, 3):
        if pos == op_span_at:
            spans.append(op_span)
        else:
            spans.append(A.FORWARD_SPAN)
    return spans


def run_inject(spans, k, forced):
    pred, _ = run(spans, substitute={(k, k): forced.unsqueeze(0)})
    return pred


rep = {"ckpt": ARGS.ckpt, "tag": f"{ARM}_m{ck.get('model_seed')}_"
       f"o{ck.get('order_seed')}", "n_donors": ARGS.n_donors,
       "donor_coverage_counts": coverage_counts}
if ARM == "G":
    rep["note"] = ("G comparator: S5 is globally contextualized "
                   "primitive execution (all cells see the D "
                   "operator phrase), not a clean single-cell test")

# ---- S3: pure packet portability through D context -----------------
s3 = {"same": [0, 0], "cf": [0, 0]}
for k in (0, 1, 2):
    for x in range(17):
        for dv in donors[k][x][:MIN_DONORS_PER_VALUE]:
            pred = run_inject(d_context(x=x), k, dv)
            s3["same"][0] += int(pred == x); s3["same"][1] += 1
        v2 = (x + 8) % 17
        for dv in donors[k][v2][:MIN_DONORS_PER_VALUE]:
            pred = run_inject(d_context(x=x), k, dv)
            s3["cf"][0] += int(pred == v2); s3["cf"][1] += 1
assert s3["same"][1] > 0 and s3["cf"][1] > 0
rep["S3_same_value"] = round(s3["same"][0] / s3["same"][1], 4)
rep["S3_counterfactual"] = round(s3["cf"][0] / s3["cf"][1], 4)
print("[s3]", rep["S3_same_value"], rep["S3_counterfactual"],
      flush=True)

# ---- S4: mouth portability (inject at last interface, fwd cell 3) --
s4 = {"D": [0, 0], "A": [0, 0], "agreement": [0, 0]}
for v in range(17):
    for dv in donors[3][v][:MIN_DONORS_PER_VALUE]:
        pred_d = run_inject(d_context(x=v), 3, dv)
        s4["D"][0] += int(pred_d == v); s4["D"][1] += 1
        spans_a = [A.VALUE_SPAN.format(x=v), A.FORWARD_SPAN,
                   A.FORWARD_SPAN, A.FORWARD_SPAN]
        pred_a = run_inject(spans_a, 3, dv)
        s4["A"][0] += int(pred_a == v); s4["A"][1] += 1
        s4["agreement"][0] += int(pred_d == pred_a)
        s4["agreement"][1] += 1
assert s4["D"][1] > 0
rep["S4_D_context"] = round(s4["D"][0] / s4["D"][1], 4)
rep["S4_A_control"] = round(s4["A"][0] / s4["A"][1], 4)
rep["S4_D_A_agreement"] = round(
    s4["agreement"][0] / s4["agreement"][1], 4)
print("[s4]", rep["S4_D_context"], rep["S4_A_control"], flush=True)

# ---- S5: local D-primitive competence ------------------------------
s5_by_op = {}
tot = [0, 0]
for op_id in range(12):
    hits = n = 0
    for tpl in range(14):
        for pos in (1, 2, 3):
            for v in range(17):
                donor_pool = donors[pos - 1][v]
                assert donor_pool, (
                    f"missing donor: interface={pos - 1}, value={v}")
                dv = donor_pool[(op_id + tpl + pos + v)
                                % len(donor_pool)]
                spans = d_context(op_span_at=pos,
                                  op_span=D.render_op(op_id, tpl),
                                  x=v)
                pred = run_inject(spans, pos - 1, dv)
                want = D.apply_op(op_id, v)
                hits += int(pred == want); n += 1
    assert n == 14 * 3 * 17
    s5_by_op[str(D.D_OPS[op_id])] = round(hits / n, 4)
    tot[0] += hits; tot[1] += n
assert tot[1] == 12 * 14 * 3 * 17
rep["S5_overall"] = round(tot[0] / tot[1], 4)
rep["S5_by_op"] = s5_by_op
print("[s5]", rep["S5_overall"], flush=True)

Path("results/scout").mkdir(parents=True, exist_ok=True)
Path(f"results/scout/scout_s3s5_{rep['tag']}.json").write_text(
    json.dumps(rep, indent=1))
print("S3S5_DONE", rep["tag"], flush=True)

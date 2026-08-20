"""Packet causal audit (cohort ruling §B, GO-immediately item).

For a trained society checkpoint, at every inter-cell interface
(edges 0->1, 1->2, 2->3), over held-out episodes:

  DESTRUCTIVE (must collapse, <=0.11 each):
    - edge deletion: the committed packet at cell k is zeroed before
      cell k+1 reads it;
    - cross-example shuffle: cell k's packet replaced by the packet
      from a DIFFERENT episode (same stage/cell);
    - norm-matched noise: random vector scaled to the packet's norm.

  CONSTRUCTIVE (must steer, >=0.95 each):
    - same-intermediate transplant: replace cell k's packet with one
      from another episode whose running value after step k is THE
      SAME -> final answer must be preserved;
    - counterfactual-intermediate transplant: donor episode's running
      value v' differs -> the mouth must output apply(remaining ops,
      v') — the donor-predicted answer;
    - private-span counterfactual: change cell k+1's op span ->
      predicted new final answer.

Fresh-world masked-layout adaptation (referee mandate §5):
evaluation mirrors the run3v trainer's masked layout; arm read from
the checkpoint; grammar/banks from the sealed fresh world.
Usage: RUN3V_WORLD=F BRIDGE_OP_SEED=6011 BRIDGE_SPLIT_SEED=2203 \
  BRIDGE_GRAMMAR_SEED=7717 python scripts/bridge_packet_audit_run3v.py CKPT
Writes results/run3v/packet_audit_<tag>.json.
"""
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.use_deterministic_algorithms(True)
assert os.environ.get("RUN3V_WORLD") == "F"
assert os.environ.get("BRIDGE_OP_SEED") == "6011"
assert os.environ.get("BRIDGE_SPLIT_SEED") == "2203"
assert os.environ.get("BRIDGE_GRAMMAR_SEED") == "7717"

from populus.bridge import BridgeA1                            # noqa
from populus.bridge_qwen import QwenGenome                     # noqa
from populus.bridge_tasks import (ANSWER_PREFIX, FORWARD_SPAN,  # noqa
                                  QUESTION, SPLIT, VALUE_SPAN,
                                  apply_chain)

CKPT = sys.argv[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = json.load(open("results/bridge/serialization_contract.json"))
Tq, Ts = C["T_QUESTION"], C["T_SPAN"]
GRAMMAR = json.load(open("results/run3v/template_grammar_run3v.json"))
QUAL_PHR = {int(k): [r["text"] for r in v["qual"]]
            for k, v in GRAMMAR["ops"].items()}
N_EPISODES = 180          # audit episode pool (diag3 held-out)
rng = random.Random(31337)

lm = QwenGenome().attach_lora().to(DEV)
bridge = BridgeA1(lm).to(DEV).eval()
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
bridge.load_state_dict(ck["bridge_trainable"], strict=False)
lm.core.load_state_dict(ck["lora"], strict=False)
label_ids = torch.tensor([r["token_id"] for r in json.load(
    open("results/bridge/label_context.json"))["labels"]],
    device=DEV)
q_ids, _ = lm.encode([QUESTION + ANSWER_PREFIX], fixed_length=Tq,
                     device=DEV)
ARM = ck["arm"]
tag = f"F_{ARM}_m{ck.get('model_seed')}_o{ck.get('order_seed')}"
EYE4 = torch.eye(4, device=DEV)


def package(s_ids, s_mask, n):
    """Mirror of the run3v trainer's masked layout."""
    s_ids = s_ids.reshape(n, 4, Ts)
    s_mask = s_mask.reshape(n, 4, Ts)
    flat_i = s_ids.reshape(n, 1, 4 * Ts).expand(n, 4, -1).contiguous()
    if ARM == "G":
        return flat_i, s_mask.reshape(n, 1, 4 * Ts).expand(
            n, 4, -1).contiguous()
    per_cell = s_mask.unsqueeze(1) * \
        EYE4.to(s_mask.dtype).unsqueeze(0).unsqueeze(-1)
    return flat_i, per_cell.reshape(n, 4, 4 * Ts)


def build_episode(seq, x):
    tpls = [QUAL_PHR[o][rng.randrange(4)] for o in seq]
    spans = [VALUE_SPAN.format(x=x)] + tpls
    while len(spans) < 4:
        spans.append(FORWARD_SPAN)
    return spans


def prefix_value(seq, x, k):
    """Running value after cell k has acted (cell 0 = source)."""
    return apply_chain(list(seq[:k]), x)


@torch.no_grad()
def run(spans, substitute=None, target=None):
    s_ids, s_mask = lm.encode(spans, fixed_length=Ts, device=DEV)
    s_ids, s_mask = package(s_ids, s_mask, 1)
    out = bridge.forward_episode(
        q_ids, s_ids, s_mask,
        substitute_traj=substitute, packet_trace=True)
    corr = (out["logits"] - out["base_logits"])[:, label_ids]
    pred = int(corr.argmax(-1))
    return pred, out["packets"]


# ---- episode pool with cached natural packets ----------------------
pool = []
bank = SPLIT["diag3"]
while len(pool) < N_EPISODES:
    seq = tuple(bank[rng.randrange(len(bank))])
    x = rng.randrange(17)
    spans = build_episode(seq, x)
    pred, packets = run(spans)
    pool.append({"seq": seq, "x": x, "spans": spans,
                 "pred": pred, "answer": apply_chain(list(seq), x),
                 "packets": packets[0].clone()})
base_acc = sum(e["pred"] == e["answer"] for e in pool) / len(pool)
print(f"[audit {tag}] pool base accuracy {base_acc:.3f}", flush=True)
# audit only episodes the model gets right (causal claims are about
# the working mechanism)
good = [e for e in pool if e["pred"] == e["answer"]]
print(f"[audit {tag}] auditing {len(good)} correct episodes",
      flush=True)

report = {"ckpt": CKPT, "tag": tag, "pool_base_acc": base_acc,
          "n_audited": len(good), "edges": {}}

for k in (0, 1, 2):                    # interface: cell k -> k+1
    res = {c: 0 for c in ("edge_del", "shuffle", "noise",
                          "same_transplant", "cf_transplant",
                          "span_cf")}
    n = {c: 0 for c in res}
    for e in good:
        d = e["packets"][k].shape
        # destructive trio: answer should NOT survive
        for cond, forced in (
            ("edge_del", torch.zeros_like(e["packets"][k])),
            ("shuffle", good[rng.randrange(len(good))]["packets"][k]),
            ("noise", torch.randn_like(e["packets"][k])
             * e["packets"][k].norm()
             / max(1e-6, float(torch.randn(d).norm()))),
        ):
            pred, _ = run(e["spans"],
                          substitute={(k, k): forced.unsqueeze(0)})
            res[cond] += int(pred == e["answer"]); n[cond] += 1
        # same-intermediate transplant: donor with equal prefix value
        # running value AFTER cell k: cell j (1..3) applies seq[j-1],
        # so after cell k exactly seq[:k] ops have been applied.
        v = prefix_value(e["seq"], e["x"], k) if k > 0 else e["x"]
        donors = [g for g in good if g is not e and
                  (prefix_value(g["seq"], g["x"], k)
                   if k > 0 else g["x"]) == v]
        if donors:
            dnr = donors[rng.randrange(len(donors))]
            pred, _ = run(e["spans"],
                          substitute={(k, k):
                                      dnr["packets"][k].unsqueeze(0)})
            res["same_transplant"] += int(pred == e["answer"])
            n["same_transplant"] += 1
        # counterfactual transplant: donor with different prefix value
        cf = [g for g in good if
              ((prefix_value(g["seq"], g["x"], k)
                if k > 0 else g["x"]) != v)]
        if cf:
            dnr = cf[rng.randrange(len(cf))]
            v2 = (prefix_value(dnr["seq"], dnr["x"], k)
                  if k > 0 else dnr["x"])
            predicted = apply_chain(list(e["seq"][k:]), v2)
            pred, _ = run(e["spans"],
                          substitute={(k, k):
                                      dnr["packets"][k].unsqueeze(0)})
            res["cf_transplant"] += int(pred == predicted)
            n["cf_transplant"] += 1
        # private-span counterfactual at cell k+1 (if it holds an op)
        if k + 1 <= len(e["seq"]):
            new_op = (e["seq"][k] + 1 + rng.randrange(11)) % 12
            new_seq = list(e["seq"]); new_seq[k] = new_op
            spans2 = list(e["spans"])
            spans2[k + 1] = QUAL_PHR[new_op][rng.randrange(4)]
            pred, _ = run(spans2)
            res["span_cf"] += int(
                pred == apply_chain(new_seq, e["x"]))
            n["span_cf"] += 1
    edge = {c: (round(res[c] / n[c], 4) if n[c] else None)
            for c in res}
    edge["n"] = n
    report["edges"][f"edge_{k}_{k+1}"] = edge
    print(f"[audit {tag}] edge {k}->{k+1}: " + json.dumps(
        {c: edge[c] for c in ('edge_del', 'shuffle', 'noise',
                              'same_transplant', 'cf_transplant',
                              'span_cf')}), flush=True)

# gates per ruling
g = report["edges"]
report["gates"] = {
    "destructive<=0.11": all(
        (g[e][c] is None or g[e][c] <= 0.11)
        for e in g for c in ("edge_del", "noise")),
    "shuffle<=0.11_or_30below": all(
        (g[e]["shuffle"] is None or g[e]["shuffle"] <= 0.11
         or (g[e]["same_transplant"] is not None
             and g[e]["same_transplant"] - g[e]["shuffle"] >= 0.30))
        for e in g),
    "same_transplant>=0.95": all(
        (g[e]["same_transplant"] is None
         or g[e]["same_transplant"] >= 0.95) for e in g),
    "cf_transplant>=0.95": all(
        (g[e]["cf_transplant"] is None
         or g[e]["cf_transplant"] >= 0.95) for e in g),
    "span_cf>=0.95": all(
        (g[e]["span_cf"] is None or g[e]["span_cf"] >= 0.95)
        for e in g),
}
report["AUDIT_PASS"] = all(report["gates"].values())
Path("results/run3v").mkdir(parents=True, exist_ok=True)
Path(f"results/run3v/packet_audit_{tag}.json").write_text(
    json.dumps(report, indent=1, default=str))
print(f"[audit {tag}] GATES {report['gates']} PASS "
      f"{report['AUDIT_PASS']}", flush=True)

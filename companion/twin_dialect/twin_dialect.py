"""Twin-dialect probe (2026-08-25, exploratory, non-gating).

Question: do a tag-rescued global society (GT) and its same-birth
masked sibling (P) share a packet dialect? Loads DONOR checkpoint,
harvests natural packets on correct diag3 episodes (P1 test values,
diag phrasing rotation 1), frees it, loads RECIPIENT, injects raw
same-value packets (and a 3-shift shuffle control) at each interface.
Arms P/PT/GT supported on either side. Directional only.
"""
import argparse, gc, json, os, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

P_ARGS = argparse.ArgumentParser()
P_ARGS.add_argument("--donor", required=True)
P_ARGS.add_argument("--recipient", required=True)
A_RGS = P_ARGS.parse_args()

assert os.environ.get("RUN3V_WORLD") == "F"
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from populus import bridge_tasks as A
from populus.bridge import BridgeA1
from populus.bridge_qwen import QwenGenome

DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = json.load(open("results/bridge/serialization_contract.json"))
Tq, Ts = C["T_QUESTION"], C["T_SPAN"]
TEST_VALUES = [0, 5, 7, 9, 10, 13, 14, 16]
SHIFT = {v: TEST_VALUES[(i + 3) % len(TEST_VALUES)]
         for i, v in enumerate(TEST_VALUES)}
MIN_PER = 4
PROGS = [tuple(s) for s in A.SPLIT["diag3"]]


def a_running(seq, x, k):
    return x if k == 0 else A.apply_chain(list(seq[:k]), x)


def build_stack(ckpt_path):
    lm = QwenGenome().attach_lora().to(DEV)
    bridge = BridgeA1(lm).to(DEV).eval()
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    bridge.load_state_dict(ck["bridge_trainable"], strict=False)
    lm.core.load_state_dict(ck["lora"], strict=False)
    arm = ck["arm"]
    assert arm in ("P", "PT", "GT"), arm
    ids = None
    if arm in ("PT", "GT"):
        im = lm.tok(" mine", add_special_tokens=False)["input_ids"]
        io = lm.tok(" other", add_special_tokens=False)["input_ids"]
        assert len(im) == 1 and len(io) == 1
        ids = (im[0], io[0])
    label_ids = torch.tensor([r["token_id"] for r in json.load(
        open("results/bridge/label_context.json"))["labels"]],
        device=DEV)
    q_ids, _ = lm.encode([A.QUESTION + A.ANSWER_PREFIX],
                         fixed_length=Tq, device=DEV)
    tag = f"{arm}_m{ck.get('model_seed')}_o{ck.get('order_seed')}"
    return lm, bridge, arm, ids, label_ids, q_ids, tag


def package(lm, arm, ids, s_ids, s_mask, n):
    s_ids = s_ids.reshape(n, 4, Ts)
    s_mask = s_mask.reshape(n, 4, Ts)
    if arm in ("PT", "GT"):
        MINE_ID, OTHER_ID = ids
        ci_all, cm_all = [], []
        for c in range(4):
            si, sm = [], []
            for j in range(4):
                mk = MINE_ID if j == c else OTHER_ID
                mkcol = torch.full((n, 1), mk, device=DEV,
                                   dtype=s_ids.dtype)
                si.append(torch.cat([mkcol, s_ids[:, j]], dim=1))
                mono = torch.ones(n, 1, device=DEV,
                                  dtype=s_mask.dtype)
                sm.append(torch.cat([mono, s_mask[:, j]], dim=1))
            ci = torch.cat(si, dim=1); cm = torch.cat(sm, dim=1)
            if arm == "PT":
                keep = torch.zeros_like(cm); w = Ts + 1
                keep[:, c * w:(c + 1) * w] = 1
                cm = cm * keep
            ci_all.append(ci.unsqueeze(1)); cm_all.append(cm.unsqueeze(1))
        return torch.cat(ci_all, 1), torch.cat(cm_all, 1)
    flat_i = s_ids.reshape(n, 1, 4 * Ts).expand(n, 4, -1).contiguous()
    eye = torch.eye(4, device=DEV, dtype=s_mask.dtype)
    per_cell = s_mask.unsqueeze(1) * eye.unsqueeze(0).unsqueeze(-1)
    return flat_i, per_cell.reshape(n, 4, 4 * Ts)


def spans_for(seq, x):
    tpl = [A.TPL_DIAG[1]] * len(seq)
    spans = [A.VALUE_SPAN.format(x=x)]
    spans += [A.render_op(o, t) for o, t in zip(seq, tpl)]
    while len(spans) < 4:
        spans.append(A.FORWARD_SPAN)
    return spans


@torch.no_grad()
def run_one(lm, bridge, arm, ids, q_ids, label_ids, spans, sub=None):
    s_ids, s_mask = lm.encode(spans, fixed_length=Ts, device=DEV)
    s_ids, s_mask = package(lm, arm, ids, s_ids, s_mask, 1)
    out = bridge.forward_episode(q_ids, s_ids, s_mask,
                                 substitute_traj=sub,
                                 packet_trace=True)
    corr = (out["logits"] - out["base_logits"])[:, label_ids]
    return int(corr.argmax(-1)), out["packets"]


# ---------- phase 1: harvest donor ----------
lm, bridge, d_arm, d_ids, label_ids, q_ids, d_tag = \
    build_stack(A_RGS.donor)
print(f"donor {d_tag}", flush=True)
rng = random.Random(8300)
bank = {k: {v: [] for v in TEST_VALUES} for k in (0, 1, 2)}
attempts = 0
while min(len(bank[k][v]) for k in bank for v in bank[k]) < MIN_PER:
    attempts += 1
    assert attempts < 30000, "coverage failure"
    seq = PROGS[rng.randrange(len(PROGS))]
    x = rng.randrange(17)
    pred, packets = run_one(lm, bridge, d_arm, d_ids, q_ids,
                            label_ids, spans_for(seq, x))
    if pred != A.apply_chain(list(seq), x):
        continue
    for k in (0, 1, 2):
        v = a_running(seq, x, k)
        if v in bank[k] and len(bank[k][v]) < MIN_PER + 2:
            bank[k][v].append(packets[0][k].detach().cpu().clone())
print(f"harvest done after {attempts} attempts", flush=True)
del lm, bridge
gc.collect(); torch.cuda.empty_cache()

# ---------- phase 2: inject into recipient ----------
lm, bridge, r_arm, r_ids, label_ids, q_ids, r_tag = \
    build_stack(A_RGS.recipient)
print(f"recipient {r_tag}", flush=True)
rng2 = random.Random(8400)
res = {k: {"same": [0, 0], "shuf": [0, 0]} for k in (0, 1, 2)}
per_iface_needed = 120
for k in (0, 1, 2):
    done = 0
    guard = 0
    while done < per_iface_needed:
        guard += 1
        assert guard < 20000
        seq = PROGS[rng2.randrange(len(PROGS))]
        x = rng2.randrange(17)
        v = a_running(seq, x, k)
        if v not in bank[k]:
            continue
        spans = spans_for(seq, x)
        base_pred, _ = run_one(lm, bridge, r_arm, r_ids, q_ids,
                               label_ids, spans)
        ans = A.apply_chain(list(seq), x)
        if base_pred != ans:
            continue
        donor_pk = bank[k][v][rng2.randrange(len(bank[k][v]))]
        sub = {(k, k): donor_pk.unsqueeze(0).to(DEV)}
        pred, _ = run_one(lm, bridge, r_arm, r_ids, q_ids,
                          label_ids, spans, sub=sub)
        res[k]["same"][0] += int(pred == ans); res[k]["same"][1] += 1
        sv = SHIFT[v]
        donor_shuf = bank[k][sv][rng2.randrange(len(bank[k][sv]))]
        sub = {(k, k): donor_shuf.unsqueeze(0).to(DEV)}
        pred, _ = run_one(lm, bridge, r_arm, r_ids, q_ids,
                          label_ids, spans, sub=sub)
        res[k]["shuf"][0] += int(pred == ans); res[k]["shuf"][1] += 1
        done += 1
    print(f"iface {k}: same {res[k]['same'][0]}/{res[k]['same'][1]}"
          f" shuf {res[k]['shuf'][0]}/{res[k]['shuf'][1]}", flush=True)

summary = {f"iface{k}": {
    "same": round(res[k]["same"][0] / max(res[k]["same"][1], 1), 4),
    "shuffle": round(res[k]["shuf"][0] / max(res[k]["shuf"][1], 1), 4)}
    for k in (0, 1, 2)}
out = {"donor": d_tag, "recipient": r_tag, "n_per_iface":
       per_iface_needed, "values": TEST_VALUES, "results": summary}
Path("results/run3v").mkdir(parents=True, exist_ok=True)
p = f"results/run3v/twin_dialect_{d_tag}_to_{r_tag}.json"
Path(p).write_text(json.dumps(out, indent=1))
print("TWIN_DIALECT_DONE", p, json.dumps(summary), flush=True)

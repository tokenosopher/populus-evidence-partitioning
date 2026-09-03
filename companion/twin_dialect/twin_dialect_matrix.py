"""Twin-dialect 2x2 matrix battery (2026-08-25, referee-prescribed, exploratory).

Full self-vs-cross transplant matrix on the m200/o950 twin pair:
donors {P, GT} x recipients {P, GT}, interfaces 0-2, all 17 values,
both diag phrasing rotations. Per (recipient, iface, episode), all
conditions are scored on the SAME base-correct recipient episode:
  same          - donor packet with the episode's running value v
  cf            - donor packet with v' = (v+3) mod 17; scored against
                  the suffix-implied counterfactual target (cf_follow)
                  AND the original answer (shift_retain)
  deranged      - donor packet with random value w != v (retention)
  deletion      - zero packet (donor-independent)
Per-case predictions are persisted (referee mandate after the v1 probe
saved only aggregates). Same harvest/eval seeds and math as
twin_dialect.py; only the battery structure is new.
"""
import argparse, gc, json, os, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

P_ARGS = argparse.ArgumentParser()
P_ARGS.add_argument("--ckpt-p", required=True)
P_ARGS.add_argument("--ckpt-gt", required=True)
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
TEST_VALUES = list(range(17))
CF_OFFSET = 3
MIN_PER = 4
N_PER_IFACE = 120
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


def spans_for(seq, x, phr):
    tpl = [A.TPL_DIAG[phr]] * len(seq)
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


def harvest(ckpt_path):
    lm, bridge, arm, ids, label_ids, q_ids, tag = build_stack(ckpt_path)
    print(f"harvest donor {tag}", flush=True)
    rng = random.Random(8300)
    bank = {k: {v: [] for v in TEST_VALUES} for k in (0, 1, 2)}
    attempts = 0
    while min(len(bank[k][v]) for k in bank for v in bank[k]) < MIN_PER:
        attempts += 1
        assert attempts < 120000, "coverage failure"
        seq = PROGS[rng.randrange(len(PROGS))]
        x = rng.randrange(17)
        phr = attempts % 2
        pred, packets = run_one(lm, bridge, arm, ids, q_ids,
                                label_ids, spans_for(seq, x, phr))
        if pred != A.apply_chain(list(seq), x):
            continue
        for k in (0, 1, 2):
            v = a_running(seq, x, k)
            if len(bank[k][v]) < MIN_PER + 2:
                bank[k][v].append((packets[0][k].detach().cpu().clone(),
                                   {"seq": list(seq), "x": x, "phr": phr}))
    print(f"harvest {tag} done after {attempts} attempts", flush=True)
    del lm, bridge
    gc.collect(); torch.cuda.empty_cache()
    return bank, tag


def eval_recipient(ckpt_path, banks):
    lm, bridge, arm, ids, label_ids, q_ids, tag = build_stack(ckpt_path)
    print(f"recipient {tag}", flush=True)
    rng2 = random.Random(8400)
    conds = [f"{d}_{c}" for d in banks
             for c in ("same", "cf_follow", "shift_retain", "deranged")]
    conds.append("deletion")
    res = {k: {c: [0, 0] for c in conds} for k in (0, 1, 2)}
    records = []
    for k in (0, 1, 2):
        done = 0
        guard = 0
        while done < N_PER_IFACE:
            guard += 1
            assert guard < 40000
            seq = PROGS[rng2.randrange(len(PROGS))]
            x = rng2.randrange(17)
            phr = done % 2
            v = a_running(seq, x, k)
            spans = spans_for(seq, x, phr)
            base_pred, _ = run_one(lm, bridge, arm, ids, q_ids,
                                   label_ids, spans)
            ans = A.apply_chain(list(seq), x)
            if base_pred != ans:
                continue
            vcf = (v + CF_OFFSET) % 17
            target_cf = A.apply_chain(list(seq[k:]), vcf)
            wder = rng2.choice([w for w in TEST_VALUES if w != v])
            rec = {"iface": k, "seq": list(seq), "x": x, "phr": phr,
                   "v": v, "vcf": vcf, "wder": wder, "ans": ans,
                   "target_cf": target_cf, "preds": {}}
            zero_pk = None
            for d, bank in banks.items():
                pk, dmeta = bank[k][v][rng2.randrange(len(bank[k][v]))]
                if zero_pk is None:
                    zero_pk = torch.zeros_like(pk)
                rec["preds"][f"{d}_same_donor"] = dmeta
                rec["preds"][f"{d}_same_exact_overlap"] = int(
                    dmeta["seq"] == list(seq) and dmeta["x"] == x
                    and dmeta["phr"] == phr)
                rec["preds"][f"{d}_same_prog_overlap"] = int(
                    dmeta["seq"] == list(seq))
                sub = {(k, k): pk.unsqueeze(0).to(DEV)}
                pred, _ = run_one(lm, bridge, arm, ids, q_ids,
                                  label_ids, spans, sub=sub)
                res[k][f"{d}_same"][0] += int(pred == ans)
                res[k][f"{d}_same"][1] += 1
                rec["preds"][f"{d}_same"] = pred
                pkc, dmetac = bank[k][vcf][rng2.randrange(len(bank[k][vcf]))]
                rec["preds"][f"{d}_cf_donor"] = dmetac
                sub = {(k, k): pkc.unsqueeze(0).to(DEV)}
                pred, _ = run_one(lm, bridge, arm, ids, q_ids,
                                  label_ids, spans, sub=sub)
                res[k][f"{d}_cf_follow"][0] += int(pred == target_cf)
                res[k][f"{d}_cf_follow"][1] += 1
                res[k][f"{d}_shift_retain"][0] += int(pred == ans)
                res[k][f"{d}_shift_retain"][1] += 1
                rec["preds"][f"{d}_cf"] = pred
                pkd, dmetad = bank[k][wder][rng2.randrange(len(bank[k][wder]))]
                sub = {(k, k): pkd.unsqueeze(0).to(DEV)}
                pred, _ = run_one(lm, bridge, arm, ids, q_ids,
                                  label_ids, spans, sub=sub)
                res[k][f"{d}_deranged"][0] += int(pred == ans)
                res[k][f"{d}_deranged"][1] += 1
                rec["preds"][f"{d}_deranged"] = pred
            sub = {(k, k): zero_pk.unsqueeze(0).to(DEV)}
            pred, _ = run_one(lm, bridge, arm, ids, q_ids,
                              label_ids, spans, sub=sub)
            res[k]["deletion"][0] += int(pred == ans)
            res[k]["deletion"][1] += 1
            rec["preds"]["deletion"] = pred
            records.append(rec)
            done += 1
        line = " ".join(f"{c}={res[k][c][0]}/{res[k][c][1]}"
                        for c in conds)
        print(f"iface {k}: {line}", flush=True)
    summary = {f"iface{k}": {c: round(res[k][c][0] / max(res[k][c][1], 1), 4)
                             for c in conds} for k in (0, 1, 2)}
    del lm, bridge
    gc.collect(); torch.cuda.empty_cache()
    return summary, records, tag


bank_p, tag_p = harvest(A_RGS.ckpt_p)
bank_gt, tag_gt = harvest(A_RGS.ckpt_gt)
assert tag_p.startswith("P_") and tag_gt.startswith("GT_")
banks = {"P": bank_p, "GT": bank_gt}

Path("results/run3v").mkdir(parents=True, exist_ok=True)
out = {"cf_offset": CF_OFFSET, "n_per_iface": N_PER_IFACE,
       "values": TEST_VALUES, "phrasings": [0, 1],
       "donors": {"P": tag_p, "GT": tag_gt}, "recipients": {}}
for name, ck in (("P", A_RGS.ckpt_p), ("GT", A_RGS.ckpt_gt)):
    summary, records, tag = eval_recipient(ck, banks)
    out["recipients"][name] = {"tag": tag, "summary": summary,
                               "records": records}
    print(f"MATRIX_RECIPIENT_DONE {name}",
          json.dumps({k: {c: s for c, s in v.items()}
                      for k, v in summary.items()}), flush=True)

audit = {}
for name, rdata in out["recipients"].items():
    a = {}
    for d in ("P", "GT"):
        recs = rdata["records"]
        for k in (0, 1, 2):
            rk = [r for r in recs if r["iface"] == k]
            n_ex = sum(r["preds"][f"{d}_same_exact_overlap"] for r in rk)
            n_pr = sum(r["preds"][f"{d}_same_prog_overlap"] for r in rk)
            keep = [r for r in rk
                    if not r["preds"][f"{d}_same_exact_overlap"]]
            acc_ex = (sum(r["preds"][f"{d}_same"] == r["ans"] for r in keep)
                      / max(len(keep), 1))
            a[f"{d}_iface{k}"] = {"n_exact_overlap": n_ex,
                                  "n_same_program": n_pr,
                                  "same_value_excl_exact":
                                      round(acc_ex, 4),
                                  "n_excl": len(keep)}
    audit[name] = a
out["overlap_audit"] = audit
p = f"results/run3v/twin_dialect_matrix_{tag_p}_x_{tag_gt}.json"
Path(p).write_text(json.dumps(out, indent=1))
print("OVERLAP_AUDIT", json.dumps(audit), flush=True)
print("TWIN_DIALECT_MATRIX_DONE", p, flush=True)

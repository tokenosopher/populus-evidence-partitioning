"""Run-3V paired evaluator (cohort ruling §B + Run-3V-R ruling gates,
applied to the author-elected fresh world).

Run mode (per trajectory, FINAL checkpoint only):
  RUN3V_WORLD=F BRIDGE_OP_SEED=6011 BRIDGE_SPLIT_SEED=2203 \
  BRIDGE_GRAMMAR_SEED=7717 python scripts/bridge_run3v_evaluator.py \
      --ckpt results/run3v_F_P_m200_o900_ckpt_final.pt

Paired mode (after all twenty evaluations):
  python scripts/bridge_run3v_evaluator.py --paired results/run3v/

Evaluation MIRRORS the trainer's masked layout exactly: every cell
receives the identical fixed four-slot concatenation; arm P exposes
only the cell's own slot, arm G exposes all four. Metrics per run:
l0, l1_qual, diag2, diag3, allcut2/3 (cut_mail), packet_shuffle3
(committed packets rolled across the batch at all three edges),
best_single_cell3 (under all-cut, mouth reading each cell's own
committed state; max over cells), train2/train3_exact (train3
subsampled to 150 seqs, logged), packets_finite, beta check.

Paired gates (pair i = same model/order seeds, d = P - G on diag):
  (1) d2>=0.20 AND d3>=0.20 in >=8/10 pairs
  (2) median d2 >= 0.25   (3) median d3 >= 0.25
  (4) P median >= 0.70 at both depths
  (5) P allcut <= 0.11 at both depths in >=8/10 P runs
  (6) no P run < 0.40 (either depth)
Four-row interpretation emitted alongside, never instead of, rows.
"""
import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

P_ARGS = argparse.ArgumentParser()
P_ARGS.add_argument("--ckpt")
P_ARGS.add_argument("--paired")
A = P_ARGS.parse_args()


def paired(results_dir):
    def load(arm):
        rows = {}
        for f in glob.glob(os.path.join(
                results_dir, f"run3v_eval_F_{arm}_m*_o*.json")):
            r = json.load(open(f))
            rows[(r["model_seed"], r["order_seed"])] = r
        return rows
    P, G = load("P"), load("G")
    keys = sorted(set(P) & set(G))
    pairs = []
    for k in keys:
        p, g = P[k], G[k]
        pairs.append({
            "model_seed": k[0], "order_seed": k[1],
            "P_diag2": p["diag2"], "P_diag3": p["diag3"],
            "G_diag2": g["diag2"], "G_diag3": g["diag3"],
            "d2": round(p["diag2"] - g["diag2"], 4),
            "d3": round(p["diag3"] - g["diag3"], 4),
            "P_allcut2": p["allcut_diag2"], "P_allcut3": p["allcut_diag3"],
            "G_allcut2": g["allcut_diag2"], "G_allcut3": g["allcut_diag3"],
            "G_best_single_cell3": g.get("best_single_cell3"),
            "G_shuffle3": g.get("packet_shuffle3"),
        })
    n = len(pairs)
    d2s = [p["d2"] for p in pairs]
    d3s = [p["d3"] for p in pairs]
    both_ge_020 = sum(p["d2"] >= 0.20 and p["d3"] >= 0.20 for p in pairs)
    p_allcut_ok = sum(p["P_allcut2"] <= 0.11 and p["P_allcut3"] <= 0.11
                      for p in pairs)
    verdict = {
        "n_pairs": n,
        "pairs": pairs,
        "median_d2": round(statistics.median(d2s), 4) if pairs else None,
        "median_d3": round(statistics.median(d3s), 4) if pairs else None,
        "median_P_diag2": round(statistics.median(
            [p["P_diag2"] for p in pairs]), 4) if pairs else None,
        "median_P_diag3": round(statistics.median(
            [p["P_diag3"] for p in pairs]), 4) if pairs else None,
        "median_G_diag2": round(statistics.median(
            [p["G_diag2"] for p in pairs]), 4) if pairs else None,
        "median_G_diag3": round(statistics.median(
            [p["G_diag3"] for p in pairs]), 4) if pairs else None,
    }
    verdict["gates"] = {
        "pairs==10": n == 10,
        "d2&d3>=0.20_in>=8": both_ge_020 >= 8,
        "median_d2>=0.25": (verdict["median_d2"] or -1) >= 0.25,
        "median_d3>=0.25": (verdict["median_d3"] or -1) >= 0.25,
        "P_median>=0.70_both": (verdict["median_P_diag2"] or -1) >= 0.70
                               and (verdict["median_P_diag3"] or -1) >= 0.70,
        "P_allcut<=0.11_in>=8": p_allcut_ok >= 8,
        "no_P_run<0.40": all(p["P_diag2"] >= 0.40 and p["P_diag3"] >= 0.40
                             for p in pairs),
    }
    verdict["VISIBILITY_EFFECT_CONFIRMED"] = all(
        verdict["gates"].values())
    g_solves = [p for p in pairs if min(p["G_diag2"], p["G_diag3"]) >= 0.40]
    g_bypass = [p for p in pairs
                if max(p["G_allcut2"], p["G_allcut3"]) > 0.11]
    verdict["interpretation_hints"] = {
        "row": ("P>>G structural-privacy-regularizer"
                if verdict["VISIBILITY_EFFECT_CONFIRMED"]
                else "see_pairs_and_referee"),
        "n_G_solving_pairs": len(g_solves),
        "n_G_allcut_bypass": len(g_bypass),
        "note": "four-row mapping is the referee's to apply; hints only",
    }
    out = os.path.join(results_dir, "run3v_verdict.json")
    Path(out).write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items() if k != "pairs"},
                     indent=1))
    return verdict


if A.paired:
    paired(A.paired)
    sys.exit(0)

assert os.environ.get("RUN3V_WORLD") == "F"
assert os.environ.get("BRIDGE_OP_SEED") == "6011"
assert os.environ.get("BRIDGE_SPLIT_SEED") == "2203"
assert os.environ.get("BRIDGE_GRAMMAR_SEED") == "7717"
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from populus.bridge import M_P, BridgeA1                       # noqa
from populus.bridge_qwen import QwenGenome                     # noqa
from populus.bridge_tasks import (ANSWER_PREFIX, FORWARD_SPAN,  # noqa
                                  QUESTION, SPLIT, VALUE_SPAN,
                                  apply_chain, render_op)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = json.load(open("results/bridge/serialization_contract.json"))
Tq, Ts = C["T_QUESTION"], C["T_SPAN"]
GRAMMAR = json.load(open("results/run3v/template_grammar_run3v.json"))
QUAL_PHR = {int(k): [r["text"] for r in v["qual"]]
            for k, v in GRAMMAR["ops"].items()}

lm = QwenGenome().attach_lora().to(DEV)
bridge = BridgeA1(lm).to(DEV).eval()
ck = torch.load(A.ckpt, map_location=DEV, weights_only=False)
bridge.load_state_dict(ck["bridge_trainable"], strict=False)
lm.core.load_state_dict(ck["lora"], strict=False)
ARM = ck["arm"]
assert ARM in ("P", "G")
assert abs(float(bridge.beta) - 0.003) < 1e-9
label_ids = torch.tensor([r["token_id"] for r in json.load(
    open("results/bridge/label_context.json"))["labels"]],
    device=DEV)
q_ids, _ = lm.encode([QUESTION + ANSWER_PREFIX], fixed_length=Tq,
                     device=DEV)
packet_finite = [True]
EYE4 = torch.eye(4, device=DEV)


def package(s_ids, s_mask, n):
    """Byte-for-byte mirror of the trainer's masked layout."""
    s_ids = s_ids.reshape(n, 4, Ts)
    s_mask = s_mask.reshape(n, 4, Ts)
    flat_i = s_ids.reshape(n, 1, 4 * Ts).expand(n, 4, -1).contiguous()
    if ARM == "G":
        flat_m = s_mask.reshape(n, 1, 4 * Ts).expand(n, 4, -1) \
                       .contiguous()
        return flat_i, flat_m
    per_cell = s_mask.unsqueeze(1) * \
        EYE4.to(s_mask.dtype).unsqueeze(0).unsqueeze(-1)
    return flat_i, per_cell.reshape(n, 4, 4 * Ts)


@torch.no_grad()
def eval_cases(cases, cut=False, batch=64, shuffle=False,
               single_cell=None):
    """shuffle: roll committed packets across the batch at all three
    edges (destructive coherence test). single_cell=k: under all-cut,
    read the mouth from cell k's committed state instead of C3."""
    hits = 0
    for i in range(0, len(cases), batch):
        chunk = cases[i:i + batch]
        texts = [s for sp, _ in chunk for s in sp]
        s_ids, s_mask = lm.encode(texts, fixed_length=Ts, device=DEV)
        B = len(chunk)
        s_ids, s_mask = package(s_ids, s_mask, B)
        q = q_ids.expand(B, -1)
        sub = None
        if shuffle:
            base = bridge.forward_episode(q, s_ids, s_mask,
                                          packet_trace=True)
            pk = base["packets"]              # (B, cells, M_P, d)
            sub = {(k, k): pk[:, k].roll(1, dims=0) for k in (0, 1, 2)}
        if single_cell is None:
            out = bridge.forward_episode(q, s_ids, s_mask,
                                         cut_mail=cut,
                                         substitute_traj=sub)
            if not torch.isfinite(out["packets"]).all():
                packet_finite[0] = False
            corr = (out["logits"] - out["base_logits"])[:, label_ids]
        else:
            out = bridge.forward_episode(q, s_ids, s_mask,
                                         cut_mail=True,
                                         packet_trace=True)
            h_base = lm.forward_embeds(
                lm.embed(q), torch.ones(B, q.shape[1], device=DEV),
                adapter_mode="base")[:, -1]
            pk = out["packets"][:, single_cell].reshape(B, -1)
            logits = lm.lm_head(
                h_base + bridge.beta * bridge.W_mouth(pk))
            corr = (logits - out["base_logits"])[:, label_ids]
        tgt = torch.tensor([a for _, a in chunk], device=DEV)
        hits += int((corr.argmax(-1) == tgt).sum())
    return round(hits / len(cases), 4)


def l0_cases():
    return [([VALUE_SPAN.format(x=x)] + [FORWARD_SPAN] * 3, x)
            for x in range(17)]


def l1_cases():
    cases = []
    for op in range(12):
        for x in range(17):
            for pi, phr in enumerate(QUAL_PHR[op]):
                pos = (pi + x) % 3 + 1
                spans = [VALUE_SPAN.format(x=x)] + [FORWARD_SPAN] * 3
                spans[pos] = phr
                cases.append((spans, apply_chain([op], x)))
    return cases


def diag_cases(name):
    cases = []
    for si, seq in enumerate(SPLIT[name]):
        for x in range(17):
            for j in range(4):
                spans = [VALUE_SPAN.format(x=x)]
                spans += [QUAL_PHR[o][(j + i) % 4]
                          for i, o in enumerate(seq)]
                while len(spans) < 4:
                    spans.append(FORWARD_SPAN)
                cases.append((spans, apply_chain(list(seq), x)))
    return cases


def train_cases(name, limit=None):
    seqs = SPLIT[name][:limit] if limit else SPLIT[name]
    cases = []
    for si, seq in enumerate(seqs):
        for x in range(17):
            spans = [VALUE_SPAN.format(x=x)]
            spans += [render_op(o, (si + x + i) % 8)
                      for i, o in enumerate(seq)]
            while len(spans) < 4:
                spans.append(FORWARD_SPAN)
            cases.append((spans, apply_chain(list(seq), x)))
    return cases


rep = {"ckpt": A.ckpt, "arm": ARM,
       "model_seed": ck.get("model_seed"),
       "order_seed": ck.get("order_seed"),
       "batch_stream_hash": ck.get("batch_stream_hash"),
       "tokens_per_cell_call": 4 * Ts,
       "cell_calls_per_episode": 4}
rep["l0"] = eval_cases(l0_cases())
print("l0", rep["l0"], flush=True)
rep["l1_qual"] = eval_cases(l1_cases())
print("l1_qual", rep["l1_qual"], flush=True)
rep["diag2"] = eval_cases(diag_cases("diag2"))
print("diag2", rep["diag2"], flush=True)
d3 = diag_cases("diag3")
rep["diag3"] = eval_cases(d3)
print("diag3", rep["diag3"], flush=True)
rep["allcut_diag2"] = eval_cases(diag_cases("diag2"), cut=True)
rep["allcut_diag3"] = eval_cases(d3, cut=True)
print("allcut", rep["allcut_diag2"], rep["allcut_diag3"], flush=True)
rep["packet_shuffle3"] = eval_cases(d3, shuffle=True)
print("shuffle3", rep["packet_shuffle3"], flush=True)
sc = {k: eval_cases(d3, single_cell=k) for k in range(4)}
rep["single_cell3"] = sc
rep["best_single_cell3"] = max(sc.values())
print("best_single_cell3", rep["best_single_cell3"], sc, flush=True)
rep["train2_exact"] = eval_cases(train_cases("train2"))
rep["train3_exact_sub150"] = eval_cases(train_cases("train3", 150))
print("train", rep["train2_exact"], rep["train3_exact_sub150"],
      flush=True)
rep["packets_finite"] = packet_finite[0]
rep["beta_buffer"] = float(bridge.beta)

tag = f"F_{ARM}_m{rep['model_seed']}_o{rep['order_seed']}"
Path("results/run3v").mkdir(parents=True, exist_ok=True)
Path(f"results/run3v/run3v_eval_{tag}.json").write_text(
    json.dumps(rep, indent=1))
print("EVAL_DONE", tag, flush=True)

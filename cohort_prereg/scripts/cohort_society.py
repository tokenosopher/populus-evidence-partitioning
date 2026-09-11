"""Role-marked cohort five-condition society trainer (PREREG v0.3 §4).

Derived from scripts/bridge_scout_tag.py (the reviewed pilot trainer) with
the pilot's single-token markers replaced by the frozen 2-token header
blocks and the five preregistered conditions. THIN-WRAPPER DISCIPLINE:
every layout/mask decision lives in torch_package() below, which is
machine-checked at every startup (and by --parity-only) against the pure
torch-free reference cohort/condition_layout.build_cell_input — bit-exact,
all conditions, all cells. Episode sampling, optimizer protocol, LR/aux
schedules, batch 32 as fixed 2x16 gradient accumulation, determinism
flags, and bit-exact resume are carried forward from the pilot unchanged.

Streams hashed per PREREG: the COMMON SEMANTIC stream (episode metadata,
identical across the five conditions at fixed order seed) and the
CONDITION-SPECIFIC SERIALIZED stream (packaged ids+mask bytes).
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from cohort.condition_layout import (CONDITIONS, MARKER_ARMS,      # noqa
                                     build_cell_input)
from cohort.filler_constructor import build_length_banks           # noqa
from cohort.torch_package import (parity_check, select_fillers,    # noqa
                                  torch_package)
from cohort.sampler import MIX, N_TRAIN_TPL, sample_spans          # noqa
from cohort.neutral_grammar import (FORWARD_SPAN_NEUTRAL,          # noqa
                                    VALUE_SPAN_NEUTRAL)

P = argparse.ArgumentParser()
P.add_argument("--condition", choices=CONDITIONS, required=True)
P.add_argument("--model-seed", type=int)
P.add_argument("--order-seed", type=int)
P.add_argument("--steps", type=int, default=20_000)
P.add_argument("--device", default="cuda")
P.add_argument("--parity-only", action="store_true",
               help="CPU layout/mask parity battery only; no model, no env "
                    "seeds needed")
P.add_argument("--resume", default=None,
               help="full-state ckpt: bit-exact continuation")
A = P.parse_args()

CSAFE = {"R-": "Rm", "R+": "Rp", "G-": "Gm", "G+": "Gp", "N+": "Np"}
BETA = 0.003
BATCH = 32

CFG = json.load(open(ROOT / "cohort" / "condition_config.json"))
HDR_IDS = {k: list(map(int, CFG["header_token_blocks"][k]))
           for k in ("mine", "other", "slot")}
W = CFG["header_token_blocks"]["width"]
assert all(len(v) == W for v in HDR_IDS.values())


def parity_battery():
    """CPU battery over all five conditions on synthetic tensors."""
    g = torch.Generator().manual_seed(20260911)
    Ts, n, pad = 33, 6, 151643
    s_ids = torch.randint(100, 5000, (n * 4, Ts), generator=g)
    # realistic right-padded masks: contiguous readable prefix
    lens = torch.randint(5, Ts + 1, (n * 4,), generator=g)
    s_mask = (torch.arange(Ts)[None, :] < lens[:, None]).long()
    banks = {L: [[6000 + L * 40 + k] * L for k in range(4)]
             for L in range(5, Ts + 1)}
    fill = select_fillers(s_mask, n, Ts, banks, 444, "opX_splitY", 12345, pad)
    for cond in CONDITIONS:
        parity_check(cond, s_ids, s_mask, n, Ts, HDR_IDS,
                     fill if cond == "N+" else None)
        print(f"parity {cond}: OK (n={n}, Ts={Ts}, W={W})")
    # N+ filler: length-matched masks -> N+ mask == G+ mask exactly
    _, m_np = torch_package("N+", s_ids, s_mask, n, Ts, HDR_IDS, fill)
    _, m_gp = torch_package("G+", s_ids, s_mask, n, Ts, HDR_IDS)
    assert torch.equal(m_np, m_gp), "N+/G+ mask identity broken"
    # PRF: ordinal-sensitive and deterministic
    f2 = select_fillers(s_mask, n, Ts, banks, 444, "opX_splitY", 12346, pad)
    f3 = select_fillers(s_mask, n, Ts, banks, 444, "opX_splitY", 12345, pad)
    assert not torch.equal(fill, f2) and torch.equal(fill, f3)
    print("parity battery: ALL PASS (incl. N+/G+ mask identity)")


if A.parity_only:
    parity_battery()
    sys.exit(0)

# ---------------------------------------------------------------------
# training path (GPU box)
# ---------------------------------------------------------------------
for _v in ("BRIDGE_OP_SEED", "BRIDGE_SPLIT_SEED",
           "COHORT_GRAMMAR_PATH", "COHORT_FILLER_SEED"):
    assert os.environ.get(_v), f"missing env {_v}"
assert A.model_seed is not None and A.order_seed is not None

from populus.bridge_tasks import SPLIT, apply_chain                # noqa
from populus.bridge_train import Trainer, aux_lambda, lr_scale     # noqa

GRAMMAR_PATH = os.environ["COHORT_GRAMMAR_PATH"]
GRAMMAR_RAW = open(GRAMMAR_PATH, "rb").read()
GRAMMAR_SHA = hashlib.sha256(GRAMMAR_RAW).hexdigest()
GRAMMAR = json.loads(GRAMMAR_RAW)
TRAIN_PHR = {int(k): v["train"] for k, v in GRAMMAR["ops"].items()}
QUAL_PHR = {int(k): v["qual"] for k, v in GRAMMAR["ops"].items()}
assert all(len(v) == N_TRAIN_TPL for v in TRAIN_PHR.values())
assert GRAMMAR["meta"]["value_span"] == VALUE_SPAN_NEUTRAL
assert GRAMMAR["meta"]["forward_span"] == FORWARD_SPAN_NEUTRAL
FILLER_SEED = int(os.environ["COHORT_FILLER_SEED"])
CFG_SHA = hashlib.sha256(
    open(ROOT / "cohort" / "condition_config.json", "rb").read()).hexdigest()

TAG = f"{CSAFE[A.condition]}_m{A.model_seed}_o{A.order_seed}"
DEV = A.device


def log(msg):
    print(f"[cohort {TAG}] {msg}", flush=True)


class CohortTrainer(Trainer):
    """V2.1.2 sampler on the fresh sealed world + neutral grammar;
    packaging delegated to torch_package (parity-checked)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.counts = {"depth": {}, "op": {}, "value": {},
                       "l1_pos": {}, "template": {}}
        self.semantic_hash = hashlib.sha256()
        self.serial_hash = hashlib.sha256()
        self._parity_done = False
        # frozen filler bank at slot width, from the runtime tokenizer
        class _T:
            def __init__(s, tok): s.tok = tok
            def encode(s, text, add_special_tokens=False):
                class R: pass
                r = R()
                r.ids = s.tok(text, add_special_tokens=False)["input_ids"]
                return r
            def decode(s, ids): return s.tok.decode(ids)
        self.pad_id = int(self.lm.tok.pad_token_id)
        assert self.pad_id == 151643, self.pad_id
        self.filler_banks, rep = build_length_banks(
            _T(self.lm.tok), range(5, self.Ts + 1))
        self.world_id = (f"op{os.environ['BRIDGE_OP_SEED']}"
                         f"_split{os.environ['BRIDGE_SPLIT_SEED']}")
        log(f"filler banks: lengths {min(rep)}..{max(rep)} "
            f"(n per length {rep[max(rep)]['n']})")
        # header ids must match the frozen config at the runtime tokenizer
        for name, surface in CFG["header_token_blocks"]["surface_forms"].items():
            ids = self.lm.tok(surface, add_special_tokens=False)["input_ids"]
            assert ids == HDR_IDS[name], (name, ids, HDR_IDS[name])
        log(f"headers verified: {HDR_IDS}")

    def _bump(self, key, val):
        d = self.counts[key]
        d[str(val)] = d.get(str(val), 0) + 1

    def sample_spans(self):
        return sample_spans(self.rng, SPLIT, TRAIN_PHR, apply_chain,
                            VALUE_SPAN_NEUTRAL, FORWARD_SPAN_NEUTRAL,
                            counts=self.counts)

    def package(self, s_ids, s_mask, n, ep_base):
        fill = None
        if A.condition == "N+":
            fill = select_fillers(s_mask.cpu(), n, self.Ts,
                                  self.filler_banks, FILLER_SEED,
                                  self.world_id, ep_base, self.pad_id)
        out_i, out_m = torch_package(
            A.condition, s_ids, s_mask, n, self.Ts, HDR_IDS, fill)
        if not self._parity_done:
            self._parity_done = True
            parity_check(A.condition, s_ids.cpu(), s_mask.cpu(), n,
                         self.Ts, HDR_IDS, fill)
            log("startup parity vs condition_layout: PASS")
        return out_i, out_m

    ACC_CHUNK = 16

    def train_step(self, n=BATCH):
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
            wgt = (sl.stop - sl.start) / n
            out = self.bridge.forward_episode(q[sl], s_ids[sl], s_mask[sl])
            corrected = (out["logits"]
                         - out["base_logits"].detach())[:, self.label_ids]
            loss = F.cross_entropy(corrected, tgt[sl])
            if lam > 0:
                loss = loss + lam * F.cross_entropy(out["aux_logits"], tgt[sl])
            (loss * wgt).backward()
            total += float(loss) * wgt
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.groups for p in g["params"]], 1.0)
        self.opt.step()
        self.step_count += 1
        return total

    def make_batch(self, n=BATCH, **_):
        eps = [self.sample_spans() for _ in range(n)]
        s_ids, s_mask = self.lm.encode(
            [s for sp, _, _ in eps for s in sp],
            fixed_length=self.Ts, device=self.dev)
        ep_base = self.step_count * BATCH
        s_ids, s_mask = self.package(s_ids, s_mask, n, ep_base)
        q = self.q_ids.expand(n, -1)
        tgt = torch.tensor([a for _, a, _ in eps], device=self.dev)
        self.semantic_hash.update(json.dumps(
            [m for _, _, m in eps]).encode())
        self.serial_hash.update(s_ids.cpu().numpy().tobytes())
        self.serial_hash.update(s_mask.cpu().numpy().tobytes())
        return q, s_ids, s_mask, tgt

    @torch.no_grad()
    def probe(self):
        def run_case(spans, answer):
            s_ids, s_mask = self.lm.encode(
                spans, fixed_length=self.Ts, device=self.dev)
            s_ids, s_mask = self.package(s_ids, s_mask, 1, ep_base=0)
            out = self.bridge.forward_episode(self.q_ids, s_ids, s_mask)
            corr = (out["logits"] - out["base_logits"])[:, self.label_ids]
            return int(corr.argmax(-1).item() == answer)

        r = {}
        hits = tot = 0
        for x in range(0, 17, 2):
            hits += run_case([VALUE_SPAN_NEUTRAL.format(x=x)]
                             + [FORWARD_SPAN_NEUTRAL] * 3, x)
            tot += 1
        r["l0"] = round(hits / tot, 4)
        hits = tot = 0
        for op in range(0, 12, 2):
            for pos in (1, 2, 3):
                x = (op * 5 + pos) % 17
                spans = [VALUE_SPAN_NEUTRAL.format(x=x)] \
                    + [FORWARD_SPAN_NEUTRAL] * 3
                spans[pos] = QUAL_PHR[op][pos % 4]
                hits += run_case(spans, apply_chain([op], x))
                tot += 1
        r["l1_qual"] = round(hits / tot, 4)
        for name in ("diag2", "diag3"):
            hits = tot = 0
            for seq in SPLIT[name][:12]:
                for x in range(0, 17, 4):
                    spans = [VALUE_SPAN_NEUTRAL.format(x=x)]
                    spans += [QUAL_PHR[o][(i + x) % 4]
                              for i, o in enumerate(seq)]
                    while len(spans) < 4:
                        spans.append(FORWARD_SPAN_NEUTRAL)
                    hits += run_case(spans, apply_chain(list(seq), x))
                    tot += 1
            r[name] = round(hits / tot, 4)
        return r

    def full_ckpt(self, path):
        torch.save({
            "bridge_trainable": {n: t for n, t in
                                 self.bridge.state_dict().items()
                                 if not n.startswith("lm.")},
            "lora": {n: t for n, t in self.lm.core.state_dict().items()
                     if "lora_" in n},
            "opt": self.opt.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "data_rng": self.rng.getstate(),
            "step": self.step_count,
            "condition": A.condition,
            "model_seed": A.model_seed, "order_seed": A.order_seed,
            "beta": BETA,
            "op_seed": os.environ["BRIDGE_OP_SEED"],
            "split_seed": os.environ["BRIDGE_SPLIT_SEED"],
            "filler_seed": FILLER_SEED,
            "grammar_sha": GRAMMAR_SHA, "config_sha": CFG_SHA,
            "semantic_stream_hash": self.semantic_hash.hexdigest(),
            "serialized_stream_hash": self.serial_hash.hexdigest(),
        }, path)


from populus.bridge_qwen import QwenGenome                         # noqa
torch.manual_seed(A.model_seed)
lm = QwenGenome().attach_lora().to(DEV)
tr = CohortTrainer(lm, data_seed=A.order_seed, device=DEV)
tr.bridge.beta.fill_(BETA)
assert tr.optimizer_audit()["all_clear"]
RESUMED_FROM = None
if A.resume:
    rck = torch.load(A.resume, map_location=DEV, weights_only=False)
    assert rck["condition"] == A.condition
    assert rck["model_seed"] == A.model_seed
    assert rck["order_seed"] == A.order_seed
    assert rck["grammar_sha"] == GRAMMAR_SHA
    assert rck["config_sha"] == CFG_SHA
    tr.bridge.load_state_dict(rck["bridge_trainable"], strict=False)
    lm.core.load_state_dict(rck["lora"], strict=False)
    tr.opt.load_state_dict(rck["opt"])
    torch.set_rng_state(rck["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([t.cpu() for t in rck["cuda_rng"]])
    tr.rng.setstate(rck["data_rng"])
    tr.step_count = rck["step"]
    RESUMED_FROM = {
        "ckpt": A.resume, "step": rck["step"],
        "pre_resume_semantic_hash": rck["semantic_stream_hash"],
        "pre_resume_serialized_hash": rck["serialized_stream_hash"]}
    log(f"RESUMED bit-exact from step {rck['step']}")
log(f"start: cond={A.condition} m={A.model_seed} o={A.order_seed} "
    f"op={os.environ['BRIDGE_OP_SEED']} "
    f"split={os.environ['BRIDGE_SPLIT_SEED']} grammar={GRAMMAR_SHA[:12]}")

traj = []
t0 = time.time()
outdir = Path("results/cohort")
outdir.mkdir(parents=True, exist_ok=True)
for step in range(A.steps - tr.step_count):
    loss = tr.train_step()
    s = tr.step_count
    if s == 3:
        # REVIEW round-2 §3.4: model-STATE prefix fingerprint — digest of
        # trainable params, optimizer state, RNG states, step counter
        # after the fixed three-batch prefix (bit-exact across admitted
        # machines at fixed seeds/condition).
        _h = hashlib.sha256()
        for _n, _t in sorted(tr.bridge.state_dict().items()):
            if not _n.startswith("lm."):
                _h.update(_n.encode())
                _h.update(_t.detach().cpu().numpy().tobytes())
        for _n, _t in sorted(lm.core.state_dict().items()):
            if "lora_" in _n:
                _h.update(_n.encode())
                _h.update(_t.detach().cpu().numpy().tobytes())
        for _gi, _g in enumerate(tr.opt.state_dict()["state"].items()):
            _h.update(str(_g[0]).encode())
            for _k in sorted(_g[1]):
                _v = _g[1][_k]
                _h.update(_k.encode())
                _h.update(_v.detach().cpu().numpy().tobytes()
                          if hasattr(_v, "detach") else str(_v).encode())
        _h.update(torch.get_rng_state().numpy().tobytes())
        if DEV == "cuda":
            for _t in torch.cuda.get_rng_state_all():
                _h.update(_t.cpu().numpy().tobytes())
        _h.update(json.dumps(tr.rng.getstate()).encode())
        _h.update(str(tr.step_count).encode())
        log(f"ADMISSION_PREFIX3 sem={tr.semantic_hash.hexdigest()} "
            f"ser={tr.serial_hash.hexdigest()} state={_h.hexdigest()}")
    if s % 500 == 0:
        p = tr.probe()
        traj.append({"step": s, "loss": round(loss, 3), **p})
        log(f"step {s}: loss {loss:.3f} probe {p}")
    if s % 1000 == 0:
        log(f"realized counts @ {s}: "
            f"{json.dumps(tr.counts, sort_keys=True)}")
    if s % 5000 == 0:
        tr.full_ckpt(outdir / f"cohort_{TAG}_ckpt_{s}.pt")

tr.full_ckpt(outdir / f"cohort_{TAG}_ckpt_final.pt")
rep = {"condition": A.condition, "model_seed": A.model_seed,
       "order_seed": A.order_seed, "trajectory": traj,
       "wall_s": round(time.time() - t0),
       "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
       "realized_counts": tr.counts, "resumed_from": RESUMED_FROM,
       "op_seed": os.environ["BRIDGE_OP_SEED"],
       "split_seed": os.environ["BRIDGE_SPLIT_SEED"],
       "filler_seed": FILLER_SEED,
       "grammar_sha": GRAMMAR_SHA, "config_sha": CFG_SHA,
       "semantic_stream_hash": tr.semantic_hash.hexdigest(),
       "serialized_stream_hash": tr.serial_hash.hexdigest()}
(outdir / f"cohort_{TAG}.json").write_text(json.dumps(rep, indent=1))
Path(f"DONE_COHORT_{TAG}").touch()
log(f"COHORT TRAJECTORY COMPLETE: {traj[-1] if traj else None}")

# What You Can't See Is What You Learn

**Restricted Evidence Visibility Favors Compositional Generalization in Shared-Genome Language-Model Societies**

Artifact release for the paper (arXiv link to follow). Author: Narcis Marincat.

Four-cell societies built from one frozen Qwen2.5-0.5B-Instruct backbone with a single shared rank-8 LoRA communicate through learned two-vector continuous packets in a fixed relay. Ten matched restricted/global twin pairs — identical initialization bytes, training order, token layout, parameters, and computation, differing **only** in an attention mask — test whether restricting each cell's evidence visibility causally changes what training learns. It does: 9/10 pairs show a ≥20-point restricted advantage at both composition depths (median paired advantages 0.7648 / 0.6050), every restricted society collapses to exact chance when communication is cut, and value-indexed packet transplants show the restricted societies learn an approximately interchangeable value code. The complete preregistered battery formally fails on one absolute floor (restricted median depth-three 0.6988 vs 0.70), reported as a binding outcome.

## Contents

- `populus/` — model code: the society architecture (`bridge.py`), frozen-genome wrapper (`bridge_qwen.py`), task/bank machinery (`bridge_tasks.py`), trainer (`bridge_train.py`), phrasing grammar (`bridge_grammar.py`).
- `scripts/` — the exact run scripts: paired-battery trainer (`bridge_run3v_society.py`), evaluator (`bridge_run3v_evaluator.py`), post hoc collision stratification (`bridge_run3v_stratified.py`), packet audits (`bridge_packet_audit_run3v.py`), bit-exact machine gate (`run3v_golden_check.py`), staged centralized comparator (`bridge_posctl_staged.py`), and the earlier qualification-cohort trainer (`bridge_qual_society.py`).
- `results/run3v/` — every evaluation, stratification, and audit JSON reported in the paper, plus the formal verdict (`run3v_verdict.json`), sealed template grammar, and split audit.
- `results/bridge/` — serialization contract and tokenizer-verified label mapping the scripts consume.
- `PREREGISTRATION.md` — the anonymized preregistration: sealed-instance seeds and hashes, frozen gates, design, and protocol history.
- `INCIDENTS.md` — incident and compute-lineage ledger (bit-exact resume protocol; disclosed losses).

## Checkpoints

All twenty society final checkpoints (~50 MB each: trainable adapter, projections, optimizer state for bit-exact resume) and three staged-comparator finals are hosted on Hugging Face: **https://huggingface.co/tokenosopher/populus-evidence-partitioning-checkpoints**. One restricted final (init 202, order 952) additionally remains marooned on an unreachable provider instance, disclosed in the paper; its evaluation JSONs are present here.

## Reproducing evaluations

Evaluation is deterministic (`torch.use_deterministic_algorithms(True)`, TF32 disabled). Example:

```bash
RUN3V_WORLD=F BRIDGE_OP_SEED=6011 BRIDGE_SPLIT_SEED=2203 BRIDGE_GRAMMAR_SEED=7717 \
python scripts/bridge_run3v_evaluator.py --ckpt <checkpoint.pt>
```

The task world regenerates from the seeds above; hashes must match `PREREGISTRATION.md` (split `94e506d408b6def1`, grammar `def14eb4182e2949`).

## License

Code: MIT (see `LICENSE`). Result data and documents: CC BY 4.0.

# What You Can't See Is What You Learn

**Restricted Evidence Visibility Favors Compositional Generalization in Shared-Genome Language-Model Societies**

Artifact release for the paper: **[arXiv:2608.20054](https://arxiv.org/abs/2608.20054)** (cs.AI; cross-listed cs.LG, cs.MA). Author: Narcis Marincat.

Four-cell societies built from one frozen Qwen2.5-0.5B-Instruct backbone with a single shared rank-8 LoRA communicate through learned two-vector continuous packets in a fixed relay. Ten matched restricted/global twin pairs — identical initialization bytes, training order, token layout, parameters, and computation, differing **only** in an attention mask — test whether restricting each cell's evidence visibility causally changes what training learns. It does: 9/10 pairs show a ≥20-point restricted advantage at both composition depths (median paired advantages 0.7648 / 0.6050), every restricted society collapses to exact chance when communication is cut, and value-indexed packet transplants show the restricted societies learn an approximately interchangeable value code. The complete preregistered battery formally fails on one absolute floor (restricted median depth-three 0.6988 vs 0.70), reported as a formal preregistered outcome.

## Contents

- `populus/` — model code: the society architecture (`bridge.py`), frozen-genome wrapper (`bridge_qwen.py`), task/bank machinery (`bridge_tasks.py`), trainer (`bridge_train.py`), phrasing grammar (`bridge_grammar.py`).
- `scripts/` — the exact run scripts: paired-battery trainer (`bridge_run3v_society.py`), evaluator (`bridge_run3v_evaluator.py`), post hoc collision stratification (`bridge_run3v_stratified.py`), packet audits (`bridge_packet_audit_run3v.py`), bit-exact machine gate (`run3v_golden_check.py`), staged centralized comparator (`bridge_posctl_staged.py`), and the earlier qualification-cohort trainer (`bridge_qual_society.py`).
- `results/run3v/` — every evaluation, stratification, and audit JSON reported in the paper, plus the formal verdict (`run3v_verdict.json`), sealed template grammar, and split audit.
- `results/bridge/` — serialization contract and tokenizer-verified label mapping the scripts consume.
- `PREREGISTRATION.md` — the anonymized preregistration: sealed-instance seeds and hashes, frozen gates, design, and protocol history.
- `INCIDENTS.md` — incident and compute-lineage ledger (bit-exact resume protocol; disclosed losses).
- `companion/` — artifacts for the companion study *Portable Semantics, Private Dialects* (P1–P4 + Appendix A):
  - `companion/p1/` — sealed cross-model transplant audit: launch/sealed manifests, fit/test packet banks per checkpoint, per-direction evaluation JSONs, frozen classification output, lane logs. The fitted alignment maps (`p1_maps.pt`, ~4.4 GB, float64) are hosted in the Hugging Face repository.
  - `companion/p34/` — P3 interface-adaptation factorial and P4 second-stream replication: configurations, learning-curve records, final evaluations, logs (adapted-run checkpoints on Hugging Face).
  - `companion/twin_dialect/` — Appendix A: `twin_dialect.py` and `twin_dialect_matrix.py` (RNG seeds 8300 harvest / 8400 recipient), directional probe JSONs, the full 2×2 matrix JSON with per-case predictions, donor episode metadata, and the donor–recipient exact-overlap audit.
  - `companion/tag_pilot/` — the three-run role-marked pilot behind Appendix A: tagged trainer/evaluator/stratifier scripts, evaluation and map-novel stratification JSONs (GT checkpoint on Hugging Face).

## Checkpoints

All twenty society final checkpoints and all three staged-comparator finals (~50 MB each: trainable adapter, projections, optimizer state for bit-exact resume) are hosted in the linked Hugging Face repository: **https://huggingface.co/tokenosopher/populus-evidence-partitioning-checkpoints**. The restricted final for initialization 202, order 952 was recovered from its provider instance on 2026-08-31 (SHA-256 `88bc8a1654828ecd70b28e5c812e193fb3ac2f72d3f8be2ea1a13269685305a8`) and uploaded; its training-result record (`results/run3v/run3v_F_P_m202_o952.json`) is now also included here, completing the run3v record set.

## Reproducing evaluations

Evaluation is deterministic (`torch.use_deterministic_algorithms(True)`, TF32 disabled). Example:

```bash
RUN3V_WORLD=F BRIDGE_OP_SEED=6011 BRIDGE_SPLIT_SEED=2203 BRIDGE_GRAMMAR_SEED=7717 \
python scripts/bridge_run3v_evaluator.py --ckpt <checkpoint.pt>
```

The task world regenerates from the seeds above; hashes must match `PREREGISTRATION.md` (split `94e506d408b6def1`, grammar `def14eb4182e2949`).

## License

Code: MIT (see `LICENSE`). Result data and documents: CC BY 4.0.


## v2: Universality scout (Appendix F)

Appendix F of v2 reports a prospectively frozen near-transfer scout on a
sealed second task family (family D, `populus/bridge_tasks_d.py`, bank
hash `2bd0b301a5b079b1`). `scout/` holds the frozen plan, launch
manifest, and final report; `scripts/bridge_scout_*.py` are the exact
evaluators and the transfer/learnability driver; `results/scout/`
contains every evaluation JSON. Adaptation checkpoints are in the
Hugging Face repository alongside the main release.

## Citation

```bibtex
@article{marincat2026restricted,
  title={What You Can't See Is What You Learn: Restricted Evidence
         Visibility Favors Compositional Generalization in
         Shared-Genome Language-Model Societies},
  author={Marincat, Narcis},
  journal={arXiv preprint arXiv:2608.20054},
  year={2026}
}
```

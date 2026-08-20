# Anonymized preregistration — paired-visibility battery (Run-3V)

All elements below were fixed in writing before any training on the sealed
task instance and before any outcome inspection. Analyses added afterward
(collision stratification, inferential tests) are labeled post hoc in the
paper and are not part of this preregistration.

## Sealed task instance

- Operator seed 6011 (12 affine bijections over Z_17).
- Split seed 2203 — split hash `94e506d408b6def1`. Held-out depth-2: 12
  ordered programs; held-out depth-3: 60 ordered programs; forbidden
  train/test intersections verified zero by bank audit
  (`results/run3v/split_audit_run3v.json`).
- Phrasing-grammar seed 7717 — grammar hash `def14eb4182e2949`; held-out
  surface phrasings; four rotations per operator
  (`results/run3v/template_grammar_run3v.json`).
- Answer alphabet: 17 tokenizer-verified single-token labels; chance = 1/17.

## Design

- Ten twin pairs: five model initializations (seeds 200–204) x two
  data-order streams (900–904 / 950–954). Within a pair, the restricted (P)
  and global (G) arm share initialization bytes and the ordered example
  stream (verified by running stream hashes).
- Sole treatment: the per-cell attention mask. Both arms receive the
  identical fixed four-slot layout, positions, packet slots, parameters,
  computation, and Transformer calls.
- Exactly 20,000 updates per model; final checkpoints only; no model
  selection.
- Curriculum mixture (frozen from the pre-battery development ladder):
  identity 0.10, single-operation 0.25, two-operation 0.325,
  three-operation 0.325.
- Secondary comparators: three staged centralized-scan models with
  cumulative evidence visibility (fresh seeds); not pair-matched, not
  compute-matched.
- Machine admission: bit-exact reproduction of two golden fingerprints
  (restricted `c7fcb21aad26951b`, global `b7c6157b1b3b79af`) computed over
  internal state after a fixed training prefix; two gates per machine.

## Frozen outcome gates (conjunctive battery)

1. P−G ≥ 0.20 at both depths in ≥ 8 of 10 pairs.
2. Median paired advantage ≥ 0.25 at depth two.
3. Median paired advantage ≥ 0.25 at depth three.
4. All-packets-cut accuracy ≤ 0.11 in ≥ 8 restricted models
   (communication necessity).
5. No restricted run below 0.40 at depth three.
6. Restricted median ≥ 0.70 at both depths.

Outcome: gates 1–5 passed; gate 6 failed at depth three (median 0.6988).
The conjunction therefore formally fails, reported as binding
(`results/run3v/run3v_verdict.json`).

## Frozen packet-audit gates (per audited model)

- Destructive (deletion / norm-matched noise) ≤ 0.11.
- Cross-example shuffle ≤ 0.11, or ≥ 30 points below intact accuracy.
- Same-value transplant ≥ 0.95.
- Counterfactual transplant fidelity ≥ 0.95.
- Span-counterfactual control ≥ 0.95.

Audit model selection (six restricted societies spanning the performance
range, plus global comparators) occurred after behavioral evaluation and is
labeled post hoc in the paper; the gate thresholds above were fixed before
any audit ran.

## Protocol history

The battery follows a restricted-only qualification cohort on a separately
sealed world (ten societies; 0/10 complete gate passes; one model met all
task-performance gates; all ten failed ordinary-language preservation) and
a development ladder that established the curriculum atoms. The paired
design, fresh world, and all thresholds above were frozen in response to a
restricted/global performance contrast observed on development worlds.

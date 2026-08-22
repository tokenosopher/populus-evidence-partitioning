# PLAN — Universality Scout (frozen-protocol transfer test)

Status: FROZEN AS AMENDED (referee ruling 2026-08-20, reviews/REVIEW_scout_ratification.md). Amendments 1-10 incorporated below; this file is now canonical. Implements the scout mandated
by REVIEW_analyses_ruling (priority after paper) and refined by
REVIEW_appendix4_priorart_sweep (scout-first confirmed; failure
localization + sample-efficiency endpoints). Derived from
PROPOSAL_UNIVERSALITY_ARM.md (author-originated 2026-08-11). No frozen
protocol is modified; all Run-3V artifacts are read-only inputs.

## Question

Is the trained translator (shared LoRA + write/read projections +
mouth) of the Run-3V restricted societies a task-specific codebook or
a general-purpose serialization of the running 17-value state? The
scout measures this on EXISTING checkpoints; it trains no new
societies (one pre-verification run excepted, below).

## Held-out family D (ring-pointer world)

Shares the frozen interface exactly: 17-class label space A-Q, frozen
QUESTION + ANSWER_PREFIX serialization, T_SPAN=33, T_Q=54, four-slot
episode shape, identity/forwarder slots, same masked layout.

- State: pointer position on a 17-slot ring (0-16), presented as
  "The pointer starts at slot x." (surface distinct from family A's
  "The value is x.").
- Operators (12 bijections on Z17, chosen so composite-map overlap
  with family A's TRAINED bank is minimized and reported):
  forward k (k in {2,3,5,7}); back k (k in {1,4,6}); reflect through
  slot c, i.e. x -> (c - x) mod 17 (c in {0,3,8}); stretch m, i.e.
  x -> (m*x) mod 17 (m in {2,3}).
- Phrasing grammar: fresh templated grammar, 4 surface rotations per
  operator, generated and hashed before any evaluation.
- Seeds (fixed now): D_OP_SEED=8021, D_SPLIT_SEED=3407,
  D_GRAMMAR_SEED=9313. Bank hashes reported at generation; banks
  sealed before any checkpoint sees an episode.
- Attributability gate: frozen base Qwen2.5-0.5B-Instruct with FULL
  visibility must score ~chance on D held-out episodes (as for A).
- Learnability gate: ONE fresh restricted society trained on D alone
  (frozen Run-3V training protocol verbatim, new seeds) must reach
  diag3 >= 0.40. Without this, D-transfer failure is uninterpretable.

## Scout measurements (existing checkpoints: six audited P societies
## + P:m201_o951 emphasized; G outlier m204_o954 as comparator)

S1. Strict zero-shot: full D evaluation (diag2/diag3 banks), all
    weights frozen. Metric: accuracy vs chance (0.0588).
S2. Cut-mail control on D: packets severed; must sit at chance if S1
    shows any signal (localizes transfer to the channel).
S3. Packet-readability transfer (receiving side): value-indexed
    donor packets harvested from FAMILY-A episodes injected at each
    interface of D episodes whose running value matches (and
    counterfactual mismatches). If A-born packets steer D episodes
    toward their value, the packet code is context-portable even
    where local parsing fails.
S4. Mouth control: teacher-forced final packet (A-born donor with the
    D episode's correct final value) -> mouth must decode the label.
S5. Local-primitive control: single-cell episodes where the cell's
    span is a D operator phrase and the incoming packet is a correct
    A-born donor; measures whether cells can parse NEW surface
    language at all (expected failure; this is the localization).
S6. Packet probe: ridge/linear readout trained on packets captured
    during S1 runs, predicting the true D running value at each
    interface. Distinguishes "packets carry nothing about D" from
    "packets carry D-state the mouth cannot express".
S7. Sample-efficiency transfer (PRIMARY quantitative endpoint, per
    sweep ruling): brief fine-tuning on D (2,000 updates, frozen
    Run-3V optimizer/hyperparameters) in three arms per checkpoint
    (2 checkpoints: m201_o951, m203_o953):
      a) frozen translator (LoRA+projections frozen; nothing else
         trainable exists -> arm a is pure mouth+none; if literally
         no trainable surface remains, arm a reduces to S1 and is
         reported as such);
      b) full fine-tune from Run-3V weights;
      c) fresh-channel control: same architecture, LoRA+projections
         reinitialized, trained 2,000 updates on D.
    Metric: diag3(D) after 2,000 updates + area under the learning
    curve (probe every 250); the b-vs-c gap is the value of the
    A-born protocol.

## Interpretation table (pre-registered)

- S1 >> chance with S2 at chance: task-trained protocol generalizes
  (would re-aim Appendix 4 at reliability/canonicality).
- S1 ~ chance, S3+S4 pass, S5 fails: packet code is portable; only
  local parsing is missing -> strongest justification for Appendix 4
  (text pretraining teaches parsing diversity).
- S1 ~ chance, S3/S4 fail: code is context-bound (episode-entangled
  beyond its home task) -> revises the value-indexed claim's scope;
  Appendix 4 becomes a test of whether a broader diet fixes code
  form itself.
- S7 b>>c: protocol reuse accelerates learning even when zero-shot
  fails (CORAL-shaped positive).

## Not in scope

No new multi-family training (the diversity curve R1-R3 remains a
separate, later decision). No protocol changes. No Appendix-4 work.

## Venue and cost (projection, 1.3x buffer)

One RTX 5090 (Vast, ~$0.40/h): bank generation + attributability +
S1-S6 evals ~6h ($2.4); learnability run 20k updates (~$8); S7 six
short runs 2k updates (~$5). Projection: $15.4 x 1.3 = ~$20. Balance
check + author notification precede any rental per standing rule.

## Generation record (draft banks; regenerate + reseal at freeze if
## the referee amends the operator set)

- D bank hash (seeds 8021/3407): `3c2c339935ff4316`; bank sizes
  train2 94, train3 445, diag2 12, diag3 60, ho2 29, f_ho 18
  (results/scout/d_family_manifest.json).
- Composite-map overlap disclosure: family A's Run-3V world trained
  208 of the 272 affine maps, so affine-D overlap is structural:
  41/60 D-diag3 and 10/12 D-diag2 composite maps coincide with
  A-trained maps; 1/12 D primitives coincides with an A primitive.
  Consequence: all D results will ALSO be reported on the D map-novel
  stratum (19/60 diag3), mirroring the paper's collision discipline.
  Note for ruling: a lookup shortcut still requires parsing ring
  vocabulary A never saw; the overlap weakens only map-level purity,
  not the parsing-generality reading.

## FROZEN AMENDMENTS (ruling of 2026-08-20; canonical)

1. FAMILY D: affine ring-pointer family RETAINED; claim ceiling: "a
   near-transfer test of value-protocol portability, not
   algebra-independent universality." Non-affine permutations deferred
   to a later far-transfer family C. Operators frozen with fwd 5
   replaced by fwd 8 pre-freeze (primitive collision with A);
   final primitive overlap 0/12. Bank hash 2bd0b301a5b079b1
   (seeds 8021/3407; pre-launch erratum 2026-08-20: reflection
   operators corrected to x -> 2c-x, matching the template language
   "reflect through slot c" — the draft implemented c-x, i.e.
   reflection through c/2; caught in pre-launch verification, banks
   resealed before any checkpoint contact; primitive overlap with A
   remains 0/12). All D results reported stratified: A-map-novel
   vs A-map-overlapping (diag3: 13/60 novel vs A-trained maps,
   3/60 vs all A-observed maps — near-transfer reading stands).
2. GRAMMAR: 14 renderings per operator kind, 8 train / 2 diag /
   4 sealed, generated and hashed before any evaluation (implemented
   in populus/bridge_tasks_d.py).
3. LEARNABILITY GATE (one fresh restricted society, full 20,000
   updates, no seed retries): L1 held-out wording >= 0.95; diag2 >=
   0.70; diag3 >= 0.60; all-cut diag2 and diag3 <= 0.11; finite
   packets throughout. Failure => reject or redesign D, no S7
   interpretation.
4. Native-base full-info accuracy: REPORTED, not gated.
5. S2 (cut-mail on D): run UNCONDITIONALLY.
6. S3-S5 hierarchy (replaces draft): S3 = pure packet portability
   (A-born donors traverse D-context relays of structural forwarders
   only; >=0.90 same-value and counterfactual). S4 = mouth
   portability (A-born final packet in D mouth context; >=0.95; A
   context positive control). S5 = local D-primitive competence
   (A-born incoming packet + one D operator phrase + forwarders +
   frozen mouth, all operators x phrasings x positions; >=0.90
   descriptive). Localization table per ruling.
7. S6 probe: cross-program ridge (train/test disjoint in sequences,
   phrasings, start values); report CV accuracy, permuted-label
   baseline, per-interface accuracy, norm-matched-random control.
   Diagnostic only.
8. S7 ARMS (per checkpoint; fresh optimizer; identical D stream;
   aux classifier reinitialized identically in every arm):
   S7a interface-frozen/LoRA-adapted (reader, writer, mouth, beta,
   base frozen; LoRA trains). S7b full adaptation from A init.
   S7c A-LoRA init + REINITIALIZED reader/writer/mouth (isolates the
   A-born interface; S7b - S7c is the protocol-initialization value).
   Horizon exactly 5,000 updates, eval every 250, Run-3 schedules
   over first 5,000 updates, no extension. Endpoints: final diag3,
   trapezoidal AUC 0-5,000, steps to 0.40/0.60/0.80.
   Criteria: strong frozen-interface transfer = S7a diag3(5000) >=
   0.40 with all-cut <= 0.11, both checkpoints. Initialization
   benefit = AUC(S7b)-AUC(S7c) >= 0.10 AND final-diag3 gap >= 0.10,
   both checkpoints. Replicated scout result, not population claim.
9. CHECKPOINTS: S7 factorial on m201_o951 (strongest; passed all
   audit gates) and m203_o953 (mid-range) — purposive selection.
   S1-S6 on all six audited P checkpoints (m200_o950, m201_o951,
   m203_o903, m203_o953, m204_o904, m204_o954-P) plus global
   comparator G m204_o954. G excluded from primary S7.
10. COST CEILING: $30 (S1-S6 + preflight ~$2-3; D learnability run
    ~$8; six 5,000-update S7 runs ~$12-13; contingency ~$5-6).
    Standing rule: measure throughput, present projection and balance
    to the author, no rental without explicit confirmation.

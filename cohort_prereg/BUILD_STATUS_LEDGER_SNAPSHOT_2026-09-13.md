# Cohort Build Status — what is built+validated vs what needs the admission box

Everything in the FROZEN DECISION PATH (classify, count, judge) is built and
validated locally with zero spend. Everything GPU/tokenizer-coupled is
specified and pure-logic-tested; its ONE irreducibly box-dependent check is
isolated and labelled, and runs in the mandatory rehearsal before the fleet.

## BUILT + LOCALLY VALIDATED (no GPU)
- `phenotype_classifier.py` — Appendix A decision tree. 8/8 known-value cases pass
  (pilot GT->deep-only inversion; parent outlier->intermediate/mixed; etc.).
- `stats_engine.py` — cluster bootstrap (100k, seed 271828) + exact two-sided
  sign-flip. Fixture-tested; 0.0625 five-cluster floor confirmed.
- `verdict.py` + `test_verdict.py` — frozen §5.2–5.5 gates. 9/9 adversarial cases
  pass: INCOMPLETE-not-reduced, 0.70 floor, tag-audit independence, comm-dependence,
  practical-null, MINIMUM-tier-C1-only.
- `seed_rule.py` — 22 seeds from one hashed master string; deterministic.
- `neutral_grammar.py` — 384 third-person renderings, zero second-person leaks
  (regex-asserted); seeded split producer.
- `marker_filler.py` — per-condition header layout + N+ filler leakage audit;
  structural checks pass across all 5 conditions × 4 slots.
- `OPS_PLAYBOOK.md` — m202 rule (sync-verify-ledger before lane advance),
  mule rule (independent spend watch + box deadlines), verify-before-destroy,
  deterministic resume, mandatory rehearsal.

## SPECIFIED, NEEDS ADMISSION-BOX FINALIZATION (labelled in code)
- Tokenizer preflight (`marker_filler.tokenizer_preflight`): equal header-token
  counts for mine:/other:/slot: at the pinned Qwen tokenizer; pad plan if unequal.
- Grammar token-length fit (`neutral_grammar`): filter renderings to T_SPAN budget
  at pinned tokenizer BEFORE the frozen split take — a pure filter, not re-authoring.
- Five-condition trainer integration: teach bridge_train to dress episodes per
  condition using marker_filler layout (own-slot mask for R±; global for G±;
  own+neutral-filler for N+). Reuses frozen architecture/optimizer/budget.
- Collision-free primary bank + G+ permutation-audit bank + monitoring bank
  constructors: pure-logic selection over SPLIT, but must run against the FRESH
  world's operator table (generated post-freeze from the seed rule).
- Machine-admission fixture: per-condition bit-exact prefix fingerprints.

## FREEZE COMMIT CHECKLIST (resolves every [AT FREEZE] bracket)
- [ ] literal SHA-256 of: neutral_grammar, marker_filler, bank constructors,
      phenotype_classifier, verdict, stats_engine, seed_rule, trainer, evaluator,
      machine-admission fixture
- [ ] immutable-archive identifier of the scientific-content deposit (pre-seed)
- [ ] TIER_DECLARATION.json (GOLD) included in the deposit  [DONE: committed]
- [ ] companion arXiv id 2609.11365 in prereg header via pointer commit
- [ ] currency-sweep citations (DIAL, Terry et al., CVRR) noted for the cohort paper

## SEQUENCE TO LAUNCH
build (this) -> Oracle build review -> self review pass -> resolve sixth-family
question -> freeze commit (hashes+archive+timestamp) -> seed generation ->
pre-run manifest -> ADMISSION BOX (tokenizer preflight + fingerprints) ->
REHEARSAL society (full sync-verify-ledger pipeline on one box) -> 50-society fleet.

## 2026-09-11 — batch 2 continued
- filler_constructor.py: token-exact N+ filler bank (36 distinct, width Ts=33, leak-free, deterministic; deny-list caught its own pool sentence "remainder" — audit works).
- condition_config.json FROZEN CONTENT: header blocks mine=[10485,25] other=[1008,25] slot=[9446,25], W=2, no padding needed (local pinned-tokenizer preflight); tokenizer.json sha c0382117...
- neutral_grammar.py: + VALUE_SPAN_NEUTRAL / FORWARD_SPAN_NEUTRAL. **[FOR REFEREE BUILD REVIEW]** prereg's "no template addresses the reader" applied to ALL spans (parent VALUE/FORWARD spans are second-person; keeping them would leave the addressing aggravator in every episode).
- grammar_build.py: deterministic grammar JSON builder; token-length filter leaves 340-376 fit candidates/op (need 36). Fixture-tested.
- scripts/cohort_society.py: five-condition trainer, thin wrapper around torch_package(); startup + --parity-only bit-exact parity vs condition_layout on all 5 conditions PASS; dual stream hashes (semantic + serialized); bit-exact resume carried from pilot.
- world.py TWO FIXES **[FOR REFEREE BUILD REVIEW]**:
  1. stale-module bug: candidate-seed loop silently rebuilt the SAME world (package attribute survived sys.modules purge). Fixed via importlib + delattr; seed-sensitivity now verified.
  2. collision-free pool corrected from diag banks (60/12 programs; >=20/>=8 mechanically unattainable, 0/30 seeds valid; diag also consumed by in-run probes) to HELD-OUT pools (ho2+ho_fn pairs; ho3_familiar+ho3_h2_only+ho_fn triples): 30/30 seeds valid, median 210 cf d3 programs. Permutation audit bank capped at first 24 cf d3 programs (frozen sorted order, ~3.7k cases, parent-eval scale).
- manifest_validator.py: §4 pre-run manifest completeness gate — 10/10 adversarial mutations rejected (incl. audit-sample lower-order rule, serialized-hash collisions, world_id/seed mismatch).
- result_compiler.py: denominators split primary_d2/primary_d3 (latent bug under corrected banks); manifest shape unified (bank_hashes.tag_audit); 6/6 adversarial rejects incl. float counts, wrong step, dup records.
- sampler.py: single shared sampling implementation (trainer + fixture; drift impossible by construction).
- cohort_admission_fixture.py: model-free per-condition prefix fingerprints; CROSS-CHECK vs real HF-tokenizer/torch path: ALL 5 CONDITIONS MATCH bit-exactly. Semantic hash identical across conditions (common stream), serialized hashes distinct — matched-twin discipline machine-verified.
- cohort_eval.py + cohort_tag_audit.py: final-ckpt scorer (raw integer counts; step-20000 asserted; frozen renderings documented) and §5.4 G+ permutation audit (full-bank denominators, mask-invariance asserted per batch, headers-only diff).

## BUILD COMPLETE pending: referee build review round 2 → freeze ceremony → seed gen → admission box → fleet.

## 2026-09-11 — round-2 referee amendments (REVIEW_cohort_build_round2.md)
Ruling: D1 APPROVE / D2+D3 approve-with-amendments / D4 rejected as written / HOLD freeze. All code amendments now implemented and re-verified:
- world.py: train_fns includes identity + all primitive maps; no_identity_primitive; primary-vs-train/diag disjointness EXPLICIT in VALID (no asserts); DEV_EXPOSED_OP_SEEDS denylist (31 seeds); monitoring rule frozen (diag2/diag3 programs x QUAL phrasings only); 12/12 fresh seeds valid under stricter filter.
- filler: per-length banks (exact active length L for L in 5..33), PRF assignment sha256(seed:world:ordinal:slot) — no semantic inputs, no modular side channel; FillerError not assert; " indeed" single-token verified; N+ masks now IDENTICAL to G+ (length-matched).
- 4 sealed rotations in primary, S1, tag audit. Audit cap min(24,N); per-permutation stratified intact/cf (unweighted mean of 5 fidelities), nonzero denominators enforced end-to-end (world -> tag evaluator -> compiler -> manifest).
- Three-legged cross-check REDONE post-amendment: fixture == stream-hasher == real HF/torch path, ALL 5 conditions bit-exact (FIXTURE2, world 910000/920000).
- Compiler: recomputes stratified means from integer per-perm counts; 5 new adversarial rejects.
- test_world_regression.py: permanent stale-module + determinism + VALID-composition tests.

## STILL OPEN before freeze (round-2 §2-3):
1. PREREG v0.4 text: six families wording (D1 span language, held-out pool construction, tag cap + stratified scoring, 4 rotations, filler PRF + length matching, dev-seed denylist note, protocol history of pool fix + stale-module bug).
2. §6.1 mechanism audits + §6.2 dialect matrix evaluators (adapt twin_dialect machinery) + fixtures.
3. DONE scripts/cohort_mask_admission.py — cell-0 packet invariance rule; LOCAL CPU RUN WITH REAL MODEL: PASS (invariance/readability/global-control all correct).
4. DONE model-state fingerprint in ADMISSION_PREFIX3 (params+opt+RNG+step digest); two-machine reproduction happens at admission.
5. DONE scripts/cohort_verdict_lock.py (atomic O_EXCL lock, archives cmd/env/stdout, --fixture dry-run path).
6. Phase-1 decision-module evidence bundle for referee (verdict/stats/classifier tests + hashes — code exists, must be presented).

## 2026-09-11 — round-3 verdict (NO-GO, 3 blockers) + same-day closure
Archived reviews/REVIEW_cohort_build_round3.md. §2.1-2.7, 2.9, 3.1, 3.4-3.6 closed by referee. Blockers all closed same day:
1. manifests/development_seed_exposures.json (single source of truth; world.py derives denylist at import; regression asserts equality; 49 op seeds incl. today's 930000-5/940000-5).
2. §6.1 menu complete (intact/allcut_bank/same/cf_follow/shift_retain/deranged/PRF-seeded norm-matched noise/span-rewrite); coverage-intersection common-case discipline (coordinator pre-pass + case-id hash equality assertion); cohort_dialect_matrix.py + cohort_dialect_aggregate.py (rho ratios, undefined-on-zero-den, four indicators with literal frozen thresholds .90/.60/.30-drop/.50).
3. Mask admission extended to downstream reader path (cells 1-3, incoming packet held by substitution, packet-path liveness, direct own-slot-only mask assertion) — LOCAL PASS on real model, all 8 fields true.
Clarifications: operator-support + tag-viability in VALID (5/6 fresh seeds pass full criterion); PRF stream_id namespace (three-legged check REGENERATED: FIXTURE3 ALL MATCH); prefix-fingerprint boundary + §6.2 thresholds + common-case rule now literal in prereg §9.
ROUND 4 sent (ratification of new thresholds + GO/NO-GO). Awaiting verdict.

## 2026-09-11 — PRE-OUTCOME AMENDMENT 1 (before any seed generation)
scripts/cohort_seedgen.py: two-phase orchestration flow (world/grammar/banks -> stream hashes -> manifest). Operational fix only — the one-shot flow could not run because stream hashes require the world that seedgen itself discovers. No scientific content, seed rule, threshold, or gate changed. Amended freeze inventory + deposit follow; classified pre-outcome amendment per PREREG deposit-and-timing clause. No seed value has been generated at the time of this amendment.
## PRE-OUTCOME AMENDMENT 2: seedgen filler section updated to the round-2 per-length bank API (build_length_banks). Orchestration-only; frozen filler constructor untouched; no seed regenerated by this change (seed values derive solely from seed_rule.py). World candidate 0 (op 3843534058) VALID first try — no rejections to ledger beyond denylist skips (none: candidate 0 not dev-exposed).

## 2026-09-12 — reader comment on parent preprint (idle-cell relay pressure) — FOR THE COHORT PAPER
A reader observed that under the mixed-depth curriculum, restricted idle (forwarder) cells can only reduce loss by relaying the packet faithfully, while globally visible idle cells can recompute — so restriction bundles "courier practice" with visibility. Ruling for our write-up:
1. NOT a confound within the cohort: depth mix is bit-identical across all 5 conditions (frozen, hash-verified). Whether idle cells must relay is determined by the SAME manipulated variable (readable foreign evidence): R-/R+/N+ idle cells must relay; G-/G+ idle cells can shortcut. His mechanism is a refinement of HOW evidence removal works, not a rival factor.
2. Development records answer his "why not full depth": the atom-free protocol memorized and stayed at chance on primitives/held-out (REVIEW_d2_v212_freeze_ruling.md approved ledger wording); shallow episodes fixed it within ~4k updates. A depth-3-only arm would likely produce non-learners.
3. TODO for cohort paper: discussion paragraph presenting idle-relay as mechanism refinement, credit the commenter; link to value-canonical audit evidence (§6.1) as what forced relaying is expected to build.
4. FOLLOW-UP REGISTER: idle-fraction manipulation — vary the proportion of idle-cell episodes while keeping atoms, decoupling courier practice from evidence visibility (added to task #38 queue).

## 2026-09-12 — IDEAS LEDGER: language-acquisition framing (Narcis brainstorm)
The curriculum + relay findings map onto language-emergence literature with unusual precision:
1. "Starting small" (Elman 1993): full-complexity-only training -> memorizers; simple-first ramp -> learners. Matches our atom-free failure exactly.
2. Depth-0 pass-along rounds = naming games (Steels, Lewis signaling): grounding a shared code for the basic currency before composition.
3. Bottleneck-as-teacher (Kirby iterated learning): compositional structure emerges BECAUSE meaning must squeeze through a narrow channel. Masking = forcing use of the packet bottleneck instead of routing around it. Punchline framing: humans lack telepathy — that constraint may be why language exists; our global cells GOT telepathy and it ruined them.
4. Passive vs interactive exposure (Kuhl live-tutor vs video phoneme learning): rich passively-visible input teaches ~nothing; constrained interactive exchange teaches — a developmental twin of G+ vs R+.
USES: (a) LinkedIn/pop framing for the cohort result; (b) discussion-section paragraph (analogy, carefully bounded, with citations); (c) FOLLOW-UP: iterated-learning POPULUS — transmit packet dialects across generations of societies through a bottleneck, test whether the code sharpens/compositionalizes (Kirby-style chains). Pairs naturally with the idle-fraction manipulation.

## 2026-09-12 — COHORT PAPER TITLE (chosen by Narcis)
"What You Can't See Is Still What You Learn: A Preregistered Sixty-Society Confirmation That Evidence Masking Drives Compositional Generalization"
Framing directive: lead with the confirmed law (masking -> generalization, tested across every combination). N+ = a third implementation of masking (informational, not attentional) and a robustness probe — supporting cast in the abstract, NOT the headline. Labels-unread + N+ bimodality in the abstract; no overindexing on the gibberish arm.

## Standing directive (2026-09-12, pre-audit-completion)
User authorization: once the 30-job audit phase completes, proceed DIRECTLY to full paper
drafting from paper/cohort/main_skeleton.tex without waiting for user sign-off. While drafting,
post plain-language findings summaries as audit results are digested. Sequence: harvest final
audits -> verify-destroy boxes -> findings rundown -> section 5 (mechanism/dialect) -> full
draft -> referee pass -> user read -> arXiv (final Submit click = user, always).

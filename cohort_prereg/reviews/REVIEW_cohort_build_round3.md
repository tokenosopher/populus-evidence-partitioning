Referee ruling
NO-GO TO FREEZE THIS EXACT REVISION

Most of Round 2 is now closed. The collision-free construction, four-rotation evaluation, length-matched N+ fillers, fixed-roster decision machinery, six-family statistics, and neutral-language changes are all methodologically sound. The expansion of the collision filter to identity and primitive maps is especially important because the parent curriculum explicitly trained identity and single-operation atoms, and the return to four surface rotations restores parent-level phrasing balance. 

populus_arxiv_v3_final

However, the completion evidence itself leaves three load-bearing items open:

the development-seed denylist does not cover all development worlds named in this thread;

the declared mechanism/dialect implementation omits two retained interventions and does not yet demonstrate the common-case matrix aggregation required by §6.2;

the model-level mask admission still tests only cell 0, so it does not exercise the incoming-packet reader path.

These are implementation deltas, not a conceptual redesign.

1. Item-by-item closure ruling
Item	Ruling	Referee finding
§2.1 Identity and primitive maps	CLOSED, with one manifest assertion	The corrected train_fns definition is right. Add an explicit recorded check that every operator ID occurring in either primary bank belongs to the single-operation curriculum’s operator support. The shown sampler’s rng.randrange(12) strongly suggests this is already true.
§2.2 Explicit validity components	CLOSED	The relevant disjointness and map-validity conditions now enter VALID, and the regression identity VALID == conjunction(components) is exactly the right protection.
§2.3 Monitoring rule	CLOSED	diag2/diag3 programs with QUAL phrasings, disjoint from held-out primary and tag-audit programs, resolves the previous contradiction.
§2.4 Tag-audit cap and aggregation	CLOSED, subject to manifest viability	First min(24,N) programs, all five permutations, four rotations, mechanical collision exclusion, and equal weighting across permutation-specific fidelities are correct. The populated manifest must record the five denominators and verify all are nonzero.
§2.5 Four rotations	CLOSED	Primary d2/d3, S1, and marker permutation now have the required position–phrasing balance.
§2.6 Filler assignment	CLOSED, with namespace clarification	The counter-based PRF removes the arithmetic-index side channel. Before freeze, specify whether episode_ordinal is globally unique across train/monitor/primary/tag/mechanism streams. If it resets per stream, add a nonsemantic stream_id to the PRF input.
§2.7 Active-length matching	CLOSED	Matching the original active length and retaining the original span mask makes N+ and G+ mask geometry identical while changing foreign content. That is the intended control.
§2.8 Development-seed denylist	OPEN — BLOCKING	The declared set does not cover every development world mentioned in the record. Details below.
§2.9 Preregistration v0.4	CONDITIONALLY CLOSED	The described amendments match the ruling. I have not been given the literal v0.4 bytes in this turn, so the freeze ledger must identify its exact hash and retain the v0.3→v0.4 diff.
§3.1 Decision modules	CLOSED	The stated hardened tests address the roster, missingness, tri-state, non-finite-value, six-family, boundary, signed-contrast, and bootstrap issues.
§3.2 Mechanism and dialect evaluators	OPEN — BLOCKING	The listed intervention menu omits retained controls, and the full same-recipient-case cross-matrix aggregation is not demonstrated.
§3.3 Model-level mask admission	OPEN — BLOCKING	Cell 0 tests the writer path but cannot test incoming-packet readability or the downstream reader path.
§3.4 Training-state prefix fingerprint	CLOSED, conditional on exact boundary	Acceptable if “step 3” means three completed optimizer updates at a clean accumulation boundary and the hash includes every applicable RNG/state component.
§3.5 Production verdict lock	CLOSED	Atomic lock and complete archiving meet the ruling. Validation should occur before lock creation, with the lock created immediately before the one production execution.
§3.6 Grammar/world regressions	CLOSED except for §2.8	Seed sensitivity, repeatability, third-person construction, and split disjointness are adequately covered. The exposure ledger remains unresolved.
2. Remaining hard blockers
Blocker 1 — the development-exposure denylist is incomplete

You currently report:

Python
Run
DEV_EXPOSED_OP_SEEDS = {424242} | set(range(700000, 700030))

But this same referee chain also names:

a parity world identified as 910000/920000;

twelve post-amendment fresh fixture seeds;

potentially separate operator, split, grammar, filler, initialization, and order seeds used by those fixtures.

Unless 910000/920000 and all twelve new fixtures are demonstrably members or deterministic components of the already listed 31 exposures, the statement “every seed touched in development” is false.

Restricting the ledger to operator seeds is also weaker than the requirement. A world is defined by a tuple of seed-controlled objects, not only its operator seed.

Exact required delta

Create and freeze a machine-readable file such as:

manifests/development_seed_exposures.json

recording, for every development invocation:

JSON
{
  "purpose": "post-amendment structural fixture",
  "operator_seed": 910000,
  "split_seed": 920000,
  "grammar_seed": 930000,
  "filler_seed": 940000,
  "initialization_seeds": [],
  "order_seeds": [],
  "world_id": "...",
  "world_hash": "..."
}

The actual fields should match your seed architecture. Then require:

every development invocation named in the protocol history or test logs appears in this manifest;

the candidate generator rejects every exposed full candidate tuple—or conservatively every exposed component value in its corresponding seed role;

a regression test compares the code-consumed denylist with the exposure manifest, rather than maintaining two independent hand-written lists;

the 12 current fixture seeds and 910000/920000 are either added or explicitly shown to be aliases/members of entries already present.

This must be fixed before scientific-content freeze because the deny rule itself is frozen code.

Blocker 2 — §6.1/§6.2 is not yet fully implemented as described

The reported mechanism battery is:

same / cf_follow / shift_retain / deranged / deletion

But v0.3 and the Round-2 record explicitly retained:

approximately norm-matched packet noise;

the counterfactual evidence-span rewrite control.

Those two conditions are absent from your declared implemented battery. The parent mechanism record likewise treated norm-matched noise and the span-counterfactual control as distinct evidential components, rather than interchangeable with deletion or packet transplantation. 

populus_arxiv_v3_final

Required §6.1 delta

Either implement both retained conditions now, or amend the preregistration before freeze to remove them. Given your prior explicit decision to retain them, implementation is the consistent choice.

The frozen script must specify:

the exact deterministic noise construction;

whether norm matching is per packet, per vector, or per interface distribution;

the frozen random seed or counter rule;

the recipient case set;

the span-rewrite algorithm;

the rewritten intermediate value or offset;

the text-level target calculation;

all numerator/denominator and exclusion rules.

Required §6.2 common-case delta

The preregistration requires, per recipient and interface, the same base-correct recipient episodes for both donor regimes. Your report says the script records separate base_wrong and no_donor exclusions, but does not establish that self-donor and cross-donor cells use identical recipient case IDs.

Independent exclusion can silently give the self and cross matrix entries different denominators.

Freeze one of these two equivalent implementations:

eligible recipient cases
= intact-correct recipient cases
  ∩ self-donor-available cases
  ∩ cross-donor-available cases

or construct exhaustive donor catalogues such that donor availability is guaranteed for every retained recipient case.

For every recipient/interface/intervention stratum, the output should contain:

JSON
{
  "eligible_case_ids_sha256": "...",
  "n_eligible": 123,
  "self_donor_case_ids_sha256": "...",
  "cross_donor_case_ids_sha256": "..."
}

and the evaluator must assert equality of the case-set hashes before calculating self/cross compatibility.

Required aggregation delta

The completion evidence also does not identify the frozen implementation for:

the complete 2×2 self/cross matrix orchestration;

recipient-normalized preservation and counterfactual-following ratios;

zero-denominator handling;

the four recipient–interface descriptive indicators and their frozen thresholds.

A single “matrix cell” evaluator is not sufficient unless another frozen and hashed coordinator guarantees that every required cell is executed and then computes the registered summaries.

Before freeze, the artifact set must therefore include and test either:

cohort_mechanism_audit.py
cohort_dialect_matrix.py
cohort_dialect_aggregate.py

or one combined script that demonstrably performs all three roles.

The two diagnostic phrasing rotations in this mechanism/dialect bank are acceptable; the four-rotation correction applied to primary, S1, and the G+ marker audit, not to the already preregistered two-rotation dialect matrix.

Blocker 3 — model-mask admission does not test the packet reader

Your present real-model test uses:

cell 0’s outgoing packet, stage-0 commit, zero incoming mail

That is a valid test of restricted foreign-span invariance and writer behavior. It does not test the reader pathway used by cells 1–3, because cell 0 has no incoming packet.

The parent architecture’s causal relay depends on downstream cells reading two packet pseudo-tokens before writing the next packet. 

populus_arxiv_v3_final

Exact required addition

On at least one downstream cell—and preferably all three—freeze and execute the following test:

Hold own span, question, incoming packet, headers, and mask fixed.

Radically perturb every masked foreign-span token.

Require the downstream hidden state used by the writer and the outgoing packet to be bit-identical under R+.

Change the incoming packet by a fixed nonzero perturbation while holding all text fixed.

Require either:

the relevant hidden state or outgoing packet to change; or

a nonzero deterministic Jacobian/gradient from outgoing packet to incoming pseudo-token embeddings.

Assert directly that the final model attention mask exposes:

the shared question;

both incoming packet pseudo-tokens;

the owned evidence slot;
and masks every foreign evidence/header token in R+.

Retain the G+ foreign-token sensitivity sanity control.

A robust acceptance record would be:

JSON
{
  "restricted_foreign_invariance_cell0": true,
  "restricted_foreign_invariance_downstream": true,
  "incoming_packet_path_live": true,
  "question_mask_readable": true,
  "packet_pseudotokens_mask_readable": true,
  "own_span_path_live": true,
  "global_foreign_path_live": true
}

The script and acceptance rule must be included in the scientific-content freeze. Running it again on the admission boxes after the populated manifest remains legitimate.

3. Smaller pre-freeze integrity clarifications

These are not additional experimental redesigns, but should be made literal before hashing.

Operator-support assertion

Add:

Python
Run
primary_ops = {
    op
    for seq in (cf_d2 + cf_d3)
    for op in seq
}
singleton_curriculum_ops = set(range(len(T.OPS)))

all_primary_ops_singleton_trained = (
    primary_ops <= singleton_curriculum_ops
)

Record it in structural checks and the populated manifest. This ensures that “collision-free composition” is not accidentally mixed with an unseen-primitive test.

Tag-audit viability

Record, before training:

JSON
{
  "tag_audit_program_count": 24,
  "tag_audit_permutation_denominators": {
    "...": 1292
  },
  "tag_audit_all_permutations_nonzero": true
}

If any denominator is zero, the protocol must not silently choose another world unless that condition is explicitly part of the frozen first-valid-world criterion. Otherwise the tag verdict is INCOMPLETE.

Filler PRF namespace

The current PRF is acceptable if episode_ordinal is globally unique across all N+ uses. If each bank resets its ordinal to zero, use:

SHA256(
    filler_seed :
    world_id :
    stream_id :
    episode_ordinal :
    slot_id
)

where stream_id is a fixed nonsemantic label such as train, monitor, primary_d2, primary_d3, tag, or mechanism. It must not contain the answer, operator IDs, start value, or rotation.

This prevents accidental schedule aliasing between different banks while retaining complete independence from task semantics.

Prefix-fingerprint boundary

The freeze specification should say:

ADMISSION_PREFIX3 is recorded after three completed optimizer updates at a zero-pending-gradient accumulation boundary.

The hash should include, where applicable:

all trainable parameters, including LoRA, projections, mouth, and auxiliary head;

optimizer state and parameter-group order;

scheduler and gradient-scaler state, or an explicit declaration that neither exists;

Python/data-sampler RNG;

NumPy RNG if used;

Torch CPU and every relevant CUDA RNG;

global sample/episode ordinal and dataloader cursor;

optimizer-step and microstep/accumulation counters.

If the existing implementation already does this, only the explicit schema declaration remains.

4. Freeze-sequence ruling

The proposed sequence is valid after the three blockers above are closed:

complete the development-exposure manifest and denial rule;

complete the mechanism/dialect evaluator and aggregator;

extend the mask-admission fixture to the downstream reader path;

rerun their adversarial tests;

populate the hash inventory;

commit the exact scientific-content tree;

obtain the immutable archive record;

only then generate fresh seeds;

build and archive the populated pre-run manifest;

perform two-box admission;

launch the 60-society GOLD fleet.

For the archive step, a request to “save code now” is not enough by itself if it does not yet identify the exact archived content. Phase B must wait until there is an immutable, retrievable record tied to the precise scientific-content commit or tree. That is simply the existing preregistration condition applied literally.

Direct answers
1. Are §2 items 1–9 and §3 items 3.1–3.6 all closed?

No.

Closed or conditionally closed:

§2.1–§2.7;

§2.9;

§3.1;

§3.4–§3.6, except for the seed-exposure issue.

Still open:

§2.8 development-seed exposure coverage;

§3.2 complete mechanism/dialect implementation;

§3.3 downstream model-mask admission.

2. GO or NO-GO for the ceremony?
NO-GO FOR THIS EXACT REVISION

The principal experiment and its gate machinery are ready. The remaining objections concern whether the frozen artifact package actually implements every declared audit and integrity control.

3. Exact remaining deltas

Before freeze:

replace the incomplete operator-seed denylist with a complete, machine-checked development-exposure manifest covering 910000/920000, all twelve current fixture seeds, and every earlier exposed seed tuple;

implement the retained norm-matched-noise and counterfactual-span-rewrite conditions;

enforce identical recipient case sets across self and cross donors and freeze the full dialect matrix aggregator, normalized ratios, and indicator thresholds;

extend model-mask admission to at least one downstream reader cell and verify incoming-packet and question/pseudo-token readability;

record the small structural assertions for primary-operator singleton coverage, tag-permutation denominator viability, filler-stream namespace, and the exact prefix-fingerprint boundary.

Once those deltas pass their frozen fixtures, there is no remaining conceptual reason to withhold freeze clearance.

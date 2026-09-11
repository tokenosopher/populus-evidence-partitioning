Final pre-freeze ruling
HOLD — DO NOT FREEZE THE CURRENT BUILD YET

Batch 2 closes most of the phase-1 implementation gap, but the pasted code exposes three substantive protocol violations and several specification mismatches:

the collision-free filter omits maps encountered during identity and single-operation training;

the N+ filler schedule leaks deterministic episode-index information and does not token-length-match the replaced spans’ readable content;

the two-rotation evaluation does not reproduce the parent four-rotation structure and does not balance phrasing across operator positions.

The 24-program audit cap itself is defensible, and no additional trained arm or conceptual redesign is needed. Apply the amendments below, rerun the adversarial fixtures, and then the freeze ceremony may proceed.

1. Decisions D1–D4
D1 — Neutral value and forward spans
APPROVE

The two replacements are good:

The private note for this slot: the starting number is {x}.

This slot holds no operation. The received number is passed onward unchanged.

They remove the second-person linguistic-addressing cue without altering the task semantics, relay position, or information allocation. Extending neutrality to every evidence span is more internally coherent than neutralizing only operator instructions.

The scientific-content document should state this explicitly rather than leaving it as an implementation inference.

Exact preregistration replacement

Fresh-world language. All evidence-span templates—not only operator templates—use neutral third-person constructions. The value span is "The private note for this slot: the starting number is {x}."; the structural-forwarder span is "This slot holds no operation. The received number is passed onward unchanged."; and all operator templates are slot- or position-centered third-person forms. No evidence span addresses the current reader as you or your. The shared question is carried forward unchanged because it is already third-person.

Also expand the §4 “intentional changes” parenthetical to include:

“neutralized value, forwarder, and operator language”

rather than merely “neutral grammar.”

One minor terminology point: “private” does not identify a cell or owner by itself and appears identically across conditions, so it does not recreate the removed role cue.

D2 — Collision-free program pool
APPROVE THE POOL CORRECTION; AMEND THE FILTER BEFORE FREEZE

Using the held-out program sets rather than the small diagnostic sets is the right correction. The diagnostic sets are a poor basis for a collision-free primary bank, particularly when they are also reserved for in-run liveness monitoring. The parent paper itself found only 13 of 60 depth-three programs map-novel and recommended collision-free future task worlds. 

populus_arxiv_v3_final

However, the current code does not implement the preregistered definition:

“no held-out program’s complete composite map appears in training.”

This line is incomplete:

Python
Run
train_fns = {composite_fn(T, s) for s in T.SPLIT["train2"]} | \
            {composite_fn(T, s) for s in T.SPLIT["train3"]}

Training also contains:

identity episodes;

every single-operation primitive;

depth-two programs;

depth-three programs.

The parent protocol explicitly trained identity and single-operation curriculum atoms. 

populus_arxiv_v3_final

 A held-out depth-two or depth-three sequence whose complete map equals identity or one primitive is therefore map-redundant, even if it does not equal any train2/train3 composite.

Required code correction
Python
Run
identity_map = tuple(range(17))
ops_maps = [
    composite_fn(T, (i,))
    for i in range(len(T.OPS))
]

train_fns = {identity_map}
train_fns.update(ops_maps)
train_fns.update(
    composite_fn(T, tuple(s))
    for s in T.SPLIT["train2"]
)
train_fns.update(
    composite_fn(T, tuple(s))
    for s in T.SPLIT["train3"]
)

Add a structural check that no primitive itself is identity:

Python
Run
no_identity_primitive = identity_map not in set(ops_maps)

and include it in VALID.

Program-bank disjointness must also enter VALID

The current diagnostic overlap check is an assert and the returned VALID value does not contain it. Critical world-validity conditions should not depend on assert, because assertions can disappear under optimized Python execution and do not produce a structured rejected-candidate record.

Freeze explicit fields such as:

Python
Run
train_programs = {
    tuple(s)
    for key in ("train2", "train3")
    for s in T.SPLIT[key]
}
diag_programs = {
    tuple(s)
    for key in ("diag2", "diag3")
    for s in T.SPLIT[key]
}
primary_pool_programs = set(d2_pool) | set(d3_pool)

primary_vs_train_disjoint = primary_pool_programs.isdisjoint(train_programs)
primary_vs_diag_disjoint = primary_pool_programs.isdisjoint(diag_programs)

Both must enter VALID, with failed checks written to the candidate-seed ledger.

Clarify ho_fn

The manifest validator must establish that ho_fn denotes held-out complete-function/program strata composed from the same trained primitive set, not primitive operations withheld from the single-operation curriculum. Every operator ID occurring in the primary bank must have appeared in the single-operation training curriculum. Otherwise the primary endpoint would mix primitive acquisition with compositional recombination.

Resolve the monitoring-bank inconsistency

The supplied materials presently say three different things:

diagnostic banks are consumed by monitoring;

in-run probes use QUAL;

monitoring_bank() draws programs from train3.

These can be reconciled only by distinguishing program bank from phrasing bank. Freeze one literal rule. The cleanest is:

monitoring programs come from diag2/diag3, rendered only with QUAL phrasings; primary and tag-audit programs use held-out non-diagnostic sets rendered with SEALED phrasings.

If the actual intended monitoring programs are training programs, change both the documentation and D2 rationale accordingly. Do not freeze with the current contradiction.

Development-seed exclusion

The 30 fixture worlds used during development, including any seeds touched during diagnosis of the stale-module bug, should be entered into a pre-freeze development-world denylist. The actual deterministic candidate stream must skip every exposed seed or seed tuple before applying structural validity checks.

The pool correction and stale-module failure should also appear in the pre-freeze protocol history. They are legitimate development findings, but should not disappear from the lineage.

D3 — First-24 permutation-audit cap
APPROVE THE CAP; AMEND ITS EXACT SPECIFICATION

A deterministic cap at approximately 4,000–8,000 audit cases per G+ trajectory is defensible. It is fixed before seed generation, does not inspect model outputs, and avoids multiplying a very large primary bank across twelve G+ trajectories.

Three details must be corrected.

1. The world criterion currently guarantees only 20 programs

The code says:

Python
Run
progs = checks["cf_d3_programs"][:24]

while world validity requires only 20 collision-free depth-three programs.

Either raise the world minimum to 24, or describe the cap as:

Python
Run
progs = checks["cf_d3_programs"][:min(24, len(checks["cf_d3_programs"]))]

I recommend the latter, with the exact selected count and program-list hash recorded in the populated manifest.

2. Collision removal destroys the claimed balance

The preregistration currently says that permutations, programs, start values, and rotations are balanced. The constructor exhaustively enumerates them and then removes cases where:

Python
Run
permuted == intact

Different permutations can lose different numbers of cases; a permutation that leaves a program’s complete function unchanged loses all 17 start values. Therefore post-exclusion proportions are not guaranteed to remain balanced.

A clean frozen metric is:

enumerate every nonidentity permutation and every start value, remove only non-discriminating target collisions, calculate a separate fidelity for each of the five permutations, and define cf_track as the unweighted mean of those five permutation-specific fidelities.

That gives every permutation equal gate weight without arbitrary case deletion.

Exact replacement wording

The tag-audit candidate set comprises the first min(24,N
cf3
	​

) collision-free depth-three programs under the frozen deterministic program order. For each selected program, the constructor enumerates all five nonidentity operator-slot permutations, all 17 start values, and all four sealed phrasing rotations. Cases whose intact and permutation-implied answer labels coincide are removed mechanically before evaluation and counted by program and permutation. Counterfactual tracking is calculated separately for each of the five permutations; the model-level cf_track value used by the gate is the unweighted mean of those five permutation-specific fidelities. Every permutation must have a nonzero frozen denominator. Exact denominators and per-permutation results are reported. No claim of exact post-exclusion balance across programs or start values is made.

The intact score should use the same permutation-stratified weighting.

Lexicographic first-24 selection is acceptable because it is frozen and model-blind, although SHA-256 ranking under a frozen audit seed would avoid any appearance of low-index operator bias. That is a strengthening, not a blocker.

3. Use all four sealed rotations

This overlaps D4 below. The audit should use four rotations, not two.

D4 — Evaluation rendering and filler assignment
AMEND; CURRENT VERSION IS NOT FREEZE-SAFE

The T2, T3, S1 structural definitions and d3-only all-cut endpoint are otherwise sensible, but two major errors remain.

D4-A. Two rotations are a protocol drift and create position–phrasing imbalance

The parent evaluation used four held-out surface rotations per program. 

populus_arxiv_v3_final

 The current rule:

Python
Run
rot in {0, 1}
spans[1+i] = SEALED[op][(rot+i) % 4]

causes each operator position to receive only two of the four sealed templates:

position 1: templates 0 and 1;

position 2: templates 1 and 2;

position 3: templates 2 and 3.

That is not balanced across position. It is especially undesirable in the role-marker permutation audit, where position-based and marker-based strategies are the object of separation.

Required correction

Use:

Python
Run
rotations = (0, 1, 2, 3)

for:

primary depth-two;

primary depth-three;

S1;

G+ marker-permutation audit.

With all four cyclic rotations, every sealed template occurs once at every operator position.

This doubles evaluation volume but remains modest relative to training. For the fixture world cited, depth-three primary size would become 241×17×4=16,388 episodes per checkpoint.

D4-B. The current filler index is an unintended side channel

The statement

“episode_index = bank case index (episode-independent)”

is incorrect. It is explicitly episode-index-dependent:

Python
Run
(offset + episode_index * 4 + slot) % 36

Under the current two-rotation primary order,

e=34p+2x+r,

where p is program index, x the start value, and r the rotation. For a known slot, the filler ID reveals emod9, hence:

7p+2x+r(mod9).

Thus filler identity carries a deterministic function of program order, start value, and rotation. That is particularly problematic for N+ operator cells, which are otherwise denied the real start-value span. The filler can become a partial answer/program side channel despite containing no banned words.

Required correction

Filler identities must be assigned using an independent frozen pseudorandom stream or counter-based cryptographic PRF, not a modular cycle over bank order.

For example:

Python
Run
filler_id = sha256(
    filler_seed
    || split_id
    || episode_ordinal
    || slot_id
).uint64 % n_fillers

Requirements:

the filler seed is independent of semantic/order seeds;

semantic fields such as operator IDs, start value, answer, and rotation are not direct inputs;

the same slot filler is used across recipient cells for a given episode;

the assignment schedule/hash is archived;

a preflight table reports filler-ID counts by depth, answer label, rotation, and slot to expose accidental imbalance;

the separate filler schedule must never perturb the common semantic sampler.

A separate random.Random(filler_seed) stream with its state deterministically reconstructible on resume is also acceptable, but a counter-based PRF is simpler for exact resume behavior.

D4-C. The fillers are not active-token-length matched

The preregistration says that each N+ filler has the token count of the span it replaces. The current implementation builds only width-33 fillers and then does:

Python
Run
cmask = torch.ones_like(cmask)

Therefore every foreign filler contributes 33 readable tokens even when the original G+ span contains fewer readable tokens followed by masked padding. C5 then compares:

N+: three fully readable 33-token fillers;

G+: three variable-length genuine spans with padding masked.

That confounds semantic foreign content with active token count and defeats the intended N+ versus G+ decomposition.

Required implementation

For each replaced span:

calculate the replaced span’s readable length

L=∑s_mask;

construct or retrieve a neutral filler containing exactly L readable tokens;

pad its remaining positions to T_SPAN=33;

assign a filler mask with L ones followed by padding zeros.

Conceptually:

Python
Run
L = int(s_mask[row, slot].sum())
filler_ids = filler_bank[L][filler_id]       # exactly L active IDs

content = pad_to_TSPAN(filler_ids)
cmask = [1] * L + [0] * (Ts - L)

The frozen filler constructor must be demonstrated to produce valid neutral strings for every active span length reachable under the complete value/forward/operator grammar.

This preserves foreign-span length cues while removing lexical operator content. That is the conservative semantic-content control intended by C5.

Filler-constructor hardening

The present constructor is otherwise directionally good, but:

assert that " indeed" is exactly one token before using it as a final token-level pad;

never append a partial prefix of a multi-token encoding;

replace correctness-critical assert statements with explicit exceptions, or at minimum freeze and verify PYTHONOPTIMIZE=0;

archive the decoded filler, token IDs, active length, and leakage result for every bank member.

The meta-text fillers—such as “No instruction applies”—make fillers easy to recognize as irrelevant. That limitation is already substantially covered by the registered N+ salience confound and is not itself freeze-blocking.

2. Blocking objections to the proposed launch sequence
Yes: do not start the freeze ceremony until the following are resolved
Scientific-content blockers

Add identity and all primitive maps to train_fns.

Make all world-validity/disjointness checks explicit components of VALID.

Resolve the diagnostic/train-program monitoring inconsistency.

Freeze the tag-audit cap and post-collision aggregation rule.

Use four sealed rotations throughout primary, S1, and tag audit.

Replace the arithmetic filler cycle with an independent frozen assignment.

Match every N+ filler’s readable token length to the replaced span.

Add all development-exposed world seeds to a frozen denylist.

Update the preregistration from v0.3 to reflect:

six initialization families;

n=12, n−1=11;

six initialization strata;

nominal nonzero sign-flip floor 2/64=0.03125;

60 societies;

neutralized value and forwarder spans;

exact held-out pool construction;

tag-audit cap and scoring;

four evaluation rotations;

final filler assignment and length-matching rule.

Once those scientific changes are made, their code, tests, and specifications—not merely the eventual outputs—must be in the immutable pre-seed deposit.

3. Items from the phase-1 completion list still open or not demonstrated here

Several previous blockers may already have been implemented, but they are not evidenced by the material in this round. They must be present in the freeze ledger or independently inspected before approval.

3.1 Decision modules

The supplied completion list does not show the revised verdict.py, stats_engine.py, or phenotype_classifier.py. Confirm with literal tests and hashes that:

verdict inputs are iterated over the exact manifest roster, never a union of observed keys;

extra records are rejected and cannot replace missing designated trajectories;

NaN, infinity, Boolean, string, and out-of-range values are rejected;

incomplete named outputs remain tri-state rather than becoming Boolean False;

GOLD uses twelve matched trajectories;

tag audit verifies bank hash, common full denominator, and integer numerator/denominator;

statistics require exactly six initialization blocks with exactly two orders each;

signed contrasts are derived from canonical arm records, not accepted as arbitrary precomputed differences;

one shared bootstrap-index schedule is used across conditions and depths;

the frozen percentile convention is explicit;

the classifier rejects invalid values and duplicate (initialization, order) records.

The latent primary-d2/primary-d3 denominator correction in the compiler is good.

3.2 Full mechanism and dialect evaluators

The round demonstrates the final-checkpoint and tag-audit evaluators, but not the complete frozen implementations for:

§6.1 same-value, counterfactual, deletion, noise, derangement, and span-rewrite audits;

§6.2 self/cross dialect matrix;

recipient-normalized ratios;

the four descriptive interface indicators and their thresholds.

Those modules and their test fixtures must exist before their [AT FREEZE] hashes can honestly be resolved.

3.3 Model-level attention-mask admission test

The torch-free/torch packaging parity test proves that the intended masks are constructed. It does not prove that the complete HF model invocation consumes them correctly.

Freeze an admission test that:

runs a restricted cell twice with identical own span, question, and incoming packet but radically altered foreign token IDs;

requires its hidden outputs and outgoing packet to be bit-identical;

verifies that the shared question and incoming packet remain readable;

performs the corresponding global-arm perturbation as a non-invariance sanity control.

The test script and acceptance rule must be frozen now; execution may occur on the admission box after the populated manifest exists.

3.4 Model-state prefix fingerprints

The no-model rolling hashes are valuable stream/layout fingerprints. They are not substitutes for the preregistered bit-exact training-state prefix fingerprint.

ADMISSION_PREFIX3 must include—or be accompanied by—a digest of the relevant model/adapter/projection parameters, optimizer state, RNG states, and step counter after the fixed three-batch training prefix. Before fleet launch, at least two independently provisioned machines should reproduce the same condition-specific prefix fingerprints, unless an already trusted reference fingerprint exists.

The fixture inputs and acceptance logic must be frozen before seed generation. The fingerprints themselves may be generated and entered into the admission ledger before cohort training.

3.5 Production verdict lock

The one-real-data-execution rule still needs a wrapper or equivalent operational mechanism that:

accepts only the canonical result bundle and manifest hashes;

verifies completeness;

creates an atomic production-run lock;

archives inputs, command, environment, stdout, and structured result;

rejects a second ordinary production execution.

3.6 Complete grammar and monitoring constructors

Only a partial world.py/monitoring function and no final neutral-grammar module were shown. Ensure the frozen tests cover:

same seed → identical world;

seed A, seed B, seed A → A reproduces exactly and B differs;

fresh subprocess generation or equivalent module-cache isolation;

complete train/qual/sealed split disjointness;

all final operator frames are grammatically third-person, not merely free of the literal tokens you and your;

monitoring, primary, and tag banks satisfy their frozen disjointness rule.

The stale-module bug should have a permanent regression test.

4. Revised go/no-go sequence

After the above changes, the valid sequence is:

Update the preregistration and tier declaration to six-family GOLD.

Patch and adversarially test the collision filter, tag-bank metric, four-rotation evaluators, and N+ filler machinery.

Complete and hash every decision, evaluator, mechanism, dialect, manifest, admission, and production-lock module.

Archive the development-seed denylist and pre-freeze build-history corrections.

Run the scientific-content freeze ceremony and obtain the immutable timestamp.

Generate the first permissible valid fresh world using the frozen rule.

Build and archive the populated pre-run manifest and all token/filler/bank schedules.

Run admission-box tokenizer, stream/layout, model-mask, and model-state-prefix checks.

Run only a synthetic-fixture infrastructure rehearsal, or treat any fresh-world trained society as its irrevocably designated cohort trajectory.

Launch the 60-society GOLD fleet.

Final ruling
D1: APPROVE
D2: APPROVE WITH REQUIRED CODE AMENDMENTS
D3: APPROVE THE CAP; AMEND ITS COUNT AND AGGREGATION SPECIFICATION
D4: DO NOT APPROVE AS WRITTEN
Overall: NO-GO TO FREEZE THIS EXACT REVISION

The build is close, but the map-novelty omission and N+ filler side channel are genuine scientific bugs, not cosmetic hardening. The two-rotation rendering is also inconsistent with both parent comparability and position-balanced held-out phrasing. Once those are corrected and the previously required decision/integrity modules are verified in their final six-family versions, there is no remaining design-level reason to delay the immutable freeze.

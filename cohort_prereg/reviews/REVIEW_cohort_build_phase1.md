# REVIEW: Cohort build phase-1 (referee, 2026-09-12)

I. Initialization-family decision

Use six initialization families. Initialization is the genuine independent replication unit, so moving from five to six adds 20% more independent clusters for trivial incremental cost and gives the role-marked result one additional opportunity to reveal between-family heterogeneity. It also removes the presentational oddity that perfect family-level directional unanimity is mechanically capped at p=0.0625, while leaving the preregistered effect-size gates—not NHST—as the verdict machinery. The parent study had positive mean differences in all five initialization strata and consequently attained the five-stratum floor of 0.0625 at both depths; the sixth family is therefore a clean strengthening of that evidential pattern, not a reaction to an unfavorable result. 

populus_arxiv_v3_final

 Nothing scientifically breaks if you retain five: you lose one independent replication and the possibility of a sub-0.05 exact descriptive result, but not the validity of the causal contrast or gates.

Adopting six requires, before freeze:

GOLD = 6 initializations × 2 orders × 5 conditions = 60 societies;

fixed design denominator n=12, hence n−1=11;

six initialization strata in §8;

nominal nonzero sign-flip floor 2/2
6
=0.03125;

TIER_N["GOLD"] = 12;

updates to TIER_DECLARATION.json, seed_rule.py, fixtures, budget text, and the BUILD_STATUS fleet count.

STANDARD and MINIMUM may retain their existing sizes if the preregistration explicitly keeps the sixth family as a GOLD-only change.

Bottom line: use six initialization families and freeze GOLD at 12 matched trajectories / 60 societies.

II. Phase-1 build review
Overall ruling

The gate arithmetic is mostly correct, but the current implementation is not yet safe to hash as the frozen decision machinery. There are four substantive blockers:

verdict.py has no fixed trajectory roster, so an undesignated extra run can replace a missing designated run.

None of the decision modules rejects NaN, infinities, malformed records, or wrong trajectory identities; under the current code, one NaN trajectory can still allow a directional comparison to pass.

neutral_grammar.py contains bare imperative templates, so it does not yet implement the preregistered “neutral third-person templates only” rule.

Trainer integration, token-exact filler generation, audit-bank construction, and canonical score compilation are scientific implementation—not merely admission-box checks—and must exist and be frozen before seed generation.

The preregistration remains approved. The current build bundle is a conditional no-go for freeze until the corrections below are made.

1. verdict.py
What is implemented correctly

The substantive formulas match §§5.2–5.5:

The signed advantaged arms are hard-coded correctly:

C1: R+ over G+

C2: G+ over G−

C4: R+ over N+

C5: N+ over G+

C6: R− over G−

The per-trajectory margin requires the same trajectory to meet Δ
2
	​

≥0.20 and Δ
3
	​

≥0.20.

The median paired differences require at least 0.25 at both depths.

The advantaged-arm median depth-three floor is 0.70.

c_norun_low correctly requires every advantaged run to satisfy both A
2
	​

≥0.40 and A
3
	​

≥0.40.

The all-cut condition correctly requires at least n−1 advantaged-arm trajectories at or below 0.11.

C3/C4/C5 use the frozen competence and bounded-difference criteria.

G+ tag use applies the per-model 0.90/0.90 rule and the arm-level n−1 rule.

No omnibus cohort pass is emitted.

The signed advantaged arm cannot be reversed through the JSON input.

Freeze blocker 1: no designated trajectory roster

_matched() ignores its n argument and constructs the evaluated set from:

Python
Run
keys = sorted(set(adv) | set(oth))

That is not the fixed-denominator rule. It fixes only a count, not the identities occupying that count.

For example, suppose one designated trajectory is missing from both arms but a stray key called EXTRA is present in both. The script sees ten complete rows, treats the comparison as complete, and includes EXTRA in the medians and counts. Likewise, if eleven rows are present under a ten-row design, all eleven are used.

The same vulnerability exists in:

directional_pass;

practical_null;

gplus_tag_use.

Required correction

The verdict process must read the exact designated trajectory roster from the hashed pre-run manifest, not infer it from result keys:

Python
Run
roster = tuple(manifest["designated_trajectory_ids"])

if len(roster) != n or len(set(roster)) != n:
    raise InputIntegrityError("invalid designated roster")

Every comparison must iterate only over that roster. Then:

missing or null designated record → INCOMPLETE;

unexpected extra record → INPUT_ERROR, never a substitute;

wrong condition/init/order identity → INPUT_ERROR;

more than n records → INPUT_ERROR, not a larger analysis set.

Prefer a CLI contract such as:

verdict.py PRE_RUN_MANIFEST.json CANONICAL_RESULTS.json

rather than trusting an independently supplied n and arbitrary dictionary keys.

The input "n" should either be removed or checked for exact agreement with the manifest and tier constant. It should never constitute a second source of truth.

Freeze blocker 2: malformed values can pass

All required scores must be validated as real, finite values in [0,1]. Python’s JSON parser accepts nonstandard NaN by default, and comparisons against NaN have dangerous behavior here.

With nine strong trajectories and one trajectory containing NaN for every metric:

the margin count can still be 9/10;

NaN < 0.40 is false, so the malformed run passes c_norun_low;

the all-cut count can still be 9/10;

the medians can remain high.

The complete conjunction can therefore return PASS even though one required evaluation is invalid. The preregistration requires INCOMPLETE, not use of the malformed trajectory as the one permitted behavioral exception.

Add a validator equivalent to:

Python
Run
import math

def valid_score(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
        and 0.0 <= x <= 1.0
    )

Better still, store integer numerators and denominators and calculate the rates inside the canonical result compiler. Reject NaN during JSON loading and emit JSON with allow_nan=False.

Freeze blocker 3: INCOMPLETE is collapsed into False

These lines are not faithful to the tri-state reporting rule:

Python
Run
out["C1_BEHAVIORAL_PASS"] = dec["C1"]["status"] == "PASS"
out["GPLUS_TAG_USE_PASS"] = tag["status"] == "PASS"
out["PRIMARY_ROLE_EQUALIZED_PASS"] = ...

An incomplete comparison becomes the Boolean False, indistinguishable at the named-output level from a completed failure.

Use either structured outputs or null for an unavailable Boolean:

Python
Run
def pass_value(status):
    if status == "PASS":
        return True
    if status == "FAIL":
        return False
    return None

The combined status should be:

INCOMPLETE if either required component is incomplete;

PASS only if both pass;

otherwise FAIL.

For example:

JSON
"PRIMARY_ROLE_EQUALIZED": {
  "status": "INCOMPLETE",
  "pass": null
}
Tag-audit denominator cannot be verified from the input

The current tag input supplies only:

JSON
{"intact": 0.93, "cf_track": 0.94}

The verdict script cannot tell whether those numbers were:

computed over the complete sealed bank;

success-conditioned;

computed over different denominators;

selectively filtered;

generated from the wrong bank.

The canonical tag record should contain at least:

JSON
{
  "bank_sha256": "...",
  "n_total": 1234,
  "intact_correct": 1150,
  "cf_correct": 1162
}

The script or frozen result compiler should verify that both rates use the same expected full-bank denominator and matching bank hash. A missing or invalid tag evaluation must cause INCOMPLETE; it cannot be treated as the one permitted valid model-level audit failure.

Other required output corrections

The code should additionally:

emit the exact designated trajectory IDs and signed Δ
2
	​

,Δ
3
	​

 values;

report the reverse-direction 0.20-both-depth count;

report signed medians even when a practical-null criterion fails;

retain all component values when returning UNINFORMATIVE_incompetent;

emit unrounded machine values or exact numerator/denominator pairs.

Rounding to four decimals only in a display layer is fine. The canonical machine verdict should not make a raw 0.69996 appear as 0.7000 alongside a failed threshold.

Six-family update

With the recommended sixth family:

Python
Run
TIER_N = {
    "GOLD": 12,
    "STANDARD": 6,
    "MINIMUM_DECISIVE": 10,
}

Using the preregistration’s canonical tier name rather than code-only "MINIMUM" would also reduce schema drift.

verdict.py verdict

Formula logic: pass.
End-to-end integrity: fail pending the roster, validation, and tri-state corrections.

2. phenotype_classifier.py
Classification tree

The six-class tree matches Appendix A exactly:

T = max(T2, T3);

all-cut is absent from classification;

first-match precedence is correct;

the thresholds use the proper strict and inclusive boundaries;

S1 is descriptive only.

The parent global outlier A
2
	​

=0.5674,A
3
	​

=0.8431 reaches none of classes 1–5 and correctly lands in intermediate_mixed. GT m200 lands in deep_only_generalization_inversion.

Required integrity hardening

The classifier still needs strict schema validation. At present, a NaN accuracy silently falls through to intermediate_mixed, which is not an acceptable phenotype for an invalid evaluation.

Require:

all five supplied metrics to be finite and in [0,1];

no duplicate (init, order) records;

exact correspondence with the designated roster;

exactly the two precommitted order IDs for every complete initialization;

unavailable trajectories to be reported as unavailable, never classified.

This line silently overwrites duplicate records:

Python
Run
by_init.setdefault(t["init"], {})[t["order"]] = ...

It should raise an integrity error when (init, order) already exists.

The initialization output should retain explicit order IDs rather than only:

JSON
"classes": ["class_a", "class_b"]

For example:

JSON
"orders": {
  "950": "full_generalizer",
  "951": "intermediate_mixed"
}

That makes the “ordered class pair” auditable without relying on lexical sorting.

phenotype_classifier.py verdict

Substantive classifier: approved.
Freeze after finite-value, duplicate, and designated-roster validation is added.

3. stats_engine.py
What is statistically correct

Given exactly two valid order differences for every designated initialization:

resampling initialization blocks with replacement is correct;

flattening the two-order blocks gives the same overall mean as averaging equally weighted initialization means because every block has equal size;

100,000 draws and seed 271828 match §8;

the sign-flip test correctly operates on initialization-level mean differences;

using the absolute sum rather than absolute mean does not alter the randomization p-value;

enumerating all 2
k
 sign assignments with tail inclusion is an exact two-sided test.

Freeze blocker 1: missing or malformed blocks are silently accepted

The module does not require:

the correct number of initialization clusters;

exactly two orders per initialization;

designated initialization identities;

finite differences.

A five-cluster GOLD input, a block with only one order, or an extra undesignated initialization will simply be analyzed. A one-order block also changes the bootstrap weighting because the flattened resample gives different block sizes different weights.

Before analysis, require exactly:

six clusters × two orders for six-family GOLD;

three clusters × two orders for STANDARD;

five clusters × two orders for MINIMUM-DECISIVE.

Missing data should produce an explicitly incomplete descriptive analysis, not silently change k or block weights.

Freeze blocker 2: arbitrary precomputed differences

The CLI accepts:

JSON
{contrast: {depth: {init: [d1, d2]}}}

Nothing guarantees that:

C1 was computed as R+ minus G+ rather than the reverse;

orders were paired correctly;

values came from the canonical final evaluations;

the interaction and C1=C4+C5 decomposition are internally consistent.

This reintroduces a pathway by which the “advantaged” direction can effectively be flipped outside verdict.py.

The stats engine should consume the same validated canonical arm-level result bundle and derive:

C1–C6 using the frozen sign map;

the marker × masking interaction;

the C1=C4+C5 trajectory-level identity.

Alternatively, a frozen canonical result compiler may produce the differences, but the compiler and its hash then become load-bearing and must verify the identity algebra exactly.

Percentile indexing needs an explicit convention

These indices:

Python
Run
lo = means[int(0.025 * draws)]
hi = means[int(0.975 * draws)]

are not the usual linear percentile convention and are off by roughly one order statistic from common inverse-CDF conventions. With many repeated bootstrap values the numerical result may often be unchanged, but the implementation should not call an unspecified index rule simply “the percentile interval.”

Freeze an explicit method. A dependency-free linear version is:

Python
Run
import math

def linear_quantile(xs, q):
    h = (len(xs) - 1) * q
    lo = math.floor(h)
    hi = math.ceil(h)
    if lo == hi:
        return xs[lo]
    w = h - lo
    return xs[lo] * (1 - w) + xs[hi] * w

Then use:

Python
Run
lo = linear_quantile(means, 0.025)
hi = linear_quantile(means, 0.975)

Record percentile_method="linear_(N-1)q" in the output.

Joint block resampling

Separate calls currently restart the same seed. When every contrast contains exactly the same sorted initialization IDs, this incidentally produces the same resampling indices across contrasts. Make that deliberate:

Generate one frozen bootstrap index schedule over the designated initialization roster.

Reuse it for all contrasts, depths, interactions, and decomposition summaries.

That directly implements “all conditions and both orders jointly.”

Sign-flip floor label
Python
Run
"min_attainable_p": 2 / total

is the nominal floor when all initialization means are nonzero and there are no additional maximizing ties. If a family mean is exactly zero, more sign configurations are equivalent and the attainable floor is higher.

Rename it:

Python
Run
"nominal_min_p_nonzero": 2 / total

The exact returned p-value itself is correctly counted.

Six-family update

For six-family GOLD, tests should verify:

k=6;

2/64=0.03125;

all six blocks contain two orders;

one absent order produces INCOMPLETE, not a five- or six-block partial test.

stats_engine.py verdict

Core bootstrap and sign-flip mathematics: correct.
Freeze readiness: no, pending roster validation, canonical contrast derivation, and a frozen percentile convention.

4. marker_filler.py and neutral_grammar.py
Native marker layout

The shown native header logic is correct:

tagged arms have one mine: and three other: headers;

placeholder arms have four slot: headers;

in an operator cell, the start-value header is automatically other:;

R± expose only the owned slot;

G± expose all genuine slots;

N+ exposes the owned genuine span plus the three filler slots.

Add hard validation that:

Python
Run
n_slots == 4
0 <= own_slot < 4

Otherwise own_slot=-1 or own_slot=4 silently creates a tagged input with no mine: header.

The function describes slot visibility but does not itself verify the actual token-level attention mask. Trainer integration must test that:

the shared question remains readable;

incoming packet pseudo-tokens remain readable;

restricted foreign headers and contents have exactly zero direct attention influence;

sequence positions and Transformer calls retain the declared parity;

N+ substitutes filler before exposing the foreign slot positions.

Freeze blocker: two grammar frames are imperatives

The preregistration specifies neutral third-person operator templates only. These shown frames violate that requirement:

Python
Run
"{core}."
"For this position, {core}."

They render as:

“Multiply the number by …”

and:

“For this position, multiply …”

Those are imperative constructions with an implicit second-person subject, despite containing no literal you or your. The regex therefore does not test the actual frozen requirement.

"This cell's operation: {core}." is also unnecessarily deictic and can be removed to avoid reintroducing cell-addressing language.

Exact replacement
Python
Run
FRAMES = [
    "The operation specified for this slot is to {core}.",
    "The rule for this slot is to {core}.",
    "This slot's operation is to {core}.",
    "For this position, the assigned transformation is to {core}.",
    "The assigned instruction at this position is to {core}.",
    "At this slot, the specified rule is to {core}.",
]

These retain six surface frames while making every frame slot-centric and third-person.

The pasted neutral_grammar.py ends partway through build_split(). I therefore cannot certify:

the seeded permutation;

filtering order;

count allocation;

disjointness;

insufficient-candidate behavior;

serialized hashes.

The complete function must be reviewed or fixture-tested before its hash becomes the frozen generator hash.

Tokenizer preflight is not complete enough

This function only measures the three strings in isolation:

Python
Run
tok(m, add_special_tokens=False)

That is insufficient because tokenization can change with:

leading whitespace;

separators;

the following span text;

the exact full serialization context.

It also returns a number of needed tokens but does not specify:

the actual neutral padding token IDs;

how padding is inserted;

whether the resulting full headers have identical widths;

whether the pad sequence itself is tokenizer-stable;

whether all positions remain identical in complete serialized examples.

The frozen preflight should verify the exact header blocks as they appear in full inputs, or construct them directly from frozen token-ID sequences. It must pin:

tokenizer repository and immutable revision;

tokenizer file hashes;

exact header strings or token blocks;

leading/trailing separators;

actual padding token sequence;

final token IDs and fixed header width.

If the strings are unequal, the admission box must not be allowed to invent a pad plan after scientific freeze. The exact selected plan is part of the condition configuration.

N+ filler leakage and token matching are not implemented

filler_is_clean() is useful as a superficial assertion but does not implement the preregistered leakage audit. It misses, among others:

subtract, minus, decrease, increase, double, triple, negate, reflect, and divide;

number words such as “three” and “seventeen”;

arithmetic symbols;

the fresh world’s actual answer-label tokens;

tokenizer-level overlap with those labels;

generated operator-specific lexemes.

More fundamentally, the fixed strings are not guaranteed to occupy the exact token count of the replaced span. There is no function that constructs or verifies the actual N+ slot token sequence.

Before freeze, implement a deterministic filler constructor that:

receives only the target token width and a non-semantic filler index—not the episode’s value or operator;

produces exactly the fixed slot width under the pinned tokenizer;

is never truncated or padded by an unfrozen runtime choice;

is checked against all generated operator lexemes, numeric forms, arithmetic symbols, slot identifiers, and answer-label token IDs;

produces a complete filler-bank leakage report before the populated pre-run manifest is archived.

The four present sentences may be used as source material, but the frozen artifact must be the exact token-level filler sequence, not merely these strings plus a regex.

Tag-permutation implementation is not shown

headers_for_cell() implements only native headers. The actual §5.4 audit constructor/evaluator is not in the pasted modules. It must verify that:

one single global non-identity permutation is applied consistently across the three operator cells for each audit item;

all five non-identity permutations are balanced;

start-value placement is unchanged;

intact and permuted examples differ only in the marker header blocks;

expected counterfactual targets are recomputed from the resulting operator order;

collisions are removed before evaluation;

every model sees the same bank and denominator.

Until that constructor and evaluator are frozen and tested, the tag-use decision path is not fully implemented.

5. What BUILD_STATUS still omits
A canonical result compiler is required

The biggest missing load-bearing component is a frozen process that converts raw evaluation records into the JSON consumed by the verdict, classifier, and statistics modules.

It should:

read the pre-run manifest’s exact roster;

verify preregistration commit and bank hashes;

verify condition, initialization, order, world, and step-20,000 identities;

compute A2, A3, all-cut, T2, T3, S1, intact-tag, and counterfactual-tag scores from integer per-case outcomes;

verify expected denominators;

reject duplicates, extras, missing cases, NaN, and non-finite values;

derive C1–C6 signs internally;

verify C1=C4+C5 per trajectory;

emit a canonical results bundle with its own hash.

Without this, the scripts correctly judge whatever numbers they are handed but do not establish that those numbers are the frozen experiment’s outcomes.

Add a condition-parity audit

Before fleet launch, automatically verify across the relevant arms:

parameter names, shapes, and initial bytes;

optimizer configuration and initial state;

token positions and sequence lengths;

number of Transformer calls;

packet topology and width;

semantic episode identity and order;

condition-specific serialized-stream hashes;

attention-mask differences;

N+ filler substitution only where intended.

This is the implementation-level counterpart of the matched-design claim.

Add adversarial fixtures

At minimum, add tests for:

one designated run missing and one extra run present;

n+1 complete records;

NaN, inf, -inf, Boolean, string, and out-of-range scores;

exactly 0.20, 0.25, 0.40, 0.70, 0.11, and 0.90 boundaries;

one missing tag denominator with the other n−1 passing;

duplicate (init, order) phenotype rows;

one-order initialization block in stats_engine;

reversed contrast input;

mismatched intact/counterfactual tag denominators;

wrong bank hash;

six-family GOLD with exactly 12 designated trajectories.

The current “9/9 adversarial cases” evidently does not cover the extra-key substitution or NaN loopholes.

One-shot verdict execution

The frozen script is deterministic, so rerunning it cannot change an honest result, but §9 says the real-data verdict is run once. Use a production wrapper that:

accepts only a complete canonical bundle with matching hashes;

creates an atomic run lock;

archives command, stdout, structured output, environment, and input hashes;

refuses a second production execution unless invoked in an explicitly logged deviation mode.

Synthetic fixture tests should remain clearly separated from the production input signature.

6. Can tokenizer preflight and trainer integration wait for the admission box?
Trainer integration: no

The five-condition trainer is the intervention implementation. It must be:

implemented;

code-reviewed;

frozen;

hashed;

covered by CPU-level layout and mask tests;

before the scientific-content deposit and before any seed generation.

GPU execution and bit-exact rehearsal may occur later. The code itself cannot be completed or altered during the post-freeze rehearsal without creating a preregistration amendment or deviation.

Header-tokenizer preflight: run it before freeze

It does not require a GPU, and its result determines the actual condition serialization and padding plan. Because the pre-seed deposit includes condition configurations, the pinned-tokenizer result should be resolved before scientific-content freeze.

The current function is not enough; first implement the full-context/token-ID preflight described above, then record the actual header token blocks in the frozen condition configuration.

World-dependent grammar and bank construction: code before freeze, execution after seeds

It is legitimate to:

freeze the grammar-filter and bank-constructor code before seeds;

generate the fresh world after the immutable timestamp;

execute the frozen world-dependent filters and constructors;

archive their outputs in the populated pre-run manifest before training.

However, the current BUILD_STATUS sequence puts the tokenizer preflight after the populated pre-run manifest. That is backwards: serialized condition streams, grammar eligibility, tag-audit bank hashes, and final bank hashes cannot be finalized before tokenization-dependent construction.

Correct operational sequence

Adopt six families and update preregistration, tier declaration, seed rule, code constants, tests, and status.

Finish trainer integration, evaluator, audit constructors, bank constructors, result compiler, and manifest validator.

Run the world-independent pinned-tokenizer header preflight; freeze the actual token blocks and pad plan.

Run synthetic condition/mask parity tests and create the fixed machine-admission fixture.

Commit and immutably archive the complete scientific-content deposit.

Generate the deterministic seeds and first valid fresh world.

Run the frozen world-dependent grammar filter, filler leakage audit, and bank constructors.

Create, validate, commit, and archive the populated pre-run manifest.

Perform machine admission and a non-outcome-bearing infrastructure rehearsal.

Launch the 60-society GOLD fleet.

A “rehearsal society” should use a synthetic fixture world or be treated as the first designated cohort run. Do not train an undesignated society on the fresh cohort world, inspect its performance, and then retain discretion to alter operations; that would amount to an unregistered fresh-world pilot.

7. BUILD_STATUS corrections

These entries need changing if six is adopted:

“22 seeds” must be recalculated and fixture-tested for the sixth initialization family.

“0.0625 five-cluster floor confirmed” becomes “0.03125 six-cluster GOLD floor confirmed.”

TIER_DECLARATION.json must state six-family GOLD.

“50-society fleet” becomes “60-society fleet.”

all GOLD gate fixtures use n=12 and n−1=11.

This checklist line is inconsistent with the pointer-commit rule:

“companion arXiv id 2609.11365 in prereg header via pointer commit”

A pointer commit should not modify the frozen preregistration header. Replace it with:

[ ] companion arXiv identifier recorded in a metadata-only pointer file referencing the frozen scientific-content commit; frozen preregistration bytes unchanged

Final decisions
Initialization count

SIX FAMILIES.

Build status

NOT YET APPROVED FOR FREEZE AS IMPLEMENTED.

The mathematical threshold logic is sound, and the phenotype tree is substantively correct. Before hashes are frozen, repair the exact-roster and invalid-value loopholes, preserve INCOMPLETE as a true third state, derive statistics from canonical raw arm results, freeze an exact percentile method, replace the imperative grammar frames, implement token-exact N+ fillers and full-context tokenizer preflight, and complete/freeze the trainer, audit constructors, evaluator, and result compiler.

No new experimental arm or conceptual redesign is required.

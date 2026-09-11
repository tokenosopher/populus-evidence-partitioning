Final pre-freeze ruling
NO-GO TO FREEZE THIS EXACT REVISION

Round 4 closes most of the Round-3 work, but the evidence still exposes three material mismatches:

the development-exposure manifest is complete, but the described rejection rule still consumes only the operator and split exposure sets;

the span-rewrite control is not counterfactually matched to the packet intervention after the first interface;

the downstream packet-reader liveness test is confounded by the architecture’s direct P
in
	​

→P
out
	​

 residual path, and the shared-question mask is asserted at the mouth rather than in the cells.

The new §6.2 indicator thresholds also need a small but important amendment before they become frozen scientific content.

These are narrow implementation changes. No additional trained arm, initialization, endpoint, or conceptual redesign is required.

1. Ruling on the new mechanism and dialect definitions
1.1 Approximately norm-matched noise
APPROVE, subject to making the exact draw rule and zero-norm behavior literal

Matching the whole two-vector packet’s Frobenius norm to the recipient’s intact packet is a defensible interpretation of “approximately norm-matched packet noise.” A case-keyed deterministic random direction is preferable to consuming a mutable global RNG stream. Reusing the same case/interface direction across models is acceptable and can improve comparability.

The frozen specification should literally define:

Z
e,i
	​

∼N(0,I),N
e,i
	​

={
∥P
e,i
	​

∥
F
	​

Z
e,i
	​

/∥Z
e,i
	​

∥
F
	​

,
0,
	​

∥P
e,i
	​

∥
F
	​

>0
∥P
e,i
	​

∥
F
	​

=0.
	​


It should also record:

the exact hash-to-seed conversion;

distribution (standard_normal, not merely “Torch Generator”);

generation device and dtype;

cast/rescaling order;

count of zero-norm intact packets;

an error if the sampled direction itself has zero or non-finite norm.

The parent paper treats deletion, approximately norm-matched noise, and natural-packet transplantation as distinct controls, so retaining noise separately is faithful. 

populus_arxiv_v3_final

If the committed implementation already specifies those details, the noise construction is closed.

1.2 Counterfactual span-rewrite control
AMEND — the current x↦x+3 rewrite is not the corresponding control

The packet intervention changes the running value at a specified interface by +3. Rewriting the initial value as x+3 does not generally create a +3 change at a later interface, because the preceding affine operators can scale or reflect that difference.

Let F
i
	​

 denote the prefix map from the initial value through the sender at interface i. For a base episode:

v
i
	​

=F
i
	​

(x),v
i
cf
	​

=v
i
	​

+3(mod17).

The corresponding text-level intervention must use:

x
i
cf
	​

=F
i
−1
	​

(v
i
cf
	​

).

Then rewrite only the start-value span from x to x
i
cf
	​

, leaving operators, phrasings, masks, and native packet generation unchanged. Because every primitive is a bijection over Z
17
	​

, this inverse exists.

Exact replacement specification

Counterfactual evidence-span rewrite. For each audited packet interface i, let F
i
	​

 be the complete operator prefix through the sender at that interface and let v
i
	​

=F
i
	​

(x). The packet-counterfactual target is v
i
cf
	​

=(v
i
	​

+3)mod17. The matched text control rewrites the start-value span to x
i
cf
	​

=F
i
−1
	​

(v
i
cf
	​

), retains the original operators and sealed phrasings, and allows all packets to be generated natively. The evaluator asserts both

F
i
	​

(x
i
cf
	​

)=v
i
cf
	​


and equality between the rewritten episode’s final target and the suffix-implied packet-counterfactual target. Any failure aborts the audit record.

At the first interface this reduces to x+3; at later interfaces it generally does not.

Your present x↦x+3 implementation remains a valid independent initial-value sensitivity control, but it cannot be called the “corresponding” span rewrite and cannot serve as text-level ground truth for the fixed-offset packet intervention.

1.3 §6.2 descriptive indicator thresholds
AMEND THE OPERATIONAL DEFINITIONS

The values 0.90 and 0.60 are reasonable descriptive cutoffs. The current labels and formulas are not fully safe.

A. “Exact canonicality” at 0.90 is not exact

Rename it:

strong within-regime canonicality

with:

self-same≥0.90∧self-cf≥0.90.

If the label exact canonicality is retained, both integer-count fidelities must equal exactly 1.000.

B. Partial canonicality

This is acceptable as:

self-same≥0.60∧self-cf≥0.60∧¬strong-canonical.
C. Context entanglement currently mixes denominators

The interventions are success-conditioned. On the common eligible set, the intact reference is therefore 1.000 by construction. Do not subtract a full-bank intact score from a success-conditioned deletion score.

Use:

self-same<0.60∧deletion≤0.70.

Equivalently, deletion causes at least a 0.30 loss from the eligible-set intact baseline of 1.000.

Call it an:

operational context-entanglement indicator

and state that it does not uniquely identify context entanglement as the only possible mechanism.

D. Cross-regime interoperability cannot rely only on counterfactual ratios

A ratio of 0.50 is uninformative if recipient-self fidelity is 0.10 and cross fidelity is 0.05. Preservation must also be included.

Exact replacement

Cross-regime interoperability. A matched twin pair is marked cross-regime interoperable at an interface only if, in both donor directions:

recipient-self same-value preservation is at least 0.60;

recipient-self counterfactual following is at least 0.60;

raw cross-donor same-value preservation is at least 0.50;

raw cross-donor counterfactual following is at least 0.50;

the recipient-normalized preservation ratio is defined and at least 0.50;

the recipient-normalized counterfactual-following ratio is defined and at least 0.50.

Both directions must satisfy the complete conjunction.

E. Low denominators

For every indicator:

if n
eligible
	​

<30, report the raw metrics with status LOW_DENOMINATOR; do not emit a positive or negative phenotype assignment.

This is stronger and cleaner than calculating a categorical indicator and merely appending a warning.

2. Status of the three Round-3 blockers
Blocker 1 — development exposure
PARTIALLY CLOSED

The machine-readable exposure manifest, single source of truth, stale-module lineage, and inclusion of newly exposed fixtures are all excellent.

However, the described operational rejection rule is:

candidate_is_dev_exposed(op, split)

and rejects only the operator and split components. Your manifest deliberately records exposures for:

grammar;

filler;

order;

initialization;

synthetic world identifiers;

but the described candidate rule does not consume those role-specific sets.

This does not satisfy the prior requirement that the generated scientific seed bundle be checked against all development-exposed components.

Exact remaining delta

Add a role-aware complete-bundle check in the frozen seed orchestrator:

Python
Run
def seed_bundle_is_dev_exposed(bundle, exposures):
    for role in (
        "operator",
        "split",
        "grammar",
        "filler",
    ):
        if bundle[role] in exposures[role]:
            return True

    if any(s in exposures["order"] for s in bundle["orders"]):
        return True

    if any(s in exposures["initialization"]
           for s in bundle["initializations"]):
        return True

    if bundle["world_id"] in exposures["world_ids"]:
        return True

    return False

Numeric equality across different roles need not cause rejection; the comparison should be role-specific.

The cleanest deterministic rule is:

derive the complete six-family candidate seed bundle from candidate counter c; reject the whole bundle if any component is development-exposed in its corresponding role; increment c; repeat.

Do not accept a bundle and then improvise replacement seeds field by field.

Once this is implemented and regression-tested, Blocker 1 is closed.

Blocker 2 — mechanism and dialect implementation
MOSTLY CLOSED, BUT THE SPAN CONTROL AND INDICATORS REMAIN OPEN

The following are now satisfactory:

the complete intervention menu exists;

norm-matched noise is distinct from deletion;

donor-coverage intersection is applied before matrix execution;

self and cross donors use the same recipient case set;

case-set hashes are asserted equal;

degenerate matrices abort rather than relax;

all four donor–recipient cells are orchestrated;

preservation and counterfactual ratios are separate;

raw numerator and denominator are retained;

zero denominator produces undefined;

no ratio clipping occurs;

two diagnostic rotations are retained as preregistered;

low-denominator cases are identified.

The blocker remains open only because:

the span rewrite is not mathematically matched to the packet counterfactual;

the descriptive indicator definitions require the amendments above.

After those two changes, Blocker 2 is closed.

Blocker 3 — downstream reader-path admission
NOT YET CLOSED

The foreign-token invariance tests are good. Holding the incoming packet fixed while perturbing masked foreign tokens gives a valid downstream invariance check.

The incoming-packet liveness test is not sufficient:

“a fixed +0.25 perturbation of the incoming packet changed each downstream packet”

The architecture directly computes:

P
out
	​

=RMSNorm(P
in
	​

+0.05ΔP).

Therefore P
out
	​

 changes when P
in
	​

 changes even if the reader pseudo-tokens are completely masked or the downstream Transformer entirely ignores them. The direct packet residual makes this test positive by construction. The paper’s architecture section explicitly contains this skip pathway. 

populus_arxiv_v3_final

Required functional test

Perturb P
in
	​

 while holding all text fixed, but compare either:

the downstream hidden states used by the writer; or

the writer residual ΔP before adding P
in
	​

;

not merely the final normalized packet.

For each downstream cell, require:

incoming perturbation changes writer hidden state or ΔP

A deterministic nonzero Jacobian of ΔP with respect to the incoming packet is also acceptable.

Required direct mask assertion

The current field:

question_mask_readable: mouth consumes the question under an explicit all-ones mask

checks the mouth, not the four adapter-active cell calls. The preregistered architecture says the cells receive the shared question.

Inspect the final cell-level attention mask and assert, for every downstream R+ cell:

question-token positions are readable;

both packet pseudo-token positions are readable;

own header and active own-span tokens are readable;

foreign headers and foreign active-span tokens are masked;

padding behavior matches the frozen layout.

Similarly, packet pseudo-token readability should be established both:

directly in the final mask tensor; and

functionally through a change in hidden state or ΔP.

Revised required record
JSON
{
  "restricted_foreign_invariance_cell0": true,
  "restricted_foreign_invariance_downstream": true,
  "cell_question_positions_readable": true,
  "packet_pseudotoken_positions_readable": true,
  "incoming_packet_reader_changes_delta_p": true,
  "own_span_path_live": true,
  "global_foreign_path_live": true,
  "mouth_question_path_readable": true
}

Once those assertions pass locally and are retained for admission-box replay, Blocker 3 is closed.

3. Smaller observations

These do not independently block freeze once the deltas above are applied.

Noise key namespace

The noise key should include an audit-version/domain separator, for example:

noise:v1:world_id:recipient_id:bank_idx:interface

Including the recipient/model identity is optional. Omitting it deliberately reuses the same random direction across recipients, which is acceptable if stated.

Common-case intersection

The frozen intersection file is an audit-derived artifact produced after training. That is acceptable because its construction is mechanical and outcome-independent apart from packet availability. Preserve:

the complete pre-intersection coverage maps;

the intersection script hash;

all exclusion counts;

the intersection-file hash;

the exact case IDs.

Matrix failure

A degenerate intersection should mark the affected secondary matrix:

MATRIX_NOT_VIABLE

It should not invalidate the primary cohort verdict or silently trigger another trained model, world, or donor-selection rule.

New fixture exposures

Any new seeds used while implementing the three final corrections must be appended to development_seed_exposures.json immediately. Reusing already exposed fixture bundles avoids expanding the list.

4. Final freeze decision
Items genuinely closed

I regard the following as closed, conditional on archived code matching the reported behavior:

identity, primitive, train2, and train3 map exclusion;

explicit structural-validity conjunction;

monitoring-program and phrasing separation;

24-program mechanical tag cap;

five-permutation stratification;

four primary/tag rotations;

filler PRF domain separation by stream;

token-active-length matching;

N+/G+ mask identity;

six-family gate and statistics machinery;

canonical integer result compilation;

signed contrast derivation;

mechanism/dialect matrix orchestration;

production verdict lock;

three-step training-state fingerprint specification;

stale-module regression;

neutral third-person grammar;

operator-support and tag-denominator viability checks.

Remaining exact deltas

Before scientific-content freeze:

make development-seed rejection role-complete across the entire generated seed bundle;

replace x↦x+3 with the interface-matched inverse-prefix span rewrite;

amend the four dialect indicators as specified, especially renaming 0.90 “exact,” aligning the deletion denominator, and requiring raw preservation plus counterfactual performance for interoperability;

inspect the final cell attention masks directly and test reader liveness using hidden states or ΔP, not final P
out
	​

;

make the exact noise distribution and zero-norm behavior literal if they are not already frozen in code.

After those changes pass their fixtures, update the exposure manifest for any newly touched seeds, populate the hash inventory, and run the freeze ceremony.

Final verdict
NO-GO FOR THIS EXACT REVISION
GO IMMEDIATELY AFTER THE FIVE DELTAS ABOVE PASS

The remaining work is narrow and mechanical, but two points—the unmatched span rewrite and the residual-confounded packet-reader test—would create incorrect mechanism claims if frozen unchanged.

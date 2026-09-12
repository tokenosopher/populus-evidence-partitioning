RULING: RATIFIED — GO for the narrowly bounded manifest correction

On the facts submitted, this is a correction of derived manifest metadata, not a change to the frozen experiment or analysis. The proposed correction is approved, with the provenance and execution requirements below.

1. Exact scope of authorization

The authorized changes are only:

Manifest field	Original value	Corrected value
expected_denominators.S1	1224	2448
bank_sizes.S1	1224	2448

The controlling derivation is:

12 operators×3 positions×17 values×4 sealed rotations=2448.

The justification is not that all 60 outputs report 2448. Agreement among outputs could also reflect a shared implementation error. The justification is your reported agreement between the pre-existing frozen four-rotation specification and the hash-matching frozen evaluator, which independently determines 2448 without consulting any performance result. The output denominators corroborate that diagnosis.

This is therefore one incorrect derived count represented in two manifest fields, not permission to revise an evaluation bank from two rotations to four after execution.

2. Binding requirements for the correction deposit

Preserve the original record; append the correction. Retain the original populated manifest byte-for-byte, retrievable through its immutable commit or an explicitly retained original file. Do not move the original freeze tag, replace its archival identity, or present the corrected manifest as the artifact that existed before training. The corrected operational manifest must have a new, explicitly post-evaluation provenance record.

Your MANIFEST_CORRECTION note must identify:

The original and corrected manifest paths, immutable commit references, and full SHA-256 hashes.

The precise frozen specification location establishing four rotations, the corresponding frozen evaluator construction, and the defective manifest-population expression.

The two authorized field changes, with a machine-checked structural diff confirming that every other field is identical.

The actual discovery and correction chronology, including the compiler’s failed invocation and this ruling.

Put the correction timestamp and explanation in the addendum—not into additional manifest fields under this authorization. Preserve historical references to the original manifest hash in existing outputs, checkpoints, and admission records; do not rewrite those artifacts to imply that they were produced under the corrected manifest.

Establish that the defect was confined to metadata. The deposited diagnosis should document, through the relevant code path, that the stale values did not determine training, truncate or select evaluation cases, or normalize stored S1 scores. In particular, the existing S1 scores must already have been computed over the frozen four-rotation bank. This requires a structural/dataflow check, not another model evaluation or a new outcome analysis.

Uniformly reported denominators alone are not a substitute for that check. If the stale value affected execution or stored scoring, this authorization does not cover repairing those additional consequences.

3. Execution and “runs once” semantics

Preserve the failed invocation and describe the sequence literally:

One result-compiler invocation aborted at its first integrity gate, before verdict computation. Following deposition of the ratified manifest correction, the unchanged compiler was invoked against the corrected manifest and the unchanged evaluation bundle.

That wording distinguishes compiler invocation from completed verdict computation. Do not retrospectively claim that the compiler was invoked only once.

The retained log and control flow must support the assertion that the aborted invocation did not compute or emit a verdict, contrast decision, or phenotype classification. Also distinguish pre-verdict from outcome-blind: disclose any performance information already inspected, without upgrading the former into the latter. The mechanical justification for this correction does not require inventing a stronger blinding claim.

After the correction is deposited, use the same frozen compiler and the same 60-society evaluation bundle. Retain a content-hash inventory of those input files so their identity is auditable. Do not retrain, reevaluate, select replacement outputs, drop societies, modify classifiers, or weaken an integrity check.

Any subsequent integrity failure remains fail-closed. This ruling is not blanket permission to keep adjusting the manifest until compilation succeeds. A new discrepancy requires its own diagnosis and bounded ruling.

4. S1’s descriptive status does not remove its integrity obligations

I accept your stated distinction that S1 gates no preregistered contrast. Nevertheless, a classifier input can affect the phenotype census and its interpretation, even when it does not affect the primary contrast.

Accordingly, the defensible claim is:

The correction changes neither the frozen S1 bank, its scoring rule, nor the classifier definitions; it corrects the manifest’s erroneous expected count to the value already required by the frozen specification.

Do not broaden this to “S1 is irrelevant to the conclusions” or “the correction cannot affect any reported classification.” The latter would require knowing how the erroneous count would otherwise have been consumed; the compiler correctly prevented that inconsistent path from proceeding.

Final disposition

GO — deposit the corrected manifest and correction note. GO — execute the frozen verdict procedure once the requirements above are recorded and satisfied. No further approval round is required for these exact two field changes. No other repair, analytical change, or relaxation of the protocol is authorized.

The appropriate classification is “post-evaluation, pre-verdict administrative correction to derived manifest metadata.” Do not silently erase it, backdate it, or describe it as a new four-rotation design decision.

Verification boundary: I could access the public repository’s landing page, but could not retrieve the underlying frozen cohort files in this turn. This ruling therefore evaluates the facts in your submission; it is not an independent re-certification of the manifest/evaluator hashes or the 60 output files. 
GitHub

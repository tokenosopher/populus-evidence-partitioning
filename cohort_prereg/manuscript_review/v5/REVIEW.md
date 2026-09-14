# Readiness review: v4 and revised v5

Reviewed 13 September 2026. Three subagents checked statistics, methods, and venue/prior-art requirements, alongside manuscript and rendered-PDF review. This is AI-assisted review, not journal peer review or an independent experimental replication.

## Recommendation

**I would post the revised v5 as a bounded confirmation preprint. I would not post v4 unchanged.** The central empirical result survives the checks: the tested masking regime has a large, consistent advantage on the fresh held-out composition world. V5 fixes the reporting errors found, supplies standalone methods and related work, and presents that result without claiming a general mechanism or universality.

**TMLR is a plausible venue, but the present artifact is an identified arXiv manuscript, not a submission-format TMLR manuscript.** An actual submission needs the official template and anonymized manuscript, PDF metadata, and supporting materials. The main scientific review risks remain six initialization clusters on one confirmation world; a failed role-following manipulation; inconclusive filler decomposition; success-conditioned mechanism comparisons between models of different competence; and the disclosed evaluation-timing/record-retention limitations. These do not make the reported masking comparison wrong. They limit what the paper can claim and may prompt requests for additional experiments.

TMLR emphasizes supported claims and clear communication, rather than a requirement to beat a benchmark or invent a new method. Its reproducibility scope makes an honestly bounded confirmation appropriate in principle. Acceptance is an editorial judgment, not established by this review. See the [acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html) and [author guide](https://jmlr.org/tmlr/author-guide.html).

## What is supported

- All 360 behavioral raw-count ratios match the compiled bundle. Bundle and corrected-manifest hashes match the archived verdict inputs.
- Independently recomputed contrast vectors, medians, means, 100,000-draw cluster bootstrap intervals, exact six-cluster sign-flip results, gate components, phenotype counts, and order concordance agree with the reported results.
- C1's median masking advantages are 0.846/0.859 at depths two/three; C6's are 0.884/0.917. Every matched trajectory exceeds the directional margins at both depths in both contrasts. The marked and unmarked contrasts reuse the initialization/order structure.
- Across 90,779 released mechanism per-case records, reconstructed numerator/denominator summaries agree. All eighteen audited masked trajectories have perfect same-value and counterfactual fidelity at the tested interfaces on their eligible sets. The 11/12 bidirectional interoperability failures are correctly reported.
- The implementation supports a matched effect of the masking regime. It does not isolate the causal contribution of foreign semantic content, establish semantic role rescue, or prove that canonicality mediates generalization.

[Detailed numerical audit](statistics-review.md) records exact values and evidence paths. Public audit records were pinned to GitHub revision `ebe2ebf80d55d52a6ef9b6c237e2fee1e6c99294`.

## Corrections made in v5

| Issue in v4 | Revision |
|---|---|
| N+ described as having two perfectly audited competent trajectories | One was perfect; the other scored 815/816 at one interface and 816/816 at the others. Both meet the operational threshold. |
| G+ deletion range used trajectory maxima while R+ used all-interface extrema | Both now use all-interface ranges; G+ is 0.000–0.175. |
| G− shift-retain rounded after averaging rounded rates | Recomputed from integer counts: 0.170, replacing 0.171. |
| C3 labelled “NULL CONFIRMED” | “CRITERION MET,” with the actual frozen machine status `PRACTICAL_NULL` explained; no statistical-equivalence claim. |
| “Derangement” described as if it were a permutation | Correctly described as different-value donor sampling with replacement. |
| Chance after deletion described as proof of answer content | Described as sensitivity to zero-packet replacement on eligible cases. |
| N+ described as removing task-relevant foreign information without qualification | Explicitly preserves original span lengths and their possible task-correlated cues. |
| Missing task, architecture, curriculum, readout and optimizer detail | Added a verified held-out worked example, packet/readout equations, residualized 17-label scoring, parameter totals, training mix, singleton-position sampling, split/bank sizes, and optimizer/LoRA settings. |
| Sparse context and dependence on earlier papers | Added a standalone opening and verified related work; clarified the contribution relative to prior restriction/compositionality studies. |
| Long abstract and strong subtitle | Shortened to 1,542 ASCII characters / 209 whitespace-delimited words; changed “Drives” to a confirmation of the tested masking advantage. |
| Overstrong machine-equivalence/lock claims | Limited retrace evidence to the tested prefix and locking to the retained lock/path. |
| Dense procedural narrative before the experiment | Moved detailed machinery to the appendix, keeping salient execution deviations in the abstract and main text. Full correction chronology remains. |
| Small statistical ambiguity | Specified that intervals concern means, added unadjusted descriptive/multiplicity language and sign-symmetry assumptions, and displayed the small C3 effect in scientific notation. |
| AI use disclosure covered protocol review only | Added the manuscript checking/editing performed in this review. |

No experiments, code, frozen records, decision rules, or existing manuscript versions were changed. The substantive corrections do not change the primary C1/C6 conclusion.

## Readability and excitement

The strongest story is simple: **under this controlled training setup, restricting direct evidence access produces markedly better held-out composition than giving every cell all the evidence.** The result is interesting on its own. It does not need claims about human language acquisition, a general law of modular intelligence, or uniquely canonical representations.

V5 leads with that question, defines “society” as repeated calls to a shared network, and shows a real task example before asking readers to interpret the result tables. It distinguishes final-task accuracy from unrestricted language-model generation and from the separate packet interventions. The detailed protocol history remains available but no longer dominates the first reading. The paper is still a technical confirmation report; the most specialized audit and provenance detail is intentionally retained.

## arXiv package and remaining upload checks

The included `arxiv_source_v5.zip` contains a single top-level `main.tex`, all five input tables, both PDF figures, the cited bibliography and compiled `main.bbl`. It excludes the review notes, old PDFs, logs, and experimental data. `arxiv_abstract.txt` is the exact manuscript abstract in ASCII and below the 1,920-character metadata limit. V4's source abstract was 2,397 characters. See [arXiv metadata guidance](https://info.arxiv.org/help/prep.html#abstract-required).

LaTeX-origin papers require their source, not a PDF-only submission. The package was rebuilt after extraction in a clean temporary directory using Tectonic, and its extracted PDF text was compared with the delivered manuscript. The final 16-page PDF was rendered and visually inspected; there are no unresolved references or overflowing boxes. This is a local build check, not a test of arXiv's production TeX environment. The uploader must inspect arXiv's generated preview before completing submission. See [arXiv source instructions](https://info.arxiv.org/help/submit_tex.html).

The upload also requires the author's account/category access and a distribution-license choice; no account action, endorsement request, publication, or license selection was performed here. A reasonable subject fit to consider is cs.LG, with cs.CL as a possible cross-list subject to arXiv moderation. Internal manuscript v5 does not mean this paper already has an arXiv v5 record.

## TMLR-specific preparation

Prepare a separate rendition using the [official TMLR template](https://github.com/JmlrOrg/tmlr) and remove author-identifying text, metadata and links from the submitted manuscript and supplements. An identified arXiv preprint is permitted; the TMLR submission itself must maintain anonymity. References precede appendices in v5 already. Do not claim the anonymous submission package has been prepared by this review.

Check the archival submission/publication status of the parent and companion before journal submission, and explain the new cohort, controls and outcomes clearly. That status was not established by finding their arXiv records. See [TMLR editorial policies](https://jmlr.org/tmlr/editorial-policies.html).

For review strategy, the most valuable further study would make marker following informative and validate it, because that directly addresses the failed manipulation check. Additional task worlds would test robustness beyond initialization/order variation. Competence-matched mechanism comparisons would better address representational differences. These are ways to strengthen journal evidence, not unstated prerequisites for reporting the present bounded result.

## External verification and scope limits

- The [public GitHub confirmation directory](https://github.com/tokenosopher/populus-evidence-partitioning/tree/main/cohort_prereg) is accessible and contains the stated protocol, result and audit records.
- The [checkpoint repository at revision a3b066e7](https://huggingface.co/tokenosopher/populus-evidence-partitioning-checkpoints/tree/a3b066e7/cohort_prereg) lists sixty final checkpoints and the inventory. File presence was checked; checkpoint bytes were not all downloaded or evaluated.
- Both cited Software Heritage snapshots resolve. The [origin visits](https://archive.softwareheritage.org/api/1/origin/https://github.com/tokenosopher/populus-evidence-partitioning/visits/) record the freeze snapshot at 2026-09-11 15:50:29.194 UTC and manifest snapshot at 15:57:12.805 UTC. This verifies archive timestamps, not every event claimed to precede/follow them.
- The [parent paper](https://arxiv.org/abs/2608.20054) and [companion](https://arxiv.org/abs/2609.11365) exist; the parent citation was corrected to its current title.
- Added related work was checked against primary venue records: [Lake and Baroni](https://proceedings.mlr.press/v80/lake18a.html), [Kottur et al.](https://aclanthology.org/D17-1321/), [Chaabouni et al.](https://aclanthology.org/2020.acl-main.407/), and [Geiger et al.](https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html).
- [arXiv's generative-AI policy](https://info.arxiv.org/help/moderation/index.html#policy-for-authors-use-of-generative-ai-language-tools) informed the actual assistance disclosure. Any material historical assistance beyond what the retained record establishes should be disclosed accurately by the author.

The review did not retrain the sixty models, perform independent checkpoint inference, re-execute the protected production verdict, recover missing admission transcripts, or validate unretained duplicate audit copies. Strong internal numerical consistency is valuable evidence; it is not a substitute for those unperformed checks.

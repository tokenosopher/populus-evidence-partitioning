# Independent statistical audit of cohort v4

Audit performed read-only against local `paper/cohort/main.tex`, tables, preregistration, locked bundle/raw counts, frozen code; remote mechanism records downloaded only to `/tmp/cohort_stat_audits/`, pinned to public GitHub revision `ebe2ebf80d55d52a6ef9b6c237e2fee1e6c99294`. No repository file changed. Locked verdict was not rerun. No training or checkpoint inference was run.

## Verdict

Behavioral numerical results and the stated complete-pass decisions are correct. No numerical blocker to the primary behavioral conclusion was found. Two mechanism reporting corrections and one table interpretation correction should be made before submission; they do not change the central conclusion.

## Required corrections

1. **Moderate factual error: two N+ trajectories are described as perfect.** `paper/cohort/main.tex:148` says “two competent trajectories were perfect on the same-value and counterfactual tests.” Only `m2392236931_o1842565541` is perfect. `m608985416_o1942292852` has 815/816 for both same-value and counterfactual following at interface 1; its other interfaces are 816/816. Evidence: remote `cohort_prereg/results_cohort_audits/self_Np_m608985416_o1942292852.json`, lines 2557–2580, locally `/tmp/cohort_stat_audits/self_Np_m608985416_o1942292852.json`. Per-case verification identifies the counterfactual error at bank_idx 116/interface 1 and the same-value error at bank_idx 124/interface 1. Proposed replacement: “Among the six preselected N+ self-audits, two competent trajectories met strong canonicality at every interface: one was perfect on both tests, and the other scored 815/816 at one interface and 816/816 at the others. A third showed high fidelity but missed the strong-canonicality threshold at one interface (counterfactual following 655/730 = 0.897 < 0.90).”

2. **Moderate aggregation/reporting inconsistency: G+ deletion range.** `paper/cohort/table_mechanism.tex:9` gives 0.051–0.175 while the R+ row gives 0.035–0.076. The R+ range is the min/max over all 36 interfaces; the G+ range is the range of trajectory-wise maxima. Actual G+ min/max over all 36 interfaces is 0.000–0.175. Evidence: `.../matrix_Rp_m1333584616_o380366532__Gp_m1333584616_o380366532/cell_self_G.json:324-328` has deletion 0/42 at interface 2. Proposed change: G+ deletion 0.000–0.175; caption `main.tex:139`: “Matrix rows report ranges over twelve recipients of their minimum interface rate for same-value and counterfactual following; deletion reports the full range across all recipient–interface combinations.” Alternatively report trajectory-wise maxima for BOTH rows (then R+ becomes 0.059–0.076) and say so explicitly, but full ranges are clearer.

3. **Moderate interpretation inconsistency: NULL CONFIRMED.** `paper/cohort/table_verdict.tex:8` labels C3 “NULL CONFIRMED,” while `main.tex:125` and preregistration section 5.3 explicitly distinguish the frozen practical-redundancy criterion from equivalence/no-effect inference. Replace result label with “CRITERION MET” (claim column already says marker redundancy) or “PRACTICAL REDUNDANCY.”

## Minor improvements

4. `table_mechanism.tex:8` G− shift-retain should round to **0.170** when computed from the raw integers using the stated order (mean over interfaces per society, then median over societies). Exact value 0.17049962796101575; current 0.171 comes from aggregating already-rounded 4-decimal rate fields, whose median is 0.17051666666666665. Compute publication summaries from integer counts, rounding only at display.

5. `table_secondary.tex:11` hides the nonzero C3 depth-three effect as −0.000 and CI [−0.000,+0.000]. Use mean −3.71×10^-5 and CI [−1.11×10^-4,0], or retain rounded table with a footnote giving these exact-scale values. Actual one-trajectory difference −0.00044563279857401383; mean −0.00003713606654783449; bootstrap lower −0.00011140819964350346, upper 0.

6. Add to `main.tex:118` caption, following preregistration section 8: “Intervals and p-values are unadjusted descriptive summaries; no familywise significance claim is made.” The current descriptive language is sound, but the paper omits this useful explicit multiplicity statement.

7. `table_verdict.tex:9`: replace “12/12 at chance (0.04–0.06 vs. 0.90)” with “0/12 meet criterion; cf-track 0.038–0.060” to avoid an untested no-above-chance inference and identify which of the two gate metrics is ranged.

8. `main.tex:139` deletion caption “values at chance ... mean the channel was carrying the answer” is stronger than necessary. Prefer “low deletion accuracy indicates dependence on the tested packet in this intervention.” Chance comparisons on success-conditioned/coverage-filtered case sets are descriptive, and an answer-carrying semantic claim does not follow from deletion alone.

## Independently verified

- Every behavioral raw-count ratio: 60 societies × 6 measures = 360 exact matches to compiled bundle.
- SHA-256 of local compiled bundle and corrected manifest exactly match the one-shot archive (7a8a920b... and 7632aa76...). Current stats engine, verdict and phenotype classifier hashes match the frozen declared hashes.
- All per-permutation raw marker counts reproduce compiled unweighted means, up to floating-point roundoff.
- All 12 contrast-by-depth trajectory vectors, medians, means, six initialization means, 100,000-draw seed-271828 cluster percentile intervals, exact six-cluster two-sided sign-flip p-values independently recomputed; all agree with the publication and secondary v2 outputs.
- C1/C6: all 12 trajectories pass both directional margins; all other components pass. C1 smallest margins 0.7140522876 (d2), 0.3877641966 (d3). C6 medians 0.8843954248, 0.9174942704. C2 0/12 joint margins; C4 5/12; C5 7/12 with 2 joint reversals. All-cut counts 12/12 for every designated advantaged arm.
- Phenotype census and order concordance exactly match the paper: R−/R+ 12 full each, 6/6 concordant; G− 1 deep-only, 8 memorizers, 3 mixed, 5/6 concordant; G+ 5 deep-only, 7 memorizers, 5/6 concordant; N+ 7 full, 4 non-acquirers, 1 shallow, 3/6 concordant.
- Reconstructed mechanism summary numerators and denominators from all **90,779 per-case records across 66 evaluation-cell JSON files** (18 self-audits + 24 matrix self + 24 matrix cross): zero mismatches. All matrix self/cross recipient eligibility hashes match.
- All 18 audited masked trajectories achieve exactly 1.000 same-value and counterfactual following at all tested interfaces on eligible sets.
- No G+ recipient meets strong canonicality at all interfaces; 11/12 have at least one context-entanglement-indicator interface. All named matrix minima agree.
- Bidirectional interoperability qualifies only at interface 2 of m1276321262_o3715708347; all exact rates and 11/12 pair counts agree.
- Low-denominator unsuccessful N+ case counts indeed range from 7 to 26. Third competent N+ counterfactual interface 655/730 is correct.

The sign-flip procedure is implemented correctly as the frozen finite sign enumeration over six initialization means. Its “exact” character is conditional on the sign-symmetry/exchangeability null; it is not a randomized-treatment exact test. The paper already treats it descriptively, which is appropriate. The study remains six initialization clusters on one new world, and neither the behavioral gates nor p-values establish universality or multiplicity-adjusted significance.

## Abstract and additional rounding verification

The v4 abstract is exactly **2,397 characters** between the LaTeX abstract delimiters after stripping outer whitespace (it is already one line). Independently extracted from the PDF, with ligatures normalized and layout line hyphenation removed, the rendered abstract is **2370 characters / 299 whitespace-separated words**. The source count includes LaTeX citation/math syntax; character-limited arXiv metadata must be counted in its final submitted plain-text representation.

All 18 displayed self-audit table cells were recomputed from integer numerators/denominators. The only 3-decimal rounding discrepancy is G− shift-retain (0.170 instead of 0.171). All other R−, G−, N+ self-audit cells agree. All arm-level table medians, all phenotype counts, and all secondary table quantities agree at published precision except the separately noted C3 negative-zero presentation. Matrix ranges agree except the inconsistent G+ deletion aggregation described above.

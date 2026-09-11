# REVIEW: Cohort currency sweep (referee, 2026-09-12, pre-freeze)

VERDICT: FREEZE-AND-LAUNCH — GO, with narrower novelty wording and updated citations

I found no external paper that preempts the cohort’s central comparison as described: a matched restricted/global × explicit-ownership-marker comparison, supplemented by a neutral-filler condition, with order-sensitive counterfactual marker auditing and a prospectively defined phenotype census.

No additional training arm, changed primary contrast, or new performance threshold is required by this literature sweep. Keep R+ versus G+ primary and the full dialect matrix secondary. The necessary changes are to attribution, related work, and the scope of the claims—not the experiment.

Date qualification: this search is verified through 11 September 2026, not 12 September. Stamp the record “searched through 2026-09-11.” A 12 September coverage statement requires a same-day delta check.

I used the attached parent v3 to establish the existing claim boundaries and live paper listings, version histories, and methods for the currency check. This is a ruling on the design you describe, not a second audit of the complete preregistration or deposit bytes.

1. Visibility, role information, and context load: what has—and has not—been anticipated
There is substantial trained-agent role/identity prior art

Agent identifiers in shared-parameter communication systems are not new. Foerster et al.’s DIAL gives agents an index alongside their private observations and messages, allowing a shared network to specialize. Its messages are differentiable during training, although discretized for execution. This is direct precedent for identity-conditioned learned communication—not merely prompted LLM role-playing. 
arXiv

Identity-present versus identity-absent training controls are also not new. Terry et al. explicitly compare agent-indication methods in parameter-shared reinforcement learning, including a no-indicator baseline. A terminology trap matters here: their method called “Identity” means leaving the observation unchanged, not supplying an agent identifier. Their experiments concern distinguishing agents with different functions, not ownership of particular evidence spans or the semantics of a learned packet relay. 
arXiv

These papers rule out “first use of role information,” “first identity-conditioned shared policy,” and “first training intervention adding agent identity.” They do not establish that somebody has already performed your ownership-cue/visibility/load comparison.

Local/global visibility and input separation already have close precedents

CoFlow remains the closest direct masking neighbor. Its centralized and decentralized variants support full teammate visibility versus attention-masked agent-local observations within the same architecture. That is a genuine trained multi-agent visibility comparison, not simply centralized training followed by a test-time mask change. It does not supply your marker-present/absent and filler controls, held-out operator-composition task, or value-indexed packet audit. 
arXiv

Béna and Goodman manipulate input separability, modular connectivity, and resource constraints in trained neural modules. They belong in the explanation of why architectural separation need not itself guarantee functional specialization. Kaszyński supplies a complementary emergent-communication precedent involving distributed senders and compositional codes, but changes sender organization/message factorization rather than isolating your within-architecture intervention. 
arXiv
+1

A further relevant bridge paper is Chen et al., “See What I See, Know What I Think”: it distinguishes context-aware from context-unaware communication and trains transformations between heterogeneous models’ KV representations. It therefore anticipates the importance of what the receiver already knows. It does not independently manipulate computational ownership markers, visibility, and filler in a jointly learned relay. 
arXiv

A September addition concerns trained role-marked languages—but a different meaning of “role”

The older NeLLCom work by Lian, Bisazza, and Verhoef already trains neural speakers/listeners through supervised learning followed by reinforcement-learning communication, studying the relationship between grammatical case marking and word order. Thus “role marking in trained emergent communication” is much too broad a novelty claim. 
arXiv

The new Zhang et al. paper, arXiv:2609.06025v1, submitted 5 September, is particularly worth adding. It combines neural-agent language learning with processing constraints, including noise, capacity, and incremental listening. It uses role-specific subject/object lexical forms; those forms are atomic vocabulary items, not separately supplied “subj/obj” tokens. Its roles are grammatical roles within communicated events, not the identity of the computational cell owning an evidence slot. It does not implement the target masking × ownership-marker × filler comparison or a continuous latent relay. 
arXiv
+1

Q1 ruling: there are relevant trained identity interventions, trained grammatical-role interventions, and trained visibility/context-availability comparisons. I found no verified external study combining those ingredients in the causal decomposition your cohort targets.

The important limit on your own decomposition

The parent paper correctly identifies three coupled consequences of its mask: foreign-evidence access, implicit ownership information, and active-context load. That is explicit in §7, pp. 10–11. 

populus_arxiv_v3_final

The new experiment should nevertheless be called a 2×2 intervention design plus a neutral-filler control, not a fully orthogonal three-factor factorial.

In particular:

Marker absent does not mean role information absent. R− still supplies an ownership cue through its mask.

R+ versus G+ equates explicit marker availability, not necessarily the total amount or effective use of role information.

N+ identifies the specified filler contrast, not a universal, representation-independent quantity called “pure context load.”

You can estimate the effects of the implemented interventions and their interaction without claiming that the five means uniquely partition performance into three additive psychological mechanisms. That distinction preserves the scientific value of the experiment while preventing an identification overclaim.

2. New causal audits since 28 August
The important new neighboring audit: CVRR

Park, Jung, and Kang’s “Reason Through the Latent!”, 2609.06746v2, was submitted 6 September and revised 10 September. Its CVRR system trains a recurrent latent visual reasoner while removing alternative image-conditioned paths before decoding. Interventions include recurrent-state replacement and manipulations of the evidence available during recurrence; fixed-question comparisons address whether effects are merely caused by changing the question. 
arXiv
+1

This is an important new precedent for making a latent state a necessary information route and then testing whether its content, rather than just its presence, matters. But it is a single-model visual-reasoning system, not an emergent inter-agent protocol. It does not perform ownership-tag permutation scored against an exact operator-composition counterfactual. Cite it; do not treat it as an exact preemptor. 
arXiv

The existing causal-audit citations remain mandatory

The two most directly relevant latent-agent audits are not new September developments:

Zhang and Emu, 2607.26773v1, distinguishes no message, another example’s message, self-generated messages, and current-example sender information. Cheng, Das, and Ramnath, 2608.04893v2, compares native, deranged, zeroed, and moment-matched KV relays. Cheng’s latest listed revision is 27 August, explicitly a metadata/abstract-formatting revision—not a new post-28-August experiment. 
arXiv
+1

Hidden APIs, 2607.27617v1, remains a close precedent for reusable causal interfaces, role-aligned transplantation, locality, and counterfactual downstream use. Geiger et al., 2106.02997, supplies the foundational interchange-intervention logic: the mechanistic claim is supported when replacing a low-level state produces the corresponding high-level counterfactual effect. Neither generic counterfactual testing nor semantic interchange should be presented as newly invented here. 
arXiv
+1

Does anything preempt the G+ marker-permutation audit?

No exact external match was found for the combination of:

Reassigning ownership markers in a trained globally visible latent relay, preserving the relevant evidence content, and scoring predictions against the mathematically implied, order-sensitive composition.

The defensible distinction is counterfactual semantic target following, not the mere use of a permutation.

There are three different conclusions an intervention might support:

Sensitivity: performance changes when tags change. This establishes dependence on something affected by the intervention, but damage or distribution shift can explain it.

Intended assignment following: predictions track the particular counterfactual answer implied by reassigned ownership. This is substantially stronger evidence for functional tag semantics.

Locally factored execution: each cell executes its assigned transformation through the proposed intermediate-value interface. Even successful tag-counterfactual following does not establish this by itself; the packet battery remains necessary.

Keep the preregistered marker-audit denominator. An answer-preserving permutation—because operations commute, composite maps coincide, or the particular input yields the same answer—cannot by itself demonstrate order following. Such cases must be interpreted under the declared rules, not silently removed to improve the score.

Q2 ruling: no newly found external paper kills the specific marker-audit design. CVRR should be added to its methodological context. The claim must remain an application and integration of established causal-intervention principles, not “the first causal audit of latent communication.”

3. Twin/dialect instruments since 25 August
Your own companion is now a public antecedent

Portable Semantics, Private Dialects, 2609.11365v1, submitted 10 September, already reports a post hoc same-lineage restricted/tagged-global case study with a complete self/cross packet matrix, exact counterfactual scoring, and recipient-self normalization. It is success-conditioned, concerns one pair, and explicitly reports no marker-permutation audit. Thus the cohort’s dialect component is a prospective extension, not the instrument’s first report. This is not external preemption; it is an attribution requirement within your programme. 
arXiv
+1

The wider instrument family already has antecedents

Self-play versus cross-play and independently learned conventions are established territory; Other-Play is a foundational reference. Model Alignment Search supplies a closer activation-level precedent: cross-model state transfer evaluated through its causal behavioral effects. A complete cross-model matrix is not, by itself, a new instrument category. 
arXiv
+1

StateBridge studies hidden-state alignment into a receiver’s input space, including vector-norm calibration. That is not the same operation as normalizing a behavioral transplant score by the recipient’s self-transplant score. Cross-Model Memory Transfer via Target-Side Reader Adaptation further motivates taking the receiving interface seriously, but concerns transferred memory and reader adaptation rather than your raw emergent-packet audit. 
arXiv
+1

Wolski et al.’s 2608.03644v1, submitted 4 August, also cautions against treating inter-seed cross-play within one implementation as a complete measure of coordination robustness. It is relevant background, not a new post-25-August result. 
arXiv

What survives as the cohort contribution

I found no new external post-25-August match to the specific same-lineage, cross-training-regime, semantically indexed natural-packet matrix with recipient-self calibration.

But the useful contribution is not the division operation in a normalization formula. It is the prospectively controlled comparison and its interpretation:

Poor cross-transplant performance is not automatically a private dialect. If a recipient also fails same-value self-transplants, the proposed intermediate value may not be a sufficient description of its own packet state. Cross-system failure then cannot be attributed solely to incompatible coordinate conventions.

Recipient normalization is descriptive calibration, not a causal donor/receiver decomposition. Raw directional scores, self-transplant baselines, and normalized scores should remain visible together. A near-chance self baseline can make a ratio unstable or misleading.

A same-seed match controls starting conditions; it does not guarantee aligned learned coordinates or establish that initialization causes the resulting dialect.

Q3 ruling: retain the matrix as secondary. Describe its novelty as prospective cohort-level extension and comparison across controlled training regimes, not first introduction of self/cross transplantation, dialect testing, or recipient normalization.

4. Precise characterization of the two manually identified neighbors
GlossoGen — the “no factorial decomposition” description needs correction

GlossoGen, 2609.01491v1, studies text-convention emergence and transmission among LLM agents with complementary information in the SaveVeyru scenario. Its reported experiments concern in-context language formation, not end-to-end training of a continuous inter-agent relay. 
arXiv

However, it does have a factorial manipulation: communication budget × availability of postmortem discussion. The platform also supports replay/forking and counterfactual message editing, and the paper studies transmission to newcomer agents. Therefore, “no factorial” and “no interventions” would both misdescribe it. 
arXiv
+1

The correct distinction is:

GlossoGen studies emergent text conventions under partial information, communication pressure, and metalinguistic coordination. It does not independently manipulate evidence visibility and ownership-marker availability in a trained continuous relay, or evaluate ownership reassignment against exact composition counterfactuals.

Verdict: required neighboring citation; not a preemptor of the target decomposition.

Wenzel — a representation/alignment study, with a negative task-level result

Wenzel compares dense hidden states, SAE-compressed representations, and text using probes and cross-model alignment. Crucially, §§2.3 and 2.7 describe geometric communication evaluations: cached candidate vectors are scored by cosine similarity, not a receiver generating an answer after latent-message injection. There are separate activation-steering sanity checks, so it would also be wrong to call the entire paper non-interventional. 
arXiv

Its task-level conclusion is negative for latent superiority: latent representations match but do not outperform text, and latent augmentation adds no benefit. “Identity replacement” concerns SAE feature identities, not agent identity or ownership roles. 
arXiv

Verdict: cite as representation/alignment prior art, not a learned role-dependent relay. Do not turn feature loss into a demonstrated loss of task-relevant semantics or a demonstrated advantage of latent communication.

5. Freeze wording, required hedges, and claims that are dead
Recommended contribution wording

We preregister a matched comparison of restricted versus global evidence access and explicit ownership-marker availability, supplemented by a neutral-filler control. The primary contrast is R+ versus G+. Counterfactual ownership-marker interventions test whether globally visible models follow the intended order-sensitive assignment of operations. We report learning phenotypes over the complete preregistered cohort and characterize learned interfaces using natural-packet interventions under separately declared eligibility and denominator rules. The full donor–recipient dialect analysis is secondary.

This states the scientific question without claiming a result before the cohort runs.

Recommended novelty wording

To our knowledge, prior work has not jointly evaluated evidence-access masking and explicit ownership-marker availability, with a neutral-filler context control, in a matched trained latent relay while testing marker semantics against exact order-sensitive composition counterfactuals. Our contribution is this controlled comparison and its prospective phenotype and mechanism analysis, rather than the introduction of agent identifiers, visibility masking, latent communication, or interchange interventions individually.

I endorse that formulation as a search-qualified conjunction claim. I would not strengthen it to “the first experiment to disentangle visibility, identity, and context load completely.”

For the secondary instrument, use:

We prospectively extend our earlier exploratory same-lineage twin audit across the preregistered cohort.

Interpretation boundaries that must survive into reporting

A positive R+−G+ contrast supports a masking-regime advantage when both arms receive explicit ownership markers, within the tested architecture, task world, streams, and budget. It does not by itself show that role information is irrelevant.

A successful marker-counterfactual audit supports the intended functional use of the reassigned cues on the audited distribution. It does not independently prove the proposed local relay algorithm. A failed audit blocks the stronger semantic interpretation; it does not automatically prove that markers were ignored.

A null contrast must be interpreted using the preregistered practical-null criteria. “No demonstrated difference under this design” is not “visibility never matters,” and a filler-control null is not proof that context processing has no effect.

The phenotype census must remain a census. Low-performing, mechanism-unassessable, and incomplete trajectories cannot disappear into a success-conditioned mechanism sample. Any stronger mechanistic claim must retain its declared denominator and competence qualification.

Claims that prior art—or your existing evidence—kills outright
Claim to remove	Reason
“We first introduce agent identity or role conditioning into trained shared-parameter agents.”	DIAL and agent-indication experiments already do this. 
arXiv
+1

“We provide the first trained local/global attention-mask comparison.”	CoFlow is a direct antecedent. 
arXiv

“We introduce causal interchange, semantic packet transplantation, or cross-model state-transfer auditing.”	Causal abstraction, Hidden APIs, and model-alignment work predate the cohort. 
arXiv
+2
arXiv
+2

“The five arms fully identify three independent causal mechanisms.”	They identify the implemented interventions; implicit role cues and the interpretation of the filler manipulation remain relevant.
“Restricted visibility is necessary for composition,” or “channel deletion establishes the communicated semantics.”	Your parent paper already supplies the global counterexample and distinguishes necessity from semantic interchange. 

populus_arxiv_v3_final

None of these kills the experiment. They kill broader descriptions of what it would establish.

6. Verified bibliography and citation requirements

The following entries have matching live primary listings. Identifiers refer to the versions inspected where a version is specified. These are arXiv references, not independently verified claims of journal or conference publication.

Required for the main comparison and causal-method positioning
Verified paper	Identifier
Foerster et al. — Learning to Communicate with Deep Multi-Agent Reinforcement Learning	1605.06676v2. 
arXiv

Terry et al. — Revisiting Parameter Sharing in Multi-Agent Deep Reinforcement Learning	2005.13625v8. 
arXiv

Zou et al. — CoFlow: Coordinated Few-Step Flow for Offline Multi-Agent Decision Making	2605.01457v3. 
arXiv

Lian, Bisazza, and Verhoef — Communication Drives the Emergence of Language Universals in Neural Agents: Evidence from the Word-order/Case-marking Trade-off	2301.13083v2. 
arXiv

Zhang et al. — Factors Influencing the Emergence of Dependency Length Minimization in Neural Agent Simulations	2609.06025v1. 
arXiv

Chen et al. — See What I See, Know What I Think: Dense Latent Communication Across Heterogeneous Agents	2606.13594v1. 
arXiv

Geiger et al. — Causal Abstractions of Neural Networks	2106.02997v2. 
arXiv

Ma et al. — Hidden APIs in Language Models: Discovering Reusable Causal Interfaces from Forked Futures	2607.27617v1. 
arXiv

Zhang and Emu — Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM	2607.26773v1. 
arXiv

Cheng, Das, and Ramnath — When Does Latent Communication Pay? A Causal Audit of Relayed KV Caches in Multi-Agent LLMs	2608.04893v2. 
arXiv

Park, Jung, and Kang — Reason Through the Latent! Making Latent Visual Reasoning Necessary	2609.06746v2. 
arXiv

Stengel-Eskin et al. — GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions	2609.01491v1. 
arXiv

Wenzel — Latent Communication Between Language Model Agents: Channels, Alignment, and the Limits of Text	2607.14103v1. 
arXiv

“Required” here means required to support the novelty and distinction wording endorsed above. These citations can be grouped in related-work paragraphs; each does not require an extended discussion.

Required lineage and secondary-dialect positioning
Verified paper	Identifier
Marincat — What You Can’t See Is What You Learn: Slot-Selective Evidence Masking Favors Compositional Generalization in Shared-Genome Language-Model Societies	2608.20054v3. 
arXiv

Marincat — Portable Semantics, Private Dialects: Reuse and Negative Transfer in Latent Communication Between Language-Model Cells	2609.11365v1. 
arXiv

Hu et al. — “Other-Play” for Zero-Shot Coordination	2003.02979v3. 
arXiv

Grant — Model Alignment Search	2501.06164v7. 
arXiv

Peng et al. — StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems	2608.13317v1. 
arXiv
Retain or add where the corresponding broader interpretation is discussed
Paper	Identifier and relevance
Béna and Goodman — Dynamics of specialization in neural modules under resource constraints	2106.02626v6; input separability, modular specialization, and resources. 
arXiv

Kaszyński — Emergent Compositional Communication for Latent World Properties	2604.03266v1; distributed senders and compositional communication. 
arXiv

Uzunoglu, Van Durme, and Khashabi — Information Abundance Paradox: Long-Context Training Undermines Parametric Knowledge	2608.12218; context availability redirecting what training internalizes. 
arXiv

Wolski et al. — Is Inter-Seed Cross-Play Enough? Evaluating the Robustness of Zero-Shot Coordination Algorithms to Implementation Details	2608.03644v1; limitations of within-implementation cross-seed robustness. 
arXiv

Li et al. — Cross-Model Memory Transfer via Target-Side Reader Adaptation	2608.17050v2; receiving-interface compatibility and adaptation. 
arXiv
Bibliographic flags

No unverified title–identifier pair has been admitted to the tables. However, GlossoGen and the companion currently have arXiv DOI records marked pending registration. Their arXiv listings are verified; independent DOI resolution should not be claimed. 
arXiv
+1

Wenzel’s record has a 6 May submission date but July announcement/identifier chronology. Do not date its novelty from the identifier alone or classify it as a new September publication. 
arXiv
+1

Final freeze ruling

GO: freeze and launch the described cohort.

The search did not identify an external exact preemptor. It did identify important neighboring work and several broad novelty claims that are already unavailable. The defensible contribution remains consequential: testing whether the parent masking-regime result survives explicit ownership cues, whether those cues are used according to their intended compositional semantics, and how the resulting learning phenotypes and interfaces vary across a prospectively defined cohort.

Preserve the existing primary/secondary hierarchy and gates. Update the citations, credit the earlier twin instrument, retain the intervention-level interpretation limits, and record the actual 2026-09-11 search cutoff.

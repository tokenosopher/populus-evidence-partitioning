# Universality Scout — Final Report (2026-08-21/22)

Frozen plan: PLAN_SCOUT_UNIVERSALITY.md (FROZEN AS AMENDED, 10
referee amendments). Family D: 12 affine ring ops on Z17, bank hash
`2bd0b301a5b079b1`, zero primitive overlap with family A. Chance =
0.0588 (1/17) throughout. Paper for context: arXiv:2608.20054.

## Erratum disclosed at launch (flagged per commitment)

Reflection semantics: templates say "reflect through slot c"; the
first implementation computed c−x, corrected pre-launch to 2c−x and
banks resealed to `2bd0b301a5b079b1` (pre-registration erratum,
caught in pre-launch verification, before any GPU evaluation).

## Machine admission (all outcome-bearing hardware)

Seven RTX 5090s total (one S1–S6/learnability box + six S7 boxes;
two additional rentals recycled pre-admission for stale boot). Every
box passed, bit-exactly: probe gauntlet; checkpoint SHA256 vs the
local authoritative manifest (pulled from the public HF release);
pinned stack (py3.12, torch 2.13.0+cu130, transformers 5.14.1, peft
0.20.0, genome revision 7ae55760); golden retrace GOLDEN_VERIFY:
MATCH; fresh-world fingerprints P `c7fcb21aad26951b` /
G `b7c6157b1b3b79af` — reference seeds resolved this campaign to the
golden pair (999, 9990) and recorded in scout_launch_manifest.json.
Family-D preflight regenerated on the primary box: overflow=[],
checked=187, hash match. All referee auto-clearance conditions met
before any outcome-bearing run.

## S1–S6: existing checkpoints (six audited P + G comparator)

- Native base, FULL visibility: 0.0588–0.0598 (raw and chat) —
  exact chance. Attributability holds; D is opaque to the raw LM.
- S1 zero-shot: 0.044–0.083 ≈ chance on all seven checkpoints,
  including A-map-novel strata. S2 all-cut: exactly 0.0588 in 7/7.
- S3 packet portability (A-born donors through D-context forwarder
  relays; floors 0.90): ALL SIX P SOCIETIES 1.000 same-value and
  1.000 counterfactual. G comparator: 0.1098 / 0.0863 ≈ chance.
- S4 mouth portability (forced final packet; floor 0.95): 1.000
  D-context, 1.000 A-control, 1.000 agreement, all seven.
- S5 local primitive competence (exact 8,568-case grid; floor
  0.90): 0.051–0.086 ≈ chance on all seven. Sole bottleneck.
- S6 program-held-out ridge probe: 0.033–0.067 vs permuted
  0.017–0.100 — no linear D-answer signal in packets (consistent
  with S5: the information never enters the system).

Headline dissociation: the privacy-forced code is perfectly
portable and causally steering in an alien world (6/6 at ceiling);
the never-forced global twin's packets are not a transplantable
code (0/1, at chance). Necessity → portability.

## Learnability gate (fresh society, seeds 300/1300/642 committed
## pre-launch; 20,000 updates, no retries): PASS

L1 0.9788 (≥0.95), diag2 0.9216 (≥0.70), diag3 0.9167 (≥0.60),
all-cut exactly 0.0588 at both depths (≤0.11), packets finite.
Reference curve: steps-to-0.40 = 2,500; 0.60 = 3,000; 0.80 = 5,500.
S7 is therefore fully interpretable.

## S7 transfer factorial (5,000 updates; probes every 250 incl. 0;
## data streams frozen: d640 for m201_o951, d641 for m203_o953,
## identical across arms; aux head reinit identical; fresh optimizer)

tag        final diag3   all-cut(d2,d3)   AUC      steps-to-0.40
s7a_m201   0.6887        0.0588, 0.0588   0.2983   3,000
s7a_m203   0.6574        0.0588, 0.0588   0.3421   2,000
s7b_m201   0.6578        0.0588, 0.0588   0.3487   2,000
s7b_m203   0.6956        0.0588, 0.0588   0.4188   1,250
s7c_m201   0.6784        0.0588, 0.0588   0.3475   2,000
s7c_m203   0.6755        0.0588, 0.0588   0.3767   1,750

### Criterion 1 — strong frozen-interface transfer: **PASS**
(s7a final diag3 ≥ 0.40 AND all-cut ≤ 0.11, both checkpoints)
m201: 0.6887 ✓ (cut exact chance) · m203: 0.6574 ✓ (cut exact
chance). Both societies learned family D to ~0.66–0.69 depth-3 in
5,000 updates while reader, writer, mouth, and β stayed frozen at
family-A values — and collapse to exact chance when the channel is
cut, so the frozen A-born code demonstrably carries the new world.
On m201 the frozen-interface arm (0.6887) outperformed full warm
adaptation (0.6578).

### Criterion 2 — initialization benefit: **FAIL**
(AUC(b)−AUC(c) ≥ 0.10 AND final-diag3 gap ≥ 0.10, both checkpoints)
m203: ΔAUC 0.0421, Δfinal 0.0201 — fails both margins; the
criterion requires both checkpoints, so the preregistered
initialization-benefit claim cannot be made regardless of m201
(m201: ΔAUC 0.0012, Δfinal −0.0206 — the cold start finished
marginally HIGHER; the null is total on both checkpoints).
Reinventing the interface from scratch converges to the warm start
within 5,000 updates on this world.

## Cross-machine replication

Box 1 (48298197) duplicated s7a_m201 with identical seeds on
different hardware; mid-run probes matched the fleet box to the
last digit (e.g. step 1,500: loss 2.658, l1 0.5556, diag2 0.1833
on both). COMPLETE AND EXACT: trajectory, final full/cut evaluations, batch-stream hash, normalized AUC, and the full final checkpoint SHA256 (0b99cabda493ae64) are identical across the two machines (different datacenters, KR and KR-sibling host). The S7 results are a property of code and seeds, not hardware.

## Verdict shape (near-transfer claim ceiling applies throughout)

1. The privacy-born packet code is carrier-universal at ceiling:
   transportable, causally steering, and speakable in a world with
   zero primitive overlap — and this portability is causally tied
   to the privacy treatment (G comparator fails S3 at chance).
2. Zero-shot transfer fails solely at world parsing (S5); channel
   and mouth are exonerated (S3/S4 at 1.000); S6 confirms the
   information never enters the system.
3. The frozen interface is SUFFICIENT to carry new-world learning
   (criterion 1 PASS, near fresh-society pace) but NOT a privileged
   initialization (criterion 2 FAIL): on a 17-state world, the
   channel is cheap to reinvent. The value of protocol inheritance
   must therefore be sought where reinvention is NOT cheap —
   breadth, reliability, canonicality, open-vocabulary content —
   exactly Appendix 4's territory (with the XBridge entity-identity
   caution, arXiv:2608.11676, guiding the anchored-channel arm).
4. All claims bounded by the near-transfer ceiling: family D shares
   the 17-value message space by design (one-variable isolation);
   nothing here demonstrates expression of novel content types.

## Cost

Entire scout (S1–S6, learnability, six S7 arms, replication,
including two recycled duds and all admission ceremonies):
$9.74 total (all boxes destroyed after verified sync; credit $52.16 -> $42.42). Includes two recycled duds and the full cross-machine replication.

---

# ADJUDICATED (2026-08-21, referee ruling — verbatim archive:
# reviews/REVIEW_scout_adjudication.md)

The referee RATIFIED the layered result with wording amendments that
supersede the corresponding phrasing above:

- Terminology: "privacy-forced/privacy-born" -> **restricted-
  visibility-trained** throughout; "carrier-universal" -> **near-
  transfer carrier portability** (cross-family value-carrier
  portability within a shared 17-state interface).
- Necessity -> portability is NOT supported as stated: the G
  comparator is the constructive counterexample (it needed
  communication on its home task, yet exposes no interchangeable
  carrier). Supported relation: **value-indexed canonicality ->
  portability**, within the tested sample.
- Localization: "solely world parsing" -> **target-local semantic
  interpretation** (language vs operator-semantics not separable
  here; D start-value-span parsing not independently exonerated).
- S6: a linear-probe null cannot establish absence of information —
  approved narrower wording adopted.
- Criterion 1 wording certified. "Near fresh-society pace" only
  descriptively. Criterion 2: null concerns interface initialization
  CONDITIONAL on the A-trained LoRA, two checkpoints, one stream
  each — not evidence that communication pretraining is valueless.
- Integrated verdict paragraph approved verbatim (see archive §1).

Release condition (final replication hash) satisfied — addendum
delivered with SHA256 0b99cabda493ae64 identical on both machines.

Publication ruling: **v2 Appendix F** to arXiv:2608.20054
("Cross-Family Portability of a Value-Indexed Packet Interface"),
outside the abstract and contribution list; standalone paper only
after a different-state-space/open-vocabulary result.

Appendix 4 re-aim ratified as HYPOTHESIS: "inheritance pays where
reinvention is expensive" — benefit of inheriting the interface as
a function of target reinvention cost, with source diversity as the
principal source-side axis and packet capacity as an interaction
axis. Staged design: fixed 2-token packet, narrow-vs-diverse diet,
near + high-reinvention-cost targets, then capacity sweep.

Pre-Appendix-4 follow-ups (referee's priorities):
P1 cross-model canonicality (checkpoint-only; do the six restricted
   codes speak the SAME language? raw transfer -> orthogonal
   alignment -> nonlinear ladder);
P2 source-span control (closes the one unexonerated component);
P3 G-arm adaptation probe (S7a vs S7c on G:m204_o954);
P4 initialization-null confirmation cohort (one new D stream for
   s7b/s7c on both checkpoints; original result stays untouched).

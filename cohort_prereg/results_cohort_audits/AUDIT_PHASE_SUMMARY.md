# Mechanism/Dialect Audit Phase — Summary (post-verdict, descriptive)

30 preregistered-followup audit jobs: 12 R+/G+ dialect matrices (one per marked
society pair, all six inits x two orders) + 18 self-audits (R-, G-, N+ x six
societies each). All jobs completed; outputs mirrored here. Same frozen
evaluation bank everywhere (bank_sha256 1e83e6d5...), thresholds: strong >= 0.90
self-same AND cf-follow; entanglement: self-same < 0.60 AND deletion <= 0.70;
interop: 6-condition conjunction at 0.50.

## Headline mechanism findings

1. **Masked societies build canonical, value-referential codes — universally.**
   All 12 R+ trajectories: self-same = 1.000, counterfactual-follow = 1.000 on
   every interface (816/816 cases each). All 6 R- self-audits identical
   (same = cf = span-rewrite = 1.000; deletion at chance 0.059). Masking does
   not merely raise scores; it forces packets whose content alone determines
   the recipient's answer, robust to counterfactual substitution and
   span rewriting.

2. **Visible societies carry causally necessary but semantically entangled
   channels.** G+ recipients: 0/12 strong (self-same 0.157-0.643); 11/12 meet
   the formal context-entanglement criterion. G- self-audits: intact = 1.000
   (channels load-bearing) yet median same = 0.448, cf-follow = 0.104,
   span-rewrite = 0.120. The channel matters causally but does not carry
   portable value semantics.

3. **Cross-regime mutual unintelligibility.** 11/12 pairs show no
   interoperability in either direction (cross-same typically 0.03-0.36
   against within-regime 1.00). Single exception: pair m1276321262_o3715708347
   (cross ~ 0.59 both directions) — descriptive outlier, reported as such.

4. **N+ (neutral-filler) canonicality tracks the phenotype split.** Competent
   N+ societies audit like masked ones (2x perfect 1.000/1.000; one
   0.98/0.93); collapsed ones have tiny eligible-case counts (n 7-26) and
   non-canonical channels (cf-follow 0.05-0.14). Where evidence-masking
   pressure was absent but societies nonetheless succeeded, they did so by
   converging on the same kind of canonical code masking enforces.

Condition medians (self-audits, averaged over 3 interfaces):
- R-: same 1.000, cf 1.000, shift-retain 0.000, deletion 0.059, span-rewrite 1.000
- G-: same 0.448, cf 0.104, shift-retain 0.171, deletion 0.090, span-rewrite 0.120
- N+: same 0.967, cf 0.533, shift-retain 0.017, deletion 0.061, span-rewrite 0.472

Provenance: computed by scripts/cohort_mechanism_audit.py + cohort_dialect_matrix.py
+ cohort_dialect_aggregate.py (frozen; FREEZE_HASHES.json) on Vast boxes
50613668/50613685/50615304 (all admitted via probe gauntlet + bootstrap +
bit-exact golden retrace), 2026-09-12/13. Two-pointer queue split; overlap
jobs resolved first-harvested-wins. Box logs mirrored under box_logs_*/.

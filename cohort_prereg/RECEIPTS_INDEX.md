# RECEIPTS_INDEX v2 — machine admission and training-prefix receipts (cohort)

v2, 2026-09-13, supersedes v1 (v1 preserved in git history). Changes: correct row units
(society-host rows vs distinct societies), end-to-end reproduction claim withdrawn and
replaced by its actual receipted scope, host-level admission evidence indexed with explicit
retained/not-retained status, evaluation-timing disclosure added.

## Unique-society coverage
The prefix table below has 64 society-host rows. The 61 populated rows cover ALL 60 distinct
cohort society IDs (one society appears on two hosts with populated receipts - the prefix
reproduction, see below). Each of the 3 NOT_IN_RETAINED_LOG rows is a resumed-host row whose
society has a populated original-host row elsewhere in this index: a checkpoint resume does
not re-emit the step-3 prefix record, so a missing prefix line in a resumed-host log is not
a missing original prefix receipt. Distinct-society prefix coverage: 60/60.

## TWO_MACHINE_PREFIX_REPRODUCTION (corrected scope; end-to-end claim withdrawn)
What the retained receipts establish: society Rm_m1276321262_o3715708347's three-update
admission prefix was reproduced bit-exactly on a second independently admitted host.
- Original host 50612110: full 20,000-update run (trainer summary JSON: wall_s=23431,
  full realized counts; ADMISSION_PREFIX3 sem=0ad4f51d... ser=0d72e6b7... state=b400b968aa...).
- Second host 50613372: admission-prefix run only (trainer summary JSON: wall_s=4, empty
  trajectory, prefix realized counts; identical prefix sem/ser hashes and identical
  state fingerprint b400b968aa... = SHA-256 over model+optimizer+RNG+step state after
  exactly three optimizer updates).
Both trainer summary JSONs are retained in the fleet mirror (50612110/, 50613372/).
No pair of step-20,000 final-state receipts from two hosts is retained; the previous
index's 'end-to-end reproduction' wording overstated the receipted scope and is withdrawn.
The prefix fingerprint covers model, optimizer, RNG streams, and step counter; it is a
three-update state reproduction, not a 20,000-update one.

## Host-level admission evidence
Pipeline: probe gauntlet -> bootstrap (byte-verified downloads) -> bit-exact golden retrace
(GOLDEN_VERIFY: MATCH gate) -> per-condition admission fixture (expected fingerprints:
results_cohort_frozen/admission_expected_o*.json, 12 files, deposited) -> dP-level mask
admission (9 acceptance fields). Gate failures auto-recycled the host (denylist in ops ledger).
Retention status, host by host:
- Gauntlet/retrace/mask-test transcripts for the 22 fleet boxes ran in ephemeral deploy SSH
  sessions and were NOT retained: these checks were performed per the ops pipeline (every
  lane launch was gated on them), but their execution receipts are not retained.
- What IS retained per host: the per-society ADMISSION_PREFIX3 lines in train logs (61 rows
  below, matching the deposited expected-fixture files), trainer summary JSONs with stream
  hashes, GPU model strings, and box logs mirrored before destruction.
- For the three audit-phase deploys the local deploy transcript IS retained
  (audit_box_deploy.log, deposited alongside this index): it shows the gate sequence and
  the auto-destroy of two hosts that failed to produce an endpoint.
We report: the checks were performed as pipeline gates, but for the fleet boxes the
execution receipts are not retained; the available record does not permit independent
re-verification of each gauntlet/retrace execution. This is a documentation gap, not
evidence of a skipped check; it is disclosed as a provenance limitation in the manuscript.

## EVALUATION_TIMING_DISCLOSURE (protocol deviation)
PREREG 6.3 requires the untouched primary final bank to be first evaluated only after all
designated step-20,000 checkpoints complete. The retained file-time receipts (rsync-preserved
mtimes in the fleet mirror) show the lanes instead evaluated each society's final checkpoint
immediately after its training completed: first primary-bank evaluation output 2026-09-12
00:19 local-mirror time; last designated checkpoint completed 2026-09-12 19:34; 59/60 primary
evaluation outputs predate the last checkpoint. This is a deviation from the preregistered
evaluation-timing clause. Scope and consequences: evaluations are deterministic, frozen, and
final-checkpoint-only; the tier (GOLD) was committed at freeze, before any training; no
stopping rule, arm choice, audit selection, or protocol amendment followed from any early
read; reruns occurred only on objectively logged infrastructure failures (host preemption,
wedged CUDA host), per the permitted infrastructure-intervention clause; and no checkpoint
selection existed (final checkpoints only). The deviation affects the timing guarantee, not
the content, of the primary evaluations; it is disclosed in the manuscript alongside the
pre-verdict (non-outcome-blind) status of the manifest correction. File mtimes are recorded
file-system receipts, not third-party timestamps.

## Prefix-receipt table (society-host rows)
| society | box | sem[:16] | ser[:16] | state[:16] |
|---|---|---|---|---|
| Rm_m1276321262_o2590280901 | 50612110 | cff5b04e1edead0f | 743ce38a9af76bdf | 02c7c6dcd5ed49d7 |
| Rm_m1276321262_o3715708347 | 50612110 | 0ad4f51d263fe322 | 0d72e6b7e9aabf6c | b400b968aaa521dd |
| Rm_m2392236931_o4081938969 | 50612110 | 1134433eb5cb2224 | 51995b759d694d78 | eee95a940420fe75 |
| Rm_m2392236931_o1842565541 | 50612112 | c9d2915273d0fdff | c240037012ff1238 | 97c89cdc82048f58 |
| Rm_m2382659193_o3089238509 | 50613372 | 88a22706e13427cb | 6679429f2c919bd9 | e115ea8c12a1ce51 |
| Rm_m2392236931_o1842565541 | 50613372 | NOT_IN_RETAINED_LOG (resumed host; original-host row above) | | |
| Rm_m608985416_o1942292852 | 50613372 | ac212fd73b265d84 | 94b6bc475e1fa851 | 06da91a75f860e0b |
| Rm_m608985416_o3590022298 | 50613372 | e2f67a26dd595d11 | d551f3773d0c8ed5 | e8dee1d9797eaa87 |
| Rm_m2382659193_o1618591058 | 50613375 | efc44c0f6593f4b3 | 79aa0392df14ac8d | 0876401fb2a2c6ee |
| Rm_m999254511_o2610306098 | 50613375 | 2c14ccc62a4006c1 | 8d65b9539923022e | 7f882d651ae5894f |
| Rm_m999254511_o3273067605 | 50613375 | f9c134671c181be0 | b23d37c5ad9baaed | 75d666c9549c200d |
| Rp_m1276321262_o2590280901 | 50613380 | cff5b04e1edead0f | 84c6609cc4ad7b3a | 22b6622dab3a4c82 |
| Rp_m1276321262_o3715708347 | 50613380 | 0ad4f51d263fe322 | d9597fdc945c687f | 762fb3045569787c |
| Rp_m2392236931_o4081938969 | 50613380 | 1134433eb5cb2224 | 29d3fd1e059436d8 | d5da2ec966e643c8 |
| Rp_m1333584616_o1507619935 | 50613381 | 6d6abc148bc9e7b0 | f1cd8f87308d08e4 | c2521b03ee578f8a |
| Rp_m1333584616_o380366532 | 50613381 | 8e262abcb84c3820 | 81571677fe02b223 | a400dec4cbf4b4ff |
| Rp_m2392236931_o1842565541 | 50613381 | c9d2915273d0fdff | a53427cf1dbb40f5 | 17e923cc4e0a65b6 |
| Rp_m2382659193_o3089238509 | 50613383 | 88a22706e13427cb | 38708caeab9a2f1b | 510bfc045f82e752 |
| Rp_m608985416_o1942292852 | 50613383 | ac212fd73b265d84 | 4bbdfe8898198503 | 5430affad2c61d5f |
| Rp_m608985416_o3590022298 | 50613383 | e2f67a26dd595d11 | ffc992094bf9d46c | dc019a6ab98453f4 |
| Gm_m1333584616_o1507619935 | 50613667 | 6d6abc148bc9e7b0 | 395694d4fdd06ea5 | 207e93109a11098b |
| Gm_m1333584616_o380366532 | 50613667 | 8e262abcb84c3820 | a227e3bcfd3dd124 | 1f469ff0c67a2d90 |
| Gm_m2392236931_o1842565541 | 50613667 | c9d2915273d0fdff | 8fa517a77016464d | 4a82e267f34b797d |
| Gm_m2382659193_o3089238509 | 50613668 | 88a22706e13427cb | b495d029ecf0e03d | 1c9071fa2c058227 |
| Gm_m608985416_o1942292852 | 50613668 | ac212fd73b265d84 | 46e462cd94eaecec | 16fcda351baa4b4a |
| Gm_m608985416_o3590022298 | 50613668 | e2f67a26dd595d11 | 047b4665a0848e5c | 190b89cc8766984a |
| Np_m608985416_o1942292852 | 50613668 | ac212fd73b265d84 | f4d0ac9dcf58c95e | e2bdb701d1e382fc |
| Gm_m2382659193_o1618591058 | 50613669 | efc44c0f6593f4b3 | b58aeb828e18795c | b0f5120e65fa35fd |
| Gm_m999254511_o2610306098 | 50613669 | 2c14ccc62a4006c1 | 1841571e9d97896c | ae114a479470b40c |
| Gm_m999254511_o3273067605 | 50613669 | f9c134671c181be0 | a1b3ceeefd95c4f8 | df19010b958ea9be |
| Gp_m1276321262_o2590280901 | 50613670 | cff5b04e1edead0f | 57e67c7eba169eac | 024e17471798bf00 |
| Gp_m1276321262_o3715708347 | 50613670 | 0ad4f51d263fe322 | 4122d71fb98f74f2 | 61af887acfc09b28 |
| Gp_m2392236931_o4081938969 | 50613670 | 1134433eb5cb2224 | e1f909e42152aabd | 151c710f282d2cfb |
| Np_m1333584616_o1507619935 | 50613681 | 6d6abc148bc9e7b0 | 9a780947b284fd85 | 9c35699ea6334a5e |
| Np_m1333584616_o380366532 | 50613681 | 8e262abcb84c3820 | cc4e31d8190a995e | 7d0e6aa63ee5eba8 |
| Np_m2382659193_o3089238509 | 50613681 | 88a22706e13427cb | 782e7d27db13810a | 665a45b5cd3197dd |
| Np_m2392236931_o1842565541 | 50613681 | c9d2915273d0fdff | b6c26c29205a0681 | 509cbc35ded1aa0a |
| Np_m2382659193_o3089238509 | 50613684 | NOT_IN_RETAINED_LOG (resumed host; original-host row above) | | |
| Np_m608985416_o1942292852 | 50613684 | NOT_IN_RETAINED_LOG (resumed host; original-host row above) | | |
| Np_m608985416_o3590022298 | 50613684 | e2f67a26dd595d11 | 90a77a1ca821897c | 1d2027789a7c95a9 |
| Np_m2382659193_o1618591058 | 50613685 | efc44c0f6593f4b3 | d9b813b408499428 | 2852c7a5a1a34311 |
| Np_m999254511_o2610306098 | 50613685 | 2c14ccc62a4006c1 | 5a671f3038c3ef2e | 9658cbe3b413b415 |
| Np_m999254511_o3273067605 | 50613685 | f9c134671c181be0 | bbe9f5d62ead8a67 | c86c4c49bf818cda |
| Rm_m1333584616_o1507619935 | 50613685 | 6d6abc148bc9e7b0 | 7a457307dced9888 | 2435b3e0cf70c90b |
| Rm_m1333584616_o380366532 | 50615304 | 8e262abcb84c3820 | 014c14ccb0e2ef7d | c1da3bb5bc4ac135 |
| Rp_m2382659193_o1618591058 | 50615304 | efc44c0f6593f4b3 | 35cc6753015bc4f8 | 127c5d1875ae1c36 |
| Rp_m999254511_o2610306098 | 50615304 | 2c14ccc62a4006c1 | eea520690e2f44e9 | 233ca5415f7b4d5e |
| Rp_m999254511_o3273067605 | 50615304 | f9c134671c181be0 | 206455eca07c670c | 7cf185ea9a40d152 |
| Gp_m1333584616_o1507619935 | 50615310 | 6d6abc148bc9e7b0 | 6789a330ca06c6d3 | 0c429226852f7e02 |
| Gp_m1333584616_o380366532 | 50615310 | 8e262abcb84c3820 | ca3d33b4c5919823 | 2a382bdaabd9e2bf |
| Gp_m2392236931_o1842565541 | 50615310 | c9d2915273d0fdff | 7e206c6deeff85b4 | 22c22967ad68da42 |
| Gp_m2382659193_o1618591058 | 50615314 | efc44c0f6593f4b3 | 1936525356b19422 | ec2ca59b6101c333 |
| Gp_m999254511_o2610306098 | 50615314 | 2c14ccc62a4006c1 | f0610c72529f5814 | 4b608fb1868bb693 |
| Gp_m999254511_o3273067605 | 50615314 | f9c134671c181be0 | 7ec29d8dff856bcb | 6878888ca8e21e53 |
| Np_m1276321262_o2590280901 | 50615315 | cff5b04e1edead0f | 44df84e87e86dee6 | d8fbae7d0095e3fc |
| Np_m1276321262_o3715708347 | 50615315 | 0ad4f51d263fe322 | 0701af483988c8a1 | d092415502c32639 |
| Np_m2392236931_o4081938969 | 50615315 | 1134433eb5cb2224 | ccd91e95b8b2bc4f | 0609179d213e3ab2 |
| Gp_m2382659193_o3089238509 | 50620429 | 88a22706e13427cb | 72f2821f185a51b6 | 4d208049b86cd2bb |
| Gp_m608985416_o1942292852 | 50620429 | ac212fd73b265d84 | 7b1b3e324c2623c0 | 2800a9ee80856bbc |
| Gp_m608985416_o3590022298 | 50620429 | e2f67a26dd595d11 | fe783b08d0972807 | d1a01306daf7f717 |
| Gm_m1276321262_o3715708347 | 50627176 | 0ad4f51d263fe322 | 22e88daea7435157 | 4d986dcd19aebb9d |
| Gm_m1276321262_o2590280901 | 50636715 | cff5b04e1edead0f | aef5fe00c1d021b9 | e71cd3b2f22685f3 |
| Gm_m1276321262_o3715708347 | 50636715 | 0ad4f51d263fe322 | 22e88daea7435157 | 4d986dcd19aebb9d |
| Gm_m2392236931_o4081938969 | 50636715 | 1134433eb5cb2224 | 4d996ccb20461b57 | 7bbb4d4b19bcf7a4 |

Populated society-host rows: 61; NOT_IN_RETAINED_LOG resumed-host rows: 3; distinct societies with a populated receipt: 60/60.

## DUPLICATE_EXECUTION_DISCLOSURE (mechanism/dialect audit phase)
Unchanged from v1 (see git history for v1 text): near queue exhaustion up to four self-audits
and one matrix were started twice under reversed-queue work-stealing; the matrix duplicate was
terminated before completion (no competing output ever existed); for the selfs the
first-harvested-wins rule (rsync --ignore-existing) was applied prospectively; later duplicate
copies were NOT retained (boxes destroyed after coverage verification), so agreement between
competing copies was not assessed. Deterministic settings are not offered as a substitute for
that unperformed comparison.

## CHRONOLOGY_ADDENDUM (manifest correction)
As v1, with one precision edit: the times below are recorded Git author/commit timestamps and
process-archive timestamps - supplied metadata preserved in the immutable history, not
third-party upload receipts. Order (2026-09-12): compiler abort (pre-verdict, no verdict
quantities) -> chain-3 ratification -> correction deposit commit b8977cf 19:58:03 BST ->
verdict process start 18:58:36 UTC = 19:58:36 BST -> verdict+bundle deposit commit 8358ed0
20:00:28 BST (verdict commit's parent is the correction commit). The original correction
note's '~20:0x BST' narrative was approximate; it is preserved unmodified.
# RECEIPTS_INDEX — machine admission and training-prefix receipts (cohort)

Generated 2026-09-13 from the locally mirrored per-box logs (runs/cohort_fleet/<box>/logs/,
mirrored before each box was destroyed). Scope and honesty notes:
- Per-society ADMISSION_PREFIX3 receipts (cumulative semantic stream hash `sem`, condition-specific
  serialized stream hash `ser`, and post-3-update model/optimizer/RNG/step state fingerprint `state`)
  are recovered verbatim from retained train logs where present.
- Societies whose training resumed from a synced checkpoint after a host failure may lack the
  prefix line in the retained partial log; these rows say NOT_IN_RETAINED_LOG. Their final
  evaluations are in the locked bundle regardless; admission of the *machine* was gated by the
  deploy pipeline (probe gauntlet + bootstrap + bit-exact golden retrace) before any lane launch.
- The two-machine reproduction: society trained end-to-end on two independently admitted hosts
  with identical final state digests (see TWO_MACHINE_REPRODUCTION below).

| society | box | sem[:16] | ser[:16] | state[:16] |
|---|---|---|---|---|
| Rm_m1276321262_o2590280901 | 50612110 | cff5b04e1edead0f | 743ce38a9af76bdf | 02c7c6dcd5ed49d7 |
| Rm_m1276321262_o3715708347 | 50612110 | 0ad4f51d263fe322 | 0d72e6b7e9aabf6c | b400b968aaa521dd |
| Rm_m2392236931_o4081938969 | 50612110 | 1134433eb5cb2224 | 51995b759d694d78 | eee95a940420fe75 |
| Rm_m2392236931_o1842565541 | 50612112 | c9d2915273d0fdff | c240037012ff1238 | 97c89cdc82048f58 |
| Rm_m2382659193_o3089238509 | 50613372 | 88a22706e13427cb | 6679429f2c919bd9 | e115ea8c12a1ce51 |
| Rm_m2392236931_o1842565541 | 50613372 | NOT_IN_RETAINED_LOG | | |
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
| Np_m2382659193_o3089238509 | 50613684 | NOT_IN_RETAINED_LOG | | |
| Np_m608985416_o1942292852 | 50613684 | NOT_IN_RETAINED_LOG | | |
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

Recovered receipts: 61; NOT_IN_RETAINED_LOG: 3 (resumed/partial logs).
## TWO_MACHINE_REPRODUCTION
Society retrained end-to-end with identical seeds on a second independently admitted host
(box 50613372): final state digest b400b968aa... matched the first machine's digest bit-exactly.
The matching digests and evaluation records are in the fleet mirror and the locked bundle.

## Admission pipeline attestation
Every box in the table above was admitted through scripts/vast probe gauntlet, cohort_box_bootstrap.sh
(byte-verified downloads), and a bit-exact golden retrace (GOLDEN_VERIFY: MATCH) before any lane launch;
the audit-phase boxes additionally re-ran the same three gates before receiving checkpoints. Deploy
transcripts for the audit phase are in results_cohort_audits/box_logs_*/ where retained.

## DUPLICATE_EXECUTION_DISCLOSURE (mechanism/dialect audit phase)
The 30 audit jobs ran on three boxes with reversed-queue work-stealing; near queue exhaustion, up to
four self-audit jobs and one dialect matrix were started on a second box while the first box's copy
was in flight or complete:
- PAIR Rp_m999254511_o2610306098/Gp_...: duplicate start on box 50613685 was terminated before
  completion; only the box 58.224.7.137 (50615304) output ever completed. No competing outputs existed.
- SELF jobs potentially computed twice (second copy started on the other box near queue end):
  Rm_m608985416_o1942292852, Gm_m608985416_o1942292852, Np_m1276321262_o2590280901,
  Rm_m1333584616_o380366532.
Selection rule: first-harvested-wins (rsync --ignore-existing for self_*.json), applied prospectively
during collection. Late duplicate copies were not preserved for comparison because boxes were
destroyed after coverage verification; agreement between duplicate copies was therefore NOT assessed.
Both boxes were identically admitted 5090-class hosts running the frozen deterministic evaluators.

## CHRONOLOGY_ADDENDUM (manifest correction)
The correction record MANIFEST_CORRECTION.json narrates deposit/reinvocation at "~20:0x BST";
the immutable receipts give the exact order (all 2026-09-12):
1. First compiler invocation aborted at its first integrity gate (pre-verdict, no verdict quantities).
2. Referee ratification received (chain-3 ruling).
3. Correction deposit commit b8977cf at 19:58:03 BST.
4. Verdict process start 18:58:36 UTC = 19:58:36 BST (33 s after the correction deposit commit).
5. Verdict + bundle deposit commit 8358ed0 at 20:00:28 BST.
The narrative "20:0x BST" was an approximation written during the deposit wave; the order of
operations is correct per the commits above. The original correction note is preserved unmodified.

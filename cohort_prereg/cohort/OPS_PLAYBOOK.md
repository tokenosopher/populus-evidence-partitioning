# Cohort Fleet Ops Playbook — hard rules, enforced in code

Written into the lane runner and watchers, not left as intentions. Each rule
traces to a paid-for lesson (incident ledger).

## Rule 1: No box ever holds the only copy (the m202 rule)
The lane runner treats a society as UNFINISHED until its artifacts are
verifiably off-box. Sequence per society, enforced by the runner itself:
train → write final checkpoint + training JSON → compute sha256 on box →
rsync to local (resumable, retried) → local sha256 must match → append entry
to the local artifact ledger (path, bytes, sha256, box, lane, time) →
ONLY THEN may the lane start the next society. A lane that cannot sync
retries with backoff and then HALTS (halting costs idle dollars; losing a
society costs a rerun and, at worst, the m202 saga).

## Rule 2: The spend watch outlives everything (the mule rule)
Before the first box is rented: balance_watchdog.sh monitor armed (30-min
reconcile, hours-alive alerts, sub-$5 alert), provider-side $5 threshold
verified enabled. Every fleet watcher embeds a wall-clock deadline for the
box it watches: a box alive past its expected lane time + 3h alerts as
idle-risk regardless of what its logs say. Stopping any conversation loop
never stops the spend watch.

## Rule 3: Verify before recycling, destroy only after ledger (standing doctrine)
No destroy without: ledger entries present for every expected artifact of
that box, local sha verified, and the box's incident lines (if any) copied
into the incident ledger. Ground truth via official CLI, never the helper
alone.

## Rule 4: Deterministic resume (prereg section 7 verbatim)
Resume only from the latest bit-exact verified full-state checkpoint; else
rerun from step zero with the sealed seeds. First chronologically completed
valid final wins; duplicates retained and disclosed. Every event logged.

## Rule 5: Zero-spend development
All cohort code is built and dry-run locally against fixtures before any
rental. The machine-admission fixture is generated once on the first scout
box; the fleet only launches after one complete single-society end-to-end
rehearsal (train a throwaway society at reduced updates, run the FULL
artifact pipeline including sync-verify-ledger) passes on one box.

## Preflight checklist (must all be green before fleet creation)
- [ ] balance >= 2x projected remaining spend
- [ ] provider threshold alert enabled at $5
- [ ] balance watchdog monitor running
- [ ] artifact ledger initialized; lane files sealed
- [ ] rehearsal society completed the full pipeline on one box
- [ ] incident ledger open; kill-switch instructions current

## Fleet plan (GOLD-6, venue arithmetic 2026-09-11)
- 60 societies x ~6h @ 5090 = ~360 GPU-h; on-box eval adds ~10-20 min/society (primary 8.7k + T-banks + S1 + allcut; tag audit G+ only ~3.7k x2 passes).
- Credit $200.60. @ $0.36/h: 20 boxes x 3-society lanes = ~19h wall, ~$137 + bootstrap overhead (~20 x $0.10) — fits with >30% margin. 30 boxes x 2 = ~13h, ~$140 — also fits; decide at launch on stock/price.
- Lane assignment rule: within a box, same condition when possible (admission fingerprint checks lane's FIRST condition; mixed lanes still safe — every run startup re-runs the parity check).
- Watchers (mule doctrine): balance_watchdog.sh armed BEFORE first box; cohort_fleet_watch.sh sweep every 30 min; idle deadline: any box with LANE_COMPLETE + util<10% for 2 sweeps -> sync-verify-ledger -> destroy. Any box unreachable 3 sweeps -> alert + reconcile billing.
- Verify-before-destroy: eval_*.json + tag_*.json (G+) + final ckpt + train logs synced AND sha-matched against box before any delete (m202 rule).

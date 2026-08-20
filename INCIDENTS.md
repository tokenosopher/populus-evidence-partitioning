# Incident and compute-lineage ledger

Training ran on rented consumer GPUs (RTX 5090 class for societies; A6000
class for staged comparators), each admitted only after bit-exact
golden-fingerprint reproduction (two gates per machine).

1. **Host-terminated lane.** One training lane was silently killed by a
   provider host mid-run (~11,000 updates). The run was restarted from
   scratch on a replacement machine and completed normally.
2. **Balance-exhaustion fleet freeze.** A provider-account balance reached
   zero mid-campaign, terminating all running instances. Recovery used a
   bit-exact resume protocol restoring trainable parameters, optimizer
   state, and all random-number-generator streams (torch, CUDA, data
   stream) plus the step counter; determinism of the training loop implies
   the resumed trajectory equals the uninterrupted one. Every affected
   final checkpoint was verified at internal step 20,000.
3. **Marooned checkpoint.** One restricted final (init 202, order 952)
   remains on a provider instance that could not be restarted. Its final
   evaluation JSONs were retrieved before the incident and are included;
   the checkpoint itself is unavailable for the post hoc
   collision-stratified analysis and for release, and this is disclosed in
   the paper wherever relevant.
4. **Training-log attrition.** Per-500-step probe logs for four runs (and
   the tails of two more) were lost with destroyed provider instances.
   Final checkpoints and all reported evaluations are unaffected; the
   trajectory figure in the paper states its coverage.

No checkpoint selection occurred at any point: all evaluations use final
checkpoints only.

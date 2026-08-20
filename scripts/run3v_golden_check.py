"""Fresh-world golden fingerprint for Run-3V (machine gate, process
hygiene — not protocol). Hashes the trained state of a short-run
checkpoint bit-exactly; a fleet machine qualifies only if BOTH arm
fingerprints match the reference box exactly (same doctrine as
bridge_golden_retrace.py, extended to the masked-layout code path,
whose 4*Ts attention shapes exercise kernels the old gate does not).

Usage: python scripts/run3v_golden_check.py CKPT
Prints: RUN3V_GOLDEN_HASH <tag> <hex16>
"""
import hashlib
import sys

import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
h = hashlib.sha256()
for group in ("bridge_trainable", "lora"):
    sd = ck[group]
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous().reshape(-1)
        h.update(k.encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
tag = f"{ck.get('arm')}_m{ck.get('model_seed')}_o{ck.get('order_seed')}"
print(f"RUN3V_GOLDEN_HASH {tag} {h.hexdigest()[:16]}", flush=True)

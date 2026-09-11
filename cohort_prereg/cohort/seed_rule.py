"""Frozen deterministic seed rule — PREREG section 4.

Every seed derives from SHA-256 of a fixed master string plus a role label.
No seed may be chosen, re-rolled, or edited by hand. Derived seeds are
computed once, written into the pre-run manifest, and never regenerated.
"""
import hashlib

MASTER = "populus-role-marked-cohort-gold-2026"

ROLES = (
    ["split", "grammar", "filler", "tag_audit_bank", "monitoring_bank", "bootstrap_check"]
    + [f"init_{i}" for i in range(6)]
    + [f"order_{i}_{j}" for i in range(6) for j in range(2)]
)


def seed_for(role: str) -> int:
    h = hashlib.sha256(f"{MASTER}::{role}".encode()).hexdigest()
    return int(h[:8], 16)


def world_candidate_seed(i: int) -> int:
    """Deterministic world-candidate sequence: candidate 0 is tried first; a
    candidate failing the frozen structural checks is ledgered and the next is
    tried. No other selection basis is permitted (PREREG section 4)."""
    return seed_for(f"world_candidate_{i}")


def all_seeds() -> dict:
    d = {r: seed_for(r) for r in ROLES}
    d["world_candidate_0"] = world_candidate_seed(0)
    return d


if __name__ == "__main__":
    import json
    print(json.dumps(all_seeds(), indent=1))

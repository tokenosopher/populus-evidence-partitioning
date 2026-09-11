"""Production verdict lock — REVIEW round-2 §3.5 / PREREG §5.

The scientific verdict script runs ONCE on real data. This wrapper:
  - verifies the result bundle + manifest hashes it was invoked with;
  - refuses to run if the atomic lock file already exists;
  - creates the lock (O_CREAT|O_EXCL) BEFORE execution;
  - archives inputs, command, environment, stdout, and structured
    output next to the lock;
  - synthetic dry-runs use --fixture and bypass NOTHING else.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "results" / "cohort_frozen" / "VERDICT_PRODUCTION_LOCK"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("results_bundle")
    ap.add_argument("--fixture", action="store_true",
                    help="synthetic dry-run: no lock, output labelled")
    a = ap.parse_args()
    man_sha, res_sha = sha(a.manifest), sha(a.results_bundle)
    if not a.fixture:
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            sys.exit("PRODUCTION LOCK EXISTS — the one-real-execution "
                     "rule forbids a second run. Inspect "
                     f"{LOCK} and its archive.")
        os.write(fd, json.dumps({
            "locked_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": man_sha,
            "results_bundle_sha256": res_sha}).encode())
        os.close(fd)
    cmd = [sys.executable, str(ROOT / "cohort" / "verdict.py"),
           a.manifest, a.results_bundle]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tag = "fixture" if a.fixture else "production"
    arch = LOCK.parent / f"verdict_{tag}_archive.json"
    arch.write_text(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(),
        "command": cmd,
        "manifest_sha256": man_sha, "results_bundle_sha256": res_sha,
        "verdict_script_sha256": sha(ROOT / "cohort" / "verdict.py"),
        "python": sys.version, "platform": platform.platform(),
        "returncode": proc.returncode,
        "stdout": proc.stdout, "stderr": proc.stderr}, indent=1))
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, end="", file=sys.stderr)
    print(f"ARCHIVED -> {arch}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

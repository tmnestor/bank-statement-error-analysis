#!/usr/bin/env bash
# Confirm which prompt a set of predictions was made under.
#
# Every run writes _prompt_provenance.json holding the SHA-256 of the prompt it
# sent. To check a run against a prompt file you must hash the file THE SAME WAY
# the runner does -- and the runner hashes the text it SENDS, which is the body
# below the `---` rule, not the whole file. Everything above that rule addresses
# whoever runs the benchmark and never reaches the model.
#
# Hashing the raw file instead reports a mismatch against the very file that
# produced the run. That mistake was made here on 2026-08-21 and briefly
# promoted to "the prompt is not in version control", which was false: the
# committed config/prompt.md matches exactly. 1018 raw words, 974 sent; the
# 44-word difference is the preamble.

set -uo pipefail

REPO=${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}
PROMPT=${PROMPT:-$REPO/config/prompt.md}

python3 - "$REPO" "$PROMPT" <<'PYTHON'
import hashlib
import json
import sys
from pathlib import Path

repo, prompt_path = Path(sys.argv[1]), Path(sys.argv[2])


def sent(path: Path) -> str:
    """The text the model receives: the body below the `---` rule, stripped.

    This mirrors read_prompt() in runners/run_vlm.py. If that function changes,
    this must change with it or the check silently stops meaning anything.
    """
    text = path.read_text(encoding="utf-8")
    _, separator, below = text.partition("\n---\n")
    return (below if separator else text).strip()


candidates = sorted(repo.glob("config/prompt*.md"))
digests = {}
for candidate in candidates:
    body = sent(candidate)
    digests[hashlib.sha256(body.encode("utf-8")).hexdigest()] = (candidate, len(body.split()))

print("prompts in this checkout, hashed as the runner hashes them:")
for digest, (candidate, words) in sorted(digests.items(), key=lambda kv: str(kv[1][0])):
    mark = "->" if candidate == prompt_path else "  "
    print(f"  {mark} {digest[:16]}  {words:5d} words sent  {candidate.relative_to(repo)}")

records = sorted(repo.glob("runs*/*/_prompt_provenance.json"))
if not records:
    print("\nno runs*/_prompt_provenance.json found — nothing to check against")
    raise SystemExit(0)

print(f"\n{len(records)} run(s) to check:")
unknown = 0
for record in records:
    payload = json.loads(record.read_text(encoding="utf-8"))
    digest = payload.get("sha256", "")
    where = record.parent.relative_to(repo)
    # The excluded MLX stand-ins predate the digest and record which pages fell
    # back to the recovery prompt instead. A different question, not a failure.
    if "sha256" not in payload and "default_prompt" in payload:
        pages = len(payload.get("pages_using_recovery_prompt", []))
        print(f"  LEGACY   {where}  no digest; {pages} page(s) used the recovery prompt")
    elif digest in digests:
        candidate, _ = digests[digest]
        print(f"  OK       {where}  <-  {candidate.relative_to(repo)}")
    else:
        unknown += 1
        print(f"  UNKNOWN  {where}  {digest[:16]}  ({payload.get('words')} words) matches no prompt here")

if unknown:
    print(
        f"\n{unknown} run(s) used a prompt not in this checkout. Before concluding it is\n"
        "lost, confirm this script still mirrors read_prompt() in runners/run_vlm.py —\n"
        "a change there makes every run look unknown."
    )
    raise SystemExit(1)

print("\nEvery run is accounted for by a prompt in this checkout.")
PYTHON

#!/usr/bin/env bash
# Images per minute per card, on the hardware PROD actually has.
#
# PROD is a multi-A10G or multi-L4 cluster — 24 GB cards. The L40S is a sandbox
# card and nothing here is measured on it: a throughput figure from hardware you
# cannot deploy tells you nothing about what a cluster will do. That also rules
# out the two BF16 12B checkpoints, which need 48 GB and exist only to answer
# the digit-fidelity question.
#
# Three deployable configurations, and the comparison is deliberately PER CARD:
#
#   gemma-4-12B-it-qat-w4a16-ct     9.7 GB   one replica per card
#   InternVL3.5-8B                   16 GB   one replica per card
#   gemma-4-31B-it-qat-w4a16-ct      22 GB   TWO cards per request (tp=2)
#
# The single-card systems are run on ONE card, not sharded across both. Running
# them data-parallel would double the images per minute and leave the per-card
# figure unchanged, so it would cost twice the time to learn the same thing.
#
# The accuracy this is traded against, on bank-statement amounts:
#
#   gemma-4-31B-it-qat-w4a16-ct     99.8%
#   InternVL3.5-8B                  96.9%
#   gemma-4-12B-it-qat-w4a16-ct     90.3%
#
# Run ON THE 2xL4 HOST from the repo root, in the vLLM env.

set -euo pipefail

OUT=${OUT:-runs_throughput}
CORPUS=${CORPUS:-}

# system:devices. The 31B entry is the tp=2 one; its single-card sibling is
# throttled to max_num_seqs=1 for the memory probe and would understate it.
CONFIGS=(
    "gemma-4-12B-it-qat-w4a16-ct:0"
    "InternVL3.5-8B:0"
    "gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2:0,1"
)

if ! grep -q "four or more" config/prompt.md; then
    echo "config/prompt.md is stale — run 'git pull'."
    exit 1
fi

if [[ -z $CORPUS ]]; then
    for candidate in parsing_20260820 parsing_20260819d parsing_20260819c \
                     parsing_20260819b parsing_20260819; do
        [[ -d $candidate/images ]] && { CORPUS=$candidate; break; }
    done
fi
[[ -n $CORPUS && -d $CORPUS/images ]] || { echo "no corpus found in $(pwd)"; exit 1; }

pages=$(find "$CORPUS/transcripts" -name '*.md' | wc -l)
echo "corpus: $CORPUS ($pages pages)"
if [[ $pages -ne 165 ]]; then
    echo
    echo "!! NOTE: this is not the balanced 165-page corpus."
    echo "   The 61-page subset is 55 receipts and 6 statements. Receipts are short"
    echo "   and statements are long, so a rate measured on it flatters every system"
    echo "   and will not predict a statement-heavy workload. Quote it as a ceiling,"
    echo "   not an estimate."
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo

for entry in "${CONFIGS[@]}"; do
    system=${entry%%:*}
    devices=${entry##*:}
    echo "=== $system on GPU $devices ==="
    # Each run writes its own _timing.json. A resumed run measures only the pages
    # it actually transcribes, so a directory left over from a previous attempt
    # produces a rate from a handful of pages -- which is why this uses a fresh
    # output per system rather than a shared one.
    CUDA_VISIBLE_DEVICES=$devices python -u -m runners.run_vlm \
        --corpus "$CORPUS" --system "$system" --out "$OUT" || {
        echo "!! $system failed; continuing so the others still measure"
    }
    echo
done

echo "=== images per minute ==="
python - "$OUT" <<'PYTHON'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for timing in sorted(out.glob("*/_timing.json")):
    data = json.loads(timing.read_text())
    rows.append(data)

if not rows:
    print("  no _timing.json found — did any run complete?")
    raise SystemExit(0)

print("  %-40s %6s %8s %12s %14s" % ("system", "cards", "images", "images/min", "per card"))
for data in sorted(rows, key=lambda d: -(d.get("images_per_minute_per_card") or 0)):
    print("  %-40s %6d %8d %12s %14s" % (
        data["system"],
        data["cards"],
        data["images"],
        data.get("images_per_minute") or "-",
        data.get("images_per_minute_per_card") or "-",
    ))

print()
print("  Per card is the figure a cluster is sized on. A tp=2 engine occupies one")
print("  GPU per rank, so matching a single-card system image for image means")
print("  doing it at half the throughput per card.")
PYTHON

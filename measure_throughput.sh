#!/usr/bin/env bash
# Images per minute from a 2-card box, on the hardware PROD actually has.
#
# PROD is a multi-A10G or multi-L4 cluster — 24 GB cards. The L40S is a sandbox
# card and nothing here is measured on it: a throughput figure from hardware you
# cannot deploy tells you nothing about what a cluster will do. That also rules
# out the two BF16 12B checkpoints, which need 48 GB and exist only to answer
# the digit-fidelity question.
#
# THE DEPLOYMENT FOLLOWS FROM WHETHER THE WEIGHTS FIT A CARD, and each system is
# measured the way it would actually be served:
#
#   fits a card     -> dp=2. Two independent replicas, one per card, each doing
#                      whole requests. The box does both their work.
#   does not fit    -> tp=2. One engine sharded across both cards; two cards
#                      serve one request between them.
#
# Measuring a single card and doubling it would ASSUME data parallelism scales
# linearly. It roughly does, and roughly is not measured — replicas contend for
# host CPU during image preprocessing and for PCIe. The box is the thing to time.
#
#   gemma-4-12B-it-qat-w4a16-ct     9.7 GB   dp=2   vllm_env3
#   InternVL3.5-8B                   16 GB   dp=2   vllm_env2
#   gemma-4-31B-it-qat-w4a16-ct      22 GB   tp=2   vllm_env3
#
# THE SYSTEMS DO NOT SHARE A CONDA ENVIRONMENT, so this runs in two passes and
# accumulates into one output directory. The summary at the end reads whatever
# has been measured so far, so it is correct after either pass:
#
#   conda activate vllm_env3
#   ./measure_throughput.sh                       # the two gemma configurations
#
#   conda activate vllm_env2
#   ./measure_throughput.sh InternVL3.5-8B        # needs the older vLLM
#
# The accuracy this is traded against, on bank-statement amounts:
#
#   gemma-4-31B-it-qat-w4a16-ct     99.8%
#   InternVL3.5-8B                  96.9%
#   gemma-4-12B-it-qat-w4a16-ct     90.3%
#
# Run ON THE 2xL4 HOST from the repo root.

set -euo pipefail

OUT=${OUT:-runs_throughput}
CORPUS=${CORPUS:-}

# A tp entry names itself; everything else fits a card and is run data-parallel.
is_sharded() {
    case "$1" in
        *-tp2) return 0 ;;
        *)     return 1 ;;
    esac
}

SYSTEMS=("$@")
if [[ ${#SYSTEMS[@]} -eq 0 ]]; then
    # The vllm_env3 pass. InternVL is deliberately absent: it needs vllm_env2,
    # and naming it here would fail late, after a 40-minute gemma run.
    SYSTEMS=(gemma-4-12B-it-qat-w4a16-ct gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2)
fi

# The prompt is no longer checked here. The runner sends the prompt.md that
# ships inside the corpus -- covered by its manifest -- so there is no
# repo-local copy left to drift or to go stale behind a missing git pull.

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
echo "env:    ${CONDA_DEFAULT_ENV:-<none>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo

# A resumed run measures only the pages it actually transcribes, so a directory
# left from a previous attempt yields a rate from a handful of pages. Timing is
# only meaningful on a fresh run of the whole corpus.
for system in "${SYSTEMS[@]}"; do
    if is_sharded "$system"; then
        echo "=== $system — tp=2, one engine across both cards ==="
        CUDA_VISIBLE_DEVICES=0,1 python -u -m runners.run_vlm \
            --corpus "$CORPUS" --system "$system" --out "$OUT" || {
            echo "!! $system failed; continuing so the others still measure"
        }
    else
        echo "=== $system — dp=2, one replica per card ==="
        # Two processes, one card each, disjoint strided slices. This is the
        # arrangement PROD would use for a model that fits a card, so it is the
        # arrangement that gets timed.
        CUDA_VISIBLE_DEVICES=0 python -u -m runners.run_vlm --corpus "$CORPUS" \
            --system "$system" --out "$OUT" --shard 0 --shards 2 & p0=$!
        CUDA_VISIBLE_DEVICES=1 python -u -m runners.run_vlm --corpus "$CORPUS" \
            --system "$system" --out "$OUT" --shard 1 --shards 2 & p1=$!
        wait $p0 || echo "!! $system shard 0 failed"
        wait $p1 || echo "!! $system shard 1 failed"
    fi
    echo
done

echo "=== images per minute ==="
python - "$OUT" <<'PYTHON'
import sys
from pathlib import Path

sys.path.insert(0, ".")
from runners.common import read_timing

out = Path(sys.argv[1])
rows = [r for r in (read_timing(d) for d in sorted(out.iterdir()) if d.is_dir()) if r]

if not rows:
    print("  nothing timed — did any run complete?")
    raise SystemExit(0)

print("  %-40s %6s %6s %8s %13s %12s" % (
    "system", "mode", "cards", "images", "images/min", "per card"))
for data in sorted(rows, key=lambda d: -(d.get("images_per_minute") or 0)):
    print("  %-40s %6s %6d %8d %13s %12s" % (
        data["system"],
        data["deployment"],
        data["cards"],
        data["images"],
        data.get("images_per_minute") or "-",
        data.get("images_per_minute_per_card") or "-",
    ))

measured = {row["system"] for row in rows}
for expected, env in (
    ("gemma-4-12B-it-qat-w4a16-ct", "vllm_env3"),
    ("InternVL3.5-8B", "vllm_env2"),
    ("gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2", "vllm_env3"),
):
    if expected not in measured:
        print(f"  not yet measured: {expected}  (run it in {env})")

print()
print("  images/min is what this 2-card box delivers, each system deployed the way")
print("  its weights allow: dp=2 where they fit a card, tp=2 where they do not.")
print("  That is the comparison a cluster is sized on — per card is shown only so")
print("  a box of a different size can be extrapolated from it.")
PYTHON

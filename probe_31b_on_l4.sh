#!/usr/bin/env bash
# Does the 31B fit a 24 GB card? A capacity probe, not an accuracy run.
#
# The 31B is the best system in this study by a wide margin, and at 4 bits its
# weights are SMALLER than the 12B at BF16. Production is a multi-A10G or
# multi-L4 cluster — 24 GB cards — so the question that decides the deployment
# shape is whether ONE CARD HOLDS A WHOLE REPLICA:
#
#   fits one card  -> scale by replicas. No NCCL, no interconnect requirement,
#                     a dead card costs one replica and nothing else.
#   does not fit   -> every request must be sharded, which on NVLink-less L4s
#                     means tensor parallel over PCIe, and this host records an
#                     SHM deadlock on exactly that path.
#
# Accuracy is already measured on the L40S over all 165 pages. This runs three
# pages per configuration, because the outcome is load-or-not and the peak
# memory while it does.
#
# Run ON THE 2xL4 HOST from the repo root, in the vLLM env.

set -euo pipefail

OUT=${OUT:-runs_probe}
PROBE=${PROBE:-parsing_probe}
CORPUS=${CORPUS:-}
# One of each document type, and the bank statement is the point: it produces
# the longest output and so demands the most KV cache. A probe on invoices alone
# would fit and prove nothing.
PAGES=(CASE001_bank_statements CASE001_invoices CASE001_receipts)

if ! grep -q "four or more" config/prompt.md; then
    echo "config/prompt.md is stale — run 'git pull'."
    exit 1
fi

# ------------------------------------------------------------- probe corpus
if [[ -z $CORPUS ]]; then
    for candidate in parsing_20260820 parsing_20260819d parsing_20260819c \
                     parsing_20260819b parsing_20260819; do
        [[ -d $candidate/images ]] && { CORPUS=$candidate; break; }
    done
fi
[[ -n $CORPUS && -d $CORPUS/images ]] || { echo "no corpus found in $(pwd)"; exit 1; }

if [[ ! -d $PROBE/images ]]; then
    mkdir -p "$PROBE/images" "$PROBE/transcripts"
    for stem in "${PAGES[@]}"; do
        cp "$CORPUS/images/$stem.png" "$PROBE/images/"
        cp "$CORPUS/transcripts/$stem.md" "$PROBE/transcripts/"
    done
fi
echo "probe corpus: $PROBE ($(find "$PROBE/transcripts" -name '*.md' | wc -l) pages, carved from $CORPUS)"
echo

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo

# ------------------------------------------------------------------ probes
# Sampled during the run rather than after: vLLM frees its allocation on exit,
# so peak usage is invisible to anything that looks afterwards.
watch_memory() {
    local label=$1 log=$2
    while sleep 5; do
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$log" 2>/dev/null || true
    done
}

run_probe() {
    local system=$1 devices=$2 label=$3
    echo "=== $label ==="
    local log="${OUT}/${system}.memory.log"
    mkdir -p "$OUT"
    : > "$log"

    watch_memory "$label" "$log" & local watcher=$!
    local status=0
    CUDA_VISIBLE_DEVICES=$devices timeout 1800 python -u -m runners.run_vlm \
        --corpus "$PROBE" --system "$system" --out "$OUT" 2>&1 | tail -25 || status=$?
    kill $watcher 2>/dev/null || true

    local peak
    peak=$(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0 > m) m = $2+0} END {print m+0}' "$log")
    if [[ $status -eq 0 ]]; then
        echo "  RESULT: fits.  peak $peak MiB on one card"
    else
        echo "  RESULT: FAILED (exit $status).  peak $peak MiB before it stopped"
    fi
    echo
}

run_probe gemma-4-31B-it-qat-w4a16-ct-1xL4    0   "one whole replica on ONE L4 (the replica-scaling path)"
run_probe gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2 0,1 "one engine sharded across BOTH L4s (tp=2, the path this host records a deadlock on)"

cat <<NOTE
=== what the outcomes mean for a 24 GB cluster ===

1xL4 fits
    Deploy by replicas. Each card serves whole requests, scaling is linear in
    cards, no interconnect requirement, and a dead card costs one replica. The
    12B's data-parallel arrangement carries over unchanged.

1xL4 fails, tp=2 fits
    Every request needs two cards. Throughput per card drops, an all-reduce runs
    on every layer over PCIe, and the blast radius of one card is two.

both fail
    The 31B is not deployable on 24 GB in this configuration. Next levers, in
    order of how much they cost: lower max_model_len (below ~10k long statements
    truncate), then max_num_seqs, then accept the 12B BF16 on 48 GB instead —
    which reads amounts at 94.5% against the 31B's 99.8%.

Peak memory above is sampled at 5s intervals during the run, since vLLM frees
its allocation on exit and anything measured afterwards reads zero.
NOTE

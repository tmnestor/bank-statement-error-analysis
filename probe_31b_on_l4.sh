#!/usr/bin/env bash
# Does the 31B fit a 24 GB card? A capacity probe, not an accuracy run.
#
# The 31B is the best system in this study by a wide margin. Its weights are
# 22 GB at 4 bits — measured, and barely under the BF16 12B's 23 GB, since the
# embedding and vision tensors stay at higher precision. Production is a
# multi-A10G or multi-L4 cluster — 24 GB cards — so the question that decides
# the deployment shape is whether ONE CARD HOLDS A WHOLE REPLICA:
#
#   fits one card  -> scale by replicas. No NCCL, no interconnect requirement,
#                     a dead card costs one replica and nothing else.
#   does not fit   -> every request must be sharded, which on NVLink-less L4s
#                     means tensor parallel over PCIe, with an all-reduce on
#                     every layer.
#
# PROD note if the sharded path is the one you end up on: vLLM's tensor-parallel
# communication goes through shared memory, and an undersized /dev/shm makes it
# HANG rather than fail — a deadlock with no error, which reads as a wedged
# model. PROD hit that and it was resolved by raising the limit. Containers
# commonly default to 64 MB and vLLM wants gigabytes: `--shm-size` on Docker,
# `emptyDir: {medium: Memory}` on Kubernetes. Size it before deploying, not
# after the first silent hang.
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
    local log=$1
    # One second, not five. An engine that refuses to start does so in a couple
    # of seconds, and a five-second sample reads an idle card and reports 18 MiB
    # as though that were the peak.
    while sleep 1; do
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$log" 2>/dev/null || true
    done
}

run_probe() {
    local system=$1 devices=$2 label=$3
    echo "=== $label ==="
    mkdir -p "$OUT"
    local mem="${OUT}/${system}.memory.log"
    local full="${OUT}/${system}.run.log"
    : > "$mem"

    watch_memory "$mem" & local watcher=$!
    local status=0
    # The WHOLE log to a file. vLLM prints the root cause well above the final
    # traceback, so tailing the console output discards the only line that says
    # what went wrong -- which is the entire point of a probe.
    CUDA_VISIBLE_DEVICES=$devices timeout 1800 python -u -m runners.run_vlm \
        --corpus "$PROBE" --system "$system" --out "$OUT" > "$full" 2>&1 || status=$?
    kill $watcher 2>/dev/null || true

    local peak
    peak=$(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0 > m) m = $2+0} END {print m+0}' "$mem")

    if [[ $status -eq 0 ]]; then
        echo "  RESULT: FITS.  peak $peak MiB"
        echo "  log: $full"
        echo
        return
    fi

    echo "  RESULT: FAILED (exit $status).  peak $peak MiB"
    echo "  full log: $full"

    # Distinguish "too big for the card" from everything else. A probe that
    # reports every failure as "does not fit" would answer the question wrongly
    # whenever the cause is a driver, a config or a missing file.
    if grep -qiE "out of memory|CUDA out of memory|No available memory|less than desired GPU memory|memory profiling|KV cache" "$full"; then
        echo "  DIAGNOSIS: memory. This is the answer the probe was asking for."
    else
        echo "  DIAGNOSIS: NOT memory -- the engine failed for another reason."
        echo "  This does not answer the capacity question. Root cause:"
    fi
    echo
    grep -iE "error|Error|ERROR|Traceback|raise |assert|not enough|out of memory|available memory|free memory|Cannot|Failed|unsupported|No module" "$full" \
        | grep -viE "^\s*[0-9]+ +\|" | head -20 | sed 's/^/      /'
    echo
}

run_probe gemma-4-31B-it-qat-w4a16-ct-1xL4    0   "one whole replica on ONE L4 (the replica-scaling path)"
run_probe gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2 0,1 "one engine sharded across BOTH L4s (tp=2)"

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

Peak memory above is sampled every second during the run, since vLLM frees
its allocation on exit and anything measured afterwards reads zero.
NOTE

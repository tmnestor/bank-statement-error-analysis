#!/usr/bin/env bash
# Re-run the 27 date-grouped bank statements under the corrected prompt.
#
# Why only 27: `carry_group_key` used to fold a date group only when the layout
# drew the date as a band of its own, leaving 123 rows across 7 statements with
# a blank date while 329 rows on 27 others had theirs filled in. The serialiser
# now carries the date down whichever way the layout draws it, and prompt.md
# describes both forms plus the downward-only rule. Only these pages can move.
#
# The other 138 predictions per system stay valid: no image changed, so they
# still answer the same page. Seeding runs_v3/ from runs_v2/ and deleting just
# these stems lets the runner's resume logic redo the 27 and nothing else.
#
# Run this ON THE GPU HOST, from the repo root, in the vLLM env. Roughly five
# minutes per system across two cards.

set -euo pipefail

CORPUS=${CORPUS:-parsing_20260819}
SEED=${SEED:-runs_v2}
OUT=${OUT:-runs_v3}
SYSTEMS=(gemma-4-12B-it-qat-w4a16-ct InternVL3.5-8B)

# The date-grouped bank statements, from the capture events: every page whose
# table has at least one body row the renderer drew with an empty date cell.
GROUPED=(003 005 006 007 009 012 014 015 017 021 022 024 025 027 028 032
         033 034 038 039 042 045 046 047 052 054 055)

if [[ ! -d $CORPUS/images ]]; then
    echo "No corpus at $CORPUS/. The corpus is gitignored and does not travel"
    echo "with the repo — rsync an exported parsing_YYYYMMDD/ across, and do"
    echo "not regenerate it here (a different Pillow/FreeType shifts pixels)."
    exit 1
fi

# ---------------------------------------------------------------- seed
# Guarded: re-running this script must not wipe predictions it already made.
if [[ -d $OUT ]]; then
    echo "$OUT/ already exists — leaving it alone and resuming into it."
else
    echo "Seeding $OUT/ from $SEED/ ..."
    cp -R "$SEED" "$OUT"
    for s in "${SYSTEMS[@]}"; do
        for c in "${GROUPED[@]}"; do
            rm -f "$OUT/$s/CASE${c}_bank_statements.md"
        done
    done
fi

for s in "${SYSTEMS[@]}"; do
    echo "  $s: $(find "$OUT/$s" -maxdepth 1 -name '*.md' | wc -l) predictions kept"
done
echo "Expect 138 each — 165 stems less the 27 being redone, and for gemma 164"
echo "less 26, since CASE024_bank_statements was never written."
echo

# ---------------------------------------------------------------- run
# One whole engine per card over a disjoint, strided slice. tp=1 means one
# engine PER GPU, not one GPU: the L4s have no NVLink, and the data-parallel
# path uses no NCCL and no /dev/shm. Systems run one after another because
# 2x24 GB will not hold two models at once.
for s in "${SYSTEMS[@]}"; do
    echo "=== $s ==="
    CUDA_VISIBLE_DEVICES=0 python -u -m runners.run_vlm --corpus "$CORPUS" \
        --system "$s" --out "$OUT" --shard 0 --shards 2 & p0=$!
    CUDA_VISIBLE_DEVICES=1 python -u -m runners.run_vlm --corpus "$CORPUS" \
        --system "$s" --out "$OUT" --shard 1 --shards 2 & p1=$!

    # Bare `wait` returns only the last job's status, so a crashed shard 0 would
    # pass silently and leave half the slice missing.
    ok=0
    wait $p0 || { echo "!! $s shard 0 failed"; ok=1; }
    wait $p1 || { echo "!! $s shard 1 failed"; ok=1; }
    [[ $ok -eq 0 ]] || echo "!! re-run this script to retry the missing pages"
    echo
done

# ---------------------------------------------------------------- report
echo "=== final counts ==="
for s in "${SYSTEMS[@]}"; do
    n=$(find "$OUT/$s" -maxdepth 1 -name '*.md' | wc -l)
    echo "  $s: $n / 165"
    if [[ -d $OUT/$s/_truncated ]]; then
        find "$OUT/$s/_truncated" -name '*.md' -exec basename {} \; |
            sed 's/^/    truncated: /'
    fi
done

cat <<'NOTE'

CASE024_bank_statements is the dot-leader runaway. If it hits the token cap it
lands in _truncated/ rather than as a prediction, which is the runner refusing
to score a repetition loop as a misreading. Re-declare it with:

  python -u -m runners.run_vlm --corpus CORPUS --system SYSTEM \
      --out OUT --declare-unproducible

Then send the directory back:

  rsync -av OUT/ local:/path/to/doc-parsing-corpus/OUT/
NOTE

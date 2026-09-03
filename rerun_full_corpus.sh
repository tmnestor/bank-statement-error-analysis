#!/usr/bin/env bash
# Full 165-page re-run of both prompted systems under the corrected prompt.
#
# Why every page, not a subset: the decoration rule changed what a correct
# transcription looks like on every document type. Receipts draw separator
# rules, statements draw dot leaders, and invoices draw neither — but the rule
# also removed the visual fence around the receipt totals block, so pages with
# no glyph runs at all can still move. A subset would leave the headline numbers
# a mixture of two prompts.
#
# The predictions this produces supersede runs_v3/. Do NOT resume into that
# directory: the runner skips pages that already exist, which is right while the
# prompt is fixed and silently wrong the moment it changes. The runner now
# refuses to mix them, but starting clean is the intent.
#
# Run ON THE GPU HOST from the repo root, in the vLLM env.
# Roughly 45 minutes for both systems across two cards.

set -euo pipefail

OUT=${OUT:-runs_v4}
SOURCE=${SOURCE:-}
SYSTEMS=(gemma-4-12B-it-qat-w4a16-ct InternVL3.5-8B)

# ------------------------------------------------- the prompt must be current
# A git pull that did not happen would spend 45 minutes reproducing the old
# results, and nothing downstream would say so. These strings are asserted by
# tests/test_prompt.py against generators/decoration.py, so they cannot drift.
# The prompt is no longer checked here. The runner sends the prompt.md that
# ships inside the corpus -- covered by its manifest -- so there is no
# repo-local copy left to drift or to go stale behind a missing git pull.

# --------------------------------------------------------------- the corpus
# Any parsing_20260819* or later vintage will do: the runner reads images/ and
# the transcript FILENAMES, and every vintage since 20260819 holds
# byte-identical images. The transcripts here are never sent to the model, and
# scoring happens on the machine that holds parsing_20260820.
CORPUS=${CORPUS:-}
if [[ -z $CORPUS ]]; then
    for candidate in parsing_20260820 parsing_20260819d parsing_20260819c \
                     parsing_20260819b parsing_20260819; do
        [[ -d $candidate/images ]] && { CORPUS=$candidate; break; }
    done
fi
if [[ -z $CORPUS || ! -d $CORPUS/images ]]; then
    echo "No corpus found. Expected a parsing_20260819* or parsing_20260820 directory"
    echo "in $(pwd), or CORPUS=<dir>. Do NOT regenerate one: score verifies every"
    echo "image against its sha256 and a re-render will not match."
    exit 1
fi

pages=$(find "$CORPUS/transcripts" -name '*.md' | wc -l)
echo "corpus: $CORPUS ($pages pages)"
if [[ $pages -ne 165 ]]; then
    echo "!! expected the full 165-page corpus, found $pages"
    echo "   (parsing_*_decoration is the 61-page subset — pass CORPUS= explicitly)"
    exit 1
fi
echo "output: $OUT/"
echo

# ------------------------------------------------------------------- the runs
# One whole engine per card over a disjoint, strided slice. tp=1 means one
# engine PER GPU, not one GPU. Systems run one after another because 2x24 GB
# will not hold both models at once.
for s in "${SYSTEMS[@]}"; do
    echo "=== $s ==="
    started=$SECONDS
    CUDA_VISIBLE_DEVICES=0 python -u -m runners.run_vlm --corpus "$CORPUS" \
        --system "$s" --out "$OUT" --shard 0 --shards 2 & p0=$!
    CUDA_VISIBLE_DEVICES=1 python -u -m runners.run_vlm --corpus "$CORPUS" \
        --system "$s" --out "$OUT" --shard 1 --shards 2 & p1=$!

    # Bare `wait` returns only the last job's status, so a crashed shard 0 would
    # pass silently and leave half the slice missing.
    ok=0
    wait $p0 || { echo "!! shard 0 failed"; ok=1; }
    wait $p1 || { echo "!! shard 1 failed"; ok=1; }
    [[ $ok -eq 0 ]] || echo "!! re-run this script to retry the missing pages"
    echo "    $(( (SECONDS - started) / 60 )) min"
    echo
done

# ------------------------------------------------------------------- report
echo "=== counts ==="
incomplete=0
for s in "${SYSTEMS[@]}"; do
    n=$(find "$OUT/$s" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l)
    echo "  $s: $n / $pages"
    [[ $n -eq $pages ]] || incomplete=1
    if [[ -d $OUT/$s/_truncated ]]; then
        find "$OUT/$s/_truncated" -name '*.md' -exec basename {} \; |
            sed 's/^/    RAN AWAY: /'
    fi
done

cat <<NOTE

Expected: 165/165 for both, and NO "RAN AWAY" lines. The three pages that hit
the token cap under the old prompt (CASE003, CASE024, CASE047 bank statements)
complete under this one — that was the whole point of the rule, measured on the
61-page subset. A runaway here means the fix did not generalise beyond it, which
is worth knowing before anything else is read.

NOTE

if [[ $incomplete -eq 1 ]]; then
    echo "Some pages are missing. Re-run this script first — it retries only those."
    echo "If a page truncates repeatedly, declare it and record why:"
    for s in "${SYSTEMS[@]}"; do
        [[ -d $OUT/$s/_truncated ]] || continue
        echo "  python -u -m runners.run_vlm --corpus $CORPUS \\"
        echo "      --system $s --out $OUT --declare-unproducible"
    done
    echo
fi

echo "Then send the predictions back:"
echo
echo "  tar czf ${OUT}.tgz $OUT"
echo
echo "Scoring happens against parsing_20260820, which is not on this host:"
echo
echo "  python -m evaluation.cli --corpus parsing_20260820 \\"
echo "      --predictions $OUT --report scores_v4.json"

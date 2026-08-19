#!/usr/bin/env bash
# Does a one-shot example steer a VLM where a stated rule alone does not?
#
# The corpus strips runs of 4+ repeated punctuation glyphs (generators/
# decoration.py), while prompt.md says "do not skip repeated ... text". Models
# obey the prompt, so gemma inserts separator rules worth 67% of a receipt's
# length and runs away on the dot leaders of nab_classic. Both variants fix that
# contradiction and state the rule; ONLY arm B adds a worked example.
#
#   baseline  config/prompt.md                     already run -> runs_v3_61/
#   arm A     config/prompt_decoration_rule.md     rule, no example
#   arm B     config/prompt_decoration_example.md  rule + one-shot example
#
# A vs B is the measurement. Everything else about the two files is identical —
# verified by diffing what read_prompt() actually sends.
#
# 61 pages: the 55 receipts (separator rules) and the 6 nab_classic statements
# (dot leaders, including all three runaway pages). Every effect lives here, so
# the other 104 pages would cost four times the GPU hours for no measurement.
#
# Run ON THE GPU HOST from the repo root, in the vLLM env. ~20 minutes for all
# four runs across two cards.

set -euo pipefail

CORPUS=${CORPUS:-parsing_20260819d_decoration}
SYSTEMS=(gemma-4-12B-it-qat-w4a16-ct InternVL3.5-8B)

if [[ ! -d $CORPUS/images ]]; then
    echo "No corpus at $CORPUS/. It is gitignored and does not travel with the"
    echo "repo — rsync it across; do not rebuild it here, because score verifies"
    echo "every image against its sha256 and a re-render will not match."
    exit 1
fi

pages=$(find "$CORPUS/transcripts" -name '*.md' | wc -l)
echo "corpus: $CORPUS ($pages pages)"
echo

for arm in rule example; do
    prompt="config/prompt_decoration_${arm}.md"
    out="runs_decoration_${arm}"
    [[ -f $prompt ]] || { echo "missing $prompt — git pull?"; exit 1; }

    for s in "${SYSTEMS[@]}"; do
        echo "=== arm ${arm} / ${s} ==="
        CUDA_VISIBLE_DEVICES=0 python -u -m runners.run_vlm --corpus "$CORPUS" \
            --system "$s" --out "$out" --prompt "$prompt" --shard 0 --shards 2 & p0=$!
        CUDA_VISIBLE_DEVICES=1 python -u -m runners.run_vlm --corpus "$CORPUS" \
            --system "$s" --out "$out" --prompt "$prompt" --shard 1 --shards 2 & p1=$!

        # Bare `wait` returns only the last job's status, so a crashed shard 0
        # would pass silently and leave half the slice missing.
        ok=0
        wait $p0 || { echo "!! shard 0 failed"; ok=1; }
        wait $p1 || { echo "!! shard 1 failed"; ok=1; }
        [[ $ok -eq 0 ]] || echo "!! re-run this script to retry the missing pages"
        echo
    done
done

echo "=== counts ==="
for arm in rule example; do
    for s in "${SYSTEMS[@]}"; do
        d="runs_decoration_${arm}/$s"
        n=$(find "$d" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l)
        echo "  ${arm}/${s}: $n / $pages"
        if [[ -d $d/_truncated ]]; then
            find "$d/_truncated" -name '*.md' -exec basename {} \; | sed 's/^/    STILL RUNNING AWAY: /'
        fi
    done
done

cat <<NOTE

The headline is the _truncated/ lines above. Baseline had CASE003, CASE024 and
CASE047 running away under prompt.md; if arm A clears them the rule was enough,
if only arm B clears them the example did the work, and if neither does then the
loop is not an instruction-following failure at all.

Declare anything still truncated, then send both directories back:

NOTE
for arm in rule example; do
    for s in "${SYSTEMS[@]}"; do
        [[ -d runs_decoration_${arm}/$s/_truncated ]] || continue
        echo "  python -u -m runners.run_vlm --corpus $CORPUS \\"
        echo "      --system $s --out runs_decoration_${arm} --declare-unproducible"
    done
done
echo
echo "  rsync -av runs_decoration_rule runs_decoration_example \\"
echo "      local:/path/to/doc-parsing-corpus/"

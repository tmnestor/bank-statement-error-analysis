#!/usr/bin/env bash
# Re-run the 31B over all seven tiers under the prompt the CORPUS ships.
#
# Why this exists: the 2026-09-03 ladder in runs_v2_31b/ was produced under a
# 974-word prompt asking for Markdown pipe tables, while corpus_20260902 ships a
# 1297-word prompt asking for HTML tables with colspan and rowspan. The model
# was scored on conventions it was never told about, so those accuracy numbers
# are a floor. (Its throughput numbers are unaffected and still good.)
#
# runs_v2_31b/ IS NOT TOUCHED. It is a valid measurement of the older prompt and
# is worth keeping; this writes to runs_v3_31b/ instead.
#
#   ./rerun_31b_corpus_prompt.sh
#   CORPUS=/other/corpus ./rerun_31b_corpus_prompt.sh
#
# Expect roughly 6 hours: ~4h of generation at the ~5.5 img/min measured across
# every tier, plus about 19 minutes of model load per tier. The loads are
# unavoidable here -- run_vlm takes one --corpus, and each tier is a different
# corpus, so seven invocations mean seven loads.

set -uo pipefail

CORPUS=${CORPUS:-$HOME/nfs_share/tod_2026/evaluation_data/corpus_20260902}
STAMP=${STAMP:-20260902}
SYS=${SYS:-gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2}
OUT=${OUT:-runs_v3_31b}
REPORTS=${REPORTS:-.}
RESULTS=${RESULTS:-results_v3_31b}
LOG=${LOG:-rerun_31b_$(date +%Y%m%d_%H%M).log}

TIERS=(clean scan-light scan-moderate scan-heavy photo-light photo-moderate photo-heavy)

# The prompt every tier must answer: the body of the corpus's own prompt.md,
# hashed the way the runner hashes it (below the '---' rule, stripped).
WANT=21ea89f3b5be07c6a92a0ab61c46c729b12060fa79938c272f6e9c16f7c409f3

fail() { echo "!! $*" >&2; exit 1; }
step() { echo; echo "=== $* ==="; }

corpus_for() {
    if [[ $1 == clean ]]; then
        echo "$CORPUS/parsing_$STAMP"
    else
        echo "$CORPUS/degraded/parsing_${STAMP}_$1"
    fi
}

# --------------------------------------------------------------- pre-flight
# Every check here is cheap and needs no GPU. They all run BEFORE the first
# model load, so a wrong path or a stale corpus costs seconds rather than
# surfacing on tier six after five hours.
step "pre-flight"

command -v nvidia-smi >/dev/null || fail "nvidia-smi is not on PATH; is this the GPU host?"
[[ -d $CORPUS ]] || fail "corpus root not found: $CORPUS"

for t in "${TIERS[@]}"; do
    c=$(corpus_for "$t")
    [[ -d $c/images ]] || fail "$t: no images/ under $c"
    [[ -f $c/prompt.md ]] || fail "$t: $c ships no prompt.md — re-export this corpus."

    got=$(python3 -c "
import hashlib, pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')
_, sep, below = text.partition('\n---\n')
print(hashlib.sha256((below if sep else text).strip().encode('utf-8')).hexdigest())
" "$c/prompt.md")

    [[ $got == "$WANT" ]] || fail "$t: $c/prompt.md sends ${got:0:12}, not the expected ${WANT:0:12}.
   This corpus does not carry the HTML-table prompt, so re-running against it
   would reproduce the same mismatch. Re-export it, or check CORPUS."

    n=$(find "$c/images" -name '*.png' -o -name '*.jpg' | wc -l | tr -d ' ')
    printf "  %-16s %4s page(s)  prompt %s\n" "$t" "$n" "${got:0:12}"
done

# Refuse to write into a directory that already holds predictions. Resuming
# would silently mix two runs, and check_prompt_provenance would refuse anyway
# once it reached a directory answering a different prompt.
if [[ -d $OUT ]]; then
    existing=$(find "$OUT" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
    [[ $existing -eq 0 ]] || fail "$OUT already holds $existing prediction(s).
   Clear it, or set OUT= to a new directory. runs_v2_31b/ is deliberately left
   alone; do not write into it."
fi

echo
echo "system:  $SYS"
echo "out:     $OUT"
echo "log:     $LOG"
echo "prompt:  ${WANT:0:12} (1297 words, from each corpus's own prompt.md)"

# ------------------------------------------------------------------ the run
# No `conda run` wrapper: it captures stdout and releases nothing until the
# process exits, which on 2026-09-02 made a working six-hour run indistinguish-
# able from a hung one. `python -u` keeps the per-page lines flowing, and tee
# keeps them after the terminal scrolls.
step "transcribe"
started=$SECONDS

for t in "${TIERS[@]}"; do
    c=$(corpus_for "$t")
    echo
    echo "--- $t ---"
    # --prompt is deliberately NOT passed: run_vlm defaults to <corpus>/prompt.md,
    # which is the whole point of this re-run.
    python -u -m runners.run_vlm --corpus "$c" --system "$SYS" --out "$OUT/$t" ||
        fail "$t: transcription failed; fix it and re-run (finished tiers are skipped)"
done

echo
echo "transcription took $(( (SECONDS - started) / 60 )) min"

# -------------------------------------------------------------------- score
# CPU only, seconds per tier. Safe to re-run on its own at any time.
step "score"
for t in "${TIERS[@]}"; do
    c=$(corpus_for "$t")
    python -m evaluation.cli --corpus "$c" --predictions "$OUT/$t" \
        --report "$REPORTS/scores_v3_31b_$t.json" || fail "$t: scoring failed"
done

# ------------------------------------------------------------------- export
step "export"
python -m analysis.export_results "$REPORTS"/scores_v3_31b_*.json --out "$RESULTS" ||
    fail "export failed"

step "done"
echo "predictions: $OUT"
echo "reports:     $REPORTS/scores_v3_31b_*.json"
echo "artifact:    $RESULTS/results.json and $RESULTS/results.csv"
echo
echo "The ladder:"
python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
for r in sorted(d["runs"], key=lambda r: (r["family"] != "clean", r["family"], r["severity"])):
    docs = [x for x in r["documents"] if x["doc_type"] == "bank_statements"]
    a = sum(x["amounts"] for x in docs)
    t = sum(x["attributable"] for x in docs)
    print("  %-6s %-9s %6.2f img/min   attributable %4d/%4d = %.4f"
          % (r["family"], r["severity"], r["throughput"]["images_per_minute"], t, a, t / a))
' "$RESULTS/results.json"

#!/usr/bin/env bash
# Step 2 of 3, run ON THE 2xL4 SANDBOX: read every degraded corpus with the 31B.
#
# THE QUESTION. The deployment case says 99.0% of bank-statement amounts are
# usable -- right value, under the right heading. That was measured on pristine
# renders, and production receives scans. This measures the same thing at six
# declared image qualities, so the case can state what it costs rather than
# assuming it costs nothing.
#
# tp=2, the configuration the case proposes and the one its clean accuracy was
# re-confirmed on. Six corpora means six engine loads, roughly 12 minutes of
# overhead on a ~2 hour run; that is cheaper than orchestrating one engine
# across six corpora and getting the bookkeeping wrong.
#
# Each corpus is scored separately and never merged: a mean over six severities
# would describe an image quality that does not exist.

set -uo pipefail

SYSTEM=${SYSTEM:-gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2}
DEGRADED=${DEGRADED:-degraded}
OUT=${OUT:-runs_degraded}
MODEL=${MODEL:-/home/jovyan/nfs_share/models/gemma-4-31B-it-qat-w4a16-ct}

fail() { echo "!! $*" >&2; exit 1; }

[[ -d $DEGRADED ]] || fail "no $DEGRADED/ directory. Unpack degraded_bank_statements.tgz into it."
[[ -d $MODEL ]] || fail "checkpoint not found: $MODEL (set MODEL=)"

corpora=("$DEGRADED"/*/)
[[ ${#corpora[@]} -gt 0 ]] || fail "$DEGRADED/ holds no corpora"

# Same prompt as every scored run, or this measures the prompt as well as the
# image quality.
digest=$(python3 -c "
import hashlib, pathlib
t = pathlib.Path('config/prompt.md').read_text(encoding='utf-8')
_, s, b = t.partition('\n---\n')
print(hashlib.sha256((b if s else t).strip().encode()).hexdigest())
")
[[ $digest == 38919c6a81ee959a4d43c0cf2d6de918fee72028317983a99f6a7cc55276db61 ]] ||
    fail "config/prompt.md sends ${digest:0:12}, not the 38919c6a every scored run used."

# Corpus completeness first, before the hardware check: it is cheaper, it does
# not need a GPU to diagnose, and it is the failure that actually happens.
# Check every corpus BEFORE loading a model. Each corpus is otherwise checked
# only when its turn comes, so a truncated transfer surfaces six times, once per
# tier, after however long the earlier tiers took. On 2026-08-22 that reported
# six different diagnostics for one cause: the copy never finished.
echo "system:  $SYSTEM"
echo "corpora: ${#corpora[@]}"
incomplete=()
runnable=()
for c in "${corpora[@]}"; do
    name=$(basename "${c%/}")
    # -maxdepth 1 prunes .ipynb_checkpoints/, which Jupyter creates inside any
    # directory opened on the share and fills with copies of the corpus images.
    # Counting recursively reported 59 images against 55 transcripts and blocked
    # a complete corpus. The runners never saw them: corpus_images builds a
    # direct path per stem rather than globbing.
    images=$(find "$c/images" -maxdepth 1 \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null |
        wc -l | tr -d ' ')
    transcripts=$(find "$c/transcripts" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
    # The redirect fails before wc runs when the file is absent, so test first.
    rows=0
    [[ -f $c/manifest.jsonl ]] && rows=$(wc -l < "$c/manifest.jsonl" | tr -d ' ')
    status="ok"
    if [[ $images -ne $transcripts || $images -ne ${rows:-0} || $images -eq 0 ]]; then
        status="INCOMPLETE"
        incomplete+=("$name")
    else
        runnable+=("$c")
    fi
    printf "  %-46s img=%-4s tr=%-4s manifest=%-4s %s\n" \
        "$name" "$images" "$transcripts" "${rows:-0}" "$status"
done

# A tier that is PRESENT but short is a broken upload and must stop the run: its
# numbers would be a subset reported as a whole. A tier that is simply ABSENT is
# not an error -- the archives are uploaded a few at a time, scan before photo,
# so a scan-only run is the intended first step rather than a mistake.
if [[ ${#incomplete[@]} -gt 0 ]]; then
    echo
    echo "!! ${#incomplete[@]} corpus/corpora arrived incomplete: ${incomplete[*]}"
    echo "   Every tier must hold the same number of images, transcripts and manifest"
    echo "   rows. A browser upload truncates silently above ~300 MB, which is what"
    echo "   the single combined archive did. Re-upload just these tiers and verify"
    echo "   the bytes arrived before unpacking:"
    echo
    echo "     sha256sum -c SHA256SUMS"
    echo "     for f in <tier>.tgz; do tar xzf \"\$f\" -C $DEGRADED; done"
    exit 1
fi

[[ ${#runnable[@]} -gt 0 ]] || fail "no complete corpus in $DEGRADED/ — nothing to run"
if [[ ${#runnable[@]} -lt 6 ]]; then
    echo
    echo "   ${#runnable[@]} of 6 tiers present. Running those; upload the rest and"
    echo "   re-run — completed tiers resume rather than repeat."
fi

cards=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[[ $cards -ge 2 ]] || fail "tensor_parallel_size is 2 but $cards GPU(s) are visible"
echo

started=$SECONDS
failed=()
for corpus in "${runnable[@]}"; do
    name=$(basename "${corpus%/}")
    echo "=== $name ==="
    # --out per corpus: score treats every subdirectory of a predictions root as
    # a system, so six tiers under one root would be read as six systems of one
    # corpus and each would fail its manifest check against the other five.
    python -u -m runners.run_vlm --corpus "${corpus%/}" --system "$SYSTEM" \
        --out "$OUT/$name" || { echo "!! $name failed"; failed+=("$name"); }
    echo
done
elapsed=$(( SECONDS - started ))

echo "=== ${#corpora[@]} corpus/corpora in $(( elapsed / 60 )) min ==="
[[ ${#failed[@]} -eq 0 ]] || echo "!! failed: ${failed[*]}"

cat <<NOTE

Send $OUT back, then score each tier against its OWN corpus -- the manifests
differ by construction, and score will refuse any other pairing:

  for d in $DEGRADED/*/; do
      n=\$(basename "\${d%/}")
      conda run -n docparse python -m generators.pipeline score \\
          --corpus "\${d%/}" --predictions "$OUT/\$n" \\
          --report "scores_\$n.json"
  done

  tar czf ${OUT}.tgz $OUT
NOTE

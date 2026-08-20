#!/usr/bin/env bash
# Does 4-bit quantisation cost digit fidelity? The precision control.
#
# gemma-4-12B-it-qat-w4a16-ct leads this benchmark on character error rate, on
# table structure and on column integrity — and gets 92.5% of amounts right,
# against MinerU's 100.0% and InternVL3.5-8B's 97.2%. Its misses are digit
# substitutions, not dropped lines. Quantisation costing character fidelity
# first is the standing explanation and has never been measured.
#
# READ THIS BEFORE QUOTING THE RESULT. This runs gemma-4-12B-it, which differs
# from the 4-bit checkpoint in two ways: 16-bit rather than 4-bit, and not
# quantisation-aware-trained. The clean control would be
# gemma-4-12B-it-qat-q4_0-unquantized — the same QAT weights at BF16 — which is
# not on this host. So the outcome is asymmetric:
#
#   no better  -> strong evidence AGAINST quantisation. A model that is both
#                 higher-precision and non-QAT, failing the same way, points
#                 somewhere else entirely.
#   better     -> precision OR QAT training, and these two runs cannot say
#                 which. Do not report it as "quantisation costs digit
#                 fidelity"; that claims more than was measured.
#
# Every decoding and engine setting matches the 4-bit run, so nothing but the
# checkpoint varies between them.
#
# One L40S, one engine, no sharding: ~24 GB of BF16 weights need a 48 GB card,
# and will not fit either L4.

set -euo pipefail

SYSTEM=gemma-4-12B-it
OUT=${OUT:-runs_control}
CORPUS=${CORPUS:-}

if ! grep -q "four or more" config/prompt.md; then
    echo "config/prompt.md does not carry the decoration rule — run 'git pull'."
    echo "The comparison run used the current prompt; a stale one would confound"
    echo "precision with prompt version, which is the whole point of a control."
    exit 1
fi

if [[ -z $CORPUS ]]; then
    for candidate in parsing_20260820 parsing_20260819d parsing_20260819c \
                     parsing_20260819b parsing_20260819; do
        [[ -d $candidate/images ]] && { CORPUS=$candidate; break; }
    done
fi
if [[ -z $CORPUS || ! -d $CORPUS/images ]]; then
    echo "No corpus found. Expected parsing_20260819* or parsing_20260820 in $(pwd),"
    echo "or CORPUS=<dir>. Do NOT regenerate one — score verifies image hashes."
    exit 1
fi

pages=$(find "$CORPUS/transcripts" -name '*.md' | wc -l)
[[ $pages -eq 165 ]] || { echo "!! expected 165 pages, found $pages in $CORPUS"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "corpus: $CORPUS ($pages pages)"
echo "system: $SYSTEM  ->  $OUT/"
echo

started=$SECONDS
python -u -m runners.run_vlm --corpus "$CORPUS" --system "$SYSTEM" --out "$OUT"
echo "    $(( (SECONDS - started) / 60 )) min"

n=$(find "$OUT/$SYSTEM" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l)
echo
echo "=== $n / $pages pages ==="
if [[ -d $OUT/$SYSTEM/_truncated ]]; then
    find "$OUT/$SYSTEM/_truncated" -name '*.md' -exec basename {} \; | sed 's/^/  RAN AWAY: /'
    echo
    echo "  Declare them, so the control is scored over the same 165 transcripts:"
    echo "    python -u -m runners.run_vlm --corpus $CORPUS \\"
    echo "        --system $SYSTEM --out $OUT --declare-unproducible"
fi

cat <<NOTE

Send it back:

  tar czf ${OUT}.tgz $OUT

The comparison is numeric fidelity on the 55 bank statements, where the 4-bit
checkpoint gets 90.3% of amounts right and gets at least one amount wrong on 36
of them.

  near 90%  -> quantisation is NOT the cause, and that is a clean result.
  near 100% -> precision or QAT training, unresolved between them until the
               gemma-4-12B-it-qat-q4_0-unquantized checkpoint is available.
NOTE

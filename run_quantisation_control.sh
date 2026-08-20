#!/usr/bin/env bash
# Does 4-bit quantisation cost digit fidelity? Two controls, because one cannot say.
#
# gemma-4-12B-it-qat-w4a16-ct leads this benchmark on character error rate, on
# table structure and on column integrity — and gets 92.5% of amounts right,
# against MinerU's 100.0% and InternVL3.5-8B's 97.2%. Its misses are digit
# SUBSTITUTIONS, not dropped lines. Quantisation costing character fidelity
# first is the standing explanation and has never been measured.
#
# The 4-bit checkpoint differs from a plain BF16 model in two ways at once:
# precision, and quantisation-aware training. So this runs both BF16 halves and
# the three pairwise comparisons separate them:
#
#   w4a16-ct    vs  qat-q4_0-unquantized  ->  precision alone (QAT held fixed)
#   w4a16-ct    vs  gemma-4-12B-it        ->  precision + QAT training
#   unquantized vs  gemma-4-12B-it        ->  QAT training alone (BF16 both)
#
# Every decoding and engine setting matches the 4-bit run in both entries —
# verified, they differ in nothing but the checkpoint path.
#
# One L40S, one engine each, no sharding, sequentially: ~24 GB of BF16 weights
# need the 48 GB card and will not fit an L4. Roughly 35 minutes per system.

set -euo pipefail

OUT=${OUT:-runs_control}
CORPUS=${CORPUS:-}
SYSTEMS=(gemma-4-12B-it-qat-q4_0-unquantized gemma-4-12B-it)

if ! grep -q "four or more" config/prompt.md; then
    echo "config/prompt.md does not carry the decoration rule — run 'git pull'."
    echo "The run these are compared against used the current prompt; a stale one"
    echo "would confound the checkpoint with the prompt version, which is exactly"
    echo "what a control exists to prevent."
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

# A missing processor_config.json is silent and expensive: without it mlx/vLLM
# falls back to a bare tokenizer with no image preprocessing, the PNG never
# becomes patches, and the model politely asks to be shown the image. That is a
# non-empty file which passes every completeness check and scores as a total
# reading failure, blaming the model for a packaging omission.
for s in "${SYSTEMS[@]}"; do
    dir=$(python -c "
import sys, yaml
print(yaml.safe_load(open('config/vlm_systems.yml'))['systems']['$s']['model'])")
    [[ -d $dir ]] || { echo "!! $s: checkpoint directory missing: $dir"; exit 1; }
    [[ -f $dir/processor_config.json ]] || {
        echo "!! $s: no processor_config.json in $dir"
        echo "   Without it the image is never preprocessed and the run scores as a"
        echo "   total reading failure while looking like a completed one."
        exit 1
    }
    arch=$(python -c "
import json; print(json.load(open('$dir/config.json')).get('model_type', '?'))" 2>/dev/null || echo '?')
    echo "  $s: $arch, processor_config.json present"
done

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "corpus: $CORPUS ($pages pages)"
echo

for s in "${SYSTEMS[@]}"; do
    echo "=== $s ==="
    started=$SECONDS
    python -u -m runners.run_vlm --corpus "$CORPUS" --system "$s" --out "$OUT"
    echo "    $(( (SECONDS - started) / 60 )) min"
    echo
done

echo "=== counts ==="
for s in "${SYSTEMS[@]}"; do
    n=$(find "$OUT/$s" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l)
    echo "  $s: $n / $pages"
    if [[ -d $OUT/$s/_truncated ]]; then
        find "$OUT/$s/_truncated" -name '*.md' -exec basename {} \; | sed 's/^/    RAN AWAY: /'
        echo "    declare them so the control is scored over the same 165 transcripts:"
        echo "      python -u -m runners.run_vlm --corpus $CORPUS \\"
        echo "          --system $s --out $OUT --declare-unproducible"
    fi
done

cat <<NOTE

Send it back:

  tar czf ${OUT}.tgz $OUT

The comparison is numeric fidelity on the 55 bank statements, where the 4-bit
checkpoint gets 90.3% of amounts right and has at least one wrong on 36 of them.

  unquantized near 100%  ->  precision. The clean result, QAT held fixed.
  unquantized near  90%  ->  NOT precision. Then gemma-4-12B-it says whether QAT
                             training is implicated or the cause is elsewhere
                             entirely — the vision budget, page resolution, or
                             the model family.
NOTE

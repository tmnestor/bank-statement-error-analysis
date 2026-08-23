#!/usr/bin/env bash
# Build the comparison sheets for the meeting: eight layouts, three conditions.
#
# COVERAGE. The corpus carries eight bank-statement layouts, two for each of the
# Big Four — and the earlier sheets used three of them, all NAB and Westpac. One
# case per layout puts every bank in front of the room, which matters because
# the failures are layout-specific: the wrapped `Date of / Transaction` header
# is a Westpac thing, the date band is a NAB thing.
#
# CONDITIONS. Clean, scan-heavy and photo-heavy. The photo tiers were missing
# and they carry the rotations and perspective a scanner cannot produce — which
# is where both systems lose the most.
#
# Each sheet shows four systems and no ground-truth panel, so the width goes to
# the systems being compared. Divergence is against the truth either way.

set -uo pipefail

CORPUS=${CORPUS:-parsing_20260820}
DEGRADED=${DEGRADED:-degraded}
ENV_NAME=${ENV_NAME:-docparse}

fail() { echo "!! $*" >&2; exit 1; }

[[ -d $CORPUS/images ]] || fail "no $CORPUS — set CORPUS="

# One case per layout, chosen for size and for the failure each layout provokes.
CASES=(
    CASE002   # anz_modern
    CASE004   # anz_standard
    CASE007   # cba_date_grouped   — date heads a group
    CASE001   # cba_standard
    CASE003   # nab_classic        — dot leaders
    CASE015   # nab_dense          — date band, MinerU fragments hardest here
    CASE012   # westpac_premium    — wrapped Date of / Transaction header
    CASE041   # westpac_standard   — strongest date grouping in the corpus
)

CLEAN_SYSTEMS=(
    "runs_31b/gemma-4-31B-it-qat-w4a16-ct"
    "runs_v4/gemma-4-12B-it-qat-w4a16-ct"
    "runs_v4/InternVL3.5-8B"
    "runs_parsers_l4/mineru-vllm"
)

build() {
    local out=$1 corpus=$2
    shift 2
    local systems=("$@")
    local flags=()
    for s in "${systems[@]}"; do
        [[ -d $s ]] || { echo "  skip: no $s"; return; }
        flags+=(--system "$s")
    done
    mkdir -p "$out"
    for case in "${CASES[@]}"; do
        [[ -f $corpus/transcripts/${case}_bank_statements.md ]] || continue
        conda run -n "$ENV_NAME" python compare_transcripts.py "${case}_bank_statements" \
            --corpus "$corpus" --no-truth "${flags[@]}" --out "$out" 2>&1 |
            grep -E "^comparisons|no prediction" || true
    done
}

echo "=== clean: four systems, eight layouts ==="
build comparisons "$CORPUS" "${CLEAN_SYSTEMS[@]}"

# On the degraded corpora only the 31B and MinerU were run, so the sheets pair
# each system's clean output against its degraded output. That is the comparison
# the tier exists to support: the same system, the same page, two image
# qualities, one truth — because degradation never changes the transcript.
for tier in scan-heavy photo-heavy; do
    corpus="$DEGRADED/${CORPUS}_${tier}"
    [[ -d $corpus/images ]] || { echo "=== $tier: absent, skipped ==="; continue; }
    echo "=== $tier: clean vs degraded, both systems ==="
    build "comparisons_${tier}" "$corpus" \
        "runs_31b/gemma-4-31B-it-qat-w4a16-ct" \
        "runs_degraded/${CORPUS}_${tier}/gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2" \
        "runs_parsers_l4/mineru-vllm" \
        "runs_degraded/${CORPUS}_${tier}/mineru-vllm"
done

echo
for d in comparisons comparisons_scan-heavy comparisons_photo-heavy; do
    [[ -d $d ]] && echo "  $d: $(find "$d" -name '*.png' | wc -l | tr -d ' ') sheet(s)"
done

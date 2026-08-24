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

# The 31B is quoted from the 2xL4 tp=2 run throughout, because that is the
# deployment PROD will have -- there is no L40S there. The tp=1 run on the L40S
# is kept only as the control that says the choice costs nothing: placed 99.01%
# and attributable 97.71% under both, 20 amounts misfiled under both, and row
# alignment 1.9 points BETTER under tp=2.
CLEAN_SYSTEMS=(
    "runs_31b_tp2/gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2"
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
            grep -E "^comparisons|no prediction|refused" || true
    done
}

echo "=== clean: four systems, eight layouts ==="
build comparisons "$CORPUS" "${CLEAN_SYSTEMS[@]}"

# FOUR SYSTEMS ON THE DEGRADED PAGE, AND NO CLEAN PANELS.
#
# These sheets used to pair each system's clean output against its degraded
# output, because runs_degraded/ held the 31B and MinerU only and a four-way
# comparison was not available. It is the same four-way comparison as the clean
# sheets, one tier down -- which is what the room needs, since the clean panels
# are already on the clean sheets and repeating them spends half the width on
# something already seen.
#
# The extra two panels come from run_degraded_12b_internvl.sh, which reads every
# degraded corpus on the sandbox and writes beside the 31B and MinerU -- so all
# four systems for a tier sit in one directory. Until that has run and come
# back, `build` reports the missing directory and skips the tier rather than
# rendering a sheet with two empty panels.
#
# Panel order matches the clean sheets: 31B, 12B, InternVL, MinerU. A room
# comparing two sheets should not have to re-find which column is which.
#
# photo-LIGHT as well as the heavy tiers. photo-heavy pushes the systems past
# the point where they discriminate -- the 31B drops to 38.5% usable on CASE012
# and MinerU to 0% -- which makes it a good demonstration of a limit and a poor
# demonstration of a difference. photo-light is the realistic phone condition and
# is where a room can see one system holding and another slipping.
TIERS=${TIERS:-"scan-heavy photo-light photo-heavy"}
RUNS=${RUNS:-runs_degraded}
for tier in $TIERS; do
    corpus="$DEGRADED/${CORPUS}_${tier}"
    [[ -d $corpus/images ]] || { echo "=== $tier: absent, skipped ==="; continue; }
    echo "=== $tier: four systems, all degraded ==="
    build "comparisons_${tier}" "$corpus" \
        "$RUNS/${CORPUS}_${tier}/gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2" \
        "$RUNS/${CORPUS}_${tier}/gemma-4-12B-it-qat-w4a16-ct" \
        "$RUNS/${CORPUS}_${tier}/InternVL3.5-8B" \
        "$RUNS/${CORPUS}_${tier}/mineru-vllm"
done

echo
for d in comparisons comparisons_scan-heavy comparisons_photo-light comparisons_photo-heavy; do
    [[ -d $d ]] && echo "  $d: $(find "$d" -name '*.png' | wc -l | tr -d ' ') sheet(s)"
done

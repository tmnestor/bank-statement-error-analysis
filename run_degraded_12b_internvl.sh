#!/usr/bin/env bash
# Run ON THE SANDBOX: read every degraded corpus with the two systems that have
# never seen a degraded image.
#
# THE GAP THIS FILLS. runs_degraded/ holds gemma-4-31B and MinerU only, so the
# degradation ladder describes two systems while the deployment case compares
# four. This completes it: 6 tiers x 55 pages x 2 systems = 660 inferences, and
# the four-system degraded comparison becomes available both as sheets and as
# corpus-wide numbers.
#
# ONE CARD. Both systems declare tensor_parallel_size: 1 in config/vlm_systems.yml,
# unlike run_degraded_31b.sh which requires two. This will run on a single L4.
#
# OUTPUT LANDS IN runs_degraded/, beside the 31B and MinerU, because `score`
# treats every subdirectory of a predictions root as a system -- so putting the
# new two here is what makes a four-system score of a tier one command instead
# of a merge step. Nothing existing is touched: each system writes its own
# subdirectory.
#
# RESUMABLE. A re-run transcribes only the stems with no non-empty prediction,
# so a crash four hours in costs the current tier and not the run. That matters
# here more than on any previous run: twelve engine loads over ~660 pages is a
# long window for the share to hiccup.
#
# EACH TIER IS SCORED SEPARATELY AND NEVER MERGED. A mean over six severities
# would describe an image quality that does not exist.

set -uo pipefail

DEGRADED=${DEGRADED:-degraded}
OUT=${OUT:-runs_degraded}
ENV_NAME=${ENV_NAME:-vllm_env}
SYSTEMS=${SYSTEMS:-"gemma-4-12B-it-qat-w4a16-ct InternVL3.5-8B"}
# Every tier by default. Set TIERS to a space-separated list of directory names
# to trim the run -- the sheets need scan-heavy, photo-light and photo-heavy.
TIERS=${TIERS:-}

fail() {
    echo "!! $*" >&2
    exit 1
}

[[ -d $DEGRADED ]] || fail "no $DEGRADED/ directory. Unpack the degraded corpora into it."

if [[ -n $TIERS ]]; then
    corpora=()
    for t in $TIERS; do corpora+=("$DEGRADED/$t/"); done
else
    corpora=("$DEGRADED"/*/)
fi
[[ -d ${corpora[0]:-} ]] || fail "$DEGRADED/ holds no corpora"

# Same prompt as every scored run, or this measures the prompt as well as the
# image quality. The digest is of the body below the `---` rule, which is what
# the runner actually sends -- hashing the whole file compares a preamble the
# model never reads, and that false alarm has already cost a sandbox start here.
digest=$(python3 -c "
import hashlib, pathlib
t = pathlib.Path('config/prompt.md').read_text(encoding='utf-8')
_, s, b = t.partition('\n---\n')
print(hashlib.sha256((b if s else t).strip().encode()).hexdigest())
")
[[ $digest == 38919c6a81ee959a4d43c0cf2d6de918fee72028317983a99f6a7cc55276db61 ]] ||
    fail "config/prompt.md sends ${digest:0:12}, not the 38919c6a every scored run used."

# Every corpus is checked BEFORE any model loads. Checking each in turn reports
# a truncated transfer once per tier, after however long the earlier tiers took;
# on 2026-08-22 that produced six diagnostics for one cause, a copy that never
# finished.
echo "systems: $SYSTEMS"
echo "corpora: ${#corpora[@]}"
total=0
for c in "${corpora[@]}"; do
    name=$(basename "${c%/}")
    [[ -d $c ]] || fail "$name: no such tier under $DEGRADED/"
    # -maxdepth 1 prunes .ipynb_checkpoints/, which Jupyter creates inside any
    # directory opened on the share and fills with copies of the corpus images.
    # Counting recursively once reported 59 images against 55 transcripts and
    # blocked a complete corpus.
    images=$(find "${c}images" -maxdepth 1 -type f | wc -l | tr -d ' ')
    transcripts=$(find "${c}transcripts" -maxdepth 1 -type f | wc -l | tr -d ' ')
    records=$(wc -l < "${c}manifest.jsonl" | tr -d ' ')
    [[ $images == "$transcripts" && $images == "$records" ]] ||
        fail "$name: $images image(s), $transcripts transcript(s), $records manifest record(s) -- transfer incomplete"
    echo "  $name: $images page(s)"
    total=$((total + images))
done

systems_count=$(echo "$SYSTEMS" | wc -w | tr -d ' ')
echo "  -> $((total * systems_count)) inference(s) across $((${#corpora[@]} * systems_count)) engine load(s)"

cards=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[[ $cards -ge 1 ]] || fail "no GPU visible"

failed=()
for system in $SYSTEMS; do
    for c in "${corpora[@]}"; do
        name=$(basename "${c%/}")
        echo
        echo "=== $system / $name ==="
        conda run -n "$ENV_NAME" --no-capture-output \
            python -u -m runners.run_vlm --corpus "${c%/}" --system "$system" \
            --out "$OUT/$name" || {
            echo "!! $system / $name failed"
            failed+=("$system/$name")
        }
    done
done

echo
echo "=== per tier, per system ==="
for c in "${corpora[@]}"; do
    name=$(basename "${c%/}")
    for system in $SYSTEMS; do
        d="$OUT/$name/$system"
        printf '  %-34s %-30s %s\n' "$name" "$system" \
            "$([[ -d $d ]] && find "$d" -maxdepth 1 -name '*.md' | wc -l | tr -d ' ' || echo 0)"
    done
done

echo
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FAILED: ${failed[*]}"
    echo "Re-run this script: complete predictions are kept and only the gaps are redone."
    exit 1
fi

echo "Done. Send $OUT/ back to the laptop, then:"
echo "  ./score_degraded.sh                 # four systems per tier"
echo "  ./build_meeting_sheets.sh           # four-system degraded sheets"

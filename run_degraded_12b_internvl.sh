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
SYSTEMS=${SYSTEMS:-"gemma-4-12B-it-qat-w4a16-ct InternVL3.5-8B"}
# The ACTIVE environment runs the model, exactly as run_degraded_31b.sh does.
# An earlier version wrapped every call in `conda run -n vllm_env`, which is a
# guess at a name that varies by host -- the sandbox has vllm_env2 -- and the
# guess failed twelve times in a row before the script gave up. Set ENV_NAME to
# wrap the calls in `conda run` if you need a different env from the active one.
ENV_NAME=${ENV_NAME:-}
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

# THE ENVIRONMENT IS CHECKED HERE, BEFORE ANY MODEL LOADS -- for the same
# reason the corpora are. A wrong environment is not a per-tier fault: it fails
# identically for all twelve combinations, and discovering that twelve times
# tells you nothing the first one did not.
if [[ -n $ENV_NAME ]]; then
    RUN=(conda run -n "$ENV_NAME" --no-capture-output)
    where="conda env '$ENV_NAME'"
else
    RUN=()
    where="active env '${CONDA_DEFAULT_ENV:-none}'"
fi

"${RUN[@]}" python -c "
import importlib.util, sys
missing = [m for m in ('vllm', 'PIL', 'yaml') if importlib.util.find_spec(m) is None]
sys.exit('missing: ' + ', '.join(missing) if missing else 0)
" || fail "$where cannot run the VLM runner.
  What:    that interpreter has no vLLM (or no Pillow / PyYAML), or the named
           environment does not exist.
  Where:   run this from the environment that holds vLLM. The sandbox shows it
           in the shell prompt -- e.g. (vllm_env2).
  Example: conda activate vllm_env2
           ./$(basename "$0")
           -- or wrap the calls instead of activating:
           ENV_NAME=vllm_env2 ./$(basename "$0")
  Recover: \`conda env list\` shows the candidates; the right one is whichever
           makes \`python -c 'import vllm'\` succeed."

echo "running in: $where"

failed=()
for system in $SYSTEMS; do
    for c in "${corpora[@]}"; do
        name=$(basename "${c%/}")
        echo
        echo "=== $system / $name ==="
        "${RUN[@]}" python -u -m runners.run_vlm --corpus "${c%/}" --system "$system" \
            --out "$OUT/$name" || {
            echo "!! $system / $name failed"
            failed+=("$system/$name")
            # A tier that produced nothing at all is an environment, checkpoint
            # or config fault, and it will repeat identically for the remaining
            # combinations. Stop and say so rather than restating it twelve
            # times -- which is exactly what this script did on 2026-08-24.
            # A tier that produced SOME pages is a data or memory fault worth
            # continuing past, since the run is resumable and the rest may be
            # fine.
            produced=$(find "$OUT/$name/$system" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
            [[ $produced -gt 0 ]] || fail "$system / $name produced no predictions at all.
  What:    the failure above is not specific to this tier -- it will repeat for
           the remaining $((${#corpora[@]} * systems_count - ${#failed[@]})) combination(s).
  Where:   read the error above; the usual causes are a checkpoint path in
           config/vlm_systems.yml that does not exist on this host, or too
           little free VRAM.
  Example: nvidia-smi                       # is a card already occupied?
           ls \$(python -c \"import yaml;print(yaml.safe_load(open('config/vlm_systems.yml'))['systems']['$system']['model'])\")
  Recover: fix the cause, then re-run this script -- finished pages are kept
           and only the gaps are redone."
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

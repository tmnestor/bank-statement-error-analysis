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
# ONE ENVIRONMENT PER SYSTEM, AND THEY ARE NOT INTERCHANGEABLE. gemma-4-12B is
# a `gemma4_unified` checkpoint, whose floor is vLLM >= 0.23.0 (see the comment
# at config/vlm_systems.yml:100); InternVL3.5-8B runs under the older vLLM in
# vllm_env2, measured at 0.19.0 on 2026-08-24. A single-env run of both is
# impossible on this host, which is why each system carries its own env below
# rather than the script taking one ENV_NAME.
#
# ONE CARD. Both systems declare tensor_parallel_size: 1 in
# config/vlm_systems.yml, unlike run_degraded_31b.sh which requires two.
#
# OUTPUT LANDS IN runs_degraded/, beside the 31B and MinerU, because `score`
# treats every subdirectory of a predictions root as a system -- so putting the
# new two here is what makes a four-system score of a tier one command instead
# of a merge step. Nothing existing is touched: each system writes its own
# subdirectory.
#
# RESUMABLE. A re-run transcribes only the stems with no non-empty prediction,
# so a crash four hours in costs the current tier and not the run.
#
# EACH TIER IS SCORED SEPARATELY AND NEVER MERGED. A mean over six severities
# would describe an image quality that does not exist.

set -uo pipefail

DEGRADED=${DEGRADED:-degraded}
OUT=${OUT:-runs_degraded}
SYSTEMS=${SYSTEMS:-"gemma-4-12B-it-qat-w4a16-ct InternVL3.5-8B"}
# Every tier by default. Set TIERS to a space-separated list of directory names
# to trim the run -- the meeting sheets need scan-heavy, photo-light, photo-heavy.
TIERS=${TIERS:-}

GEMMA_ENV=${GEMMA_ENV:-vllm_env3}
INTERNVL_ENV=${INTERNVL_ENV:-vllm_env2}

fail() {
    echo "!! $*" >&2
    exit 1
}

# Which environment loads which checkpoint. Wrong here means a run that fails at
# the first engine load, or worse, one that loads and is quietly degraded.
env_for() {
    case "$1" in
        gemma-4-12B-it-qat-w4a16-ct) echo "$GEMMA_ENV" ;;
        InternVL3.5-8B) echo "$INTERNVL_ENV" ;;
        *) fail "no environment declared for system '$1'.
  What:    every system must name the conda env whose vLLM can load it; this
           script has no default, because guessing one produced twelve
           identical failures on 2026-08-24.
  Where:   the env_for() case block in $(basename "$0").
  Example: $1) echo \"\${MY_ENV:-vllm_env3}\" ;;
  Recover: add the branch, or run only the declared systems with
           SYSTEMS='gemma-4-12B-it-qat-w4a16-ct InternVL3.5-8B'." ;;
    esac
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
echo "corpora: ${#corpora[@]}"
total=0
for c in "${corpora[@]}"; do
    name=$(basename "${c%/}")
    [[ -d $c ]] || fail "$name: no such tier under $DEGRADED/"
    # -maxdepth 1 prunes .ipynb_checkpoints/, which Jupyter creates inside any
    # directory opened on the share and fills with copies of the corpus images.
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

# EVERY ENVIRONMENT IS PROVEN BEFORE ANY MODEL LOADS, for the same reason the
# corpora are: a broken environment is not a per-tier fault. It fails
# identically for all six of that system's tiers, and discovering it six times
# tells you nothing the first one did not.
#
# vLLM is IMPORTED, not merely located. `find_spec` finds a vllm whose shared
# objects cannot load, so on 2026-08-24 a find_spec check passed and the run
# died at the first engine load -- which is the failure the check exists to
# prevent.
#
# LD_LIBRARY_PATH is prepended with the env's own lib directory. vLLM's compiled
# extensions link against the environment's libstdc++, and when the system one
# is found first the import fails with a missing CXXABI/GLIBCXX symbol. That is
# what happened on 2026-08-24, on a shell whose prompt read
# `(vllm_env2) (vllm_env2)` -- a doubled activation leaves the ordering wrong.
# Prepending is what a clean activation would have done, so this repairs the
# ordering rather than overriding it.
declare -A ENV_OF PREFIX_OF
echo
for system in $SYSTEMS; do
    env_name=$(env_for "$system") || exit 1
    ENV_OF[$system]=$env_name

    prefix=$(conda run -n "$env_name" python -c "import sys; print(sys.prefix)" 2>/dev/null | tail -1)
    [[ -n $prefix && -d $prefix ]] || fail "conda env '$env_name' (for $system) not found.
  What:    \`conda run -n $env_name\` could not report a prefix, so the
           environment does not exist or is not usable.
  Where:   this host's conda installation.
  Example: conda env list
  Recover: set the override to the env that holds a vLLM able to load
           $system, e.g.
             GEMMA_ENV=vllm_env3 INTERNVL_ENV=vllm_env2 ./$(basename "$0")"
    PREFIX_OF[$system]=$prefix

    # `conda run` appends its own "... failed. (See above for error)" line on a
    # non-zero exit, which is the LAST line and says nothing. Drop it so the
    # diagnostic quotes the actual ImportError.
    version=$(conda run -n "$env_name" env "LD_LIBRARY_PATH=$prefix/lib:${LD_LIBRARY_PATH:-}" \
        python -c "import vllm; print(vllm.__version__)" 2>&1 |
        grep -vE '^ERROR conda\.|^\(See above|^$' | tail -1)
    [[ $version =~ ^[0-9]+\.[0-9]+ ]] || fail "conda env '$env_name' cannot load vLLM for $system.
  What:    importing vllm there failed: $version
  Where:   conda env '$env_name', prefix $prefix
  Example: a missing CXXABI/GLIBCXX symbol means the system libstdc++ is found
           before the environment's; this script already prepends
             $prefix/lib
           so a failure here is a genuinely broken or wrong environment.
  Recover: conda env list, then point the override at the right one:
             GEMMA_ENV=... INTERNVL_ENV=... ./$(basename "$0")"

    echo "  $system"
    echo "      env $env_name, vLLM $version"
done

failed=()
for system in $SYSTEMS; do
    env_name=${ENV_OF[$system]}
    prefix=${PREFIX_OF[$system]}
    for c in "${corpora[@]}"; do
        name=$(basename "${c%/}")
        echo
        echo "=== $system ($env_name) / $name ==="
        conda run -n "$env_name" --no-capture-output \
            env "LD_LIBRARY_PATH=$prefix/lib:${LD_LIBRARY_PATH:-}" \
            python -u -m runners.run_vlm --corpus "${c%/}" --system "$system" \
            --out "$OUT/$name" || {
            echo "!! $system / $name failed"
            failed+=("$system/$name")
            # A tier that produced nothing at all is an environment, checkpoint
            # or config fault, and it will repeat identically for the remaining
            # combinations. Stop and say so rather than restating it twelve
            # times -- which is what this script did on 2026-08-24. A tier that
            # produced SOME pages is a data or memory fault worth continuing
            # past, since the run is resumable and the rest may be fine.
            produced=$(find "$OUT/$name/$system" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
            [[ $produced -gt 0 ]] || fail "$system / $name produced no predictions at all.
  What:    the failure above is not specific to this tier -- it will repeat for
           this system's remaining tier(s).
  Where:   read the error above; with the environment already proven, the usual
           causes are a checkpoint path in config/vlm_systems.yml that does not
           exist on this host, or too little free VRAM.
  Example: nvidia-smi                       # is a card already occupied?
           conda run -n $env_name python -c \"import yaml;print(yaml.safe_load(open('config/vlm_systems.yml'))['systems']['$system']['model'])\"
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

#!/usr/bin/env bash
# Read every degraded corpus with MinerU, on the 2xL4.
#
# WHY THIS ONE. MinerU is the system that fragments rows -- 276 of them on the
# clean corpus, against zero for every gemma checkpoint -- and fragmentation is
# a row-segmentation failure, which is exactly what a smeared or skewed scan
# should make worse. The 31B's flat result across the scan ladder says nothing
# about that: it never fragments a row to begin with, so it had no fragmentation
# to lose.
#
# It is also the cheap run. MinerU manages ~22 images/min across the two cards
# against the 31B's 5.3, so all six tiers is roughly 15 minutes rather than two
# hours.
#
# CHUNKING AND RESUME. --chunk bounds peak memory and gives the run a resume
# point; a crash costs one chunk rather than a tier. Re-running transcribes only
# the pages with no non-empty prediction, so this is safe to repeat.
#
# Each tier gets its own --out for the same reason run_degraded_31b.sh does:
# score treats every subdirectory of a predictions root as a system, so six
# tiers under one root would read as six systems of one corpus and each would
# fail its manifest check against the other five.

set -uo pipefail

SYSTEM=${SYSTEM:-mineru-vllm}
BACKEND=${BACKEND:-vlm-engine}
DEGRADED=${DEGRADED:-degraded}
OUT=${OUT:-runs_degraded}
CHUNK=${CHUNK:-25}
MODEL=${MODEL:-/home/jovyan/nfs_share/models/MinerU2.5-Pro-2605-1.2B}

ENV_NAME=${ENV_NAME:-mineru_cuda}

fail() { echo "!! $*" >&2; exit 1; }

# The environment first, because everything below depends on it and the three
# exports live in it. MinerU has its own env: its mlx-vlm range is disjoint from
# the VLM envs', so vllm_env3 cannot run it however the exports are set.
[[ ${CONDA_DEFAULT_ENV:-} == "$ENV_NAME" ]] || fail "active env is '${CONDA_DEFAULT_ENV:-none}', expected '$ENV_NAME'.

   conda activate $ENV_NAME
   export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
   export MINERU_MODEL_SOURCE=local
   export MINERU_TOOLS_CONFIG_JSON=~/mineru.json

   All four are needed, and none survives a new shell."
command -v mineru >/dev/null || fail "the mineru CLI is not on PATH in $ENV_NAME"

[[ -d $DEGRADED ]] || fail "no $DEGRADED/ directory"
[[ -d $MODEL ]] || fail "checkpoint not found: $MODEL (set MODEL=)"
[[ ${MINERU_MODEL_SOURCE:-} == "local" ]] ||
    fail "MINERU_MODEL_SOURCE is not 'local'; MinerU may download weights mid-run.
   export MINERU_MODEL_SOURCE=local
   export MINERU_TOOLS_CONFIG_JSON=~/mineru.json"
config=${MINERU_TOOLS_CONFIG_JSON:-}
[[ -n $config && -f ${config/#\~/$HOME} ]] ||
    fail "MINERU_TOOLS_CONFIG_JSON is unset or points at no file.
   MINERU_MODEL_SOURCE=local says do not fetch; this says where local is."

corpora=()
incomplete=()
for c in "$DEGRADED"/*/; do
    name=$(basename "${c%/}")
    images=$(find "$c/images" \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | wc -l | tr -d ' ')
    transcripts=$(find "$c/transcripts" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
    printf "  %-46s img=%-4s tr=%-4s\n" "$name" "$images" "$transcripts"
    if [[ $images -ne $transcripts || $images -eq 0 ]]; then
        incomplete+=("$name")
    else
        corpora+=("${c%/}")
    fi
done
[[ ${#incomplete[@]} -eq 0 ]] || fail "incomplete corpora: ${incomplete[*]}"
[[ ${#corpora[@]} -gt 0 ]] || fail "no complete corpus in $DEGRADED/"

cards=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[[ $cards -ge 2 ]] || fail "expected 2 GPUs for dp=2, found $cards"

echo
echo "system:  $SYSTEM   backend: $BACKEND   chunk: $CHUNK"
echo "corpora: ${#corpora[@]}"
echo

started=$SECONDS
failed=()
for corpus in "${corpora[@]}"; do
    name=$(basename "$corpus")
    echo "=== $name ==="
    # dp=2, one replica per card, with SEPARATE workdirs: MinerU stages a chunk
    # of images into a scratch directory and deletes it afterwards, so two
    # processes sharing one would delete each other's inputs mid-run.
    CUDA_VISIBLE_DEVICES=0 python -u -m runners.run_mineru --corpus "$corpus" \
        --out "$OUT/$name" --system "$SYSTEM" --backend "$BACKEND" --chunk "$CHUNK" \
        --shard 0 --shards 2 --workdir ".mineru_deg_0" & p0=$!
    CUDA_VISIBLE_DEVICES=1 python -u -m runners.run_mineru --corpus "$corpus" \
        --out "$OUT/$name" --system "$SYSTEM" --backend "$BACKEND" --chunk "$CHUNK" \
        --shard 1 --shards 2 --workdir ".mineru_deg_1" & p1=$!
    wait $p0 || { echo "!! $name shard 0 failed"; failed+=("$name/0"); }
    wait $p1 || { echo "!! $name shard 1 failed"; failed+=("$name/1"); }
    echo
done

echo "=== ${#corpora[@]} corpus/corpora in $(( (SECONDS - started) / 60 )) min ==="
[[ ${#failed[@]} -eq 0 ]] || echo "!! failed: ${failed[*]}"

cat <<'NOTE'

Send runs_degraded back and re-run ./score_degraded.sh — it scores every system
present under each tier, so MinerU joins the 31B in the same report.

The question this answers: the 31B's flat result across the scan ladder says
nothing about fragmentation, because it never fragments a row. MinerU does, 276
times on the clean corpus, and row segmentation is what a smeared or skewed scan
should attack.
NOTE

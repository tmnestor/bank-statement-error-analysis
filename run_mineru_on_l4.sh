#!/usr/bin/env bash
# Put MinerU on the Pareto chart, by running it on the hardware PROD has.
#
# MinerU's accuracy in this study was measured on Apple Silicon via MLX, so it
# has no throughput figure comparable to the gemma runs — a rate from a laptop
# does not size a cluster. At 93.0% usable amounts on bank statements it sits
# between the two gemmas, which is exactly the region the frontier runs through:
# if it is also fast, it changes the deployment answer.
#
# READ THIS BEFORE REUSING THE OLD ACCURACY NUMBERS. This runs the SAME weights
# through a DIFFERENT inference stack — vLLM on CUDA rather than MLX on Metal.
# Same checkpoint does not guarantee same output, so the run is scored in its own
# right under its own system name rather than inheriting the Mac figures. If the
# predictions turn out identical, that is a result worth having too.
#
# MinerU is ~2.2 GB, so it fits a 24 GB card many times over: dp=2, one whole
# replica per card, same as the 12B.

set -euo pipefail

ENV_NAME=${ENV_NAME:-mineru_cuda}
MODEL_DIR=${MODEL_DIR:-/home/jovyan/nfs_share/models/MinerU2.5-Pro-2605-1.2B}
OUT=${OUT:-runs_throughput}
SYSTEM=${SYSTEM:-mineru-vllm}
CORPUS=${CORPUS:-}

cat <<'SETUP'
=== one-time setup, if you have not done it ===

  hf download opendatalab/MinerU2.5-Pro-2605-1.2B \
      --local-dir /home/jovyan/nfs_share/models/MinerU2.5-Pro-2605-1.2B

  conda create -n mineru_cuda --clone vllm_env2 -y
  conda activate mineru_cuda
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
  pip install --no-deps mineru mineru-vl-utils
  pip install mineru --no-build-isolation

CLONE vllm_env2, do not build from a bare python. mineru-vl-utils[vllm] requires
vllm<0.22.0 and vllm_env2 already carries 0.19.0, so cloning gives the resolver
nothing to search. Asking pip to resolve vLLM from nothing sends it backtracking
through every version of every transitive dependency — it downloads sdists to
read their metadata and can run for hours. If it starts doing that anyway, add
--only-binary=:all:.

Do NOT install into vllm_env2 itself. It is load-bearing for the InternVL runs,
and a clone costs disk rather than a working environment.

Check both survived before running anything:

  mineru --help | head -5
  python -c "import vllm; print(vllm.__version__)"

Point MinerU at the local weights so it cannot start downloading mid-run:

  export MINERU_MODEL_SOURCE=local

===============================================
SETUP

[[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]] || {
    echo "!! active env is '${CONDA_DEFAULT_ENV:-none}', expected '$ENV_NAME'."
    echo "   conda activate $ENV_NAME"
    exit 1
}
command -v mineru >/dev/null || { echo "!! the mineru CLI is not on PATH in $ENV_NAME"; exit 1; }
[[ -d $MODEL_DIR ]] || { echo "!! checkpoint not found: $MODEL_DIR"; exit 1; }
[[ ${MINERU_MODEL_SOURCE:-} == "local" ]] || {
    echo "!! MINERU_MODEL_SOURCE is not 'local'. Without it MinerU may download"
    echo "   weights mid-run, which times the network rather than the model."
    exit 1
}

if [[ -z $CORPUS ]]; then
    for candidate in parsing_20260820 parsing_20260819d parsing_20260819c \
                     parsing_20260819b parsing_20260819; do
        [[ -d $candidate/images ]] && { CORPUS=$candidate; break; }
    done
fi
[[ -n $CORPUS && -d $CORPUS/images ]] || { echo "no corpus found in $(pwd)"; exit 1; }

pages=$(find "$CORPUS/transcripts" -name '*.md' | wc -l)
echo "corpus: $CORPUS ($pages pages)   ->  $OUT/$SYSTEM"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo

# dp=2, one replica per card, matching how the 12B and InternVL were timed.
#
# --shard/--shards splits the pages between the processes. Without it both would
# take the WHOLE corpus and race, transcribing everything twice and producing a
# throughput figure that means nothing.
#
# Separate workdirs are required, not tidiness: MinerU stages a chunk of images
# into a scratch directory and deletes it afterwards, so two processes sharing
# one would delete each other's inputs mid-run. Both must also sit OUTSIDE --out,
# since score treats every subdirectory of the predictions root as a system.
# vlm-engine, stated explicitly and NOT left to the default. MinerU's default is
# now hybrid-engine, a different parsing method — using it would compare a
# hybrid-engine run on CUDA against a vlm-engine run on Metal and attribute the
# difference to the hardware. vlm-engine is the same backend name the Mac run
# used; MinerU picks the local accelerator itself, MLX there and vLLM here.
BACKEND=${BACKEND:-vlm-engine}
echo "backend: $BACKEND"
started=$SECONDS
CUDA_VISIBLE_DEVICES=0 python -u -m runners.run_mineru --corpus "$CORPUS" \
    --out "$OUT" --system "$SYSTEM" --backend "$BACKEND" \
    --shard 0 --shards 2 --workdir .mineru_work_0 & p0=$!
CUDA_VISIBLE_DEVICES=1 python -u -m runners.run_mineru --corpus "$CORPUS" \
    --out "$OUT" --system "$SYSTEM" --backend "$BACKEND" \
    --shard 1 --shards 2 --workdir .mineru_work_1 & p1=$!
wait $p0 || echo "!! shard 0 failed"
wait $p1 || echo "!! shard 1 failed"
elapsed=$(( SECONDS - started ))

n=$(find "$OUT/$SYSTEM" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l)
echo
echo "=== $n / $pages pages in $(( elapsed / 60 )) min ==="
python - "$n" "$elapsed" <<'PYTHON'
import sys

pages, seconds = int(sys.argv[1]), int(sys.argv[2])
if pages and seconds:
    print(f"  {60 * pages / seconds:.2f} images/min across 2 cards")
    print(f"  {60 * pages / seconds / 2:.2f} per card")
PYTHON

cat <<NOTE

run_mineru.py writes no _timing.json — it shells out to the MinerU CLI rather
than driving an engine, so the rate above is wall clock from this script and
INCLUDES model load, unlike the gemma figures. Subtract a load per invocation
before putting it beside them, or treat it as a floor.

Then send both back and re-score, because this is a different inference stack
from the Mac run and its accuracy cannot be assumed:

  tar czf ${SYSTEM}.tgz $OUT/$SYSTEM
NOTE

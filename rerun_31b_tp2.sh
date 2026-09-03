#!/usr/bin/env bash
# Confirm the 31B's accuracy on the configuration actually being proposed.
#
# THE GAP THIS CLOSES. Every accuracy figure for the 31B -- 99.0% usable amounts,
# 94.7% of statement rows, zero broken structure -- comes from `runs_31b`, which
# ran tensor_parallel_size: 1 on a 48 GB card. The throughput figure, 5.33
# images/min, comes from a separate tp=2 run on the 2xL4 whose transcripts were
# NOT kept. Production is 24 GB cards, so tp=2 is the deployment being argued
# for, and "did you measure accuracy on the configuration you are proposing?"
# currently has no answer.
#
# The two system entries in config/vlm_systems.yml are identical except for
# tensor_parallel_size, so this isolates sharding and nothing else. Temperature
# is 0.0 in both.
#
# WHAT WOULD MAKE THE PREDICTIONS DIFFER. Tensor parallelism splits each matmul
# across cards and all-reduces the partial sums, so the reduction happens in a
# different order and in a different count of steps. Floating-point addition is
# not associative, so a handful of logits land a hair apart; where two tokens are
# near-tied, that flips one. Greedy decoding then amplifies the flip for the rest
# of the page. This is expected in small amounts and is exactly why it must be
# measured rather than assumed -- it is the same class of mistake as quoting
# MinerU's Apple Silicon numbers for its CUDA run, where 41 of 165 pages differed.
#
# Run ON THE 2xL4 HOST from the repo root, in the vLLM env.

set -uo pipefail

SYSTEM=${SYSTEM:-gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2}
BASELINE=${BASELINE:-runs_31b/gemma-4-31B-it-qat-w4a16-ct}
OUT=${OUT:-runs_31b_tp2}
CORPUS=${CORPUS:-parsing_20260820}
MODEL=${MODEL:-/home/jovyan/nfs_share/models/gemma-4-31B-it-qat-w4a16-ct}

fail() { echo "!! $*" >&2; exit 1; }

# --- guards, all before any GPU is touched ---------------------------------

[[ -d $CORPUS/images ]] || fail "corpus not found: $CORPUS/images (set CORPUS=)"
[[ -d $MODEL ]] || fail "checkpoint not found: $MODEL (set MODEL=)"

# The prompt must be the one the scored run used, or this compares two things.
# The prompt is no longer checked here. The runner sends the prompt.md that
# ships inside the corpus -- covered by its manifest -- so there is no
# repo-local copy left to drift or to go stale behind a missing git pull.

# A timing figure over a resumed run is meaningless, and a half-populated
# directory would also make the diff below compare different sets of pages.
existing=$(find "$OUT/$SYSTEM" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l)
[[ $existing -eq 0 ]] || fail "$OUT/$SYSTEM already holds $existing prediction(s).
   Clear it first:  rm -rf $OUT/$SYSTEM"

[[ -d $BASELINE ]] || fail "baseline predictions not found: $BASELINE
   Without them this runs but cannot answer the question it exists to answer."
base_n=$(find "$BASELINE" -maxdepth 1 -name '*.md' | wc -l)
[[ $base_n -eq 165 ]] || fail "baseline holds $base_n predictions, expected 165"

cards=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[[ $cards -ge 2 ]] || fail "tensor_parallel_size is 2 but $cards GPU(s) are visible"

python3 -c "
import sys, yaml
spec = yaml.safe_load(open('config/vlm_systems.yml')).get('systems', {}).get('$SYSTEM')
if spec is None: sys.exit('!! $SYSTEM is not declared in config/vlm_systems.yml')
tp = spec['vllm_engine']['tensor_parallel_size']
if tp != 2: sys.exit(f'!! $SYSTEM declares tensor_parallel_size {tp}, expected 2')
print(f'  tp={tp}  max_model_len={spec[\"vllm_engine\"][\"max_model_len\"]}  '
      f'max_num_seqs={spec[\"vllm_engine\"][\"max_num_seqs\"]}  temp={spec[\"temperature\"]}')
" || exit 1

echo "corpus:   $CORPUS ($(find "$CORPUS/transcripts" -name '*.md' | wc -l) pages)"
echo "baseline: $BASELINE ($base_n predictions, tp=1)"
echo "writing:  $OUT/$SYSTEM"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

# --- the run ---------------------------------------------------------------
#
# No --shard: tp=2 is ONE engine across both cards, not two replicas. Sharding
# the page list as well would start two tensor-parallel engines on two cards
# each, which is four cards.
python -u -m runners.run_vlm --corpus "$CORPUS" --system "$SYSTEM" --out "$OUT"
status=$?
echo
[[ $status -eq 0 ]] || echo "!! run_vlm exited $status — the comparison below covers whatever landed"

# --- the answer ------------------------------------------------------------

python3 - "$BASELINE" "$OUT/$SYSTEM" <<'PYTHON'
import sys
from pathlib import Path

baseline, produced = Path(sys.argv[1]), Path(sys.argv[2])
stems = sorted(p.stem for p in baseline.glob("*.md"))

missing, identical, differing = [], [], []
for stem in stems:
    candidate = produced / f"{stem}.md"
    if not candidate.exists() or not candidate.stat().st_size:
        missing.append(stem)
        continue
    a = (baseline / f"{stem}.md").read_text(encoding="utf-8")
    b = candidate.read_text(encoding="utf-8")
    (identical if a == b else differing).append(stem)

print("=" * 70)
print(f"tp=1 vs tp=2 over {len(stems)} pages")
print(f"  byte-identical : {len(identical)}")
print(f"  differing      : {len(differing)}")
print(f"  missing        : {len(missing)}")
if differing:
    print("\ndiffering pages:")
    for stem in differing[:20]:
        a = (baseline / f"{stem}.md").read_text(encoding="utf-8")
        b = (produced / f"{stem}.md").read_text(encoding="utf-8")
        print(f"  {stem:28} {len(a):6d} -> {len(b):6d} bytes")
    if len(differing) > 20:
        print(f"  ... and {len(differing) - 20} more")
if missing:
    print(f"\nmissing: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")

print()
if not differing and not missing:
    print("IDENTICAL. Accuracy measured under tp=1 carries to tp=2 unchanged, and")
    print("the deployment case can quote one set of numbers for both.")
elif differing:
    print("NOT identical, which is the expected outcome and not a failure. Score")
    print("the new run and compare the AGGREGATES -- a few flipped tokens that")
    print("leave usable-amount rate where it was still supports the case; a")
    print("material move means the proposed configuration must be quoted from")
    print("this run, not the tp=1 one.")
PYTHON

cat <<NOTE

Then score it and put the two side by side:

  conda run -n docparse python -m evaluation.cli \\
      --corpus $CORPUS --predictions $OUT --report scores_31b_tp2.json

  # the numbers the case rests on, from the configuration being proposed
  conda run -n du python -c "
import sys; sys.path.insert(0,'.')
from analysis.figures import load
for tag, report in (('tp=1','scores_31b.json'), ('tp=2','scores_31b_tp2.json')):
    d = load(report); b = d[d.doc_type=='bank_statements']
    print(tag, 'usable', round(b.usable.sum()/b.amounts.sum(), 4),
          'aligned', round(b.aligned.sum()/b.truth_rows.sum(), 4),
          'misfiled', int(b.misfiled.sum()))
"

Send $OUT back either way. If the predictions differ, the tp=2 report becomes
the one the deployment case quotes.
NOTE

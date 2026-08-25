#!/usr/bin/env bash
# Step 3 of 3: score every degraded tier and report what scanning costs.
#
# Each tier is scored against its OWN corpus. The manifests differ by
# construction -- degradation rewrites every image, so every hash changes -- and
# `score` refuses any other pairing, which is what makes scoring a degraded
# prediction against clean ground truth impossible rather than merely
# discouraged.
#
# Tiers present but unscored are skipped with a note rather than failing the
# run, so this works after a scan-only pass and again after photo arrives.

set -uo pipefail

DEGRADED=${DEGRADED:-degraded}
RUNS=${RUNS:-runs_degraded}
ENV_NAME=${ENV_NAME:-docparse}
ANALYSIS_ENV=${ANALYSIS_ENV:-du}

fail() { echo "!! $*" >&2; exit 1; }

[[ -d $DEGRADED ]] || fail "no $DEGRADED/ directory"
[[ -d $RUNS ]] || fail "no $RUNS/ directory — run ./run_degraded_31b.sh first, or unpack runs_degraded.tgz"

scored=0
for d in "$DEGRADED"/*/; do
    name=$(basename "${d%/}")
    predictions="$RUNS/$name"
    if [[ ! -d $predictions ]]; then
        echo "  skip $name — no predictions in $predictions"
        continue
    fi
    echo "=== $name ($(find "$predictions" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ') system(s)) ==="
    # Output goes to a log rather than /dev/null. `score` refuses a corpus whose
    # manifest does not verify and names the offending stem when it does; hiding
    # that leaves "scoring failed" with no cause, which is the one message that
    # cannot be acted on.
    log="scores_${name}.log"
    if conda run -n "$ENV_NAME" python -m evaluation.cli \
        --corpus "${d%/}" --predictions "$predictions" \
        --report "scores_${name}.json" > "$log" 2>&1; then
        echo "  -> scores_${name}.json"
        scored=$((scored + 1))
    else
        echo "  !! scoring failed for $name — last lines of $log:"
        tail -12 "$log" | sed 's/^/     /'
    fi
done

echo
[[ $scored -gt 0 ]] || fail "nothing scored"
echo "$scored tier(s) scored."
echo

# The answer, per intake channel. Never averaged across severities: a mean over
# six image qualities describes one that does not exist.
conda run -n "$ANALYSIS_ENV" python -m analysis.degradation

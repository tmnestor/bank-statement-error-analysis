#!/usr/bin/env bash
# Find the prompt that produced every scored run in this study.
#
# Each run wrote _prompt_provenance.json recording the SHA-256 of the prompt it
# read. All four prompted runs -- the 31B, the 12B 4-bit, InternVL, and both BF16
# controls -- recorded the SAME digest:
#
#     38919c6a81ee959a4d43c0cf2d6de918fee72028317983a99f6a7cc55276db61   974 words
#
# That digest matches NO commit and NO file in this repository. The committed
# config/prompt.md is 1d1b22e0..., 1018 words. So the comparisons between systems
# are sound -- one prompt, four systems -- but the absolute numbers cannot be
# reproduced from a clean checkout, which is the first thing a reviewer will try.
#
# The likely explanation is an uncommitted edit in the sandbox checkout at the
# time of the runs. This script looks for it. It is READ-ONLY: it hashes files
# and prints paths, and changes nothing.

set -uo pipefail

WANTED=38919c6a81ee959a4d43c0cf2d6de918fee72028317983a99f6a7cc55276db61
REPO=${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}
SEARCH=${SEARCH:-$HOME}

sha() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

echo "looking for prompt with sha256 $WANTED"
echo "repo:   $REPO"
echo "search: $SEARCH"
echo

found=""

# 1. The obvious place, before anything clever.
echo "--- working tree ---"
for f in "$REPO"/config/prompt*.md; do
    [[ -f $f ]] || continue
    h=$(sha "$f")
    mark=" "
    [[ $h == "$WANTED" ]] && { mark="*"; found=$f; }
    printf "%s %s  %5s words  %s\n" "$mark" "${h:0:16}" "$(wc -w <"$f")" "$f"
done

# 2. Stashes. An interrupted edit is exactly what would produce this situation.
echo
echo "--- git stashes ---"
if git -C "$REPO" stash list 2>/dev/null | grep -q .; then
    git -C "$REPO" stash list
    while IFS= read -r ref; do
        blob=$(git -C "$REPO" show "${ref}:config/prompt.md" 2>/dev/null) || continue
        h=$(printf '%s' "$blob" | { command -v sha256sum >/dev/null && sha256sum || shasum -a 256; } | cut -d' ' -f1)
        printf "  %s  %s\n" "${h:0:16}" "$ref"
        [[ $h == "$WANTED" ]] && found="stash:$ref"
    done < <(git -C "$REPO" stash list --format='%gd')
else
    echo "  (none)"
fi

# 3. Every blob git has ever seen, including unreachable ones from amended or
# discarded commits. `cat-file --batch-all-objects` reaches objects no branch
# points at, which is where an edit that was committed and then rewritten lives.
echo
echo "--- all git objects (including unreachable) ---"
matches=$(git -C "$REPO" cat-file --batch-all-objects --batch-check='%(objectname) %(objecttype)' 2>/dev/null |
    awk '$2=="blob"{print $1}' |
    while read -r oid; do
        h=$(git -C "$REPO" cat-file blob "$oid" 2>/dev/null | { command -v sha256sum >/dev/null && sha256sum || shasum -a 256; } | cut -d' ' -f1)
        [[ $h == "$WANTED" ]] && echo "$oid"
    done)
if [[ -n $matches ]]; then
    echo "  FOUND as git blob(s):"
    for oid in $matches; do
        echo "    git show $oid > config/prompt.md"
        found="blob:$oid"
    done
else
    echo "  no match among git objects"
fi

# 4. Anything else on disk. Bounded to markdown so this stays quick.
echo
echo "--- filesystem sweep under $SEARCH ---"
while IFS= read -r f; do
    h=$(sha "$f")
    if [[ $h == "$WANTED" ]]; then
        echo "  FOUND: $f"
        found=$f
    fi
done < <(find "$SEARCH" -name '*.md' -size -100k -type f 2>/dev/null)
[[ -z $found ]] && echo "  no match"

echo
echo "=============================================================="
if [[ -n $found ]]; then
    echo "RECOVERED: $found"
    echo
    echo "Copy it to config/prompt.md, confirm the digest, and commit it as its"
    echo "own file -- do NOT overwrite the current prompt.md, which is a later"
    echo "version and is what future runs should use:"
    echo
    echo "    cp <found> config/prompt_v3_scored.md"
    echo "    sha256sum config/prompt_v3_scored.md   # must be $WANTED"
    echo "    git add config/prompt_v3_scored.md && git commit"
else
    echo "NOT FOUND."
    echo
    echo "Then the prompt is gone, and the honest options are:"
    echo "  a) re-run the four prompted systems against the committed prompt.md"
    echo "     and requote every number from that run; or"
    echo "  b) present as-is, stating that the runs share one prompt -- so every"
    echo "     comparison holds -- but that the absolute figures are not"
    echo "     reproducible from this checkout."
    echo
    echo "Do not quietly substitute the committed prompt.md for the scored one."
    echo "It is 44 words longer and its differences are the subject of finding 2."
fi

#!/usr/bin/env bash
# Export analysis/summary.ipynb as a single HTML file that makes NO network
# requests, for colleagues with no Python environment.
#
# WHY THIS EXISTS. `jupyter nbconvert --to html` produces a file that is
# self-contained in every way that matters -- figures are inlined as
# `data:image/svg+xml` URIs and the stylesheet is embedded -- except that its
# template unconditionally emits two CDN script tags, MathJax and require.js.
# On a managed work machine those outbound requests raise security alerts, and
# the notebook uses neither: zero LaTeX expressions and zero widget outputs, so
# both scripts are dead weight whose failure is invisible.
#
# WHY THE TAGS ARE DELETED RATHER THAN BLANKED. Setting the URLs to empty via
# --HTMLExporter.mathjax_url= leaves `<script src="">` behind, and an empty src
# resolves to the DOCUMENT'S OWN URL -- so the browser re-fetches the page and
# tries to execute the HTML as JavaScript. Quieter than a CDN call, but still a
# request, and still something a scanner can flag.
#
# The result is checked, not asserted: the script greps the output for any
# http(s) reference and fails if one survives.
#
# A general version of this lives in the `standalone-notebook` skill and works
# on any notebook. This copy is kept because it is self-contained: the repo's
# reproducibility must not depend on a skill installed in one developer's home
# directory. Keep the two in step when either changes.

set -uo pipefail

NOTEBOOK=${NOTEBOOK:-analysis/summary.ipynb}
OUT_DIR=${OUT_DIR:-analysis}
ENV_NAME=${ENV_NAME:-du}

fail() {
    echo "!! $*" >&2
    exit 1
}

[[ -f $NOTEBOOK ]] || fail "no $NOTEBOOK
  What:    the notebook to export does not exist.
  Where:   $(pwd)
  Example: NOTEBOOK=analysis/summary.ipynb ./$(basename "$0")
  Recover: run this from the repository root."

target="$OUT_DIR/$(basename "${NOTEBOOK%.ipynb}").html"

# --embed-images covers images ATTACHED to markdown cells. Cell OUTPUTS inline
# regardless, because they already live in the .ipynb; a `![](diagram.png)` in a
# markdown cell does not, and without this it stays a relative path that breaks
# the moment the file is moved.
conda run -n "$ENV_NAME" jupyter nbconvert --to html \
    --embed-images \
    --output-dir "$OUT_DIR" \
    --HTMLExporter.mathjax_url= \
    --HTMLExporter.require_js_url= \
    "$NOTEBOOK" 2>&1 | grep -vi 'alternative text' || true

[[ -f $target ]] || fail "nbconvert wrote no $target"

# Delete the emptied script tags. Both forms appear on a single line each: one
# inline after <title>, one with a space between the opening and closing tags.
# sed rather than a heredoc-fed python: heredocs are avoided throughout this
# repository, and an in-place edit through a temp file is portable to both BSD
# and GNU sed, unlike `sed -i`.
before=$(grep -o '<script' "$target" | wc -l | tr -d ' ')
tmp=$(mktemp)
sed 's|<script src="">[[:space:]]*</script>||g' "$target" > "$tmp" && mv "$tmp" "$target"
after=$(grep -o '<script' "$target" | wc -l | tr -d ' ')
echo "  removed $((before - after)) empty-src script tag(s); $after remain (inline, no network)"

remaining=$(grep -oE '(src|href)="https?://[^"]*' "$target" | sort -u)
if [[ -n $remaining ]]; then
    echo "$remaining" >&2
    fail "$target still references the network -- the export is not airgapped."
fi

figures=$(grep -o 'data:image/svg' "$target" | wc -l | tr -d ' ')
size=$(wc -c < "$target")
echo "  $target"
echo "  $((size / 1024)) KB, $figures inlined figure(s), 0 network request(s)"

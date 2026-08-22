#!/usr/bin/env bash
# Split the degraded corpora into per-tier archives, for upload through a browser.
#
# WHY. The single 353 MB archive is transferred by browser upload, which warns
# above ~300 MB and truncates silently rather than failing: the first attempt
# arrived with two tiers holding transcripts and no images, and four holding
# nothing, with no error at either end. Six archives of 50-85 MB each upload
# below the warning threshold, and a failure costs one tier instead of the set.
#
# ORDER MATTERS. The `scan` ladder is production's actual intake and is the one
# the deployment case needs; `photo` is the secondary channel. Uploading scan
# first means the run can start on 157 MB rather than 353, and the result that
# matters arrives without waiting for the rest.
#
# Every archive is checksummed. A truncated upload produces a valid-looking
# directory, so the digest is the only cheap way to know the bytes arrived.

set -uo pipefail

DEGRADED=${DEGRADED:-degraded}
OUT=${OUT:-upload}

fail() { echo "!! $*" >&2; exit 1; }

[[ -d $DEGRADED ]] || fail "no $DEGRADED/ directory — run ./make_degraded_statements.sh first"

mkdir -p "$OUT"
rm -f "$OUT"/*.tgz "$OUT"/SHA256SUMS

# scan before photo, deliberately: see ORDER MATTERS above.
ordered=()
for family in scan photo; do
    for severity in light moderate heavy; do
        for d in "$DEGRADED"/*_"${family}-${severity}"/; do
            [[ -d $d ]] && ordered+=("$d")
        done
    done
done
[[ ${#ordered[@]} -gt 0 ]] || fail "$DEGRADED/ holds no tier directories"

echo "splitting ${#ordered[@]} tier(s) into $OUT/"
echo
for d in "${ordered[@]}"; do
    name=$(basename "${d%/}")
    images=$(find "$d/images" -type f ! -name '._*' | wc -l | tr -d ' ')
    transcripts=$(find "$d/transcripts" -name '*.md' ! -name '._*' | wc -l | tr -d ' ')
    [[ $images -eq $transcripts && $images -gt 0 ]] ||
        fail "$name is already incomplete locally: $images image(s), $transcripts transcript(s)"

    # COPYFILE_DISABLE keeps macOS from writing AppleDouble ._* companions into
    # the archive. They are harmless once the runners filter them, but they
    # inflate the upload and they made the first failure harder to read.
    COPYFILE_DISABLE=1 tar czf "$OUT/${name}.tgz" -C "$DEGRADED" "$name" ||
        fail "could not archive $name"

    size=$(du -h "$OUT/${name}.tgz" | cut -f1)
    printf "  %-46s %5s  (%s pages)\n" "$name" "$size" "$images"
done

( cd "$OUT" && shasum -a 256 ./*.tgz > SHA256SUMS )

echo
echo "=== $OUT/ ==="
du -sh "$OUT"
cat <<'NOTE'

Upload the three scan-* archives FIRST -- that is production's intake and the
half the deployment case needs. Then, on the sandbox:

  mkdir -p degraded
  for f in *_scan-*.tgz; do tar xzf "$f" -C degraded; done

  # Confirm the bytes arrived. A truncated upload leaves a plausible directory,
  # so this is the check that matters, not the file listing.
  sha256sum -c SHA256SUMS        # or: shasum -a 256 -c SHA256SUMS

  ./run_degraded_31b.sh

The run script skips tiers that are absent and refuses tiers that are present
but short, so a scan-only run works and a half-uploaded one stops.

Then upload the photo-* archives and re-run; completed tiers are not redone.
NOTE

"""Normalisation policy for scoring.

The primary metric is computed over normalised text, and this module is the
only place that normalisation happens. Policy lives in config/scoring.yml
with every key required, so that reading that file alone answers what scoring
does -- the same contract config/serialisation.yml carries for transcripts.

Stripping is structural, never a character blacklist. Corpus design says
normalisation strips Markdown syntax characters; taken literally that is
wrong, because the corpus contains real content built from the same
characters -- the hyphen in a statement date range, the Westpac Credits (-)
column header, the * in the Square merchant descriptor SQ *COASTAL, and the
# beginning the receipt number #R-011FDD. Each of those would be corrupted
by a blacklist, and a model reading them correctly would be scored down for it.
So each strip below is positional and paired.
"""

import re
import unicodedata
from pathlib import Path

import yaml

REQUIRED_POLICY_KEYS: tuple[str, ...] = (
    "unicode_form",
    "fold_dashes",
    "fold_quotes",
    "strip_emphasis",
    "strip_heading_marks",
    "strip_table_marks",
    "strip_blockquote_marks",
    "collapse_whitespace",
    "fold_case",
)

_EXAMPLES: dict[str, str] = {
    "unicode_form": "NFKC",
    "fold_dashes": "true",
    "fold_quotes": "true",
    "strip_emphasis": "true",
    "strip_heading_marks": "true",
    "strip_table_marks": "true",
    "strip_blockquote_marks": "true",
    "collapse_whitespace": "true",
    "fold_case": "false",
}

_UNICODE_FORMS = ("NFC", "NFD", "NFKC", "NFKD")

# Use Unicode escapes to create the translation tables
_DASH_FOLD = str.maketrans(
    {
        c: "-"
        for c in [
            "‐",  # hyphen
            "‑",  # non-breaking hyphen
            "‒",  # figure dash
            "–",  # en dash
            "—",  # em dash
            "―",  # horizontal bar
            "−",  # minus sign
        ]
    }
)

_QUOTE_FOLD = str.maketrans(
    {
        **{
            c: "'"
            for c in [
                "‘",  # left single quote
                "’",  # right single quote
                "‚",  # single low-9 quote
                "‛",  # single high-reversed-9 quote
                "′",  # prime
            ]
        },
        **{
            c: '"'
            for c in [
                "“",  # left double quote
                "”",  # right double quote
                "„",  # double low-9 quote
                "‟",  # double high-reversed-9 quote
                "″",  # double prime
            ]
        },
    }
)

# Paired delimiters only. **a** and _a_ are emphasis; the lone * in
# SQ *COASTAL Hobart AUS is content and has no partner, so neither pattern
# matches it.
_STRONG = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_EMPHASIS = re.compile(r"(?<![\w*_])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*_])", re.DOTALL)

# An ATX heading is a hash run followed by whitespace or end of line. The
# trailing \s is what leaves the receipt number #R-011FDD intact.
_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^ {0,3}> ?", re.MULTILINE)

_SEPARATOR_CHARS = set("|:- \t")


class ScoringError(RuntimeError):
    """Raised when the scoring policy is missing, malformed, or incomplete."""


def _err(what: str, *, path: Path, key: str, expected: str, recover: str) -> ScoringError:
    """Build a four-element fail-fast diagnostic."""
    return ScoringError(
        "Invalid scoring policy.\n"
        f"  What:     {what}\n"
        f"  Where:    {path} -> {key}\n"
        f"  Expected: {expected}\n"
        f"  Recover:  {recover}"
    )


def load_scoring_policy(path: Path) -> dict:
    """Load and validate the normalisation policy.

    Every key is required. Omitting one is an error, never a silent default,
    including fold_case, whose value is a no-op -- so that reading the file
    alone answers what scoring does.

    Args:
        path: Path to scoring.yml.

    Returns:
        The validated policy mapping.

    Raises:
        ScoringError: The file is missing, unparseable, not a mapping, missing
            a required key, or names a Unicode form Python does not implement.
    """
    resolved = path.resolve()
    if not path.exists():
        raise _err(
            f"{path} does not exist.",
            path=resolved,
            key="(whole file)",
            expected="a YAML mapping declaring every key of "
            f"{list(REQUIRED_POLICY_KEYS)}, e.g.\n              unicode_form: NFKC",
            recover="create config/scoring.yml (see the copy in the repository root config/).",
        ) from None

    try:
        policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise _err(
            f"the file is not valid YAML: {err}",
            path=resolved,
            key="(whole file)",
            expected="parseable YAML, e.g.\n              fold_case: false",
            recover="fix the syntax error at the line named above.",
        ) from err

    if not isinstance(policy, dict):
        raise _err(
            f"expected a mapping, got {type(policy).__name__}.",
            path=resolved,
            key="(document root)",
            expected="a top-level mapping of policy keys, e.g.\n              unicode_form: NFKC",
            recover="wrap the settings in a top-level mapping.",
        ) from None

    for key in REQUIRED_POLICY_KEYS:
        if key not in policy:
            raise _err(
                f"'{key}' is not declared.",
                path=resolved,
                key=key,
                expected=f"every key of {list(REQUIRED_POLICY_KEYS)} present -- none has a "
                f"Python default, including no-op values, e.g.\n              {key}: "
                f"{_EXAMPLES[key]}",
                recover=f"add '{key}:' to {path}.",
            ) from None

    if policy["unicode_form"] not in _UNICODE_FORMS:
        raise _err(
            f"unknown unicode_form {policy['unicode_form']!r}.",
            path=resolved,
            key="unicode_form",
            expected=f"one of {list(_UNICODE_FORMS)}, e.g.\n              unicode_form: NFKC",
            recover="set unicode_form to one of the four Unicode normalisation forms.",
        ) from None

    return policy


def _strip_table_marks(text: str) -> str:
    """Drop pipe-table separator rows and turn cell pipes into spaces.

    A separator row is a line built only from pipes, colons, dashes and space,
    and carrying at least one of each of the first and third. Only lines whose
    stripped form begins with a pipe are treated as table rows at all, so a
    pipe appearing inside prose is left alone.
    """
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|"):
            if "-" in stripped and set(stripped) <= _SEPARATOR_CHARS:
                continue
            kept.append(line.replace("|", " "))
        else:
            kept.append(line)
    return "\n".join(kept)


def normalise(text: str, policy: dict) -> str:
    """Apply the normalisation policy, in the order the policy fixes.

    The steps do not commute -- stripping table pipes before collapsing
    whitespace leaves the padding the pipes were separating, and collapsing
    first would fold a separator row onto its neighbours -- so scoring spec
    section 3.3 fixes the order and this function implements exactly it.

    Args:
        text: Raw transcript or prediction text.
        policy: A mapping validated by load_scoring_policy.

    Returns:
        The normalised string.
    """
    text = unicodedata.normalize(policy["unicode_form"], text)  # type: ignore[arg-type]

    if policy["fold_dashes"]:
        text = text.translate(_DASH_FOLD)
    if policy["fold_quotes"]:
        text = text.translate(_QUOTE_FOLD)

    if policy["strip_emphasis"]:
        text = _STRONG.sub(r"\2", text)
        text = _EMPHASIS.sub(r"\2", text)
    if policy["strip_heading_marks"]:
        text = _HEADING.sub("", text)
    if policy["strip_table_marks"]:
        text = _strip_table_marks(text)
    if policy["strip_blockquote_marks"]:
        text = _BLOCKQUOTE.sub("", text)

    if policy["collapse_whitespace"]:
        text = re.sub(r"\s+", " ", text).strip()
    if policy["fold_case"]:
        text = text.casefold()

    return text

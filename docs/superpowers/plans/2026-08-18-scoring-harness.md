# Scoring Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `score` command that measures predictions against the corpus on two metrics and classifies every divergence as a convention mismatch or a reading error.

**Architecture:** Four units with one-way dependencies — `metrics.py` (pure arithmetic, no policy) → `scoring.py` (policy + normalisation) → `divergence.py` (align, classify, group) → `pipeline.py` (preconditions, orchestration, reporting). Classification is by construction: normalise both sides of a hunk, and equality means the difference was pure convention.

**Tech Stack:** Python 3.12, stdlib only for the new logic (`difflib`, `unicodedata`, `re`, `hashlib`, `json`), plus the repo's existing `typer`, `rich` and `PyYAML`. Tests with `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-18-scoring-harness-design.md` (read it first; it argues every decision below). Its parent is `docs/superpowers/specs/2026-08-17-document-parsing-corpus-design.md` §5.

## Global Constraints

- Conda environment is **`docparse`**, not the global `du`. Run everything as `conda run -n docparse <command>`.
- **No new runtime dependencies.** The five are `Pillow`, `PyYAML`, `typer`, `rich`, `Faker`. `numpy`, `opencv` and `rapidfuzz` stay out.
- Python 3.12 type syntax: `X | Y`, never `Union[X, Y]`. No `from __future__ import annotations`. No `TYPE_CHECKING` guards for runtime-signature types.
- Maximum line length **108**.
- All paths are `pathlib.Path`. Google-style docstrings.
- **Every config key is required** — no Python-side defaults, including no-op values.
- **Every fail-fast error carries all four elements**: What, Where (absolute path + dotted key), Expected (concrete example), Recover (one-line remediation). Tests assert this via `assert_diagnostic_error` in `tests/helpers.py`.
- In `except` blocks always `raise ... from err` or `from None` (B904).
- `tests/` is gitignored — write and run tests, but stage source and config only.
- Quality gates, all four must pass before each commit:
  ```
  conda run -n docparse pytest tests/ --cov=generators --cov-report=term
  conda run -n docparse ruff check --fix --ignore ARG001,ARG002,F841 .
  conda run -n docparse ruff format .
  conda run -n docparse mypy generators --ignore-missing-imports
  ```
- Coverage floor 80%. **No Claude attribution in commit messages.**

---

### Task 1: Edit-distance metrics

Pure arithmetic with no knowledge of Markdown, policy, or files. Spec §7 makes this isolation the boundary that structurally enforces §5.1's "`difflib` never scores" rule.

**Files:**
- Create: `generators/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `edit_distance(a: Sequence, b: Sequence) -> int`
  - `error_rate(distance: int, reference_length: int) -> float`
  - `cer(truth: str, prediction: str) -> tuple[int, float]`
  - `wer(truth: str, prediction: str) -> tuple[int, float]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
"""Edit distance and error rates, independent of any scoring policy."""

import pytest

from generators.metrics import cer, edit_distance, error_rate, wer


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("abc", "abd", 1),
        ("kitten", "sitting", 3),
        ("", "abc", 3),
        ("abc", "", 3),
    ],
)
def test_edit_distance_over_characters(a, b, expected):
    assert edit_distance(a, b) == expected


def test_edit_distance_over_word_sequences():
    """The same function scores words; `wer` depends on it accepting lists."""
    assert edit_distance(["the", "cat"], ["the", "dog"]) == 1


def test_edit_distance_is_symmetric():
    assert edit_distance("kitten", "sitting") == edit_distance("sitting", "kitten")


def test_error_rate_defines_both_empty_cases():
    """Spec §4: defined, rather than left to a ZeroDivisionError."""
    assert error_rate(0, 0) == 0.0
    assert error_rate(5, 0) == 1.0


def test_cer_divides_by_the_truth_not_the_longer_side():
    """Spec §4: a system emitting nothing scores 1.0, not 0.5."""
    distance, rate = cer("abcd", "")
    assert (distance, rate) == (4, 1.0)


def test_cer_returns_distance_and_rate():
    assert cer("abc", "abd") == (1, pytest.approx(1 / 3))


def test_wer_splits_on_whitespace():
    assert wer("the cat sat", "the dog sat") == (1, pytest.approx(1 / 3))


def test_wer_ignores_whitespace_runs():
    """Wrapping is not a word error; §4's WER splits, it does not count spaces."""
    assert wer("the cat sat", "the   cat\nsat") == (0, 0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n docparse pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'generators.metrics'`

- [ ] **Step 3: Write the implementation**

Create `generators/metrics.py`:

```python
"""Edit-distance metrics for transcription scoring (scoring spec §4).

Pure arithmetic. This module knows nothing about Markdown, about the scoring
policy, or about files, and that isolation is deliberate: it makes the reported
numbers auditable on their own, and it is what structurally enforces the rule
that `difflib` locates divergences but never computes a score (spec §5.1).
`difflib.SequenceMatcher` returns a longest-contiguous-match ratio, not an edit
distance, and reporting one as the other would be wrong.
"""

from collections.abc import Sequence


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance with unit insert/delete/substitute cost.

    Takes any sequence, not just `str`, so the same implementation scores
    characters and word lists -- `wer` passes lists.

    Two rows rather than a full matrix: the corpus's largest transcript is
    ~3,100 characters, and holding one 3,100-cell row instead of a 9.6M-cell
    matrix is what keeps a pure-Python implementation viable (spec §4).

    Args:
        a: Reference sequence.
        b: Sequence to compare against it.

    Returns:
        The minimum number of single-element edits converting `a` into `b`.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (item_a != item_b))
            )
        previous = current
    return previous[-1]


def error_rate(distance: int, reference_length: int) -> float:
    """Divide a distance by its reference length, defining the empty cases.

    An empty reference with an empty prediction is a perfect score; an empty
    reference with any prediction is a total miss. Spec §4 fixes both rather
    than leaving a `ZeroDivisionError` to surface mid-run.

    Args:
        distance: Edit distance between reference and prediction.
        reference_length: Length of the reference, in the same units.

    Returns:
        The error rate, `0.0` or `1.0` when the reference is empty.
    """
    if reference_length == 0:
        return 0.0 if distance == 0 else 1.0
    return distance / reference_length


def cer(truth: str, prediction: str) -> tuple[int, float]:
    """Character edit distance and error rate.

    The denominator is the truth, not `max(len(...))`: an error rate is
    conventionally relative to the reference, and a system that emits nothing
    should score 1.0 rather than 0.5 (spec §4).

    Args:
        truth: The canonical transcript.
        prediction: The system's output.

    Returns:
        `(distance, rate)`.
    """
    distance = edit_distance(truth, prediction)
    return distance, error_rate(distance, len(truth))


def wer(truth: str, prediction: str) -> tuple[int, float]:
    """Word edit distance and error rate, splitting on whitespace.

    Splitting rather than counting whitespace means a rewrapped line is not a
    word error, which matches the transcripts being captured pre-wrap.

    Args:
        truth: The canonical transcript.
        prediction: The system's output.

    Returns:
        `(distance, rate)`.
    """
    truth_words = truth.split()
    distance = edit_distance(truth_words, prediction.split())
    return distance, error_rate(distance, len(truth_words))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n docparse pytest tests/test_metrics.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Run the full gates**

```
conda run -n docparse pytest tests/ -q
conda run -n docparse ruff check --fix --ignore ARG001,ARG002,F841 .
conda run -n docparse ruff format .
conda run -n docparse mypy generators --ignore-missing-imports
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add generators/metrics.py
git commit -m "✨ feat: add edit-distance metrics for scoring"
```

---

### Task 2: Normalisation policy and function

**Files:**
- Create: `config/scoring.yml`
- Create: `generators/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `REQUIRED_POLICY_KEYS: tuple[str, ...]`
  - `class ScoringError(RuntimeError)`
  - `load_scoring_policy(path: Path) -> dict`
  - `normalise(text: str, policy: dict) -> str`

**Two real corpus strings drive the regexes** — both found by scanning the shipped transcripts, both would be corrupted by the naive reading of corpus spec §5 that scoring spec §3.2 rejects:

- `EFTPOS SQ *COASTAL Hobart AUS` — a Square merchant descriptor. A literal, unpaired `*` in transaction text. Emphasis stripping must require a matched pair.
- `#R-011FDD` — a receipt number at the start of a line. An ATX heading requires whitespace after its hashes, so `^#+` alone would eat this.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scoring.py`:

```python
"""Normalisation policy: loading, and each key's effect in isolation."""

from pathlib import Path

import pytest

from generators.scoring import REQUIRED_POLICY_KEYS, ScoringError, load_scoring_policy, normalise
from tests.helpers import assert_diagnostic_error

POLICY = {
    "unicode_form": "NFKC",
    "fold_dashes": True,
    "fold_quotes": True,
    "strip_emphasis": True,
    "strip_heading_marks": True,
    "strip_table_marks": True,
    "strip_blockquote_marks": True,
    "collapse_whitespace": True,
    "fold_case": False,
}


def test_the_shipped_policy_loads_and_declares_every_key():
    policy = load_scoring_policy(Path("config/scoring.yml"))
    assert set(policy) == set(REQUIRED_POLICY_KEYS)


def test_a_missing_key_fails_with_a_four_element_diagnostic(tmp_path):
    partial = tmp_path / "scoring.yml"
    partial.write_text("unicode_form: NFKC\n", encoding="utf-8")
    with pytest.raises(ScoringError) as excinfo:
        load_scoring_policy(partial)
    assert_diagnostic_error(str(excinfo.value), mentions=("fold_case", str(partial.resolve())))


def test_a_missing_file_fails_with_a_four_element_diagnostic(tmp_path):
    with pytest.raises(ScoringError) as excinfo:
        load_scoring_policy(tmp_path / "absent.yml")
    assert_diagnostic_error(str(excinfo.value), mentions=("absent.yml",))


def test_emphasis_is_stripped_when_paired():
    assert normalise("**Total**: $12.00", POLICY) == "Total: $12.00"
    assert normalise("_Total_: $12.00", POLICY) == "Total: $12.00"


def test_an_unpaired_asterisk_in_content_survives():
    """`SQ *COASTAL` is a real Square merchant descriptor in the corpus."""
    assert normalise("EFTPOS SQ *COASTAL Hobart AUS", POLICY) == "EFTPOS SQ *COASTAL Hobart AUS"


def test_heading_marks_are_stripped_only_with_following_whitespace():
    assert normalise("# TAX INVOICE", POLICY) == "TAX INVOICE"


def test_a_receipt_number_starting_with_hash_survives():
    """`#R-011FDD` is a real receipt number; it is not an ATX heading."""
    assert normalise("#R-011FDD", POLICY) == "#R-011FDD"


def test_table_marks_and_separator_rows_are_stripped():
    table = "| Date | Amount |\n| --- | --- |\n| 01/09/2023 | $12.00 |"
    assert normalise(table, POLICY) == "Date Amount 01/09/2023 $12.00"


def test_hyphens_inside_content_survive_stripping():
    """Spec §3.2: four real strings a character blacklist would corrupt."""
    for text in (
        "Statement Period: 01/09/2023 - 23/09/2023",
        "Delivery: Standard delivery, 5-7 business days",
        "Credits (-)",
        "Payment Terms: 50% deposit, balance on completion",
    ):
        assert normalise(text, POLICY) == text


def test_blockquote_marks_are_stripped():
    assert normalise("> quoted", POLICY) == "quoted"


def test_dashes_and_quotes_fold_to_ascii():
    assert normalise("— ‘a’ “b”", POLICY) == "- 'a' \"b\""


def test_whitespace_runs_collapse_and_ends_trim():
    assert normalise("  a\n\n  b  ", POLICY) == "a b"


def test_case_is_never_folded():
    """Corpus spec §5: reading identifiers with correct case is part of the job."""
    assert normalise("Robin Wood", POLICY) == "Robin Wood"


def test_each_strip_is_off_when_its_key_is_false():
    off = dict(POLICY, strip_emphasis=False)
    assert normalise("**Total**", off) == "**Total**"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n docparse pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'generators.scoring'`

- [ ] **Step 3: Write the policy file**

Create `config/scoring.yml`:

```yaml
# How a prediction and a transcript are normalised before the primary metric.
# This file is the whole normalisation policy: reading it should answer what
# scoring does without consulting Python. Every key is required — the loader
# fails fast on any omission, including the key whose value is a no-op.
#
# Normalisation lives here, and never in the generator: the corpus emits one
# canonical form and never normalises, so scoring policy can change without
# regenerating a single image (corpus design §5).

# Unicode normalisation form, applied first.
unicode_form: NFKC

# En/em/figure dashes and curly quotes fold to their ASCII equivalents. This is
# folding, not stripping: the characters become ASCII, they do not disappear.
fold_dashes: true
fold_quotes: true

# Markdown structure is removed positionally, never as a character blacklist.
# A blacklist would corrupt real content in this corpus — the hyphen in
# "01/09/2023 - 23/09/2023" and in "5-7 business days", the Westpac "Credits (-)"
# column header — and would score a model down for reading them correctly.
# So: emphasis is a matched pair of delimiters, a heading mark is a hash run
# followed by whitespace at the start of a line (leaving the receipt number
# "#R-011FDD" alone), a table mark is a cell pipe or a separator row.
strip_emphasis: true
strip_heading_marks: true
strip_table_marks: true
strip_blockquote_marks: true

# All whitespace runs become one space, and both ends are trimmed. This is what
# makes the primary metric blind to wrapping.
collapse_whitespace: true

# Deliberately false, and present rather than absent so the file states the
# decision instead of leaving it to be inferred from silence. Reading account
# names and identifiers with correct case is legitimately part of transcription
# (corpus design §5).
fold_case: false
```

- [ ] **Step 4: Write the implementation**

Create `generators/scoring.py`:

```python
"""Normalisation policy for scoring (scoring spec §3).

The primary metric is computed over normalised text, and this module is the
only place that normalisation happens. Policy lives in `config/scoring.yml`
with every key required, so that reading that file alone answers what scoring
does -- the same contract `config/serialisation.yml` carries for transcripts.

Stripping is structural, never a character blacklist. Corpus design §5 says
normalisation "strips Markdown syntax characters"; taken literally that is
wrong, because the corpus contains real content built from the same
characters -- the hyphen in a statement date range, the Westpac "Credits (-)"
column header, the `*` in the Square merchant descriptor "SQ *COASTAL", and the
`#` beginning the receipt number "#R-011FDD". Each of those would be corrupted
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

_DASH_FOLD = str.maketrans({c: "-" for c in "‐‑‒–—―−"})
_QUOTE_FOLD = str.maketrans(
    {
        **{c: "'" for c in "‘’‚‛′"},
        **{c: '"' for c in "“”„‟″"},
    }
)

# Paired delimiters only. `**a**` and `_a_` are emphasis; the lone `*` in
# "SQ *COASTAL Hobart AUS" is content and has no partner, so neither pattern
# matches it.
_STRONG = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_EMPHASIS = re.compile(r"(?<![\w*_])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*_])", re.DOTALL)

# An ATX heading is a hash run followed by whitespace or end of line. The
# trailing `\s` is what leaves the receipt number "#R-011FDD" intact.
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
    including `fold_case`, whose value is a no-op -- so that reading the file
    alone answers what scoring does.

    Args:
        path: Path to `scoring.yml`.

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
        )

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
        )

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
            )

    if policy["unicode_form"] not in _UNICODE_FORMS:
        raise _err(
            f"unknown unicode_form {policy['unicode_form']!r}.",
            path=resolved,
            key="unicode_form",
            expected=f"one of {list(_UNICODE_FORMS)}, e.g.\n              unicode_form: NFKC",
            recover="set unicode_form to one of the four Unicode normalisation forms.",
        )

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
    §3.3 fixes the order and this function implements exactly it.

    Args:
        text: Raw transcript or prediction text.
        policy: A mapping validated by `load_scoring_policy`.

    Returns:
        The normalised string.
    """
    text = unicodedata.normalize(str(policy["unicode_form"]), text)

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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `conda run -n docparse pytest tests/test_scoring.py -v`
Expected: PASS, 14 tests.

- [ ] **Step 6: Run the full gates**

```
conda run -n docparse pytest tests/ -q
conda run -n docparse ruff check --fix --ignore ARG001,ARG002,F841 .
conda run -n docparse ruff format .
conda run -n docparse mypy generators --ignore-missing-imports
```

- [ ] **Step 7: Commit**

```bash
git add generators/scoring.py config/scoring.yml
git commit -m "✨ feat: add the scoring normalisation policy"
```

---

### Task 3: Divergence classification and grouping

**Files:**
- Create: `generators/divergence.py`
- Test: `tests/test_divergence.py`

**Interfaces:**
- Consumes: `generators.scoring.normalise(text, policy)` from Task 2.
- Produces:
  - `@dataclass(frozen=True) class Hunk` with fields `truth: str`, `prediction: str`, `kind: str`
  - `CONVENTION: str = "convention"`, `READING: str = "reading"`
  - `hunks(truth: str, prediction: str, policy: dict) -> list[Hunk]`
  - `group(hunks_by_system: dict[str, list[Hunk]]) -> dict[str, list[dict]]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_divergence.py`:

```python
"""Hunk extraction, classification by construction, and grouping."""

from generators.divergence import CONVENTION, READING, Hunk, group, hunks

POLICY = {
    "unicode_form": "NFKC",
    "fold_dashes": True,
    "fold_quotes": True,
    "strip_emphasis": True,
    "strip_heading_marks": True,
    "strip_table_marks": True,
    "strip_blockquote_marks": True,
    "collapse_whitespace": True,
    "fold_case": False,
}


def test_identical_text_yields_no_hunks():
    assert hunks("Total: $12.00", "Total: $12.00", POLICY) == []


def test_an_emphasis_difference_is_a_convention_mismatch():
    """It vanishes under normalisation, so by construction it is convention."""
    found = hunks("Total: $12.00", "**Total**: $12.00", POLICY)
    assert [h.kind for h in found] == [CONVENTION]


def test_a_transposed_digit_is_a_reading_error():
    """It survives normalisation, so by construction it is a reading error."""
    found = hunks("ABN 57 773 872 148", "ABN 57 773 872 143", POLICY)
    assert [h.kind for h in found] == [READING]


def test_one_pair_can_carry_both_classes():
    truth = "# TAX INVOICE Total: $157.39"
    prediction = "## TAX INVOICE **Total**: $157.89"
    kinds = sorted(h.kind for h in hunks(truth, prediction, POLICY))
    assert kinds == [CONVENTION, READING]


def test_a_hunk_records_both_sides_verbatim():
    found = hunks("Total", "**Total**", POLICY)
    assert (found[0].truth, found[0].prediction) == ("Total", "**Total**")


def test_an_omission_is_a_hunk_with_an_empty_prediction():
    found = hunks("Payment Terms: Net 30 days", "Payment Terms:", POLICY)
    assert found and found[0].prediction == ""


def test_grouping_counts_a_repeated_hunk_once_with_both_systems():
    hunk = Hunk(truth="Total", prediction="**Total**", kind=CONVENTION)
    grouped = group({"docling": [hunk], "mineru": [hunk]})
    assert grouped[CONVENTION] == [
        {"truth": "Total", "prediction": "**Total**", "count": 2,
         "systems": ["docling", "mineru"]}
    ]


def test_grouping_separates_the_two_classes():
    grouped = group(
        {
            "gemma": [
                Hunk(truth="Total", prediction="**Total**", kind=CONVENTION),
                Hunk(truth="148", prediction="143", kind=READING),
            ]
        }
    )
    assert len(grouped[CONVENTION]) == 1
    assert len(grouped[READING]) == 1


def test_grouping_sorts_by_descending_count():
    common = Hunk(truth="Total", prediction="**Total**", kind=CONVENTION)
    rare = Hunk(truth="Date", prediction="**Date**", kind=CONVENTION)
    grouped = group({"a": [common, common, rare]})
    assert [g["count"] for g in grouped[CONVENTION]] == [2, 1]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n docparse pytest tests/test_divergence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'generators.divergence'`

- [ ] **Step 3: Write the implementation**

Create `generators/divergence.py`:

```python
"""Locate divergences and classify each one (scoring spec §5).

The classification rests on a single idea: normalise both sides of a hunk, and
if they become equal the difference was pure convention -- by definition, since
normalisation is exactly the set of differences declared not to be reading
errors. There is no pattern list to maintain and nothing to tune, and the
classifier cannot drift from the policy because it *is* the policy, applied at
hunk granularity instead of document granularity.

`difflib` locates divergences here and never scores them; the numbers come from
`generators.metrics` (spec §5.1).
"""

import difflib
from dataclasses import dataclass

from generators.scoring import normalise

CONVENTION = "convention"
READING = "reading"


@dataclass(frozen=True)
class Hunk:
    """One divergence between a transcript and a prediction.

    Attributes:
        truth: The transcript's span, verbatim.
        prediction: The prediction's span, verbatim. Empty when the system
            omitted the span entirely.
        kind: `CONVENTION` or `READING`.
    """

    truth: str
    prediction: str
    kind: str


def hunks(truth: str, prediction: str, policy: dict) -> list[Hunk]:
    """Extract and classify every divergence between two texts.

    Alignment is over whitespace-split words rather than characters: a word is
    the smallest unit a human can read a divergence report at, and character
    opcodes would fragment one wrong digit into three unreadable hunks.

    `autojunk` is off. Its heuristic treats any element appearing in more than
    1% of a sequence of 200+ elements as junk, and in a bank statement the
    words "$", "EFTPOS" and the pipe-stripped column values all clear that bar
    -- leaving alignment to skip exactly the repetitive text the corpus is made
    of.

    Args:
        truth: The canonical transcript.
        prediction: The system's output.
        policy: A mapping validated by `load_scoring_policy`.

    Returns:
        One `Hunk` per divergent span, in document order.
    """
    truth_words = truth.split()
    prediction_words = prediction.split()
    matcher = difflib.SequenceMatcher(a=truth_words, b=prediction_words, autojunk=False)

    found: list[Hunk] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        truth_span = " ".join(truth_words[i1:i2])
        prediction_span = " ".join(prediction_words[j1:j2])
        kind = (
            CONVENTION
            if normalise(truth_span, policy) == normalise(prediction_span, policy)
            else READING
        )
        found.append(Hunk(truth=truth_span, prediction=prediction_span, kind=kind))
    return found


def group(hunks_by_system: dict[str, list[Hunk]]) -> dict[str, list[dict]]:
    """Group identical hunks corpus-wide, counting and attributing them.

    Grouping is exact match on the `(truth, prediction)` pair -- no clustering,
    no fuzzy merging, no stemming -- so a count is a fact about the corpus
    rather than the output of a similarity threshold nobody will remember
    tuning (spec §5.2).

    The intended reading: a convention group appearing across every system
    indicts the convention; one appearing in a single system is that system's
    dialect and should not move `config/serialisation.yml`.

    Args:
        hunks_by_system: System name -> every hunk that system produced.

    Returns:
        `{CONVENTION: [...], READING: [...]}`, each a list of
        `{"truth", "prediction", "count", "systems"}` sorted by descending
        count then by truth text.
    """
    counts: dict[tuple[str, str, str], int] = {}
    systems: dict[tuple[str, str, str], set[str]] = {}

    for system, system_hunks in hunks_by_system.items():
        for hunk in system_hunks:
            key = (hunk.kind, hunk.truth, hunk.prediction)
            counts[key] = counts.get(key, 0) + 1
            systems.setdefault(key, set()).add(system)

    grouped: dict[str, list[dict]] = {CONVENTION: [], READING: []}
    for (kind, truth, prediction), count in counts.items():
        grouped[kind].append(
            {
                "truth": truth,
                "prediction": prediction,
                "count": count,
                "systems": sorted(systems[(kind, truth, prediction)]),
            }
        )
    for kind in grouped:
        grouped[kind].sort(key=lambda entry: (-entry["count"], entry["truth"]))
    return grouped
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n docparse pytest tests/test_divergence.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Run the full gates**

```
conda run -n docparse pytest tests/ -q
conda run -n docparse ruff check --fix --ignore ARG001,ARG002,F841 .
conda run -n docparse ruff format .
conda run -n docparse mypy generators --ignore-missing-imports
```

- [ ] **Step 6: Commit**

```bash
git add generators/divergence.py
git commit -m "✨ feat: classify divergences as convention or reading errors"
```

---

### Task 4: Input preconditions

Three hard failures before any scoring happens, per spec §6. `sha256_of` already exists in `generators/export.py` and is reused rather than reimplemented.

**Files:**
- Modify: `generators/pipeline.py` (add two private helpers near `_load_event_records`)
- Test: `tests/test_score_inputs.py`

**Interfaces:**
- Consumes: `generators.export.sha256_of(path: Path) -> str`.
- Produces:
  - `class ScoreInputError(RuntimeError)` — defined in `generators/pipeline.py`
  - `_verify_corpus(corpus: Path) -> list[dict]` — returns manifest rows
  - `_pair_predictions(corpus: Path, predictions: Path) -> dict[str, dict[str, Path]]` — system name -> {stem -> prediction path}

- [ ] **Step 1: Write the failing test**

Create `tests/test_score_inputs.py`:

```python
"""Scoring refuses to run against a corpus or prediction set it cannot trust."""

import json
import shutil
from pathlib import Path

import pytest

from generators.pipeline import ScoreInputError, _pair_predictions, _verify_corpus
from tests.helpers import assert_diagnostic_error


def _corpus(tmp_path: Path) -> Path:
    """Build a two-document corpus with a valid manifest."""
    root = tmp_path / "parsing_20260818"
    (root / "images").mkdir(parents=True)
    (root / "transcripts").mkdir(parents=True)
    rows = []
    for stem in ("CASE001_invoices", "CASE002_invoices"):
        image = root / "images" / f"{stem}.png"
        image.write_bytes(b"not-really-a-png-" + stem.encode())
        (root / "transcripts" / f"{stem}.md").write_text("# TAX INVOICE\n", encoding="utf-8")
        from generators.export import sha256_of

        rows.append(
            {
                "image": f"images/{stem}.png",
                "transcript": f"transcripts/{stem}.md",
                "doc_type": "invoices",
                "sha256": sha256_of(image),
            }
        )
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return root


def _predictions(tmp_path: Path, corpus: Path, systems=("docling",)) -> Path:
    root = tmp_path / "runs"
    for system in systems:
        directory = root / system
        directory.mkdir(parents=True)
        for transcript in (corpus / "transcripts").glob("*.md"):
            shutil.copy(transcript, directory / transcript.name)
    return root


def test_a_matching_corpus_verifies_and_returns_its_rows(tmp_path):
    rows = _verify_corpus(_corpus(tmp_path))
    assert len(rows) == 2


def test_a_tampered_image_is_refused_with_a_four_element_diagnostic(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "images" / "CASE001_invoices.png").write_bytes(b"different bytes")
    with pytest.raises(ScoreInputError) as excinfo:
        _verify_corpus(corpus)
    assert_diagnostic_error(str(excinfo.value), mentions=("CASE001_invoices.png", "sha256"))


def test_a_missing_manifest_is_refused_with_a_four_element_diagnostic(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "manifest.jsonl").unlink()
    with pytest.raises(ScoreInputError) as excinfo:
        _verify_corpus(corpus)
    assert_diagnostic_error(str(excinfo.value), mentions=("manifest.jsonl",))


def test_each_subdirectory_is_one_system(tmp_path):
    corpus = _corpus(tmp_path)
    predictions = _predictions(tmp_path, corpus, systems=("docling", "mineru"))
    paired = _pair_predictions(corpus, predictions)
    assert sorted(paired) == ["docling", "mineru"]
    assert sorted(paired["docling"]) == ["CASE001_invoices", "CASE002_invoices"]


def test_loose_markdown_files_are_an_error_not_an_anonymous_system(tmp_path):
    corpus = _corpus(tmp_path)
    predictions = tmp_path / "runs"
    predictions.mkdir()
    (predictions / "CASE001_invoices.md").write_text("x", encoding="utf-8")
    with pytest.raises(ScoreInputError) as excinfo:
        _pair_predictions(corpus, predictions)
    assert_diagnostic_error(str(excinfo.value), mentions=("subdirectory",))


def test_a_missing_prediction_is_refused(tmp_path):
    corpus = _corpus(tmp_path)
    predictions = _predictions(tmp_path, corpus)
    (predictions / "docling" / "CASE002_invoices.md").unlink()
    with pytest.raises(ScoreInputError) as excinfo:
        _pair_predictions(corpus, predictions)
    assert_diagnostic_error(str(excinfo.value), mentions=("CASE002_invoices", "docling"))


def test_an_extra_prediction_is_refused(tmp_path):
    corpus = _corpus(tmp_path)
    predictions = _predictions(tmp_path, corpus)
    (predictions / "docling" / "CASE999_invoices.md").write_text("x", encoding="utf-8")
    with pytest.raises(ScoreInputError) as excinfo:
        _pair_predictions(corpus, predictions)
    assert_diagnostic_error(str(excinfo.value), mentions=("CASE999_invoices",))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n docparse pytest tests/test_score_inputs.py -v`
Expected: FAIL — `ImportError: cannot import name 'ScoreInputError' from 'generators.pipeline'`

- [ ] **Step 3: Write the implementation**

In `generators/pipeline.py`, add `from generators.export import ExportError, export_corpus, sha256_of` to the existing export import, then add these immediately after `_load_event_records`:

```python
class ScoreInputError(RuntimeError):
    """Raised when the corpus or the predictions cannot be trusted to score."""


def _score_input_err(what: str, *, where: str, expected: str, recover: str) -> ScoreInputError:
    """Build a four-element fail-fast diagnostic."""
    return ScoreInputError(
        "Cannot score.\n"
        f"  What:     {what}\n"
        f"  Where:    {where}\n"
        f"  Expected: {expected}\n"
        f"  Recover:  {recover}"
    )


def _verify_corpus(corpus: Path) -> list[dict]:
    """Read the manifest and check every image against its recorded hash.

    The shipped README calls this "not ceremony". A mismatch means the
    predictions and the ground truth are different vintages, and any number
    computed from them is meaningless -- so scoring refuses rather than
    reporting (scoring spec §6).

    Args:
        corpus: An exported `parsing_YYYYMMDD/` directory.

    Returns:
        The manifest rows, in file order.

    Raises:
        ScoreInputError: The manifest is absent or any image hash differs.
    """
    manifest = corpus / "manifest.jsonl"
    if not manifest.exists():
        raise _score_input_err(
            f"{manifest} does not exist.",
            where=str(manifest.resolve()),
            expected="the manifest.jsonl written by `export`, one JSON row per case, e.g.\n"
            '              {"image": "images/CASE001_invoices.png", ...}',
            recover="pass --corpus pointing at an exported parsing_YYYYMMDD/ directory, "
            "or run `python -m generators.pipeline export` to produce one.",
        )

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        image = corpus / row["image"]
        if not image.exists():
            raise _score_input_err(
                f"{row['image']} is in the manifest but not on disk.",
                where=str(image.resolve()),
                expected="every manifest row to name an image present in the corpus.",
                recover="re-export the corpus so manifest and images agree.",
            )
        if sha256_of(image) != row["sha256"]:
            raise _score_input_err(
                f"{row['image']} does not match its sha256 in the manifest.",
                where=f"{manifest.resolve()} -> {row['image']}",
                expected=f"sha256 {row['sha256']}.",
                recover="score against the corpus these predictions were produced from, or "
                "re-run the predictions against this corpus. Scoring across vintages is "
                "the failure the manifest exists to prevent.",
            )
    return rows


def _pair_predictions(corpus: Path, predictions: Path) -> dict[str, dict[str, Path]]:
    """Pair every transcript with one prediction per system, by filename stem.

    Each immediate subdirectory of `predictions` is one system, named by the
    directory. The rule is stated rather than sniffed, so a directory of loose
    `.md` files is an error rather than an anonymous system (scoring spec §6).

    Args:
        corpus: An exported `parsing_YYYYMMDD/` directory.
        predictions: Directory whose subdirectories are systems.

    Returns:
        System name -> {transcript stem -> prediction path}.

    Raises:
        ScoreInputError: No subdirectories, or any system's files do not
            exactly cover the corpus's transcripts.
    """
    if not predictions.is_dir():
        raise _score_input_err(
            f"{predictions} is not a directory.",
            where=str(predictions.resolve()),
            expected="a directory whose immediate subdirectories are systems, e.g.\n"
            "              runs/docling/CASE001_invoices.md",
            recover="create one directory per system under the --predictions path.",
        )

    systems = sorted(p for p in predictions.iterdir() if p.is_dir())
    if not systems:
        raise _score_input_err(
            f"{predictions} contains no subdirectory.",
            where=str(predictions.resolve()),
            expected="one subdirectory per system, each holding predictions named for the "
            "transcript stems, e.g.\n              runs/docling/CASE001_invoices.md",
            recover="move the prediction files into a subdirectory named for the system "
            "that produced them.",
        )

    expected_stems = {p.stem for p in (corpus / "transcripts").glob("*.md")}
    paired: dict[str, dict[str, Path]] = {}
    for system in systems:
        found = {p.stem: p for p in system.glob("*.md")}
        missing = sorted(expected_stems - set(found))
        if missing:
            raise _score_input_err(
                f"system '{system.name}' has no prediction for {len(missing)} transcript(s): "
                f"{missing[:5]}{' ...' if len(missing) > 5 else ''}.",
                where=str(system.resolve()),
                expected=f"one .md per transcript stem, {len(expected_stems)} in total.",
                recover="re-run inference for the missing cases, or score a corpus subset "
                "by exporting one.",
            )
        extra = sorted(set(found) - expected_stems)
        if extra:
            raise _score_input_err(
                f"system '{system.name}' has {len(extra)} prediction(s) with no transcript: "
                f"{extra[:5]}{' ...' if len(extra) > 5 else ''}.",
                where=str(system.resolve()),
                expected="every prediction to name a transcript stem in this corpus.",
                recover="remove the extra files, or score against the corpus they came from "
                "-- an extra file usually means two corpora have been mixed.",
            )
        paired[system.name] = found
    return paired
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n docparse pytest tests/test_score_inputs.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Run the full gates**

```
conda run -n docparse pytest tests/ -q
conda run -n docparse ruff check --fix --ignore ARG001,ARG002,F841 .
conda run -n docparse ruff format .
conda run -n docparse mypy generators --ignore-missing-imports
```

- [ ] **Step 6: Commit**

```bash
git add generators/pipeline.py
git commit -m "✨ feat: refuse to score an unverified corpus or partial run"
```

---

### Task 5: The `score` command

**Files:**
- Modify: `generators/pipeline.py` (add the `score` command after `export`)
- Modify: `CLAUDE.md` (commands table + repository layout — gitignored, not staged)
- Test: `tests/test_score.py`

**Interfaces:**
- Consumes: `_verify_corpus`, `_pair_predictions` (Task 4); `cer`, `wer` (Task 1); `load_scoring_policy`, `normalise` (Task 2); `hunks`, `group`, `CONVENTION`, `READING` (Task 3).
- Produces: the `score` CLI command and `scores.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_score.py`:

```python
"""The score command end to end, including the self-scoring invariant."""

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from generators.pipeline import app

runner = CliRunner()


def _export(tmp_path: Path) -> Path:
    """Export the real corpus into tmp_path and return the directory."""
    result = runner.invoke(app, ["export", "--target", str(tmp_path), "--date", "20260818"])
    assert result.exit_code == 0, result.output
    return tmp_path / "parsing_20260818"


def _predictions_from(corpus: Path, tmp_path: Path, system: str) -> Path:
    """Copy the transcripts verbatim as one system's predictions."""
    root = tmp_path / "runs"
    directory = root / system
    directory.mkdir(parents=True)
    for transcript in (corpus / "transcripts").glob("*.md"):
        shutil.copy(transcript, directory / transcript.name)
    return root


def test_scoring_the_corpus_against_itself_is_perfect(tmp_path):
    """The invariant: if normalisation, alignment or pairing breaks, this fails."""
    corpus = _export(tmp_path)
    predictions = _predictions_from(corpus, tmp_path, "self")
    report = tmp_path / "scores.json"

    result = runner.invoke(
        app,
        ["score", "--corpus", str(corpus), "--predictions", str(predictions),
         "--report", str(report)],
    )
    assert result.exit_code == 0, result.output

    scores = json.loads(report.read_text(encoding="utf-8"))
    system = scores["systems"]["self"]
    assert system["strict"]["cer"] == 0.0
    assert system["strict"]["wer"] == 0.0
    assert system["normalised"]["cer"] == 0.0
    assert system["normalised"]["wer"] == 0.0
    assert scores["divergences"]["convention"] == []
    assert scores["divergences"]["reading"] == []


def test_an_emphasised_prediction_scores_clean_normalised_and_dirty_strict(tmp_path):
    """The whole point: perfect reading, wrong convention."""
    corpus = _export(tmp_path)
    predictions = _predictions_from(corpus, tmp_path, "emphatic")
    target = predictions / "emphatic" / "CASE001_invoices.md"
    target.write_text(target.read_text(encoding="utf-8").replace("Total", "**Total**"),
                      encoding="utf-8")
    report = tmp_path / "scores.json"

    result = runner.invoke(
        app,
        ["score", "--corpus", str(corpus), "--predictions", str(predictions),
         "--report", str(report)],
    )
    assert result.exit_code == 0, result.output

    scores = json.loads(report.read_text(encoding="utf-8"))
    system = scores["systems"]["emphatic"]
    assert system["normalised"]["cer"] == 0.0, "emphasis must not count as a reading error"
    assert system["strict"]["cer"] > 0.0, "emphasis must count against the strict metric"
    assert scores["divergences"]["convention"], "the difference must be classified as convention"
    assert scores["divergences"]["reading"] == []


def test_a_transposed_digit_is_reported_as_a_reading_error(tmp_path):
    corpus = _export(tmp_path)
    predictions = _predictions_from(corpus, tmp_path, "sloppy")
    target = predictions / "sloppy" / "CASE001_invoices.md"
    target.write_text(target.read_text(encoding="utf-8").replace("$157.39", "$157.89"),
                      encoding="utf-8")
    report = tmp_path / "scores.json"

    result = runner.invoke(
        app,
        ["score", "--corpus", str(corpus), "--predictions", str(predictions),
         "--report", str(report)],
    )
    assert result.exit_code == 0, result.output

    scores = json.loads(report.read_text(encoding="utf-8"))
    assert scores["systems"]["sloppy"]["normalised"]["cer"] > 0.0
    assert scores["divergences"]["reading"], "a wrong amount survives normalisation"


def test_scoring_a_tampered_corpus_exits_non_zero(tmp_path):
    corpus = _export(tmp_path)
    predictions = _predictions_from(corpus, tmp_path, "self")
    next(iter((corpus / "images").glob("*.png"))).write_bytes(b"tampered")

    result = runner.invoke(
        app, ["score", "--corpus", str(corpus), "--predictions", str(predictions)]
    )
    assert result.exit_code == 1
    assert "sha256" in result.output


def test_the_report_records_the_corpus_and_policy_it_used(tmp_path):
    corpus = _export(tmp_path)
    predictions = _predictions_from(corpus, tmp_path, "self")
    report = tmp_path / "scores.json"
    runner.invoke(
        app,
        ["score", "--corpus", str(corpus), "--predictions", str(predictions),
         "--report", str(report)],
    )
    scores = json.loads(report.read_text(encoding="utf-8"))
    assert scores["corpus"].endswith("parsing_20260818")
    assert scores["policy"].endswith("scoring.yml")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n docparse pytest tests/test_score.py -v`
Expected: FAIL — typer reports `No such command 'score'`.

- [ ] **Step 3: Write the implementation**

Add to the imports at the top of `generators/pipeline.py`:

```python
from generators.divergence import CONVENTION, READING, group, hunks
from generators.metrics import cer, error_rate, wer
from generators.scoring import ScoringError, load_scoring_policy, normalise
```

Add `_DEFAULT_SCORING = Path("config/scoring.yml")` beside the other defaults, then add this command after `export`:

```python
@app.command()
def score(
    corpus: Annotated[Path, typer.Option("--corpus", help="An exported parsing_YYYYMMDD/ directory.")],
    predictions: Annotated[
        Path, typer.Option("--predictions", help="Directory whose subdirectories are systems.")
    ],
    policy: Annotated[Path, typer.Option("--policy", help="Path to scoring.yml")] = _DEFAULT_SCORING,
    report: Annotated[
        Path, typer.Option("--report", help="Where to write the JSON report.")
    ] = Path("scores.json"),
) -> None:
    """Score predictions against the corpus and classify every divergence.

    Reports two metrics per system -- normalised, which measures reading, and
    strict, which measures reading plus convention adherence -- and groups the
    divergences behind them into convention mismatches and reading errors, so
    the §8.6 calibration pass can tell which of the two it is looking at.

    Raises:
        typer.Exit: With code 1 when the corpus fails verification, the
            predictions do not cover it, or the policy is invalid.
    """
    try:
        convention = load_scoring_policy(policy)
        _verify_corpus(corpus)
        paired = _pair_predictions(corpus, predictions)
    except (ScoringError, ScoreInputError) as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    transcripts = {p.stem: p for p in (corpus / "transcripts").glob("*.md")}
    systems: dict[str, dict] = {}
    hunks_by_system: dict[str, list] = {}

    for system, files in sorted(paired.items()):
        totals = {
            "strict": {"char_distance": 0, "chars": 0, "word_distance": 0, "words": 0},
            "normalised": {"char_distance": 0, "chars": 0, "word_distance": 0, "words": 0},
        }
        per_document: list[dict] = []
        system_hunks: list = []

        for stem in sorted(files):
            truth = transcripts[stem].read_text(encoding="utf-8")
            prediction = files[stem].read_text(encoding="utf-8")
            pairs = {
                "strict": (truth, prediction),
                "normalised": (normalise(truth, convention), normalise(prediction, convention)),
            }
            row = {"stem": stem}
            for metric, (left, right) in pairs.items():
                char_distance, char_rate = cer(left, right)
                word_distance, word_rate = wer(left, right)
                totals[metric]["char_distance"] += char_distance
                totals[metric]["chars"] += len(left)
                totals[metric]["word_distance"] += word_distance
                totals[metric]["words"] += len(left.split())
                row[f"{metric}_cer"] = char_rate
                row[f"{metric}_wer"] = word_rate
            per_document.append(row)
            system_hunks.extend(hunks(truth, prediction, convention))

        systems[system] = {
            metric: {
                "cer": error_rate(totals[metric]["char_distance"], totals[metric]["chars"]),
                "wer": error_rate(totals[metric]["word_distance"], totals[metric]["words"]),
                "distance": totals[metric]["char_distance"],
            }
            for metric in ("normalised", "strict")
        }
        systems[system]["macro"] = {
            f"{metric}_cer": (
                sum(row[f"{metric}_cer"] for row in per_document) / len(per_document)
                if per_document
                else 0.0
            )
            for metric in ("normalised", "strict")
        }
        systems[system]["documents"] = per_document
        hunks_by_system[system] = system_hunks

    grouped = group(hunks_by_system)
    payload = {
        "corpus": str(corpus),
        "policy": str(policy),
        "systems": systems,
        "divergences": grouped,
    }
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    _print_score_report(systems, grouped, report)


def _print_score_report(systems: dict, grouped: dict, report: Path) -> None:
    """Render the JSON payload as terminal tables.

    Derived from the same structure that was written to disk rather than
    computed separately, so the terminal and the report cannot disagree.

    Args:
        systems: Per-system aggregates and per-document rows.
        grouped: Divergence groups, keyed by class.
        report: Where the JSON was written, echoed for the reader.
    """
    table = Table(title="Scores (micro-averaged; macro in the JSON report)")
    table.add_column("system")
    table.add_column("normalised CER", justify="right")
    table.add_column("normalised WER", justify="right")
    table.add_column("strict CER", justify="right")
    table.add_column("strict WER", justify="right")
    for system, scores in sorted(systems.items()):
        table.add_row(
            system,
            f"{scores['normalised']['cer']:.4f}",
            f"{scores['normalised']['wer']:.4f}",
            f"{scores['strict']['cer']:.4f}",
            f"{scores['strict']['wer']:.4f}",
        )
    rprint(table)

    for kind, heading in ((CONVENTION, "Convention mismatches"), (READING, "Reading errors")):
        entries = grouped[kind]
        if not entries:
            rprint(f"[green]{heading}: none.[/green]")
            continue
        divergences = Table(title=f"{heading} (top 20 of {len(entries)})")
        divergences.add_column("count", justify="right")
        divergences.add_column("transcript")
        divergences.add_column("prediction")
        divergences.add_column("systems")
        for entry in entries[:20]:
            divergences.add_row(
                str(entry["count"]),
                entry["truth"][:60],
                entry["prediction"][:60],
                ", ".join(entry["systems"]),
            )
        rprint(divergences)

    rprint(f"[green]Full report written to {report}.[/green]")
```

Add `from rich.table import Table` to the imports.

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n docparse pytest tests/test_score.py -v`
Expected: PASS, 5 tests. The self-scoring test is the one that matters.

- [ ] **Step 5: Run the full gates**

```
conda run -n docparse pytest tests/ --cov=generators --cov-report=term
conda run -n docparse ruff check --fix --ignore ARG001,ARG002,F841 .
conda run -n docparse ruff format .
conda run -n docparse mypy generators --ignore-missing-imports
```
Expected: all pass, coverage at or above 80%.

- [ ] **Step 6: Run it for real against the shipped corpus**

```bash
mkdir -p /tmp/runs/self && cp parsing_20260818/transcripts/*.md /tmp/runs/self/
conda run -n docparse python -m generators.pipeline score \
    --corpus parsing_20260818 --predictions /tmp/runs \
    --report /tmp/scores.json
```
Expected: every metric 0.0000, both divergence classes empty, and the run finishes in about two minutes for one system.

- [ ] **Step 7: Update CLAUDE.md**

Add a `score` row to the commands table:

```markdown
| `score` | Predictions vs the exported corpus: normalised and strict CER/WER, plus every divergence classified as a convention mismatch or a reading error and grouped corpus-wide. Verifies the manifest first and refuses to score across vintages. | `--corpus`, `--predictions`, `--policy`, `--report` |
```

Add to the repository layout block, under `generators/`:

```
                metrics.py       edit distance, CER, WER (no policy, no I/O)
                scoring.py       scoring.yml loading + normalisation
                divergence.py    hunk extraction, classification, grouping
```

and `scoring.yml` to the `config/` line. `CLAUDE.md` is gitignored, so it is edited but never staged.

- [ ] **Step 8: Commit**

```bash
git add generators/pipeline.py
git commit -m "✨ feat: add the score command"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §2 lives in this repo as a sixth command | 5 |
| §3.1 policy keys | 2 |
| §3.2 structural stripping | 2 (tests pin all four hyphen strings, plus `SQ *COASTAL` and `#R-011FDD`) |
| §3.3 fixed order | 2 (`normalise` applies exactly §3.3's order) |
| §4 metrics, truth as denominator, empty cases, micro + macro | 1 (maths), 5 (aggregation) |
| §5 classification by construction | 3 |
| §5.1 alignment; `difflib` never scores | 3 (`autojunk=False`; metrics come from Task 1) |
| §5.2 exact-match grouping with attribution | 3 |
| §5.3 quantisation caveat | 3 (per-system attribution is what makes it legible; no code owed) |
| §6 CLI shape and three preconditions | 4 |
| §7 four modules | 1, 2, 3, 4/5 |
| §8 JSON + terminal table | 5 |
| §9 test layers | every task; end-to-end in 5 |
| §10 out of scope | nothing built for it |

**Type consistency:** `normalise(text, policy)` is called with that signature in Tasks 3 and 5. `cer`/`wer` return `tuple[int, float]` and are unpacked as `(distance, rate)` in Task 5. `hunks(...)` returns `list[Hunk]`; `group(...)` takes `dict[str, list[Hunk]]` and returns `dict[str, list[dict]]`, consumed as such in Task 5. `_verify_corpus` returns manifest rows, which Task 5 discards — it is called for its verification, not its value.

**Known gap, deliberately not filled:** the spec's key list has no `strip_code_spans`. If Docling or MinerU wrap values in backticks, the fix is a new key in `config/scoring.yml` and an amendment to spec §3.1 — not a silent Python addition, which the repo's "every config key is required" rule forbids.

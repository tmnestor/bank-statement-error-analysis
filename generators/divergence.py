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

from generators.scoring import normalise, space_html_table_tags

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
    if policy["strip_html_table_marks"]:
        truth = space_html_table_tags(truth)
        prediction = space_html_table_tags(prediction)

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
            CONVENTION if normalise(truth_span, policy) == normalise(prediction_span, policy) else READING
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

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
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (item_a != item_b)))
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

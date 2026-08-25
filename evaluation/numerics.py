"""Numeric fidelity: are the amounts right, whatever the formatting?

Character error rate weighs every character equally, so a wrong digit in a
total costs exactly what a typo in a merchant's name costs. On a financial
document those are not the same error, and no metric here separated them.

This one is deliberately **convention-blind**. Amounts are extracted from the
raw text, so a system emitting HTML tables and one emitting pipe tables are
compared on identical terms — which makes it the only measure in the harness
that puts the prompted VLMs and the dedicated parsers on the same footing.
Column integrity needs a table dialect it can parse; this needs nothing.

Measured 2026-08-20 over 165 pages, it inverted the CER ordering outright:
MinerU reproduced **all 2,507** bank-statement amounts exactly while posting
the largest convention penalty of any system, and gemma-4-12B — first on CER,
first on table structure — got at least one amount wrong on 36 of 55
statements. Reading a table, matching a house style, and getting the digits
right are three separable skills.

Nothing here reads a file or consults a policy: the caller supplies two strings.
"""

import re
from collections import Counter

# A quantity, not merely a number. Requiring either a decimal shape or a group
# separator keeps dates, times, reference numbers, ABNs and card masks out of
# the count: they are digits, but a system's handling of a card mask must not
# offset its handling of a total.
#
# One or two decimal places, not two: a system that writes 1234.5 where the page
# prints 1,234.50 has read the figure correctly and formatted it differently,
# and demanding two places would score that as the amount being absent.
# The trailing \b still excludes 345.678, which is a comma misread as a point
# and must stay visible as an error rather than matching 345.67.
_AMOUNT = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+\.\d{1,2}\b")


def amounts(text: str) -> Counter:
    """Every amount in a page, exactly as written.

    A `Counter` rather than a set because a page legitimately repeats a figure
    — a total restated at the foot, a balance carried forward — and a system
    that emits it once has not transcribed the page.

    Args:
        text: A transcript or a prediction, in any Markdown or HTML dialect.

    Returns:
        Amount string -> how many times it appears.
    """
    return Counter(match.group() for match in _AMOUNT.finditer(text))


def canonical(token: str) -> str:
    """Reduce an amount to its value, discarding presentation.

    Grouping separators go and trailing zeros after a decimal point go, so
    `1,234.50` and `1234.5` are one value. The decimal point itself stays:
    `345,678` and `345.678` differ by three orders of magnitude, and two models
    were observed reading one as the other.

    Args:
        token: An amount as written.

    Returns:
        Its canonical value, as a string.
    """
    without_grouping = token.replace(",", "")
    if "." not in without_grouping:
        return without_grouping
    return without_grouping.rstrip("0").rstrip(".")


def _one_edit_apart(left: str, right: str) -> bool:
    """Whether two values differ by a single substitution, insertion or deletion.

    Args:
        left: One canonical value.
        right: The other.

    Returns:
        True if at most one edit separates them.
    """
    if left == right:
        return True
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) == 1
    if abs(len(left) - len(right)) != 1:
        return False
    longer, shorter = (left, right) if len(left) > len(right) else (right, left)
    return any(longer[:i] + longer[i + 1 :] == shorter for i in range(len(longer)))


def numeric_fidelity(truth: str, prediction: str) -> dict:
    """Score one document's amounts against the page's.

    A truth amount with no counterpart is classified by whether the prediction
    holds an unmatched amount one edit away. That separates the two failures a
    single count would merge: **misread**, where the system saw the figure and
    got a digit wrong, and **dropped**, where it never emitted one. A model that
    silently omits a line and one that fabricates a digit fail differently, and
    only the first is recoverable by re-reading the page.

    Args:
        truth: The canonical transcript.
        prediction: The system's output, in any dialect.

    Returns:
        Mapping with `truth_amounts`, `prediction_amounts`, `matched`
        (canonical value), `literal` (matched as written), `misread`,
        `dropped` and `invented`.
    """
    written_truth, written_prediction = amounts(truth), amounts(prediction)

    value_truth: Counter = Counter()
    for token, count in written_truth.items():
        value_truth[canonical(token)] += count
    value_prediction: Counter = Counter()
    for token, count in written_prediction.items():
        value_prediction[canonical(token)] += count

    missing = value_truth - value_prediction
    unmatched = list((value_prediction - value_truth).elements())

    misread = 0
    for wanted in missing.elements():
        for index, got in enumerate(unmatched):
            if _one_edit_apart(wanted, got):
                misread += 1
                unmatched.pop(index)
                break

    return {
        "truth_amounts": sum(written_truth.values()),
        "prediction_amounts": sum(written_prediction.values()),
        "matched": sum((value_truth & value_prediction).values()),
        "literal": sum((written_truth & written_prediction).values()),
        "misread": misread,
        "dropped": sum(missing.values()) - misread,
        "invented": len(unmatched),
    }

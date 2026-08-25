"""Column integrity: did each amount land under the heading it belongs to?

The corpus's pages draw row separators but **no vertical rules**, so column
membership is never delimited on the page. It exists only as horizontal
position relative to the headers, which means a model does not transcribe the
table structure — it *infers* it and then serialises the inference as pipes.

That inference can fail for reasons unrelated to reading. On
`CASE005_bank_statements` InternVL3.5-8B read a header cell that wraps across
two lines, `Date of / Transaction`, as two separate columns; every data row
inherited the extra cell, so every amount sat one column right of the heading
that names it. On a bank statement that is the difference between money out and
money in.

**CER cannot see this.** That page scored a normalised CER of 0.0128 — three
times better than that system's corpus median of 0.0425 — because
normalisation strips pipes as table marks, discarding exactly the delimiters
that encode the answer. The whole structural failure costs a couple of
characters.

So this is scored separately and never folded into CER, which would inherit the
same blindness. It answers one question: of the amounts the page shows, how
many did the system file under the right column?

Amounts are compared by value **and** column index, without aligning rows.
Row alignment is fragile when a prediction adds or drops a row, and it is not
needed: an amount that appears in the prediction at the same column index has
been filed correctly regardless of which row it landed in.
"""

import re
from collections import Counter

# A currency amount as this corpus writes them: optional $, thousands groups,
# exactly two decimals, optional Cr suffix on a balance column.
_AMOUNT = re.compile(r"^\$?\d[\d,]*\.\d{2}(?:\s*Cr)?$")


_HTML_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")


def table_rows(text: str) -> list[list[str]]:
    """Extract table rows in either dialect, dropping separator rows.

    Both the pipe form the corpus uses and the HTML form MinerU emits are
    parsed, because **column placement is a structural question and must not
    depend on dialect**. Which dialect a system writes is a convention matter,
    already measured by the strict/normalised gap; scoring only pipes here would
    report a system that uses HTML as having filed 100% of amounts wrongly,
    which is a statement about the parser, not about the system.

    A pipe row is a line whose stripped form starts with a pipe, which leaves
    prose containing a pipe alone. The separator row is dropped: it carries no
    content, only the column count already visible in the header.

    Empty cells are preserved rather than collapsed: in a transaction table the
    empty cell *is* the debit/credit distinction.

    Args:
        text: A transcript or prediction.

    Returns:
        One list of cell strings per row, in document order.
    """
    html = [
        [_HTML_TAG.sub("", cell).strip() for cell in _HTML_CELL.findall(row)]
        for row in _HTML_ROW.findall(text)
    ]

    rows: list[list[str]] = [row for row in html if row]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(set(cell) <= set("-: ") and "-" in cell for cell in cells if cell):
            continue
        rows.append(cells)
    return rows


def column_width(rows: list[list[str]]) -> int:
    """The table's column count, taken as the most common row width.

    The mode rather than the header's width: a header that the model split or
    merged is exactly the failure being measured, and the body rows are the
    larger sample.

    Args:
        rows: Rows from `table_rows`.

    Returns:
        The modal cell count, or 0 when there are no rows.
    """
    if not rows:
        return 0
    return Counter(len(row) for row in rows).most_common(1)[0][0]


def amount_placements(rows: list[list[str]]) -> Counter:
    """Count each amount by the column index it occupies.

    Keyed by `(value, column_index)` so that the same figure appearing as a
    debit and as a credit is counted as two different placements — which is the
    distinction the whole metric exists to preserve.

    Args:
        rows: Rows from `table_rows`.

    Returns:
        Counter of `(amount, column_index)`.
    """
    placements: Counter = Counter()
    for row in rows:
        for index, cell in enumerate(row):
            if _AMOUNT.match(cell):
                placements[(cell, index)] += 1
    return placements


def attributable_placements(rows: list[list[str]]) -> Counter:
    """Count each amount by column AND by the row key that identifies it.

    `amount_placements` discards the row, so an amount counts as correctly
    placed when it sits under the right heading in *any* row. That is the right
    question for placement and the wrong one for usability: a downstream
    consumer reading rows as records cannot act on a figure it cannot attribute
    to a transaction.

    MinerU is the case that forces the distinction. It emits a group's date as a
    row of its own and the transaction beneath it with an empty date cell, so
    every amount is under the correct heading in a row that identifies nothing.
    On CASE015 that scores 56 of 56 amounts placed and 0 attributable.

    The key is the row's FIRST cell — the date on every layout in this corpus.
    Deliberately not the description as well: a misread merchant name would then
    fail the amount beside it, folding reading accuracy into a placement measure,
    which is the conflation `read` and `placed` were separated to avoid.

    Args:
        rows: Rows from `table_rows`. The header is skipped, since its cells are
            headings rather than amounts.

    Returns:
        Counter of `(amount, column_index, row_key)`.
    """
    placements: Counter = Counter()
    for row in rows[1:]:
        key = row[0].strip() if row else ""
        for index, cell in enumerate(row):
            if _AMOUNT.match(cell):
                placements[(cell, index, key)] += 1
    return placements


def column_integrity(truth: str, prediction: str) -> dict:
    """Score the two independent ways an extracted amount can be wrong.

    They are separate questions and a single rate answers neither:

    1. **Is the number on the page at all?** — `read`, the amounts whose value
       appears somewhere in the prediction's tables, wherever it sits.
    2. **Was it taken from the right place?** — `placed`, of those, how many sit
       under the heading the page puts them under.

    `misfiled` merges the two: it counts an amount as misplaced when it is
    absent from its column for *any* reason, so a misread digit is charged to
    placement. That total is what a downstream consumer feels, so it is kept —
    but it cannot rank a system on either dimension, because a model that reads
    badly and places perfectly scores the same as one that reads perfectly and
    places badly.

    `placement_rate` is `placed / read`, and is the only figure here that
    isolates dimension 2: it is conditional on the value having been read, so a
    system is not credited for placing amounts it never produced, nor charged
    for misplacing ones it never read. A system that reads nothing has no
    placement rate rather than a perfect one.

    Args:
        truth: The canonical transcript.
        prediction: The system's output.

    Returns:
        Mapping with `truth_columns`, `prediction_columns`, `columns_match`,
        `amounts`, `read`, `placed`, `placement_rate` (None where nothing was
        read) and `misfiled`.
    """
    truth_rows = table_rows(truth)
    prediction_rows = table_rows(prediction)

    truth_placements = amount_placements(truth_rows)
    prediction_placements = amount_placements(prediction_rows)

    truth_attributable = attributable_placements(truth_rows)
    prediction_attributable = attributable_placements(prediction_rows)

    # Multiset difference: an amount the truth shows twice in one column must
    # appear twice there to count as filed.
    missing = truth_placements - prediction_placements

    # Dimension 1, position discarded: the same amount in any column counts.
    truth_values: Counter = Counter()
    for (value, _), count in truth_placements.items():
        truth_values[value] += count
    prediction_values: Counter = Counter()
    for (value, _), count in prediction_placements.items():
        prediction_values[value] += count
    read = sum((truth_values & prediction_values).values())

    # Dimension 2, conditional on dimension 1: of what was read, how much sits
    # where the page puts it.
    placed = sum((truth_placements & prediction_placements).values())

    # The figure a consumer reading rows as records can act on: correct value,
    # correct column, and in a row carrying the date that identifies it.
    attributable = sum((truth_attributable & prediction_attributable).values())

    return {
        "truth_columns": column_width(truth_rows),
        "prediction_columns": column_width(prediction_rows),
        "columns_match": column_width(truth_rows) == column_width(prediction_rows),
        "amounts": sum(truth_placements.values()),
        "read": read,
        "placed": placed,
        "placement_rate": (placed / read) if read else None,
        "attributable": attributable,
        "misfiled": sum(missing.values()),
    }

"""Table structure scored in its own right (2026-08-19).

The corpus's original scope excluded table-structure scoring on the grounds
that a table here is judged as text in reading order. The calibration pass
showed that exclusion rested on a false premise: it assumed columns were
delimited on the page. They are not — these pages draw row separators but no
vertical rules, so a model *infers* the structure and serialises the inference.
CER then discards the pipes that carry it.

Two separable questions, reported apart because a single number ranks systems
arbitrarily depending on weighting. Measured on 55 bank statements:

| | gemma-4-12B | MinerU |
|---|---|---|
| row segmentation | 0 fragments | **276 fragments**, 152 width breaks |
| content recall | 60.2% | **64.6%** |

MinerU recovers slightly more cell content and shreds the structure doing it;
gemma preserves the structure. Neither ordering is wrong — they answer different
questions, so the metric reports both rather than averaging them into a ranking
that hides the trade.

Column assignment, the third question, lives in `generators.columns`.

**There is deliberately no cell-accuracy figure here, and there must not be
one.** Rows are aligned by `row_signature`, which is their content — so any
aligned pair has matching cells *by construction*, and an accuracy computed over
aligned rows only can barely fall below 1.0. A misread digit does not lower it;
it removes the row from the alignment instead, showing up in `aligned`. Such a
metric was reported here until 2026-08-20 and read as "MinerU transcribes cells
perfectly" when it meant "the rows MinerU got right, it got right". `aligned`
and `content_recall` answer the question it appeared to.
"""

import difflib
import re
from collections import Counter

from generators.columns import column_width, table_rows

_WHITESPACE = re.compile(r"\s+")


def row_signature(cells: list[str]) -> str:
    """Reduce a row to what identifies it, for alignment.

    Empty cells and whitespace runs are dropped: a pretty-printed table and an
    unpadded one describe the same row, and a cell's *position* is scored by
    `generators.columns`, not here. Case is preserved — reading identifiers with
    correct case is part of the job.

    Args:
        cells: The row's cells.

    Returns:
        A signature string.
    """
    return "".join(_WHITESPACE.sub(" ", cell).strip() for cell in cells if cell.strip())


def _is_fragment(cells: list[str]) -> bool:
    """Whether a row looks like the tail of a wrapped cell rather than a row.

    A continuation carries text in its first cell and nothing anywhere else —
    the second visual line of a description that the page wrapped and the
    system emitted as its own row. That is the dominant structural failure in
    this corpus, and it is distinct from a *merge*, where a system combines two
    truth rows into one and produces no spurious row at all.

    Args:
        cells: The row's cells.

    Returns:
        True when the row is a continuation fragment.
    """
    return bool(cells) and bool(cells[0].strip()) and not any(c.strip() for c in cells[1:])


def table_report(truth: str, prediction: str) -> dict:
    """Score one document's table structure and content separately.

    Rows are aligned by signature with `difflib`, so an inserted fragment or a
    dropped row shifts nothing after it. `autojunk` is off for the same reason
    it is off in `generators.divergence`: these tables are built from repeated
    tokens, and the heuristic would treat exactly them as junk.

    Note what `aligned` therefore means: rows match on *content*, so a misread
    cell drops its row out of the alignment rather than being reported as a
    wrong cell. Cell-level correctness within a row is not separable from row
    matching here, and no figure claiming otherwise belongs in this report —
    see the module docstring.

    Args:
        truth: The canonical transcript.
        prediction: The system's output.

    Returns:
        Mapping with `truth_rows`, `prediction_rows`, `aligned`, `fragments`,
        `width_breaks`, `recalled_cells`, `truth_cells`, and `content_recall`
        (None when there is no table — absent is not the same as zero).
    """
    truth_rows = table_rows(truth)
    prediction_rows = table_rows(prediction)

    width = column_width(prediction_rows)
    width_breaks = sum(1 for row in prediction_rows if len(row) != width)
    fragments = sum(1 for row in prediction_rows if _is_fragment(row))

    truth_signatures = [row_signature(row) for row in truth_rows]
    prediction_signatures = [row_signature(row) for row in prediction_rows]
    matcher = difflib.SequenceMatcher(a=truth_signatures, b=prediction_signatures, autojunk=False)

    aligned = sum(i2 - i1 for tag, i1, i2, _j1, _j2 in matcher.get_opcodes() if tag == "equal")

    # Row-independent: did the values survive at all, wherever they landed?
    # Anything measured over aligned rows is computed on a filtered sample and
    # flatters exactly the system whose rows do not align. This measure has no
    # such blind spot. A multiset, so a value the page shows twice must appear
    # twice.
    truth_values = Counter(
        _WHITESPACE.sub(" ", cell.strip()) for row in truth_rows for cell in row if cell.strip()
    )
    prediction_values = Counter(
        _WHITESPACE.sub(" ", cell.strip()) for row in prediction_rows for cell in row if cell.strip()
    )
    recalled = sum((truth_values & prediction_values).values())
    truth_cell_count = sum(truth_values.values())

    return {
        "truth_rows": len(truth_rows),
        "prediction_rows": len(prediction_rows),
        "aligned": aligned,
        "fragments": fragments,
        "width_breaks": width_breaks,
        "recalled_cells": recalled,
        "truth_cells": truth_cell_count,
        "content_recall": (recalled / truth_cell_count) if truth_cell_count else None,
    }

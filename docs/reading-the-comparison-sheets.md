# Reading the comparison sheets

Each sheet is one bank statement. The leftmost panel is the page image; every
panel to its right is one system's transcription, rendered back into a table so
it can be compared row against row.

There is **no ground-truth panel**. The truth is what the colours are measured
against, not another column to read.

## The four fills

| Fill | Means | What it is evidence of |
|---|---|---|
| **none** | the row matches the truth exactly | — |
| **pale red** | this cell's content differs from the truth | a **reading error** |
| **pale blue** | in the system's table, with no counterpart in the truth's | a **structural** difference |
| **grey** | the truth has this row and the system did not | a **dropped** row |

## Red and blue answer different questions

**Pale red is a misread value.** The cell paired with a truth cell and the
contents disagree. Inside a red cell there are two lines:

- the **top line, in red**, is what the system produced — the mistake
- the **bottom line, in green**, is what the page actually says

So red text is never right and green text is never wrong. If a cell shows only
one line, the system and the page agree on it.

**Pale blue is not an invented value.** It means the system put something in its
table that the truth keeps outside one. Two things produce it:

- **A whole row in blue** — the system emitted a table row that has no truth row
  to pair with. MinerU wraps an entire page into one HTML table, so a rewards
  summary the corpus keeps as paragraphs becomes seven extra table rows. *The
  text was read correctly; only its structure differs.*
- **A single cell in blue** — the row paired, but the system produced a column
  the truth does not have. This is the wrapped `Date of / Transaction` heading
  being read as two columns, which pushes every amount one place sideways.

Calling blue "invented" would accuse a system of fabricating text it in fact
read off the page. Keep the two words apart: **red is wrong, blue is elsewhere.**

### The blue row you will be asked about first

The `Totals at end of period` row at the foot of a statement is blue on some
panels. **The system read it perfectly** — same five columns, same values, in
the same cells as the truth:

```
truth : |  | Totals at end of period | $18,173.17 | $9,045.49 |  |
31B   : |  | Totals at end of period | $18,173.17 | $9,045.49 |  |
```

The corpus records that row as a **separate table**, and these systems kept it
in the transaction table. Nothing on the page marks the boundary — the totals
sit under the Debits and Credits columns, aligned to them — and `prompt.md`
never mentions totals or starting a new table. So the systems were not told, and
arguably have the better reading.

It affects **14 of 55** statements, and the systems split on it:

| system | merges into the transaction table | keeps it separate |
|---|---|---|
| gemma-4-12B | 14 | 0 |
| gemma-4-31B | 10 | 4 |
| InternVL3.5-8B | 0 | 13 (1 absent) |
| MinerU | 0 | 14 |

The 31B disagreeing with **itself** — 10 one way, 4 the other — is the tell that
this is an undefined rule rather than a model behaviour.

**Cost: one row of ROWS ALIGNED per affected page, and nothing at all on AMOUNTS
USABLE.** Column integrity does not care which table a row lives in, which is
why the 31B still reads 100% usable on `CASE002` while showing this row in blue.

Known and deliberately left as is for now; fixing it would change ground truth
rather than any image.

## The caption under each panel

```
41/44 rows exact   2 changed   1 unmatched   0 missed   |  4 columns
```

- **rows exact** — matched the truth completely
- **changed** — paired, but at least one cell differs (the red cells)
- **unmatched** — the system's row had no truth row (the blue rows)
- **missed** — the truth's row had no system row (the grey rows)
- **columns** — the table's width, or a range if the system was inconsistent.
  Anything other than the truth's width is a structural failure on its own.

## An empty panel

A panel with no table and a caption reading

```
refused: ran to the token cap (repetition loop)
```

is a page the system **failed to produce**, not one it was never asked. The
generation ran to the output cap — a repetition loop — and the runner sets such
output aside rather than scoring it, because scoring it would blame the model's
reading for the operator's cap. It counts as 0% on both figures, which is what a
consumer would receive. InternVL does this on `CASE002`.

## The two figures at the foot

**ROWS ALIGNED** is all-or-nothing per row: one wrong character anywhere fails
the whole row. It falls long before anything a consumer would notice, which
makes it a sensitive early warning and a poor headline. A low figure here does
not by itself mean the output is unusable.

**AMOUNTS USABLE** is the one to quote — *right value, right column, right row*.
It counts amounts a downstream consumer can actually act on, and it is computed
by the same code as the scoring report, so a sheet cannot disagree with the
numbers in the deck.

The row condition is the part that matters. An amount can be correct and under
the correct heading, yet sit in a row whose date is blank — so it cannot be
attributed to a transaction. On `CASE015`, MinerU places **56 of 56** amounts
correctly and **0** are attributable. Quoting placement alone would report that
page as a success.

## The one-line version, if asked

> Red is a value the system got wrong. Blue is a value it got right but filed in
> a structure the page does not have. Grey is a value it never produced. The
> figure to judge a system by is *amounts usable* — right value, right column,
> right row.

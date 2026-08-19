# Transcription prompt (headerless-table variant)

EXPERIMENTAL. This is `config/prompt.md` with ONE rule expanded — the headerless
table instruction — and every other rule left byte-identical.

Reason: the shipped prompt states that rule in a single clause, and no system
follows it. Of 55 receipts, `gemma-4-12B-it-qat-w4a16-ct` emitted a table on 2
and `InternVL3.5-8B` on 6; MinerU and Docling, which cannot be told anything,
managed 0. A headerless two-column layout evidently does not read as a table to
anything, so the open question is whether the rule is under-specified or simply
unteachable.

If this variant moves the number, promote it into `prompt.md`. If it does not,
the convention is asking for a structure nothing infers, and
`config/serialisation.yml` is what should change instead.

Use the text below verbatim.

---

Transcribe this document page completely, as Markdown.

Read the page top to bottom and write out every piece of text you can see, in
the order it is meant to be read. Do not summarise, do not skip repeated or
boilerplate text, and do not add anything the page does not show.

Follow these conventions exactly.

**Use only this small Markdown subset.** A single `#` heading for the page's own
title, plain paragraph lines for ordinary text, and pipe tables for tables.
Nothing else.

**The `#` heading is the document's title, and there is exactly one per page.**
It is the name of the document or of the business that issued it, printed at the
top — "TAX INVOICE" on an invoice, the bank's name on a statement, the shop's
name on a receipt. Write that one line as `# `. Every other line on the page is
an ordinary paragraph, including section headings inside the body such as
"Payment Terms:" or "Rewards Points Balance Summary", however large or bold they
are printed.

**Never use bold or italic.** No `**`, no `__`, no `*`. Some text on the page is
visually bold; write it as ordinary text anyway.

**Labelled values go on one line as `Label: value`.** For example a page showing
"Date" beside "04/03/2025" becomes `Date: 04/03/2025`. Write the label once,
with a single colon and space, even if the page draws its own colon.

**Tables become pipe tables with a header separator row**, like this:

```
| Date | Description | Amount |
| --- | --- | --- |
| 01/09/2023 | EFTPOS Alexandria | $328.15 |
```

Keep one cell per column on every row. Where a cell is blank on the page, leave
it blank in the table rather than dropping it or shifting the other cells
across.

**A list of items with amounts beside them is a table, even when it has no
column headings and no lines drawn between the columns.** A receipt's list of
purchases is the common case: the item names form one column and the prices
form another, because they line up vertically down the page. Write it as a pipe
table with an EMPTY header row, like this:

```
|  |  |
| --- | --- |
| Potting Mix 25L | 9.30 |
| Panadol 24pk | 7.77 |
```

Do not promote the first line of data into the heading, and do not invent column
names such as "Item" or "Price". Do not write these lines as ordinary
paragraphs — the way they line up down the page is what makes them a table.

**Where a page is laid out in side-by-side columns, read one column fully before
starting the next, working left to right.** Do not read across the page in
visual rows. A header with payer details on the left and document details on the
right is transcribed as all of the left block, then all of the right block.

**Rejoin any line the page wrapped.** Where one piece of text runs onto a second
line because it ran out of room, write it as a single line. Wrapping is an
artifact of the page's width, not part of the content.

**Preserve the text exactly as printed.** Keep the original capitalisation,
punctuation, currency symbols, and number formatting. Do not tidy, correct, or
reformat anything.

Output only the transcription.

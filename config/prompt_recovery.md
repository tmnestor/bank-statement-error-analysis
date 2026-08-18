# Transcription prompt (repetition-recovery variant)

NOT the benchmark prompt. This is config/prompt.md with ONE added instruction —
the 'write every line exactly once' paragraph below — and every convention rule
left byte-identical, so the convention being measured is unchanged.

Used only to recover pages where a model fell into a repetition loop and ran to
max_output_tokens, producing no usable transcription at all. A page transcribed
with this prompt is not directly comparable to one transcribed with prompt.md;
runs/<system>/_prompt_provenance.json records which pages used it.


This prompt and the transcripts beside it are a matched pair. Change one without
the other and the benchmark silently measures something else. Its conventions
are exactly those declared in `serialisation.yml`, which ships alongside it.

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
across. If the table has no printed column headings, use an empty header row —
do not promote the first line of data into the heading, and do not invent
column names.

**Where a page is laid out in side-by-side columns, read one column fully before
starting the next, working left to right.** Do not read across the page in
visual rows. A header with payer details on the left and document details on the
right is transcribed as all of the left block, then all of the right block.

**Rejoin any line the page wrapped.** Where one piece of text runs onto a second
line because it ran out of room, write it as a single line. Wrapping is an
artifact of the page's width, not part of the content.

**Write every line of the page exactly once.** Do not repeat a line, a table
row, or a block of text you have already written. If you find yourself writing
something you have written before, the page is finished — stop there.

**Preserve the text exactly as printed.** Keep the original capitalisation,
punctuation, currency symbols, and number formatting. Do not tidy, correct, or
reformat anything.

Output only the transcription.

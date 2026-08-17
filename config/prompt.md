# Transcription prompt

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

**Use only this small Markdown subset.** A single `#` heading for a banner
title, plain paragraph lines for ordinary text, and pipe tables for tables.
Nothing else.

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

**Preserve the text exactly as printed.** Keep the original capitalisation,
punctuation, currency symbols, and number formatting. Do not tidy, correct, or
reformat anything.

Output only the transcription.

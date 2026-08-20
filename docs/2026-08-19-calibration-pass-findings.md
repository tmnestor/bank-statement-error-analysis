# Can a prompted VLM be steered to a house style?

**Calibration pass, August 2026.** Four document-parsing systems over 165
synthetic pages of Australian business documents.

---

## The short answer

The benchmark asks a model to read a whole page and write it out as Markdown.
Doing that forces a *house style*: where headings go, how tables are drawn, how
a label joins its value. The worry was that our house style might be
idiosyncratic — that a model could read a page perfectly and still score badly
for formatting it differently.

**It is not, and the reason is that gemma-4-12B can be told.** Given the
convention in its prompt it reaches a median normalised character error rate of
**0.008**, produces the correct table width on **163 of 165 documents**, and
files amounts under the right heading more reliably than either dedicated
parser. Three separate conventions were each stated, measured, and adopted:

| Convention | before | after |
|---|---|---|
| headerless item tables | 2/55 pages | **53/55** |
| dates carried down a group | 61.6% of rows | **97.2%** (truth: 98.9%) |
| repeated glyphs are spacing | 275 stray lines | **0** |

The cumulative effect of stating all three: median normalised CER **0.0201 →
0.0081**, mean **0.3011 → 0.0178**, misfiled amounts **12.7% → 9.7%** — and
three pages the benchmark had written off as impossible now transcribe at 0.05.

Four things qualify it, and they are the substance of this document:

1. **Steerability is a property of the model, not the prompt.** The same three
   conventions, identically worded, moved the 12B and barely moved an 8B from
   another family. A prompt improvement is a property of the *pair*.
2. **A worked example does something a stated rule cannot.** Held to a
   controlled A/B, a rule alone stopped a model emitting the wrong characters
   but not improvising a replacement; the example supplied the behaviour the
   prohibition left undefined.
3. **Normalised character error rate — this benchmark's headline number — is
   blind to the error that matters most on a bank statement.** It compares two
   strings after stripping Markdown syntax, and a table's pipes are Markdown
   syntax, so the delimiters that say which column an amount sits in are
   discarded before anything is compared. Pages with *every* amount filed under
   the wrong heading score 0.08–0.11, twice a normal page and nowhere near the
   1.0 that output deserves.
4. **Ground truth and prompt are a matched pair, and both are components under
   test.** Two defects in this corpus were found only by asking whether it
   treated the same visual device the same way everywhere.

---

## What is being measured

### Full-page transcription

Given one image of a document page, output **all** of its text, in reading
order, as Markdown. Not information extraction, which asks for named fields and
ignores the rest; not OCR, which returns characters without structure; not
layout analysis, which returns boxes. The output is a document a person could
read in place of the page.

The score is a **whole-page edit distance**, so every character counts equally
and there is nowhere to hide: a system cannot do well by getting the total right
and skipping the address.

### The corpus

165 pages — 55 cases across invoices, receipts and bank statements, in 18 layout
variants. Every page is **synthetic and authored**: the ground truth is written
by the renderer at the moment it draws the page, not annotated afterwards.
Nothing is estimated or OCR'd, so label error is zero by construction.

Pages are pristine renders, not scans. The benchmark measures parsing on clean
input.

### Terms used throughout

| Term | Meaning |
|---|---|
| **convention** | a rule in the house style: `# ` for the page title, pipe tables, `Label: value` |
| **prompted system** | a VLM that reads the convention in its prompt |
| **unprompted system** | a dedicated parser with no prompt input |
| **transcript** | the ground-truth Markdown for one page |
| **prediction** | what a system produced for that page |

---

## The metrics

### CER and WER

**Character Error Rate**: edit distance between prediction and truth, divided by
the truth's length. 0.02 means about one character in fifty is wrong. The
denominator is the truth, not the longer of the two, so a system that outputs
nothing scores 1.0 rather than 0.5 — and a system that outputs far too much can
score above 1.0. **Word Error Rate** is the same over whitespace-separated
tokens.

Lower is better throughout.

### Two ways to score the same prediction

**Normalised** applies Unicode normalisation, collapses whitespace, folds dashes
and quotes, and strips Markdown syntax. It measures **reading** — did the model
see what is on the page?

**Strict** compares the text as written. It measures reading **plus** adherence
to the house style.

Case is deliberately not folded: capitalisation is on the page.

### The gap

**gap = strict CER − normalised CER.**

What normalisation removes is, by construction, formatting. So the gap *is* the
cost of the convention: how much a system is penalised for formatting
differently while reading correctly. It is the experiment.

### Column integrity

The pages draw row separators but **no vertical rules**. Column membership is
never delimited on the page; it exists only as horizontal position under a
heading. A model does not transcribe table structure — it **infers** it, then
serialises the inference as pipes.

Normalised CER cannot see that inference fail, because normalisation strips
pipes as table marks and so discards exactly the delimiters that carry the
answer. **Column integrity** counts, for every amount the page shows, whether
the system filed it under the same heading. It is reported separately and never
folded into CER, which would inherit the same blindness.

### Table structure

Column integrity presupposes the rows were segmented correctly. Often they are
not, so rows are scored in their own right. Truth rows are matched to prediction
rows by signature, so a dropped or invented row shifts nothing after it:

- **rows aligned** — how many truth rows found a counterpart;
- **fragments** — continuation rows, one logical row split in two, usually
  because a description wrapped;
- **width breaks** — rows with the wrong number of cells;
- **cell accuracy** — of the cells in *aligned* rows, how many match exactly.
  Restricting to aligned rows is essential: comparing cells across rows that do
  not correspond compares unrelated values;
- **content recall** — of every truth cell, how many appear anywhere in the
  prediction. Catches the opposite failure, a system that aligns few rows but
  did read the text.

Always read per document type. Bank statements are the only genuinely difficult
tables here; averaging them with near-saturated invoices hides everything.

---

## The systems

Split by one property: **whether they can be told the convention.**

| Group | Systems | Reads a prompt? | Measures |
|---|---|---|---|
| **Prompted** | gemma-4-12B-it-qat-w4a16-ct, InternVL3.5-8B | yes | is the convention *communicable*? |
| **Unprompted** | MinerU, Docling | no | is it *idiomatic Markdown at all*? |

You cannot ask MinerU to emit pipe tables. Its divergences are therefore
uncontaminated evidence about whether the house style matches what the Markdown
world does by default. A prompted model *is* told; if it still diverges, the
instruction failed.

The gemma checkpoint is 4-bit quantisation-aware-trained, so character-level
slips there carry a quantisation signal rather than a convention one.

---

## Results

Per-page **medians**. A handful of catastrophic pages distort the means badly,
and quoting means alone misdescribes every system here.

| System | median nCER | median sCER | **gap** | mean nCER | told? |
|---|---|---|---|---|---|
| **gemma-4-12B-it-qat-w4a16-ct** | **0.0081** | 0.0382 | **+0.0136** | **0.0178** | yes |
| mineru | 0.0404 | 0.4277 | **+0.3917** | 0.0520 | no |
| InternVL3.5-8B | 0.0420 | 0.2119 | **+0.1630** | 0.0743 | yes |
| docling | 0.4145 | 0.6120 | +0.0581 | 1.6457 | no |

**Gemma reads best and formats closest**, by a margin of five times on the
median. Its mean of 0.018 against a median of 0.008 is the tightest distribution
of any system here: no catastrophic pages remain.

**MinerU and InternVL read comparably** — 0.040 and 0.042 — and are separated
almost entirely by convention cost.

**MinerU pays 0.39 for formatting it cannot be instructed about**, principally
HTML tables on 110 of 165 pages where the corpus uses pipe tables. It reads well
and formats differently, which is exactly what an unpromptable parser should
look like.

**InternVL sits between**, and about half its gap is table *padding* — it
pretty-prints `|-----|-----|` against the corpus's `| --- |`. Collapsing
whitespace alone, changing nothing else, takes its median strict CER from 0.2116
to **0.1149**.

Whether padding should count is a fair question, and it should. These pages draw
no vertical rules, so in a space-aligned block the whitespace **is** the
structure — and `normalised` already discards it, making `strict` the only place
spacing is measured at all.

**Docling is a different category.** A median of 0.41 against a mean of 1.65 is
repetition loops, not convention. Its gap says nothing about the house style.

---

## Gemma-4 extracts tables most accurately

Across all 165 pages:

| System | rows aligned | fragments | width breaks | cell accuracy | content recall |
|---|---|---|---|---|---|
| **gemma-4-12B** | **1359/2005 (67.8%)** | 9 | 15 | **0.995** | 0.888 |
| InternVL3.5-8B | 1343/2005 (67.0%) | **0** | 8 | 0.957 | 0.867 |
| mineru | 1234/2005 (61.5%) | 292 | 156 | 1.000 | 0.888 |
| docling | 283/2005 (14.1%) | 120 | 13 | 0.850 | 0.273 |

Amounts filed under the correct heading:

| System | misfiled | **rate** | wrong column count | **correct width** |
|---|---|---|---|---|
| **gemma-4-12B** | 242/2503 | **9.7%** | **2** | **163/165** |
| mineru | 325/2503 | 13.0% | 56 | 109/165 |
| InternVL3.5-8B | 330/2503 | 13.2% | 13 | 152/165 |
| docling | 1929/2503 | 77.1% | 117 | 48/165 |

**Column-count stability is where the margin is.** Gemma produces a table of the
correct width on 163 of 165 documents, against MinerU's 109 and Docling's 48. A
system that invents or drops a column shifts every amount on the page, so this
single property dominates everything downstream of it.

By document type — and read it this way, because bank statements are the only
genuinely hard tables here:

| misfiled | bank statements | invoices | receipts |
|---|---|---|---|
| **gemma-4-12B** | 11.7% | 2.3% | **0.0%** |
| InternVL3.5-8B | 14.7% | 1.0% | 16.8% |
| mineru | **7.0%** | **0.0%** | 100% |
| docling | 83.1% | 24.0% | 100% |

Gemma misfiles **not one** of the 184 receipt amounts, and aligns 184 of 184
receipt rows. It is the only system producing a correct table on all three
document types: the parsers' 100% on receipts is not misplacement but absence —
neither emits a table there at all.

**MinerU still places bank-statement amounts better**, 7.0% against gemma's
11.7%. That is the one measurement where the CER leaderboard and the structural
metrics disagree, and it has survived every prompt revision.

### Bank-transaction tables

The hard case, and worth separating. 55 statements, 1,612 rows, 2,011 amounts.

| System | rows aligned | fragments | width breaks | cell acc | misfiled | **width ok** |
|---|---|---|---|---|---|---|
| mineru | **1025/1612** | 292 | 156 | 1.000 | **7.0%** | 54/55 |
| InternVL3.5-8B | 987/1612 | **0** | 8 | 0.947 | 14.7% | 51/55 |
| **gemma-4-12B** | 971/1612 | **0** | **0** | 0.994 | 11.7% | **55/55** |
| docling | 134/1612 | 120 | 13 | 0.704 | 83.1% | 6/55 |

**MinerU aligns the most rows and pays for them.** It recovers more content than
anything else — content recall 0.928 — but does it by splitting 292 rows in two
and emitting 156 rows of the wrong width. Gemma aligns 54 fewer rows and breaks
structure **zero times in 1,612**. Those are different things to want: MinerU
recovers more, gemma's output is trustworthy row by row.

**Gemma gets the column count right on all 55 statements.** InternVL's 4 broken
documents carry **149 of its 296 misfiled amounts** — half its total error on
7% of its pages, which is what a column-count failure does.

Row alignment tops out near 60–65% for every real reader, and the reason is not
segmentation. It is that a single misread character in a row poisons that row's
signature. Two pages illustrate both mechanisms:

- one statement lost 27 rows to a date read as `01/03/2024` instead of
  `01/01/2024`. Because the convention carries a group's date onto every row
  beneath it, one wrong digit fails every row of that group.
- another lost 24 rows to a balance read as `$15,571.32` instead of
  `$15,971.32` — and then **every subsequent balance was recomputed from the
  wrong figure**, `$16,073.50` becoming `$15,469.14`. The model is not reading
  that column; it is doing arithmetic down it.

That second one is worth dwelling on. A model reconstructing a running balance
rather than transcribing it will look correct on any single row, and will look
correct to an information-extraction task that pulls one balance. It is visible
here only because the whole column is scored against the whole page.

---

## Findings

### 1. Steerability is a property of the model, not the prompt

Three conventions, each stated in the prompt and each measured on both prompted
systems:

| Convention | gemma-4-12B | InternVL3.5-8B |
|---|---|---|
| headerless item tables (receipt rows segmented) | 4.9% → **100%** | 9.2% → **81.5%** |
| dates carried down a group | 61.6% → **97.2%** | 57.1% → 57.4% |
| repeated glyphs omitted (stray lines) | 275 → **0** | 69 → 6 |
| worked example added on top | rescued a failed page | **caused** a new failure |

Compounded over the full corpus, the three conventions take gemma's median CER
from 0.0201 to **0.0081** and its mean from 0.3011 to **0.0178** — and its
receipt tables from 9 of 184 rows correctly segmented to **184 of 184**, with
not one of 184 amounts misfiled. Over the same revisions InternVL's median moved
0.0432 → 0.0420.

The 12B adopts every one. The 8B adopts the structural rule about tables,
ignores the procedural rule about dates entirely — three phrasings, a 0.9-point
spread, no trend — and is *destabilised* by the intervention that helped the
larger model most.

The date rule is the cleanest case. It was phrased three ways across three runs,
each more explicit than the last, and InternVL sat at 57.1%, 56.5%, 57.4%
against a corpus that is 98.9% consistent. It also kept emitting the date band as
a table row of its own, 17 then 25 then 17 times, never approaching zero.

**A flat response across several genuine attempts is a capability signal, not an
invitation to write a fourth version.** Distinguishing "badly worded" from
"cannot do this" requires more than one model, which is why the pass was
specified with two families.

This also qualifies the headline. The convention costs gemma +0.008 and InternVL
+0.155 — still far better than MinerU's +0.392, but eighteen times gemma's. How
well a house style lands is a property of the model as much as of the style.

### 2. What a worked example does that a stated rule cannot

Every prompt revision here changed wording and added an example at once, so none
could say which half worked. One was run as a controlled A/B instead: two prompt
files **identical except for a single worked example**, over 61 pages.

The rule concerned runs of repeated punctuation — a line of dashes drawn as a
separator, a trail of dots padding a reference out to a fixed width. The corpus
treats both as typography and omits them.

| gemma-4-12B | rule only | **rule + example** |
|---|---|---|
| stray separator lines | **0** | **0** |
| median receipt CER | 0.0044 | 0.0043 |
| **mean** CER | 0.1108 | **0.0214** |
| amounts misfiled | 7.6% | **0.3%** |

On medians the two are indistinguishable. On the mean they differ five times
over, and **the entire difference is one page** — a statement whose truth is
1,507 characters:

| | output | CER |
|---|---|---|
| rule only | 7,334 characters | **5.52** |
| rule + example | 1,491 characters | **0.076** |

The rule-only output ends like this:

```
000000000000000000000000000000000000000000000000000000000000
```

**Told not to write repeated dots, the model wrote repeated zeros.** It obeyed
the letter of the instruction and improvised a substitute — and because that
generation finished under the token limit, no completeness check flagged it.

A prohibition says what not to emit and leaves the replacement undefined. An
example shows what to write *instead*. The shipped rule now carries both, plus
an explicit "do not replace an omitted run with anything else".

So the useful claim is not that examples help. It is that **an example
disambiguates the replacement behaviour a prohibition leaves open**, which
matters precisely where a model would otherwise invent one.

The example is not free. It perturbs generation everywhere — 52 of 61 pages
differ between the two arms — and mildly degrades about five pages the rule
alone had nearly perfect. For the 8B it is a net loss, and it produced a fresh
runaway on a page both other configurations handled.

### 3. Stating a rule is not the same as communicating it

The prompt had always said:

> If the table has no printed column headings, use an empty header row — do not
> promote the first line of data into the heading, and do not invent column
> names.

Receipts are exactly that: an item list with prices, no headings. Compliance was
**2 of 55** for gemma and 6 of 55 for InternVL. Near zero, because the rule was a
clause inside a paragraph about tables, and a receipt's item list does not
register as "a table" in the first place.

Rewriting it to name the case — *"A list of items with amounts beside them is a
table, even when it has no column headings and no lines drawn between the
columns"*, with a worked example — moved gemma to **53 of 55**. Same rule, same
model, same pages; only the wording.

**A rule can be technically present and effectively absent.** The prompt is a
component to be tested, not documentation to be written once.

### 4. Normalised CER is blind to the error that matters most on a statement

Four statements print a header cell that wraps across two lines — `Date of /
Transaction`. InternVL reads it as **two columns**. Every data row inherits the
extra cell, so every amount sits one column right of the heading naming it. On a
bank statement that is money out reported as money in.

| Page | amounts misfiled | normalised CER |
|---|---|---|
| CASE033 | **38 of 38** | 0.0829 |
| CASE005 | 36 of 39 | 0.0902 |
| CASE012 | **39 of 39** | 0.0915 |
| CASE025 | **38 of 38** | 0.1070 |

Three of those pages have **every single amount** under the wrong heading and
pay 8–11% character error for it — for output that is entirely worthless.
Normalisation strips pipes as table marks, discarding exactly the delimiters
that encode the answer, so a total structural failure registers as a mild
reading day.

This failure is also **immune to prompting**: it is unchanged across four
successive prompt revisions, at 0.3034, 0.3028 and 0.3045 on the affected
layout. Not every problem is an instruction problem.

### 5. Ground truth is a component under test

Two defects were found in this corpus, both the same fault: **the same visual
device treated two ways, decided by which piece of code drew it.**

**Repeated glyphs.** A dashed separator rule was excluded as decoration; a dot
leader inside a table cell was captured as content. Both halves cost accuracy —
three of four systems transcribed the separators anyway, and the leaders the
corpus *did* record were skipped by one model and caused another to fail
outright. Resolved: a run of four or more repeated punctuation glyphs is
decoration wherever it is drawn. Three or fewer is punctuation, so an ellipsis
and a decimal point survive.

**Grouped dates.** A statement prints a date once and leaves it blank on the
rows beneath, in two forms: on a band of its own, or in the first transaction's
own date cell. The corpus filled the date in for the first form only, so 123
rows across 7 statements kept a blank date while 329 rows on 27 others had
theirs supplied. A model following the single stated rule was correct on one
layout family and wrong on the other.

The second one produced a lesson worth generalising. Adding the date rule
*appeared* to cost gemma 5.5 points of table accuracy. Against a corrected
corpus the same predictions show a 5.2-point **gain**:

| | inconsistent corpus | corrected corpus |
|---|---|---|
| without the rule | 64.3% rows aligned | 58.6% |
| with the rule | 58.8% | **63.8%** |

**A metric regression is evidence about the pair (system, ground truth), never
about the system alone.** And internal consistency is a testable property that
no whole-page CER will surface, because both behaviours produce plausible
transcripts. Only asking "does the corpus treat this device the same way
everywhere?" finds it.

Neither fix changed a single image. Because ground truth is written at render
time and serialisation is a separate step, a convention can change and every
transcript re-emits in seconds — so every existing prediction stayed valid and
both hypotheses were tested by re-scoring rather than re-running.

### 6. A prompt is part of the eval set

The prompt ships **with** the corpus and is read by the systems being scored, so
a real cell value in a worked example hands a model part of an answer it was
supposed to read off the page. A model could reproduce those lines without
seeing them, and nothing in the results would show it.

Every example in every prompt file is now invented content, verified absent from
all 165 transcripts, with a test that checks each example value against every
transcript.

Three details worth carrying to any similar benchmark:

- **Header words are fine, values are not.** "Date" and "Description" appear on
  real pages by necessity; they are vocabulary, not answers. The guard exempts
  the header row and checks data cells.
- **Check every prompt file, not the shipped one.** An experimental variant is
  run against the same scored pages and leaks just as effectively. Widening the
  guard to all of them immediately found a stale example carrying a real date and
  a real amount.
- **Verify the guard fails on a deliberate leak.** A vacuously-passing guard
  manufactures confidence and is worse than none.

### 7. Character-level errors are the remaining real cost

The evaluated gemma checkpoint is 4-bit QAT. Theory says quantisation costs
character fidelity first, and it does. On one field across seven pages, gemma got
5 of 7 wrong: three reading a comma as a period, three substituting a digit
(`487,205` → `497,205`). InternVL, with a corrected tokenizer, got every digit
right but made the same comma-for-period substitution on 3 of 7.

Two independent models misreading the same comma glyph is worth noting. The
pages were checked at magnification and render unambiguous commas. `345,678`
versus `345.678` differ by three orders of magnitude.

This was measurable only because those figures are authored per page. Where a
layout hardcodes one number repeated across seven pages, one error repeats seven
times and a model can score well by memorising it.

---

## Limits

**Coverage.** Gemma produced all 165 pages. InternVL produced 164; the one
failure is scored as a total failure rather than dropped, so all systems are
averaged over the same 165 transcripts. Docling produced 2 empty pages, scored
the same way.

**The two prompted systems were run under the prompt shipped with this corpus**,
verified by hashing the text actually sent. **The two parsers read no prompt**,
so their figures are unaffected by prompt revisions and are directly comparable
throughout.

**Row alignment is fragile to single-character errors.** A row is matched by its
content, so one misread digit in a cell fails that row — and where the
convention carries a date onto every row of a group, one misread date fails the
whole group. Two statements lose 27 and 24 rows this way. Alignment percentages
are therefore a joint measure of segmentation *and* character fidelity, not of
segmentation alone.

**The column metric conflates two things.** `misfiled` counts an amount as
misplaced if it is absent from its column for *any* reason, including a misread
digit. It is not purely structural.

**Synthetic and pristine.** Clean renders, not photographs or scans. No claim is
made about degraded documents.

**Character errors are not separated by cause.** A misread digit could be
quantisation, resolution, or the model. Comparing quantised and unquantised
checkpoints would isolate it; that has not been done.

**One convention mismatch is known and unresolved.** On one layout the corpus
records a block's visual alignment (`Opening Balance          345,678`) while a
model writes `Opening Balance: 345,678`, obeying the prompt's instruction to put
labelled values on one line. The block is drawn as plain lines rather than
label/value pairs, so the transcript keeps the spacing. A model reads correctly,
follows the shipped rule, and still diverges.

**Two correct instructions can be jointly wrong.** Removing the separator rules
also removed the visual fence around a receipt's totals block, which then looks
exactly like the headerless item list another rule instructs the model to render
as a table. Spurious second tables on gemma's receipts went from 5 of 55 to
**18 of 55**, and they are the sole source of its 9 fragments and 15 width
breaks — every one of which is on a receipt, none on a statement.

---

## What this pass produced

**A house style that is demonstrably communicable.** Told the conventions, a
capable prompted model adopts them for **+0.0136** — the gap between its strict
and normalised CER, which is by construction what formatting costs it. An
unpromptable parser of comparable reading quality pays +0.3917 for the same
corpus. The style is not an arbitrary imposition.

**A steerability result with a sharp edge.** The 12B adopts every convention put
to it; an 8B from another family adopts one of three and is destabilised by the
technique that helps the 12B most. Prompt engineering results do not transfer
across model scales, and reporting one without the other overstates both.

**Two metrics that normalised CER cannot replace.** Column integrity, because a
page can be character-perfect and financially meaningless. Table structure,
because column integrity presupposes the rows. Together they produced the pass's most useful
result — gemma-4-12B gets the column count right on **163 of 165** documents and
on **all 55 bank statements**, breaking table structure zero times in 1,612
statement rows — and its most counter-intuitive one: MinerU, which pays the
largest convention penalty of any system, still places bank-statement amounts
more accurately than anything else. Reading a table and matching a house style
are close to independent skills.

**One observation that only a whole-page metric could surface.** A model that
misreads a running balance then **recomputes every balance below it** from the
wrong figure. It is arithmetically reconstructing the column rather than
transcribing it — invisible on any single row, invisible to an extraction task
pulling one balance, and unmistakable when the whole column is scored at once.

**Four scoring corrections**, each closing a case where formatting was scored as
misreading: HTML tables, HTML entities, trailing label colons, and repeated-glyph
decoration. Together they took MinerU's median normalised CER from **0.4740 to
0.0393** — a twelvefold correction on a system whose reading never changed, and
a measure of how badly a formatting-blind metric can misrepresent a competent
reader.

**Two corpus defects fixed and one prompt contradiction removed**, all found by
asking whether the benchmark said the same thing about the same device
everywhere.

**A working practice.** Ground truth is written at render time; serialisation and
scoring are separate steps. A convention can change and every transcript
re-emits in seconds without re-rendering an image — which is what made it cheap
enough to discover that a regression was the benchmark's fault rather than the
model's.

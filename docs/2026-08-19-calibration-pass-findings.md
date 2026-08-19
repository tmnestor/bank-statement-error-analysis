# Is the transcription convention fair to models?

**Calibration pass, 19 August 2026.** Four document-parsing systems over 165
synthetic pages. Written for readers who know machine learning but not this
project's vocabulary — every term is defined at first use.

---

## The short answer

We built a benchmark that asks a model to read a whole page of a business
document and write it out as Markdown. Doing that forces the benchmark to pick a
*house style*: where headings go, how tables are drawn, how a label joins its
value. The worry was that our house style might be idiosyncratic — that a model
could read a page perfectly and still score badly for formatting it differently.

The house style is largely fine. But the pass found seven things that matter more
than that verdict, and the last four came from measuring the prompt itself as
if it were a component under test:

1. **Stating a rule is not the same as communicating it.** One rule was written
   in the prompt and achieved **2/55** compliance. Rewritten to name the case it
   applies to — same rule, same model, same pages — it reached **53/55**.
2. **The corpus was contradicting itself**, treating the same visual device as
   content in one place and decoration in another, depending on which piece of
   code drew it. That cost one model insertions equal to **63%** of a page's
   length, on every receipt. A second, independent instance of the same fault
   was found later and is finding 7.
3. **The primary metric is blind to the error that matters most.** A page where
   every row was filed under the wrong column heading scored *better* than that
   system's average.
4. **A rule can be stated, communicated, obeyed — and still make things worse.**
   A new instruction appeared to cost 5.5 points of table accuracy. It was
   actually correct, and the corpus was wrong on 7 of 55 pages. Fixing the
   corpus flipped the sign to a 6.6-point *gain*. Nothing distinguishes those
   two situations except going and looking.
5. **Some instructions never land, however they are worded.** The same rule was
   phrased three ways across three runs. One model went from 62% to 97%
   compliance; the other stayed at 57%, 56%, 57%. Rewording is not always the
   remedy — sometimes the capability is simply absent.
6. **A worked example does something a stated rule cannot.** Held to a
   controlled A/B — two prompts differing by one example and nothing else — the
   rule alone stopped a model emitting the offending characters but not
   *running away* on them: it substituted a run of zeros. The example supplied
   the replacement behaviour a prohibition leaves undefined.
7. **The smaller model is steered less reliably and destabilised more easily.**
   Every intervention in this pass lands cleanly on the 12B and weakly, or
   backwards, on the 8B — the same example that rescued a page for one gave the
   other a fresh runaway. A prompt improvement is not a property of the prompt;
   it is a property of the pair.

One finding was **overturned**. A page-level failure recorded here as a real
document feature defeating a capable model turned out to be the prompt
instructing the model to do the opposite of what the ground truth records.

And one process finding, learned the hard way: **a prompt is part of the
evaluation set**, and putting real data in its examples silently leaks answers.

---

## What is being measured

### Full-page transcription

Given one image of a document page, output **all** of its text, in reading
order, as Markdown. Nothing summarised, skipped, or added.

This is deliberately not several adjacent tasks it gets confused with:

| Not this | Why it differs |
|---|---|
| **Information extraction (IE)** | IE asks for named fields — invoice total, ABN, due date — and returns a short bounded record. Transcription asks for the whole page. |
| **OCR** | OCR recovers characters. Transcription also decides structure: what is a heading, what is a table, what order a two-column layout is read in. |
| **Layout analysis** | Returns boxes and region labels. We score text, not geometry. |
| **Table structure recognition** | Scored elsewhere, with metrics like TEDS. Here a table is judged as text in reading order — a decision this pass calls into question. |

The IE distinction is not academic. A model can be excellent at IE on a page and
still fail to transcribe it, because transcription requires a long unbounded
generation and IE does not. We hit exactly that (see *The dot-leader runaway*).

### The corpus

165 page images: **55 cases × 3 document types** (invoices, receipts, bank
statements) across **18 layout variants**. Australian business documents,
entirely synthetic.

**Ground truth is authored, not annotated.** Nobody labelled these pages
afterwards. The renderer emits the correct answer *as it draws*: each drawing
primitive records the text it is about to put on the canvas. Image and expected
transcription come from the same pass, so they cannot disagree.

A runtime check enforces it. Every call that puts text on the canvas is tagged
with the identifier of the record that authorised it, and at page end the
renderer asserts no untagged text was drawn.

### Terms used throughout

**Transcript** — the correct answer for one page. 165 of them.

**Convention** (or **serialisation policy**) — the house style transcripts are
written in, declared in configuration: `#` for the page title, pipe tables with
a header separator row, `Label: value` for labelled values, never bold or
italic, side-by-side columns read one column at a time.

**Prompt** — the instructions shipped to systems that can accept instructions.
It states the convention in prose. Prompt and transcripts are a matched pair.

**Prediction** — what a system actually produced for a page.

---

## The metrics

### CER and WER

**Character Error Rate** is the edit distance between prediction and truth
divided by the truth's length; **Word Error Rate** the same over
whitespace-separated words. Both are *error* rates: **lower is better**, 0 is
perfect.

Two properties surprise people:

- `0.05` means one character in twenty is wrong.
- **Values above 1.0 are common here.** CER exceeds 1 when the prediction is
  *longer* than the truth, because surplus characters count as insertions. A
  model emitting 34,000 characters against 557 of truth scores about 62. The
  metric is correctly reporting that the output is mostly invented.

### Two ways to score the same prediction

**Strict** compares raw text — reading *and* formatting compliance together.

**Normalised** first passes both sides through a declared policy (Unicode
normalisation, whitespace collapsed, Markdown syntax removed positionally, HTML
table tags stripped, entities unescaped, label colons folded) and then compares.
Both sides get the same treatment, so purely *formatting* differences vanish and
only genuine misreadings remain.

Normalisation lives entirely in the scoring tool. The corpus emits one canonical
form and never normalises, so scoring policy can change without regenerating an
image — which is what made most of this pass cheap.

### The gap

> **gap = strict CER − normalised CER**

Everything normalisation removed is, by construction, formatting rather than
reading. So the gap is **the cost of the house style** to that system.

### Convention mismatch vs reading error

Every individual difference is classified the same way: if the two spans become
identical after normalisation it is a **convention mismatch**; if the difference
survives, it is a **reading error**. There is no pattern list and nothing to
tune — the classifier *is* the normalisation policy, applied per difference.

### Column integrity — added during this pass

The pages draw row separators but **no vertical rules**. Column membership is
never delimited; it exists only as horizontal position relative to the headers.
So a model does not transcribe table structure — it **infers** it, then
serialises the inference as pipes.

CER cannot see that inference fail. This metric counts, for each amount the page
shows, whether the system filed it in the same column. It is reported separately
and never folded into CER, which would inherit the same blindness.

### Table structure — also added during this pass

Column integrity assumes the rows were segmented correctly to begin with. Often
they are not, so rows are scored in their own right. Truth rows are matched to
prediction rows by signature — a dropped or invented row therefore shifts
nothing after it — and five numbers are reported:

- **rows aligned** — how many truth rows found a counterpart at all;
- **fragments** — continuation rows, where one logical row was split in two
  (usually a description that wrapped on the page);
- **width breaks** — rows with the wrong number of cells;
- **cell accuracy** — of the cells in *aligned* rows, how many match exactly.
  Restricting to aligned rows is essential: comparing cells across rows that do
  not correspond would compare unrelated values and produce a meaningless
  number;
- **content recall** — of every cell in the truth, how many appear anywhere in
  the prediction. This catches the opposite failure, a system that aligns few
  rows but did read the content.

Reported per document type, never only in aggregate. Bank statements are the
only genuinely difficult tables in this corpus, and averaging them with
near-saturated invoices hides everything interesting.

---

## Experimental design

Systems are split by one property: **whether they can be told the convention.**

| Group | Systems | Reads the prompt? | Measures |
|---|---|---|---|
| **Prompted** | gemma-4-12B-it-qat-w4a16-ct, InternVL3.5-8B | yes | Is the convention *communicable*? |
| **Unprompted** | MinerU, Docling | no | Is it *idiomatic Markdown at all*? |

A dedicated parser has no prompt input — you cannot ask MinerU to emit pipe
tables. So its divergences are uncontaminated evidence about whether the house
style matches what the Markdown world does by default. A prompted model *is*
told; if it still diverges, the instruction failed.

---

## Results

Final figures, against corpus vintage `parsing_20260819d` and the corrected
prompt. Per-page **medians**, which describe these systems better than means — a
few catastrophic pages distort the averages badly.

| System | median nCER | median sCER | **median gap** | mean nCER | told? |
|---|---|---|---|---|---|
| **gemma-4-12B-it-qat-w4a16-ct** | **0.0201** | 0.0476 | **+0.0084** | 0.3011 | yes |
| mineru | 0.0404 | 0.4277 | **+0.3917** | 0.0520 | no |
| InternVL3.5-8B | 0.0432 | 0.2299 | **+0.1552** | 0.0970 | yes |
| docling | 0.4145 | 0.6120 | +0.0581 | 1.6457 | no |

**Reading quality is close between the three real readers** — 0.020 to 0.043
median CER. Gemma reads best *and* formats closest.

Gemma's mean of 0.30 against a median of 0.020 is worth pausing on, because it
is not a reading failure: it is the 55 receipts, where the model transcribes the
page's horizontal rules and the corpus excludes them as decoration. Delete those
rule lines from its output and its receipt CER falls from **0.8164 to 0.0083**.
Almost the whole mean is one convention disagreement, quantified in finding 2.

**MinerU loses 0.39 to formatting it cannot be instructed about**, principally
emitting HTML tables (`<table><tr><td>`) on 110 of 165 pages where the corpus
uses pipe tables. It reads well and formats differently, which is exactly what
an unpromptable parser should look like.

**Gemma loses almost nothing (+0.009)** — told the style, it complies.

**InternVL sits between (+0.155), and about two-thirds of that is table
padding.** It pretty-prints its tables:

```
| Description             | Qty | Unit Price |
|-------------------------|-----|------------|
```

against the corpus's unpadded form. Collapsing whitespace alone drops its strict
CER from 0.349 to 0.121. It followed the structural rule — pipe table, header
separator row — and differs on spacing.

Whether that should count is a real question. **It should**, and the padding is
not excused, because in these documents alignment carries meaning: the pages
have no vertical rules, so in any space-aligned block the whitespace *is* the
structure, and `normalised` already discards it. `strict` is the only place
spacing is measured at all.

**Docling is in a different category.** A median of 0.41 with a mean of 1.35
means a few pages emit far more text than exists — repetition loops on thermal
receipts, one at 62× the truth length — plus pages producing nothing. Its gap
should not be read as evidence about the house style.

### Column integrity

| System | amounts | misfiled | **misfiled %** | docs with wrong column count |
|---|---|---|---|---|
| gemma-4-12B-it-qat-w4a16-ct | 2503 | 319 | **12.7%** | 4 |
| mineru | 2503 | 325 | 13.0% | 56 |
| InternVL3.5-8B | 2503 | 361 | 14.4% | 14 |
| docling | 2503 | 1929 | 77.1% | 117 |

Read this **by document type**, because the aggregate misleads badly. Bank
statements are the only genuinely hard table here; invoices are near-saturated
and receipts are a two-column list.

| misfiled | bank statements | invoices | receipts |
|---|---|---|---|
| **mineru** | **141/2011 = 7.0%** | 0/308 = 0.0% | 184/184 = 100% |
| gemma-4-12B | 313/2011 = 15.6% | 5/308 = 1.6% | 1/184 = 0.5% |
| InternVL3.5-8B | 326/2011 = 16.2% | 3/308 = 1.0% | 32/184 = 17.4% |
| docling | 1671/2011 = 83.1% | 74/308 = 24.0% | 184/184 = 100% |

The 100% for the two parsers on receipts is not misplacement: neither emits a
table there at all, so every amount is trivially absent from a column. Gemma's
bank-statement figure includes 3 pages declared unproducible and scored as total
failures; on the pages it produced, it is 12.1%.

The meaningful number is **bank statements: about 1 amount in 7 filed under the
wrong heading** by the two prompted models, while those same models post
normalised CERs near 0.02.

And note **MinerU is the most structurally faithful of the four** on the hard
tables — the reverse of what the convention metrics suggest.

### Table structure

Column integrity asks whether an amount landed under the right heading. This
asks the prior question: did the system reproduce the table's *rows* at all?
Truth rows are matched to prediction rows by signature, so a dropped or invented
row shifts nothing after it, and cell accuracy is measured only over rows that
matched — comparing cells across rows that do not correspond would compare
unrelated values.

| System | rows aligned | fragments | width breaks | cell accuracy | content recall |
|---|---|---|---|---|---|
| gemma-4-12B-it-qat-w4a16-ct | **1409/2005 (70.3%)** | **0** | **0** | 0.992 | 0.879 |
| InternVL3.5-8B | 1390/2005 (69.3%) | 17 | 9 | 0.922 | 0.902 |
| mineru | 1234/2005 (61.5%) | 292 | 156 | **1.000** | 0.888 |
| docling | 283/2005 (14.1%) | 120 | 13 | 0.850 | 0.273 |

A **fragment** is a continuation row: the system split one logical row across
two, usually because the description wrapped on the page. A **width break** is a
row with the wrong number of cells.

**No system wins outright, and the two candidates fail in opposite ways.** Gemma
never breaks structure — zero fragments, zero width breaks across 165 pages —
and when it commits to a cell it is right 99.2% of the time; but it recalls the
fewest cells. MinerU has perfect cell accuracy on the rows it aligns and the
best amount placement, but produces 292 fragments and 156 width breaks, and does
not recognise a receipt item list as a table at all.

If you had to pick one for a bank statement, the answer is **MinerU**, on the
strength of 7.0% misfiled amounts against 15.6% and 16.2%. That directly
contradicts the CER leaderboard, where MinerU sits second and pays the largest
convention penalty of any system. Reading tables and matching a house style are
close to independent skills.

---

## Findings

### 1. A rule can be stated and still not communicated

The prompt has always said:

> If the table has no printed column headings, use an empty header row — do not
> promote the first line of data into the heading, and do not invent column
> names.

Receipts have exactly that: an item list with prices, no headings. Compliance:

| System | receipts emitting a table | told the rule? |
|---|---|---|
| gemma-4-12B-it-qat-w4a16-ct | **2 of 55** | yes |
| InternVL3.5-8B | 6 of 55 | yes |
| mineru | 0 of 55 | no |
| docling | 0 of 55 | no |

Near-zero everywhere. The rule was a clause inside a paragraph about tables, and
a receipt's item list evidently does not register as "a table" in the first
place.

Rewriting it to name the case — *"A list of items with amounts beside them is a
table, even when it has no column headings and no lines drawn between the
columns"*, with a worked headerless example — moved gemma-4-12B from **2/55 to
53/55**, producing the correct empty header row rather than invented column
names. Same rule, same model, same pages; only the wording changed.

The worked example is built from invented content, verified to appear in none of
the 165 transcripts, and no prediction echoed any of it. An earlier version of
this experiment used real line items from a scored page — see *A prompt is part
of the eval set* below — and the 53/55 figure is the clean re-run.

Re-running both prompted systems confirmed it on the structural metric, and
**both models moved** — which is what separates this rule from the one in
finding 8:

| Receipt rows correctly segmented | before | after |
|---|---|---|
| gemma-4-12B-it-qat-w4a16-ct | 9/184 (4.9%) | **183/184 (99.5%)** |
| InternVL3.5-8B | 17/184 (9.2%) | **149/184 (81.0%)** |

Misfiled amounts on receipts fell with it: gemma 97.3% → 0.5%, InternVL 90.2% →
17.4%.

**The prompt is a component to be tested, not documentation to be written once.**
A rule can be technically present and effectively absent.

### 2. The corpus was contradicting itself on repeated glyphs

The corpus had two treatments of the same visual device, decided by which
drawing primitive produced it:

| Device | Corpus treatment | What systems did |
|---|---|---|
| dashed separator rule | **excluded** as decoration | 3 of 4 transcribed it |
| dot leader inside a table cell | **captured** as content | omitted it, or ran away |

Both halves cost accuracy. Gemma inserted characters equal to **63% of the whole
transcript's length** on every one of 55 receipts, transcribing separator rules
the corpus had decided not to record; docling 124%. Meanwhile the dot leaders the
corpus *did* record were skipped by InternVL and caused Gemma to fail outright.

Resolved 19 August: **repeated glyphs are decoration wherever they are drawn.** A
run of four or more repeated punctuation glyphs is no longer captured. Three is
punctuation, so an ellipsis and a decimal point survive, as do `5-7 business
days` and a statement's date range.

The glyphs are still **drawn** — 0 of 165 images changed — so every existing
prediction stayed valid and nothing was re-run. Only the truth moved. Every real
reader improved: MinerU 0.0698 → 0.0526, InternVL 0.1131 → 0.0978, Gemma 0.1860
→ 0.1710.

### 3. CER is blind to the error that matters most on a statement

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

Three of those pages have **every single amount** under the wrong heading, and
pay about 8–11% character error for it — twice the system's median of 0.043, for
a result that is entirely worthless. Normalisation strips pipes as table marks,
discarding exactly the delimiters that encode the answer, so a total structural
failure registers as a mild reading day.

It was starker still before the corpus was corrected: `CASE005` then scored
**0.0128**, three times *better* than that system's median, for the same
catastrophe. The point survives the change in magnitude — CER prices this error
somewhere between "cheap" and "free", and nothing about the number tells you
which page is unusable.

Hence the column-integrity metric above. Without it the harness could not
distinguish "read the statement correctly" from "read every character correctly
and filed every amount under the wrong heading".

### 4. The dot-leader runaway

One layout prints **dot leaders** — ~40 periods padding a transaction reference
across to the amount column. Real statements do this.

On the pages carrying them, gemma-4-12B transcribes the masthead, account details
and table header correctly (369 characters), reaches the first leader, and emits
**128,768 consecutive periods** — roughly 3,200× amplification — until cut off at
the token limit.

Neither *decoding* remedy works. Raising the token limit buys a longer run of
periods, as there is no pending content behind the loop. A repetition penalty
completes the page only by **deleting all 12 leaders** — about 480 characters
the page shows — which is worse than failing, because it looks like success.

InternVL does not loop; it omits the leaders entirely. So the runaway is
Gemma-specific, not a property of the page.

> **Superseded on 2026-08-20, and the correction is the interesting part.**
> This finding originally concluded that a real document feature had defeated a
> capable model, and the pages were written off as unproducible and scored 1.0.
> That was wrong. The runaway was an **instruction-following artefact**: the
> prompt said "do not skip repeated or boilerplate text" while the corpus strips
> repeated glyphs, so the model was doing as it was told. Told the actual
> convention, it does not loop — all three pages complete. See finding 10.
>
> Both decoding remedies failed because both treated a symptom. That is the
> lesson worth keeping: **a pathology that resists every generation parameter is
> evidence to re-read the prompt, not evidence of a model limit.**

**Which pages loop is not stable.** It is deterministic for a fixed prompt —
three attempts at temperature 0 reproduce it exactly — but the *set* of afflicted
pages moves when the prompt is edited, on byte-identical images:

| Prompt version | pages that ran away |
|---|---|
| v1 | CASE003, CASE024 |
| v2 | CASE024 |
| v3 | CASE003, CASE024, **CASE047** |

`CASE003` recovered and then relapsed; `CASE047` had never failed before. The
prompt is not a neutral wrapper around a fixed capability — it perturbs decoding
enough to move a pathological attractor on and off individual pages.

The operational consequence: **a record of unproducible pages is scoped to a
prompt version, not to a page.** Carrying one forward across a prompt change
would score 1.0 for a page the model can now read, and would have concealed
`CASE003`'s recovery entirely.

**This failure is invisible to information extraction**, which asks for a few
hundred bounded tokens and stops. The same checkpoint has high IE accuracy on
these very pages.

### 5. A prompt is part of the eval set

The first version of the rewritten rule used a worked example built from **real
line items on a scored page** — `Potting Mix 25L | 9.30` and `Panadol 24pk |
7.77`, taken from a receipt in the corpus. Those strings appear in 5 and 9 of
the 165 transcripts.

The prompt ships **with** the corpus and is read by the systems being scored, so
that handed the model part of an answer it was supposed to read off the page. A
model could reproduce those lines without seeing them, and nothing in the
results would show it. Checking the pre-existing examples found the same problem
already present, at smaller scale: an amount from the original table example
appears in one transcript.

Both examples are now invented content, verified absent from all 165
transcripts, with a test that greps every example cell against every transcript
so a future edit cannot reintroduce it.

Two details worth carrying to any similar benchmark:

- **Header words are fine, values are not.** "Date" and "Description" appear on
  real pages by necessity; they are vocabulary, not answers. The test exempts
  the header row and checks data rows only.
- **Verify the guard fails on a deliberate leak.** The first version of that
  test passed immediately — because a quoting error had left it with an empty
  regex matching nothing. A vacuously-passing guard is worse than no guard: it
  manufactures confidence. It was caught only because it *should* have flagged a
  word appearing in all 165 transcripts and didn't.

The compliance result survived decontamination: 2/55 → **53/55** with clean
examples, against 54/55 with the leaked ones, and **no prediction echoed any
example string**. The gain was the rewording, not the model copying values it
had been shown.

### 6. Character-level errors are the remaining real cost

The evaluated Gemma checkpoint is **4-bit QAT** — quantisation-aware training at
four bits per weight. Theory says this costs character fidelity first. It does.

On one field across seven pages, gemma-4-12B got 5 of 7 wrong: three reading a
comma as a period, three substituting a digit (`487,205` → `497,205`). InternVL,
with a corrected tokenizer, got **every digit right** but made the same
comma-for-period substitution on 3 of 7.

Two independent models misreading the same comma glyph is worth noting. The
pages were checked at magnification: they render unambiguous commas. The
substitution is real, and `345,678` versus `345.678` differ by three orders of
magnitude.

This was only measurable because those figures had just been changed from
hardcoded layout literals — identical on all seven pages — to per-page authored
data. Under the old arrangement one number repeated seven times would have
produced one error repeated seven times, and a model could have scored well by
memorising it.

### 7. A rule can be obeyed and still make things worse

A bank statement often prints a date once and lists that day's transactions
beneath it with the date column blank. The corpus fills the date in on every
row, so each row stands alone, and the prompt says to do the same.

Adding that rule appeared to be a mistake. Gemma's table row accuracy fell from
**64.3% to 58.8%**, and misfiled amounts rose. The obvious reading — the
instruction confused the model — was wrong.

The corpus draws a date group **two ways**: as a band across the table with the
date alone on it, or with the date in the first transaction's own date cell.
Both mean the same thing, and which one a page gets is an implementation detail
invisible to any reader. The code that fills the date in handled only the first.
So **123 rows across 7 statements kept a blank date while 329 rows on 27 others
had theirs filled in** — and a model following the shipped instruction was
correct on one layout family and wrong on the other.

Gemma had gone from replicating the date on 62% of grouped rows to **97.8%**,
against a corpus that was itself only 79.8% consistent. Every extra date it
supplied correctly was scored as an error.

Fixing the corpus flipped the sign:

| | against the inconsistent corpus | against the corrected corpus |
|---|---|---|
| without the rule | 64.3% rows aligned | 58.6% |
| with the rule | 58.8% | **63.8%** |

A 5.5-point loss became a 5.2-point gain, from the same predictions — no model
was re-run to produce that table. The rule was right all along.

Two things generalise. **A metric regression is evidence about the pair
(system, ground truth), never about the system alone**; here the natural
inference was exactly backwards. And **internal consistency of ground truth is
a testable property** that nothing in a whole-page CER will surface: both
behaviours produce plausible transcripts, and only asking "does the corpus treat
this device the same way everywhere?" finds it. That is the second time this
pass that a single question caught a defect — the first was finding 2.

### 8. Some instructions never land

Finding 1 showed a rule going from 2/55 to 53/55 on rewording, which invites the
conclusion that non-compliance is a wording problem. It is not always.

The date rule of finding 7 was phrased three ways over three runs: stated
plainly, then naming the banded case, then naming both cases with a worked
example and an explicit exception. Compliance, measured as the share of grouped
rows carrying their date:

| Prompt version | gemma-4-12B | InternVL3.5-8B |
|---|---|---|
| v1 | 61.6% | 57.1% |
| v2 | 97.8% | 56.5% |
| v3 | 97.2% | 57.4% |
| *corpus* | *98.9%* | *98.9%* |

One model heard it on the first rewrite and has tracked the corpus ever since.
The other has not moved at all — three phrasings, a 0.9-point spread, no trend.
It also kept emitting the date band as a table row of its own, 17 then 25 then
17 times, never approaching zero.

So a prompt is worth testing as a component, and rewording is worth trying — but
a flat response across several genuine attempts is a capability signal, not an
invitation to write a fourth version. **Distinguishing the two requires more
than one model**, which is precisely why the pass was specified with two
families rather than one.

That also qualifies the headline result. The claim "the convention is
communicable" rests mainly on one model: gemma pays 0.008 to follow it, InternVL
pays 0.155 — still far better than MinerU's 0.392, but eighteen times gemma's.
How well a house style lands is a property of the model as much as of the style.

### 9. A structural failure that three prompt revisions did not touch

Two statements carry a header cell that wraps: `Date of / Transaction`. Under
the original prompt InternVL read it correctly as one column. Under both later
prompts it splits into three, and merges two transactions into one row:

```
truth   | Date of Transaction | Description | Debits | Credits (-) |
v1      | Date | Transaction Description | Debits | Credits ($) |    4 columns, 0 misfiled
v2, v3  | Date | Transaction | Description | Debits | Credits ($) |  5 columns, 38 misfiled
```

Every character survives, so CER barely registers it. The row structure is
destroyed and 38 amounts per page are filed under the wrong heading. It is
stable across two prompt revisions that were aimed at other things, so it is not
a wording artefact — and it is invisible to every metric except column
integrity, which is the argument for that metric restated as a live case.

### 10. What a worked example does that a stated rule cannot

Every earlier prompt change altered wording and added an example at once, so
none of them could say which half worked. This one was run as a controlled
A/B: two prompt files, **identical below the operator preamble except for one
worked example**, over 61 pages — the 55 receipts, which draw separator rules,
and the 6 statements that draw dot leaders.

It began by fixing a contradiction. The corpus strips runs of four or more
repeated punctuation glyphs at capture, while the prompt said *"do not skip
repeated or boilerplate text"* — instructing the model to transcribe precisely
what the ground truth omits. Nothing in the prompt mentioned separator rules or
dot leaders at all. Both arms scope that sentence and state the rule; only arm B
adds the example.

| gemma-4-12B | baseline | arm A: rule | **arm B: rule + example** |
|---|---|---|---|
| statement pages completed | 3/6 | 6/6 | 6/6 |
| separator lines emitted | 275 | **0** | **0** |
| decoration characters | 21,650 | **0** | **0** |
| median receipt CER | 0.8164 | 0.0044 | 0.0043 |
| **mean** CER | — | 0.1108 | **0.0214** |
| amounts misfiled | 24.0% | 7.6% | **0.3%** |

**On medians the two arms are indistinguishable. On the mean they differ five
times over, and the entire difference is one page.**

That page is the one the benchmark had already written off. Under the original
prompt it ran to the token cap and was recorded as unproducible — a real
document feature defeating a capable model, or so finding 4 concluded. Truth is
1,507 characters:

| | output | CER |
|---|---|---|
| baseline | hit the token cap | scored 1.0 |
| arm A | 7,334 characters | **5.52** |
| arm B | 1,491 characters | **0.076** |

Arm A's output ends like this:

```
000000000000000000000000000000000000000000000000000000000000
```

**The stated rule did not stop the runaway. It changed its shape.** Told not to
write repeated dots, the model wrote repeated zeros instead — and because that
generation happened to finish under the token cap, nothing flagged it. It was
reported as a clean 6/6 completion for an hour before the mean gave it away.

The example is what fixed it, and the mechanism is legible: a prohibition says
what not to emit and leaves the substitute unspecified, while an example shows
what to write *instead*. The shipped rule now carries both, including an
explicit "do not replace an omitted run with anything else".

So the sharper claim is not that examples help. It is that **an example
disambiguates the replacement behaviour a prohibition leaves open**, and that
matters exactly where a model would otherwise improvise.

Two costs, and they fall unevenly.

The example perturbs generation everywhere — **52 of 61 pages differ between the
arms** — and mildly degrades about five receipts the rule alone had nearly
perfect (0.002 → 0.05): a duplicated business name here, a line split in two
there. Net still strongly positive for gemma.

For the smaller model it is a net loss:

| InternVL3.5-8B | baseline | arm A | arm B |
|---|---|---|---|
| separator lines | 69 | **6** | 10 |
| mean CER | — | **0.0785** | 0.1009 |
| pages produced | 61/61 | **61/61** | 60/61 |

Arm B gave it a **new** runaway — 53,547 characters of empty table rows,
`|      |` repeated to the token cap, on a page both other arms handled.

That is the pattern the whole pass has been drawing, now in its clearest form.
**The smaller model is steered less reliably and destabilised more easily.**
Every intervention tried here — the headerless rewrite, the date rule, the
decoration rule, the worked example — lands cleanly on the 12B and lands
weakly, or backwards, on the 8B. A prompt improvement is not a property of the
prompt; it is a property of the pair.

One control worth recording: InternVL's CER on those six statements was 0.3034,
0.3028, 0.3045 across the three arms. Three interventions, no movement. That is
the wrapped-header column split of finding 9, exactly as predicted before the
run — which is the closest this pass came to a pre-registered hypothesis.

---

## Limits

**The two parsers were not re-run under the final prompt, and did not need to
be.** MinerU and Docling read no prompt, so their figures come from the earlier
run, re-scored against the final corpus. The prompted systems' figures come from
the final run. Every number in this document is against `parsing_20260819d`.

**Scoring conflates two things in the column metric.** `misfiled` counts an
amount as misplaced if it is absent from its column for *any* reason, including a
misread digit. It is not purely structural.

**Synthetic and pristine.** Clean renders, not photographs or scans. The
benchmark measures parsing accuracy on clean input and does not claim to predict
performance on degraded documents.

**Coverage.** gemma-4-12B produced 162 of 165 pages; the 3 dot-leader failures
are scored as total failures rather than dropped, so all systems are averaged
over the same 165 transcripts. InternVL produced all 165. Docling produced 2
empty pages, scored the same way. Finding 10 has since shown those 3 pages are
producible under the corrected prompt, so every gemma figure in this document
**understates** it — the pages are still scored 1.0 here because the full-corpus
run predates the fix.

**The main results predate the decoration rule.** Finding 10 was measured on a
61-page subset and the rule has since been promoted into the shipped prompt, but
the 165-page figures throughout this document were produced without it. Gemma's
receipt CER in particular is 0.8164 here and 0.0043 under the current prompt.
Re-running the full corpus is the outstanding work.

**One convention mismatch is known and unresolved.** On one layout the corpus
records a block's visual alignment (`Opening Balance          345,678`) while a
model writes `Opening Balance: 345,678`, obeying the prompt's instruction to put
labelled values on one line. The block is drawn as plain lines rather than
label/value pairs, so the transcript keeps the spacing. A model reads correctly,
follows the shipped rule, and still diverges. Either the rule should be scoped to
real pairs or the block should be captured as pairs; neither has been done.

**Some divergences are classified wrongly.** Of the divergences the report calls
reading errors, 21% are format artefacts — transcribed horizontal rules and a
wrapping ` ```markdown ` fence — which are convention, not misreading. The
classification understates how well these systems read.

**Character errors are not separated by cause.** A misread digit could be
quantisation, resolution, or the model. Comparing quantised and unquantised
checkpoints would isolate it; that has not been done.

---

## What changed as a result

**The convention mostly stands.** A capable prompted model follows it at a cost
of 0.009 CER. It is not an arbitrary house style.

**Four scoring fixes**, each closing a case where formatting was scored as
misreading: HTML tables, HTML entities, trailing label colons, and repeated-glyph
decoration. The first three dropped MinerU's normalised CER from 0.495 to 0.055 —
a nine-fold correction, and a measure of how badly a formatting-blind metric can
misrepresent a competent reader.

**Two new metrics.** Column integrity, because CER could not see the error that
matters most on a bank statement. Then table structure — row alignment,
fragments, width breaks, cell accuracy, content recall — because column
integrity presupposes the rows were segmented correctly, and on the hard tables
they often are not. Together they produced the pass's most counter-intuitive
result: the system with the worst convention score reads bank-statement tables
best.

**Two corpus defects fixed**, both the same fault — the corpus treating one
visual device two ways depending on which code path drew it. Repeated glyphs are
now decoration wherever they appear; a grouped date is now carried down whichever
way the layout draws it. A third defect, fabricated figures hardcoded in a
layout, is now authored per page with its arithmetic enforced at validation.

**Three prompt revisions, individually measured.** The headerless-table rewrite
was a large win (receipt row accuracy 4.9% → 99.5% for gemma, 9.2% → 81.0% for
InternVL). The date rule looked like a loss and was actually a win once the
corpus stopped contradicting itself. Both shipped in the same revision, and the
aggregate CER concealed that they had opposite signs — which is the argument for
changing one instruction at a time.

**A working practice, not just a result.** Because ground truth is written at
render time and serialisation is a separate step, a convention can be changed and
every transcript re-emitted in seconds without re-rendering an image. Both corpus
fixes above changed **zero images**, so every existing prediction stayed valid and
the hypotheses were tested by re-scoring rather than re-running. That is what
made it affordable to find out that a regression was the corpus's fault.

**A fourth prompt revision, and the first controlled one.** The decoration rule
and its worked example are now in the shipped prompt, along with a fix to the
sentence that had been instructing models to transcribe what the corpus omits.
On the 61 pages it was measured over, gemma's receipt CER fell from 0.8164 to
0.0043, its misfiled amounts from 24.0% to 0.3%, and three pages previously
written off as unproducible now transcribe correctly.

**A finding retracted.** The dot-leader runaway was recorded as a real document
feature defeating a capable model. It was the prompt contradicting the ground
truth. Both remedies that failed — a larger token budget, a repetition penalty —
failed because both treated the symptom.

**Still open**: the `Label: value` mismatch in Limits; the misclassification of
rules and code fences as reading errors; the wrapped-header column split of
finding 9, which four prompt revisions have now failed to move; and the
receipt-totals block, which the decoration rule turned into a spurious second
table on 26 of 55 pages by removing the separator rules that had been fencing it
off — two individually correct instructions, jointly wrong.

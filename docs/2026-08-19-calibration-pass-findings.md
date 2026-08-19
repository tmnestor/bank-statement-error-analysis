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

The house style is largely fine. But the pass found three things that matter
more than that verdict:

1. **Stating a rule is not the same as communicating it.** One rule was written
   in the prompt and achieved **2/55** compliance. Rewritten to name the case it
   applies to — same rule, same model, same pages — it reached **53/55**.
2. **The corpus was contradicting itself**, treating the same visual device as
   content in one place and decoration in another, depending on which piece of
   code drew it. That cost one model insertions equal to **63%** of a page's
   length, on every receipt.
3. **The primary metric is blind to the error that matters most.** A page where
   every row was filed under the wrong column heading scored *better* than that
   system's average.

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

Per-page **medians**, which describe these systems better than means — a few
catastrophic pages distort the averages badly.

| System | median nCER | median sCER | **median gap** | told? |
|---|---|---|---|---|
| **gemma-4-12B-it-qat-w4a16-ct** | **0.0186** | 0.0591 | **+0.0092** | yes |
| mineru | 0.0368 | 0.4277 | **+0.3917** | no |
| InternVL3.5-8B | 0.0425 | 0.3492 | **+0.2322** | yes |
| docling | 0.4145 | 0.6120 | +0.0587 | no |

**Reading quality is close between the three real readers** — 0.019 to 0.043
median CER. Gemma reads best *and* formats closest.

**MinerU loses 0.39 to formatting it cannot be instructed about**, principally
emitting HTML tables (`<table><tr><td>`) on 110 of 165 pages where the corpus
uses pipe tables. It reads well and formats differently, which is exactly what
an unpromptable parser should look like.

**Gemma loses almost nothing (+0.009)** — told the style, it complies.

**InternVL sits between (+0.232), and about two-thirds of that is table
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
| mineru | 2503 | 325 | **13.0%** | 56 |
| InternVL3.5-8B | 2503 | 408 | 16.3% | 53 |
| gemma-4-12B-it-qat-w4a16-ct | 2503 | 408 | 16.3% | 56 |
| docling | 2503 | 1929 | 77.1% | 117 |

Read this **by document type**, because the aggregate misleads badly:

| gemma-4-12B | bank_statements | invoices | receipts |
|---|---|---|---|
| misfiled | 170/1957 = **8.7%** | 5/308 = 1.6% | 179/184 = 97.3% |

The receipt figure is a serialisation difference, not misplacement — the corpus
serialises receipt items as a headerless two-column table and the models emitted
plain lines (the subject of finding 1 below). The meaningful number is
**bank statements: about 1 amount in 12 filed under the wrong heading**, by both
of the best systems, while those same systems post normalised CERs near 0.02.

And note **MinerU is the most structurally faithful of the four** — the reverse
of what the convention metrics suggest.

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

On `CASE005_bank_statements`, InternVL read a header cell that wraps across two
lines — `Date of / Transaction` — as **two columns**. Every data row inherited
the extra cell, so every amount sat one column right of the heading naming it. On
a bank statement that is money out reported as money in.

That page scored a normalised CER of **0.0128** — three times *better* than that
system's corpus median of 0.0425 — because normalisation strips pipes as table
marks, discarding exactly the delimiters that encode the answer. The entire
structural failure cost a couple of characters.

Hence the column-integrity metric above. Without it the harness could not
distinguish "read the statement correctly" from "read every character correctly
and filed every amount under the wrong heading".

### 4. The dot-leader runaway

One layout prints **dot leaders** — ~40 periods padding a transaction reference
across to the amount column. Real statements do this.

On 2 of the 6 pages carrying them, gemma-4-12B transcribes the masthead, account
details and table header correctly (369 characters), reaches the first leader,
and emits **128,768 consecutive periods** — roughly 3,200× amplification — until
cut off at the token limit. Deterministic across three attempts at temperature 0.

Neither remedy works. Raising the token limit buys a longer run of periods, as
there is no pending content behind the loop. A repetition penalty completes the
page only by **deleting all 12 leaders** — about 480 characters the page shows —
which is worse than failing, because it looks like success. The pages are
recorded as unproducible and scored as total failures.

InternVL does not loop; it omits the leaders entirely. So the runaway is
Gemma-specific, not a property of the page.

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

---

## Limits

**The prompt rewrite is shipped but the scores predate it.** The headerless-table
wording is now in ; the predictions scored above were produced under
the old wording, so they understate what the prompted systems achieve on
receipts.

**Scoring conflates two things in the column metric.** `misfiled` counts an
amount as misplaced if it is absent from its column for *any* reason, including a
misread digit. It is not purely structural.

**Synthetic and pristine.** Clean renders, not photographs or scans. The
benchmark measures parsing accuracy on clean input and does not claim to predict
performance on degraded documents.

**Coverage.** gemma-4-12B produced 163 of 165 pages; the 2 dot-leader failures
are scored as total failures rather than dropped, so all systems are averaged
over the same 165 transcripts. Docling produced 2 empty pages, scored the same
way.

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

**One new metric**: column integrity, because CER could not see the error that
matters most on a bank statement.

**One corpus defect fixed**: fabricated figures hardcoded in a layout are now
authored per page with their arithmetic enforced at validation.

**One prompt rewrite proven** and awaiting promotion.

**Still open**: whether table structure deserves scoring in its own right. The
corpus scope explicitly excludes it, but that exclusion assumed columns were
delimited on the page. They are not.

# Is the transcription convention fair to models?

**Calibration pass, 19 August 2026.** Three document-parsing systems over 165
synthetic pages. Written for readers who know machine learning but not this
project's vocabulary — every term is defined at first use.

---

## The one-paragraph answer

We built a benchmark that asks a model to read a whole page of a business
document and write it out as Markdown. Doing that requires the benchmark to
pick a *house style* — where headings go, how tables are drawn, how a label and
its value are joined. The worry was that our house style might be idiosyncratic:
a model could read a page perfectly and still score badly for formatting it
differently. **It isn't.** A model that is told the style follows it almost
exactly, paying a formatting penalty of **0.0007**. A parser that cannot be told
anything pays **0.39**. The house style is learnable, so it stays as it is.

---

## What is being measured

### Full-page transcription

The task is **transcription**: given one image of a document page, output *all*
of its text, in reading order, as Markdown. Nothing is summarised, nothing is
skipped, nothing is added.

This is deliberately **not** several adjacent tasks it is often confused with:

| Not this | Why it differs |
|---|---|
| **Information extraction (IE)** | IE asks for named fields — invoice total, ABN, due date. Output is a short, bounded record. Transcription asks for the entire page. |
| **OCR** | OCR recovers characters. Transcription also has to decide structure: what is a heading, what is a table, what order to read a two-column layout. |
| **Layout analysis** | Layout analysis returns boxes and region labels. We score text, not geometry. |
| **Table structure recognition** | Scored elsewhere with metrics like TEDS. Here a table is judged as text in reading order. |

The distinction matters for interpreting these results. A model can be excellent
at IE on a page and still fail to transcribe it, because transcription requires
a long, unbounded generation and IE does not — a failure mode we hit directly
(see *The dot-leader runaway*).

### The corpus

165 page images: **55 cases × 3 document types** (invoices, receipts, bank
statements) across **18 layout variants** — different banks, different invoice
styles, thermal receipt slips. Australian business documents, entirely synthetic.

**Ground truth is authored, not annotated.** This is the unusual part. Nobody
labelled these pages after the fact. The renderer emits the correct answer *at
the moment it draws the page*: each drawing primitive records what text it is
about to put on the canvas. The image and its expected transcription are
produced by the same pass, so they cannot disagree.

A runtime check enforces this. Every call that puts text on the canvas is tagged
with the identifier of the record that authorised it, and at page end the
renderer asserts that no untagged text was drawn. If a new code path draws
something without recording it, generation fails — it is not a test that can be
skipped.

### Terms used throughout

**Transcript** — the correct answer for one page: a Markdown file stating what
that page shows. 165 of them, one per image.

**Convention** (or **serialisation policy**) — the house style the transcripts
are written in, declared in a configuration file: `#` for the page title, pipe
tables with a header separator row, `Label: value` for labelled values, no bold
or italic ever, read side-by-side columns one column at a time. Every rule is
written down; none is implicit in code.

**Prompt** — the instructions shipped to models that can accept instructions.
It states the convention in prose. The prompt and the transcripts are a matched
pair: change one without the other and the benchmark silently measures something
else.

**Prediction** — what a system actually produced for a page.

---

## The metrics, and the one that matters

### CER and WER

**Character Error Rate** is the edit distance between prediction and truth,
divided by the length of the truth. **Word Error Rate** is the same computed
over whitespace-separated words.

Both are *error* rates, so **lower is better** and 0 is perfect. Two properties
surprise people:

- `0.05` means one character in twenty is wrong.
- **Values above 1.0 are possible and common here.** CER exceeds 1 when the
  prediction is *longer* than the truth, because every surplus character counts
  as an insertion. A model that emits 34,000 characters against 557 of truth
  scores about 62. This is not a bug in the metric; it is the metric correctly
  reporting that the output is mostly invented.

### Two ways to score the same prediction

Every prediction is scored twice.

**Strict** compares the raw text. It measures reading *and* formatting
compliance together.

**Normalised** first passes both sides through a declared normalisation policy
— Unicode normalisation, whitespace collapsed, Markdown syntax removed
positionally, HTML table tags stripped, entities unescaped, label colons folded
— and then compares. Because both sides get the same treatment, purely
*formatting* differences vanish and only genuine misreadings remain.

Normalisation lives entirely in the scoring tool. The corpus emits one canonical
form and never normalises, so scoring policy can change without regenerating a
single image.

### The gap: the actual experimental measurement

> **gap = strict CER − normalised CER**

Everything normalisation removed is, by construction, formatting rather than
reading. So the gap is **the cost of the house style** for that system: how much
of its apparent error was never about reading the page at all.

A system with a large gap read the page well and formatted it differently. A
system with a gap near zero formatted the way the benchmark expects, and
whatever error remains is real misreading.

### Two classes of divergence

Every difference between a prediction and its truth is also classified
individually, by the same principle:

- **Convention mismatch** — the two spans become identical after normalisation.
  The system read correctly and wrote it differently.
- **Reading error** — the difference survives normalisation. The system got the
  text wrong.

There is no pattern list and nothing to tune. The classifier *is* the
normalisation policy, applied at the level of a single difference instead of a
whole document.

---

## Experimental design

The systems are split by one property: **whether the system can be told the
convention.**

| Group | Systems | Reads the prompt? | What it measures |
|---|---|---|---|
| **Prompted** | `gemma-4-12B-it-qat-w4a16-ct` | yes | Is the convention *communicable*? |
| **Unprompted** | MinerU, Docling | no — they emit whatever their authors chose | Is the convention *idiomatic Markdown at all*? |

The split is the point, and both halves are needed.

A dedicated document parser has no prompt input. You cannot ask MinerU to emit
pipe tables instead of HTML. So every divergence it produces is uncontaminated
evidence about whether our house style matches what the Markdown world does by
default.

A prompted model *is* told — pipe tables, one `#` per page, no bold. If it still
diverges, the instruction failed to land, and the problem is ours.

The two answer different failure modes:

- **Unidiomatic but teachable** → parsers diverge, prompted models comply.
  Acceptable; the style just has to be stated.
- **Unteachable** → prompted models diverge too. Then the convention itself is
  wrong, and it is fixable in configuration without re-rendering any image.

---

## Results

All three systems over the same 165 pages.

| System | normalised CER | strict CER | **gap** | told? |
|---|---|---|---|---|
| **MinerU** | **0.0545** | 0.4476 | **+0.3931** | no |
| **gemma-4-12B-it-qat-w4a16-ct** | 0.2926 | 0.2933 | **+0.0007** | **yes** |
| **Docling** | 1.6576 | 2.0611 | +0.4035 | no |

Per-page **medians**, which are more representative because means here are
distorted by a handful of catastrophic pages:

| System | median normalised CER | median strict CER | **median gap** |
|---|---|---|---|
| MinerU | 0.0386 | 0.4277 | **+0.3917** |
| **gemma-4-12B-it-qat-w4a16-ct** | **0.0186** | 0.0591 | **+0.0092** |
| Docling | 0.4267 | 0.6120 | +0.0587 |

### Reading the table

**MinerU reads these pages almost perfectly and loses 0.39 CER to formatting.**
A median normalised CER of 0.039 is a strong reader. Its strict CER of 0.43 is
almost entirely house style — principally that it emits HTML tables
(`<table><tr><td>`) on 110 of 165 pages, including every bank statement, where
the corpus uses pipe tables.

**The prompted model loses essentially nothing to formatting: 0.0007.** Given
the instruction, it complies.

**That is the result.** The convention is teachable, so it does not need to
change. A system that is told the style follows it; a system that cannot be told
diverges in exactly the way you would expect.

### A second finding

The prompted model is also **the better transcriber**. Its median normalised CER
of **0.0186** beats MinerU's 0.0386 — it reads more accurately *and* formats
correctly at the same time. Its much worse mean (0.2926) is two total failures
dragging the average, not typical behaviour. Where mean and median disagree this
sharply, the median is describing the system and the mean is describing its
worst days.

---

## Findings worth acting on

### 1. The dot-leader runaway

One layout, `nab_classic`, prints **dot leaders** — a run of about 40 periods
padding a transaction reference across to the amount column, the typographic
device that leads your eye across a table row. Real bank statements do this.

On 2 of the 6 pages carrying them, the prompted model transcribes the masthead,
account details and table header correctly — 369 characters — reaches the first
dot leader, and emits **128,768 consecutive periods**. A 40-character feature on
the page became a 128,768-character output: roughly **3,200× amplification**. It
never recovers, and the generation is cut off at the token limit.

This is deterministic — three attempts at temperature 0 — and neither obvious
remedy works:

- **Raising the token limit** buys a longer run of periods. There is no pending
  content behind the loop.
- **A repetition penalty** (1.05) completes the page, but only by **deleting all
  12 dot leaders** — about 480 characters the page genuinely shows. That is
  worse than failing, because it looks like success.

The pages are therefore recorded as unproducible and scored as total failures.

**Why this is kept rather than fixed.** A real document feature that reduces a
capable model to emitting nothing is exactly what an evaluation set should
surface. It is also only visible because the corpus spans layouts: a pass run on
invoices alone sees none of it, and this affects 2 of 6 `nab_classic` pages and
0 of the other 159.

**Note for anyone benchmarking VLMs on documents.** This failure is invisible to
information extraction. IE asks for a few hundred bounded tokens and stops;
transcription asks for the whole page and gives a degenerate loop room to
develop. The same checkpoint has high IE accuracy on these very pages.

### 2. A genuine convention mismatch

On the Westpac rewards summary the page prints two visually aligned columns:

```
Opening Balance          345,678
```

The corpus records that alignment. The model writes:

```
Opening Balance: 345,678
```

The model is **obeying the prompt**, which says labelled values go on one line
as `Label: value`, "even if the page draws its own colon". But the corpus emits
this block as plain aligned lines rather than as label/value pairs, so the
transcript keeps the spacing.

A model reads correctly, follows the shipped instruction, and diverges anyway.
This is the one case found where the benchmark, not the model, is at fault.
**Undecided:** either narrow the `Label: value` rule in the prompt to genuine
pairs, or capture these blocks as pairs. Either is a configuration change; no
image needs re-rendering.

### 3. Character-level errors are the real remaining cost

The evaluated checkpoint is **4-bit QAT** — quantisation-aware training at four
bits per weight, a compression that trades numeric precision for memory. Theory
says this costs character fidelity before it costs anything else. It does.

On one field across seven pages:

| Truth | Predicted | Error |
|---|---|---|
| `487,205` | `497,205` | 8 → 9 |
| `604,517` | `804.517` | 6 → 8, and comma → period |
| `159,733` | `159,753` | 3 → 5 |
| `345,678` | `345.678` | comma → period |
| `271,309` | `271.309` | comma → period |

Five of seven wrong. Three are the same substitution — a comma read as a period
— which is systematic glyph confusion rather than noise, and consequential:
`345,678` and `345.678` differ by three orders of magnitude.

This was only measurable because those figures were made **per-page authored
data** shortly before the run. They had been hardcoded literals in the layout,
identical on all seven pages; under that arrangement one number repeated seven
times would have produced one error repeated seven times, and a model could have
scored well by memorising it.

### 4. Docling's numbers are not a convention signal

Docling's mean CER of 1.66 against a median of 0.43 says a small number of pages
emit far more text than exists — repetition loops on thermal receipts, one at
62× the truth length — plus 2 pages that produced nothing. Its gap of +0.40
should not be read as evidence about the house style; it is a parser failing.

---

## Limits of this result

**One prompted system, not two.** The design called for two VLM families so that
compliance could be shown as a property of prompted models generally rather than
of one checkpoint. `InternVL3.5-8B` has not been run. The central conclusion
rests on one model, and that is the main thing that would strengthen it.

Two locally quantised stand-in models were run earlier and excluded from these
figures — different quantisation and decoding made them incomparable. They did
independently show the same near-zero gap (+0.04 and −0.01), which is
corroboration but not a substitute.

**Coverage.** The prompted model produced 163 of 165 pages; the 2 dot-leader
failures are scored as total failures rather than dropped, so all three systems
are averaged over the same 165 transcripts and the numbers stay comparable.
Docling produced 2 empty pages, scored the same way.

**Synthetic and pristine.** These are clean renders, not photographs or scans.
The benchmark measures parsing accuracy on clean input; it does not claim to
predict performance on degraded documents.

**Character-level errors are not separated by cause.** A misread digit could be
quantisation, resolution, or the model. The design intended to isolate this by
comparing quantised and unquantised checkpoints, which has not been done.

---

## What changes as a result

**The convention stays.** This was the question the pass existed to answer, and
the answer is that the house style is learnable. No change to the serialisation
policy.

**Three normalisation fixes were made during the pass**, each closing a case
where a formatting difference was being scored as a misreading: HTML tables,
HTML entities, and trailing label colons. Their combined effect on MinerU was to
drop its normalised CER from 0.495 to 0.055 — a nine-fold correction, and a
measure of how badly a formatting-blind metric can misrepresent a competent
reader. Corpus-wide, differences classified as reading errors fell by 53%.

**One corpus defect was fixed:** fabricated figures hardcoded in a layout are
now authored per page with their arithmetic enforced at validation time.

**One open decision:** the `Label: value` mismatch above.

**One open task:** run the second VLM family.

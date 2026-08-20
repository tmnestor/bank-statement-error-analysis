# Can a prompted VLM be steered to a house style?

**Calibration pass, August 2026.** Seven document-parsing systems over 165
synthetic pages of Australian business documents.

---

## The short answer

The benchmark asks a model to read a whole page and write it out as Markdown.
Doing that forces a *house style*: where headings go, how tables are drawn, how
a label joins its value. The worry was that our house style might be
idiosyncratic — that a model could read a page perfectly and still score badly
for formatting it differently.

**It is not, and the reason is that gemma-4 can be told.** Three separate
conventions were each stated, measured, and adopted:

| Convention | before | after |
|---|---|---|
| headerless item tables | 2/55 pages | **53/55** |
| dates carried down a group | 61.6% of rows | **97.2%** (truth: 98.9%) |
| repeated glyphs are spacing | 275 stray lines | **0** |

Stating all three took the 12B's median normalised CER from **0.0201 to
0.0081**, its mean from **0.3011 to 0.0178**, and its misfiled amounts from
**12.7% to 9.7%** — and three pages the benchmark had written off as impossible
now transcribe at 0.05.

**The 31B at the same 4-bit quantisation goes further than steering alone
could.** Median normalised CER **0.0000** — more than half its pages are
character-perfect once formatting is set aside — with **99.0%** of
bank-statement amounts **usable** (right value, right heading), **94.7%** of
statement rows recovered, and **1.0%** filed under the wrong heading. It is the
first system here that reads the tables, gets the digits right, and follows the
house style at once.

Four things qualify that, and they are the substance of this document:

1. **Steerability is a property of the model, not the prompt.** The same three
   conventions, identically worded, moved the 12B and barely moved an 8B from
   another family. A prompt improvement is a property of the *pair*.
2. **A worked example does something a stated rule cannot.** Held to a
   controlled A/B, a rule alone stopped a model emitting the wrong characters
   but not improvising a replacement; the example supplied the behaviour the
   prohibition left undefined.
3. **Normalised character error rate — this benchmark's headline number — is
   blind to the two errors that matter most on a bank statement.** It strips
   Markdown syntax before comparing, and a table's pipes are Markdown syntax, so
   the delimiters recording which column an amount sits in are discarded first:
   pages with *every* amount under the wrong heading score 0.08–0.11. And it
   weighs a wrong digit in a total exactly as it weighs a typo in a merchant's
   name. Scored on amounts alone, the ranking is close to reversed.
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
| **Prompted** | four gemma-4 checkpoints, InternVL3.5-8B | yes | is the convention *communicable*? |
| **Unprompted** | MinerU, Docling | no | is it *idiomatic Markdown at all*? |

You cannot ask MinerU to emit pipe tables. Its divergences are therefore
uncontaminated evidence about whether the house style matches what the Markdown
world does by default. A prompted model *is* told; if it still diverges, the
instruction failed.

Four gemma checkpoints appear: a 12B at 4-bit quantisation-aware training, the
same QAT weights at BF16, the plain BF16 instruct model, and a 31B at 4-bit.
They exist to separate precision, QAT training and capacity as explanations for
character-level error — see finding 8.

### What each system needs to run

The prompted models only. Weights measured on disk, not derived from parameter
counts. Under vLLM the memory a process actually claims is
`gpu_memory_utilization` × the card regardless — what varies is how much of that
budget the weights leave for the KV cache, and the KV cache is what fails first.

| System | weights | 24 GB card | ran on |
|---|---|---|---|
| gemma-4-12B-it-qat-w4a16-ct | **9.7 GB** | one replica per card | 2×L4 |
| InternVL3.5-8B | 16 GB | one replica per card | L4 |
| gemma-4-31B-it-qat-w4a16-ct | **22 GB** | **no — needs 2 cards** | L40S; 2×L4 tp=2 |
| gemma-4-12B-it-qat-q4_0-unquantized | 23 GB | no | L40S 48 GB |
| gemma-4-12B-it | 23 GB | no | L40S 48 GB |
| *gemma-4-31B-it (BF16)* | *59 GB* | *no* | *fits nothing available* |

Docling and MinerU are absent because they were run on Apple Silicon against
unified memory, which is not a GPU sizing figure and does not transfer to the
cluster this table exists to inform.

Two consequences worth reading off it.

**The 31B at 4 bits is 22 GB, not the ~15 GB its parameter count suggests.** The
embedding and vision tensors stay at higher precision and Gemma's vocabulary is
large. On a 23,034 MiB card that leaves nothing for a KV cache: the single-card
probe peaked at 21,908 MiB and died allocating it. Sharded across two L4s it
loads and runs, holding ~20.4 GB per card.

So on 24 GB hardware **the best system in this study is a two-card-per-request
model**, and that is what makes the throughput question worth asking.

**The 31B has no BF16 counterpart that fits anything here**, at 59 GB against a
48 GB card. That is why finding 8's precision comparison is done at 12B and its
capacity comparison at 4 bits: those are the two experiments the hardware
permits.

### What each system costs to run

Measured on the 2×L4, the hardware production has. Nothing is timed on the
L40S — a rate from a card you cannot deploy predicts nothing — which also
excludes the two BF16 12B checkpoints, since they need 48 GB.

Each system is timed **the way its weights allow it to be served**, because
that is what a box delivers:

- weights fit a card → **dp=2**, two independent replicas, one per card
- weights do not fit → **tp=2**, one engine sharded across both

Timing one card and doubling it would assume data parallelism scales linearly.
It roughly does, and roughly is not measured: replicas contend for host CPU
during image preprocessing and for PCIe.

| System | deployment | images/min | per card | usable amounts |
|---|---|---|---|---|
| **gemma-4-12B-it-qat-w4a16-ct** | dp=2 | **13.79** | **6.90** | 88.3% |
| InternVL3.5-8B | dp=2 | 5.47 | 2.74 | 85.3% |
| **gemma-4-31B-it-qat-w4a16-ct** | tp=2 | 5.33 | 2.67 | **99.0%** |

*165 pages, both cards. Usable amounts are bank statements only, where the hard
tables are.*

**InternVL3.5-8B is dominated on both axes.** It is 2.5× slower than the 12B and
delivers fewer usable amounts than it; and it runs at the 31B's speed while
delivering 13.7 points fewer. There is no workload of these three for which it
is the right choice — which is not visible from accuracy alone, where its digit
reading looked like a reasonable middle option.

That leaves a two-way decision, and both of its numbers are now measured rather
than one measured and one assumed:

| | gemma-4-12B | gemma-4-31B |
|---|---|---|
| cards per request | 1 | 2 |
| images/min per card | **6.90** | 2.67 |
| usable amounts | 88.3% | **99.0%** |
| wrong amounts per statement | about 1 in 9 | about 1 in 100 |

**2.6× the hardware for roughly an order of magnitude fewer wrong amounts.**
Which side of that is right is a costing question rather than a measurement one,
and it depends on what a wrong amount costs to catch downstream. What the
benchmark can say is that the trade is real, roughly linear in cards, and not
the trade the CER table implies — on median normalised CER the two look 0.0081
against 0.0000, which reads as a difference of no consequence.

---

## Results

Per-page **medians**. A handful of catastrophic pages distort the means badly,
and quoting means alone misdescribes every system here.

| System | median nCER | median sCER | **gap** | mean nCER | told? |
|---|---|---|---|---|---|
| **gemma 31B 4-bit** | **0.0000** | 0.0161 | **+0.0134** | **0.0016** | yes |
| gemma 12B BF16 | 0.0026 | 0.0317 | +0.0139 | 0.0082 | yes |
| gemma 12B BF16 QAT | 0.0050 | 0.0359 | +0.0138 | 0.0101 | yes |
| gemma 12B 4-bit | 0.0081 | 0.0382 | +0.0136 | 0.0178 | yes |
| MinerU | 0.0404 | 0.4277 | **+0.3917** | 0.0520 | no |
| InternVL3.5-8B | 0.0420 | 0.2119 | **+0.1630** | 0.0743 | yes |
| Docling | 0.4145 | 0.6120 | +0.0581 | 1.6457 | no |

**The 31B's median normalised CER is 0.0000** — more than half its pages are
transcribed without a single character wrong once formatting is set aside.

**The convention cost is flat across all four gemma checkpoints** — +0.0134,
+0.0138, +0.0139, +0.0136 — while their reading accuracy varies fivefold.
Following a house style is not a fidelity-limited skill: precision and capacity
move what a model *reads* and leave what it *formats* untouched. That
dissociation is easy to miss in a table and obvious in figure 1.

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

## Reading the tables

Across all 165 pages:

| System | rows aligned | fragments | width breaks | cell accuracy | content recall |
|---|---|---|---|---|---|
| **gemma 31B 4-bit** | **1919/2005 (95.7%)** | **0** | 6 | 0.997 | **0.987** |
| gemma 12B BF16 | 1528/2005 (76.2%) | 4 | 66 | 0.996 | 0.925 |
| gemma 12B BF16 QAT | 1486/2005 (74.1%) | **0** | **3** | 0.999 | 0.917 |
| gemma 12B 4-bit | 1359/2005 (67.8%) | 9 | 15 | 0.995 | 0.888 |
| InternVL3.5-8B | 1343/2005 (67.0%) | **0** | 8 | 0.957 | 0.867 |
| MinerU | 1234/2005 (61.5%) | 292 | 156 | 1.000 | 0.888 |
| Docling | 283/2005 (14.1%) | 120 | 13 | 0.850 | 0.273 |

Amounts filed under the correct heading:

| System | misfiled | **rate** | wrong column count | **correct width** |
|---|---|---|---|---|
| **gemma 31B 4-bit** | 20/2503 | **0.8%** | 3 | 162/165 |
| gemma 12B BF16 | 116/2503 | 4.6% | 8 | 157/165 |
| gemma 12B BF16 QAT | 131/2503 | 5.2% | **0** | **165/165** |
| gemma 12B 4-bit | 242/2503 | 9.7% | 2 | 163/165 |
| MinerU | 325/2503 | 13.0% | 56 | 109/165 |
| InternVL3.5-8B | 330/2503 | 13.2% | 13 | 152/165 |
| Docling | 1929/2503 | 77.1% | 117 | 48/165 |

**Column-count stability separates the families.** Every gemma checkpoint gets
the table width right on at least 157 of 165 documents; MinerU manages 109 and
Docling 48. A system that invents or drops a column shifts every amount on the
page, so this one property dominates everything downstream of it.

By document type — and read it this way, because bank statements are the only
genuinely hard tables here:

| misfiled | bank statements | invoices | receipts |
|---|---|---|---|
| **gemma 31B 4-bit** | **1.0%** | **0.0%** | **0.0%** |
| gemma 12B BF16 | 5.6% | 1.0% | **0.0%** |
| gemma 12B 4-bit | 11.7% | 2.3% | **0.0%** |
| InternVL3.5-8B | 14.7% | 1.0% | 16.8% |
| MinerU | 7.0% | **0.0%** | 100% |
| Docling | 83.1% | 24.0% | 100% |

The parsers' 100% on receipts is absence, not misplacement — neither emits a
table there at all. Every gemma checkpoint misfiles **not one** of the 184
receipt amounts.

### Bank-transaction tables

The hard case, and worth separating. 55 statements, 1,612 rows, 2,507 amounts.

| System | rows aligned | fragments | width breaks | cell acc | misfiled | **width ok** |
|---|---|---|---|---|---|---|
| **gemma 31B 4-bit** | **1527/1612 (94.7%)** | **0** | **0** | 0.997 | **1.0%** | **55/55** |
| gemma 12B BF16 | 1137/1612 (70.5%) | **0** | **0** | 0.996 | 5.6% | **55/55** |
| gemma 12B BF16 QAT | 1095/1612 (67.9%) | **0** | **0** | 0.999 | 6.4% | **55/55** |
| MinerU | 1025/1612 (63.6%) | 292 | 156 | 1.000 | 7.0% | 54/55 |
| InternVL3.5-8B | 987/1612 (61.2%) | **0** | 8 | 0.947 | 14.7% | 51/55 |
| gemma 12B 4-bit | 971/1612 (60.2%) | **0** | **0** | 0.994 | 11.7% | **55/55** |
| Docling | 134/1612 (8.3%) | 120 | 13 | 0.704 | 83.1% | 6/55 |

**The 31B is the first system to read these tables properly** — 94.7% of rows
recovered against everyone else's 60–70%, zero structure broken, and 1.0% of
amounts misfiled, an order of magnitude below the next best.

**MinerU aligns rows by breaking them.** It recovers more content than any 12B —
content recall 0.928 — but does it by splitting 292 rows in two and emitting 156
of the wrong width. Every gemma checkpoint breaks structure **zero times in
1,612 rows**. Those are different things to want: MinerU recovers more of the
text, gemma's output is trustworthy row by row.

**The alignment ceiling was a digit ceiling in disguise.** For every system
except the 31B, row recovery sits near 60–70%, and the cause is not
segmentation. Rows are matched by their content, so one misread character fails
the row it sits in — and where the convention carries a group's date onto every
row beneath it, one wrong date fails the whole group. Two statements show both
mechanisms:

- one lost 27 rows to a date read as `01/03/2024` instead of `01/01/2024`;
- another lost 24 rows to a balance read as `$15,571.32` instead of
  `$15,971.32`, after which **every subsequent balance was recomputed from the
  wrong figure**, `$16,073.50` becoming `$15,469.14`. The model is not reading
  that column; it is doing arithmetic down it.

That second failure is invisible on any single row, invisible to an extraction
task pulling one balance, and unmistakable only when the whole column is scored
at once.

The 31B, which misreads five amounts in 2,507, recovers 94.7% of rows. Fix the
digits and the structural ceiling lifts with them.

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

### 7. Getting the digits right is a separate skill, and CER cannot see it

Character error rate weighs a wrong digit in a total exactly as it weighs a typo
in a merchant's name. On a financial document those are not the same error, so
amounts are scored on their own — extracted from the raw text, which makes this
the only measure here that compares a system emitting HTML tables and one
emitting pipe tables on identical terms.

**Read the next table carefully: it measures the digits and nothing else.**
Amounts are pulled out of the raw text with a pattern match, so nothing here
knows whether a figure ended up in the right column, the right row, or in a
table at all. A system could score 100% on it and file every amount under the
wrong heading. `recall` is the share of the page's amounts the system
reproduced; `precision` is the share of what it emitted that the page actually
shows.

| System | digit recall | digit precision | misread | dropped | invented |
|---|---|---|---|---|---|
| MinerU | **100.0%** | **100.0%** | 0 | 1 | 1 |
| gemma 31B 4-bit | 99.9% | 99.9% | 3 | 2 | 2 |
| InternVL3.5-8B | 97.2% | 99.8% | 2 | 91 | 3 |
| gemma 12B 4-bit | 92.5% | 92.9% | 106 | 148 | 132 |
| Docling | 37.9% | — | 1 | 2093 | 243 |

*All 165 pages, 3,371 amounts.*

The ordering is not the CER ordering: MinerU reads every digit correctly while
paying the largest convention penalty of any system, and the 12B gemma — first
on CER, first on table structure — is last of the four real readers.

The two failure modes are reported apart rather than summed, because they call
for different remedies. **InternVL's deficit is almost entirely omission**: 91
dropped against 2 misread, and 99.8% of what it does emit is right. **The 12B's
is corruption**: 106 misread and 132 invented, so roughly one amount in fourteen
that it prints is not on the page at all.

### But digits alone are not the deployment number

An amount read perfectly and filed under the wrong heading is as wrong as one
misread — downstream, neither can be told from the other. The figure a consumer
can act on is **usable**: present, with the right value, under the right
heading. That is `100% − misfiled`, since column integrity counts an amount as
misplaced when it is absent from its column for *any* reason.

Ranking by it reverses two systems:

| System | digit recall | cell accuracy | width correct | **usable** |
|---|---|---|---|---|
| gemma 31B 4-bit | 99.8% | 0.996 | 55/55 | **99.0%** |
| MinerU | **100.0%** | 1.000 | 54/55 | 93.0% |
| gemma 12B 4-bit | 90.3% | 0.994 | **55/55** | **88.3%** |
| InternVL3.5-8B | **96.9%** | 0.947 | 51/55 | **85.3%** |

*Bank statements, 2,507 amounts.*

**InternVL reads 6.6 points more digits correctly than the 12B gemma and
delivers 3 points fewer usable amounts.** It gets the numbers right and puts
them in the wrong place — 4 statements with the wrong column count against
gemma's none, and cell accuracy 0.947 against 0.994. MinerU loses 7 points the
same way.

So the three skills dissociate cleanly, and quoting any one of them alone
misdescribes a system:

- **gemma 12B**: structure right, digits wrong
- **InternVL3.5-8B**: digits right, structure wrong
- **gemma 31B**: both right, and an order of magnitude ahead on usable amounts
- **MinerU**: perfect digits, 7% misplaced, and no table at all on receipts

One mechanism deserves naming. On several statements a single misread amount is
followed by every subsequent balance being **recomputed from the wrong figure** —
`353.68` read as `355.68`, then each balance below it off by exactly `−2.00`. The
model is not transcribing the running-balance column; it is doing arithmetic down
it. That is invisible on any single row, invisible to an extraction task pulling
one balance, and unmistakable only when the whole column is scored at once.

All of this was measurable only because these figures are authored per page.
Where a layout hardcodes one number repeated across seven pages, one error
repeats seven times and a model can score well by memorising it.

### 8. The digit deficit is capacity, not quantisation

The obvious explanation for the 12B's 90.3% was its 4-bit quantisation —
quantisation costs character fidelity first, so the story tells itself. It is
wrong, and only a control could show that.

The 12B checkpoint differs from a plain BF16 model in two ways at once, so three
runs were needed to separate them. All decoding and engine settings were held
identical; only the checkpoint varied.

| comparison | isolates | effect |
|---|---|---|
| 12B 4-bit → 12B BF16 QAT | precision, QAT held fixed | **+4.1 pts** |
| 12B BF16 QAT → 12B BF16 | QAT training, both BF16 | +0.1 pts |
| **12B 4-bit → 31B 4-bit** | **capacity, quantisation held fixed** | **+9.5 pts** |

**At the same 4 bits, the 31B reaches 99.8%.** So 4-bit was never the binding
constraint. Precision buys a real 4 points at 12B, but capacity subsumes it: the
31B at 4 bits reads amounts *more* accurately than the 12B at BF16. QAT training
contributes nothing measurable either way.

The causal direction matters. "Quantisation costs digit fidelity" describes a
model running out of capacity, with the bit-width as a symptom. Had the control
not run, that sentence would have gone into this document as a finding.

**The 31B is also the first system to read these tables properly**, and the two
results are connected. It aligns 94.7% of statement rows against everyone else's
60–70%, with zero fragments and 1.0% of amounts misfiled — an order of magnitude
below the next best. Row alignment matches rows by their content, so a single
wrong digit fails the row it sits in; where a date is carried onto every row of a
group, one wrong date fails the whole group. Every other system's alignment
ceiling was substantially a digit-fidelity ceiling wearing a structural costume.
Remove the digit errors and it lifts.

It also beats MinerU where MinerU had been unbeatable. On the digits alone
MinerU is ahead, 100.0% to 99.8% — but it files one figure in fourteen under the
wrong heading, so **93.0% of its amounts are usable against the 31B's 99.0%**.
Reading every digit correctly is not the same as delivering a usable number, and
the 31B is the first system here that does both.

One qualification. The 31B is a differently trained model, not a scaled 12B, so
"capacity" is shorthand for everything that differs between the two sizes. It is
a strong test rather than the clean isolation the precision comparison was.

---

## Limits

**Coverage.** All four gemma checkpoints produced all 165 pages. InternVL
produced 164; the one failure is scored as a total failure rather than dropped,
so every system is averaged over the same 165 transcripts. Docling produced 2
empty pages, scored the same way.

**The four gemma checkpoints differ only in the checkpoint.** Every decoding and
engine setting is identical across them, including the vision budget — varying
that would confound precision or capacity with what the model can see. All seven
systems were scored against the same corpus, and the analysis refuses reports
that were not.

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

**"Capacity" is shorthand.** The 31B is a differently trained model, not a
scaled 12B, so finding 8 attributes the digit deficit to everything that differs
between the two sizes. The precision comparison is a clean isolation — same QAT
weights, only the bit-width varies — and the capacity one is a strong test
rather than a controlled experiment.

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
capable prompted model adopts them for **+0.013** — the gap between its strict
and normalised CER, which is by construction what formatting costs it. An
unpromptable parser pays +0.392 for the same corpus. The style is not an
arbitrary imposition.

That figure is remarkably stable: **+0.0134, +0.0136, +0.0138, +0.0139** across
four gemma checkpoints whose reading accuracy varies fivefold. Precision and
capacity move what a model reads and leave what it formats untouched. Following
a house style is not a fidelity-limited skill.

**A steerability result with a sharp edge.** The 12B adopts every convention put
to it; an 8B from another family adopts one of three and is destabilised by the
technique that helps the 12B most. Prompt engineering results do not transfer
across model scales, and reporting one without the other overstates both.

**Three metrics that normalised CER cannot replace.** Column integrity, because
a page can be character-perfect and financially meaningless. Table structure,
because column integrity presupposes the rows. Numeric fidelity, because CER
prices a wrong digit in a total at a typo in a merchant's name — and it is the
only measure here that is convention-blind, so it compares a system emitting
HTML tables with one emitting pipe tables on identical terms.

Between them they produced the pass's most counter-intuitive result, and it only
appears when the metrics are kept apart. **MinerU reads every one of the 2,507
bank-statement digits correctly and still delivers fewer usable amounts than the
31B**, because it misfiles 7% of them. **InternVL3.5-8B reads 6.6 points more
digits correctly than the 12B gemma and delivers 3 points fewer usable amounts**,
for the same reason.

Reading a table, matching a house style and getting the digits right are three
separable skills, and no single one of them ranks these systems correctly for
anyone who cares about the money. The figure that does is **usable** — present,
right value, right heading — and it is the one to quote.

**One observation that only a whole-page metric could surface.** A model that
misreads a running balance then **recomputes every balance below it** from the
wrong figure. It is arithmetically reconstructing the column rather than
transcribing it — invisible on any single row, invisible to an extraction task
pulling one balance, and unmistakable when the whole column is scored at once.

**A cause established by control rather than assumed.** The 12B's digit errors
looked like a quantisation cost, and quantisation costing character fidelity
first is exactly what theory predicts. Three controlled runs say otherwise:
precision buys 4 points, quantisation-aware training buys none, and **capacity
buys 9.5** — the 31B at the same 4 bits reads amounts more accurately than the
12B at BF16. The plausible explanation was the wrong one, and only running the
control distinguished them.

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

# Can a prompted VLM be steered to a house style?

**Calibration pass, August 2026.** Six document-parsing systems over 165
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
could.** Median normalised CER **0.0000** — **128 of its 165 pages** are
character-perfect once formatting is set aside — with **99.0%** of
bank-statement amounts **usable** (right value, right heading), **94.7%** of
statement rows recovered, and **1.0%** filed under the wrong heading. It is the
first system here that reads the tables, gets the digits right, and follows the
house style at once.

Six things qualify that, and they are the substance of this document:

1. **Steerability is a property of the model, not the prompt.** The same three
   conventions, identically worded, moved the 12B and barely moved an 8B from
   another family. A prompt improvement is a property of the *pair*.
2. **A worked example does something a stated rule cannot.** Held to a
   controlled A/B, a rule alone stopped a model emitting the wrong characters
   but not improvising a replacement; the example supplied the behaviour the
   prohibition left undefined.
3. **Normalised character error rate ranks systems adequately and cannot find a
   bad page.** Across the six systems it tracks usable-amount rate closely
   (Spearman ρ = 0.943). *Within* a system it is uncorrelated with whether a
   page's numbers survive — ρ = 0.01 for two of them, and **−0.17** for the best,
   whose worst page loses 11% of its amounts at a CER of 0.0000. **Fourteen
   pages lose every amount to the wrong column, and ten of them score below
   0.11** — the worst being 50 amounts on one statement at 0.075.
   Ranking systems is something you do once; deciding which pages a human must
   check is something you do on every batch.
4. **Ground truth and prompt are a matched pair, and both are components under
   test.** Two defects in this corpus were found only by asking whether it
   treated the same visual device the same way everywhere.
5. **A wrong number and a missing number are not the same failure.** Ranked by
   what reaches a consumer as plausible-but-wrong, the 31B is at 2.0% and MinerU
   at 37.2% — and MinerU misreads *no* digits at all, losing 596 correctly-read
   amounts to rows with no date. "Which system fabricates numbers?" and "which
   system delivers wrong data?" have different answers.
6. **Robustness to a bad scan is a property of the system, not of scanning.**
   Over two degradation ladders and all four systems, the 31B leads at every
   tier and the margin *widens* as intake worsens — 17.7 points clear of the
   next system at scan-light, 33.7 clear at photo-heavy. A clean benchmark
   cannot see that. And what degradation destroys is **attribution, not
   placement**: the 31B keeps 89–99% of amounts under the correct heading all
   the way down, while the share that can be tied to a transaction falls from
   97.7% to 44.6%, because the *date cell* is what fails. Measured as placement,
   scanning looked free; measured as what a consumer can act on, `scan-heavy`
   costs 9.1 points.

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

Pages are pristine renders. A parallel set of **degraded** corpora derives from
them — two intake channels, scanner and phone camera, three declared severities
each — so the benchmark measures clean input by default and scanned input on
demand. The degraded images alter legibility only; their transcripts are copied
byte for byte from the clean corpus, so a score difference is attributable to
image quality alone. See finding 10.

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
- **content recall** — of every truth cell, how many appear anywhere in the
  prediction. Catches the opposite failure, a system that aligns few rows but
  did read the text.

There is deliberately **no per-cell accuracy** among these. An earlier draft
reported one — cells matching within aligned rows — and it was withdrawn as
near-tautological. Rows are matched by their content, so every aligned pair has
matching cells by construction: a misread digit does not lower the accuracy, it
drops the row out of the alignment. The figure sat at 0.95–1.00 for every system
regardless of how well it read, and MinerU's 1.000 was read as "transcribes
cells perfectly" when it meant "the rows it got right, it got right". **rows
aligned** is where a misread cell actually registers.

Always read per document type. Bank statements are the only genuinely difficult
tables here; averaging them with near-saturated invoices hides everything.

---

## The systems

Split by one property: **whether they can be told the convention.**

| Group | Systems | Reads a prompt? | Measures |
|---|---|---|---|
| **Prompted** | four gemma-4 checkpoints, InternVL3.5-8B | yes | is the convention *communicable*? |
| **Unprompted** | MinerU | no | is it *idiomatic Markdown at all*? |

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

MinerU is 2.2 GB and fits a card many times over. Every system here is measured
on the L4s, so nothing in this document mixes hardware.

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
| **MinerU2.5-Pro-1.2B** | dp=2 | **≥ 22.20** | **≥ 11.10** | 90.3% |
| gemma-4-12B-it-qat-w4a16-ct | dp=2 | 13.79 | 6.90 | 88.3% |
| InternVL3.5-8B | dp=2 | 5.47 | 2.74 | 85.3% |
| **gemma-4-31B-it-qat-w4a16-ct** | tp=2 | 5.33 | 2.67 | **99.0%** |

*165 pages, both cards. Usable amounts are bank statements only, where the hard
tables are.*

**MinerU's rate is a floor, marked `≥`.** It is invoked as a subprocess per
chunk and reloads the model each time, so its clock is wall clock with load
included, while the three engine-driven systems time generation only. Its true
rate is higher by an unmeasured margin. This is the one number in the study
measured on a different basis from its neighbours, so it is labelled everywhere
it appears rather than quietly compared.

**InternVL3.5-8B is dominated on every axis.** It is 2.5× slower than the 12B
and delivers fewer usable amounts than it; and it runs at the 31B's speed while
delivering 13.7 points fewer. There is no workload here for which it is the
right choice — which is not visible from accuracy alone, where its digit reading
looked like a reasonable middle option.

**And so, on these tables, is the 12B gemma.** MinerU is at least 1.6× faster
*and* 2 points more usable. That is a genuine reversal: the 12B was on the
frontier until MinerU had a comparable rate to be judged by, and prompted-VLM
convention compliance — the thing the 12B is good at and MinerU cannot do at
all — turns out not to be what buys usable amounts.

Two qualifications, and they are load-bearing rather than hedges:

- **The y axis is bank statements.** MinerU emits no table at all on receipts,
  where every gemma checkpoint misfiles not one amount. Its position here is its
  best document type, not its average.
- **It pays +0.387 strict CER** for a Markdown dialect no prompt can change,
  against the 12B's +0.030. If the consumer is a downstream parser reading
  house-style Markdown rather than a person reading amounts, that cost is real
  and this chart does not show it.

That leaves a frontier of two, and every number on it is now measured:

| | MinerU2.5-Pro | gemma-4-31B |
|---|---|---|
| cards per request | 1 | 2 |
| images/min per card | **≥ 11.10** | 2.67 |
| usable amounts | 90.3% | **99.0%** |
| wrong amounts per statement | about 1 in 10 | about 1 in 100 |
| can be told the convention | no | **yes** |
| produces a table on receipts | no | **yes** |

**At least 4× the throughput per card, against roughly an order of magnitude
fewer wrong amounts.** Which side is right is a costing question rather than a
measurement one, and it turns on what a wrong amount costs to catch downstream
and on whether the workload is bank statements alone.

#### Every 31B figure in this document is the 2xL4 tp=2 deployment

Production has no L40S, so the number quoted must be the one the deployed
configuration produces. The single-card run on the 48 GB L40S is kept only as
the control that says the choice costs nothing.

| bank statements | tp=1 (L40S) | **tp=2 (2xL4)** |
|---|---|---|
| amounts placed | 99.01% | **99.01%** |
| amounts attributable | 97.71% | **97.71%** |
| amounts misfiled | 20 | **20** |
| rows aligned | 94.73% | **96.59%** |

**Sharding costs nothing measurable and helps the one metric it moves.** 31 of
165 pages differ by a handful of bytes -- tensor parallelism all-reduces partial
sums in a different order and float addition is not associative -- but no
aggregate the deployment rests on changes, and row alignment is 1.9 points
better under tp=2.

That was worth measuring rather than assuming. Two version differences in this
study did change results: an OpenCV bump altered 2 of 9 degraded images, and
MinerU on Apple Silicon differs from MinerU on CUDA on 41 of 165 pages.

#### The 31B was re-measured on the configuration being proposed

Its accuracy figures came from a run at `tensor_parallel_size: 1` on a 48 GB
card, while the throughput came from a sharded pair on the 2×L4. Production is
24 GB cards, so tp=2 is the deployment — and accuracy had never been measured on
it. The two system entries differ only in that setting, so re-running isolated
sharding and nothing else.

**31 of 165 pages differ**, almost all by one to five bytes. That is expected:
tensor parallelism all-reduces partial sums in a different order, floating-point
addition is not associative, and a near-tied token occasionally flips. What
matters is whether the aggregates move.

| bank statements | tp=1 | tp=2 (proposed) |
|---|---|---|
| usable amounts | 1991/2011 (**99.0%**) | 1991/2011 (**99.0%**) |
| amounts misfiled | 20 | 20 |
| digit recall | 99.80% | 99.80% |
| fragments + width breaks | 0 | 0 |
| correct column count | 55/55 | 55/55 |
| **rows aligned** | 1527/1612 (94.7%) | **1557/1612 (96.6%)** |

**Every figure the deployment case rests on is identical, and row alignment is
30 rows better.** Not a contradiction: rows are matched by content, so one
recovered character can restore a whole group — the same mechanism finding 8
describes, running forwards.

Two figures move slightly the other way: 124 pages are character-perfect against
128, while the mean normalised CER improves from 0.0016 to 0.0014. Both are the
same handful of flipped tokens, and neither touches an amount.

The tables in this document continue to quote the tp=1 run, so that all six
systems are reported from one vintage. The deployment case quotes tp=2, because
that is what would be deployed. They agree.


CER does show this difference where it matters — 0.0202 against 0.0004 on bank
statements, a fifty-fold ratio. It is over all 165 pages that it flattens, to
0.0081 against 0.0000, because two thirds of the corpus is easy. Which median
gets quoted decides whether the difference looks decisive or negligible, and
finding 9 is about the more serious limitation underneath that.

---

## Results

Per-page **medians**. A handful of catastrophic pages distort the means badly,
and quoting means alone misdescribes every system here.

| System | median nCER | median sCER | **median gap** | mean nCER | told? |
|---|---|---|---|---|---|
| **gemma 31B 4-bit** | **0.0000** | 0.0161 | **+0.0134** | **0.0016** | yes |
| gemma 12B 4-bit | 0.0081 | 0.0382 | +0.0136 | 0.0178 | yes |
| MinerU | 0.0403 | 0.4277 | **+0.3917** | 0.0495 | no |
| InternVL3.5-8B | 0.0420 | 0.2119 | **+0.1630** | 0.0743 | yes |

**The gap column is the median of each page's own gap — deliberately not the
difference of the two medians beside it.** A median of differences is not the
difference of medians, and they answer different questions: subtracting the
columns says how two summary statistics of the corpus relate, while the median
gap says what the convention costs *on a page*, which is the quantity this
study is about. The two coincide only if the same pages sit at both medians,
and here they do not — the 31B's normalised median is 0.0000, drawn from a
different set of pages than its strict median. Reading the column as column
minus column is an easy mistake to make, and it was made here.

**The 31B's median normalised CER is 0.0000** — **128 of 165 pages (78%)** are
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

---

## Reading the tables

Across all 165 pages:

| System | rows aligned | fragments | width breaks | content recall |
|---|---|---|---|---|
| **gemma 31B 4-bit** | **1919/2005 (95.7%)** | **0** | 6 | **0.987** |
| gemma 12B 4-bit | 1359/2005 (67.8%) | 9 | 15 | 0.888 |
| InternVL3.5-8B | 1343/2005 (67.0%) | **0** | 8 | 0.867 |
| MinerU | 1251/2005 (62.4%) | 276 | 152 | 0.881 |

Amounts filed under the correct heading:

| System | misfiled | **rate** | wrong column count | **correct width** |
|---|---|---|---|---|
| **gemma 31B 4-bit** | 20/2503 | **0.8%** | 3 | 162/165 |
| gemma 12B 4-bit | 242/2503 | 9.7% | 2 | 163/165 |
| MinerU | 379/2503 | 15.1% | 56 | 109/165 |
| InternVL3.5-8B | 330/2503 | 13.2% | 13 | 152/165 |

**Column-count stability separates the families.** Every gemma checkpoint gets
the table width right on at least 157 of 165 documents; MinerU manages 109. A
system that invents or drops a column shifts every amount on the page, so this
one property dominates everything downstream of it.

By document type — and read it this way, because bank statements are the only
genuinely hard tables here:

| misfiled | bank statements | invoices | receipts |
|---|---|---|---|
| **gemma 31B 4-bit** | **1.0%** | **0.0%** | **0.0%** |
| gemma 12B 4-bit | 11.7% | 2.3% | **0.0%** |
| InternVL3.5-8B | 14.7% | 1.0% | 16.8% |
| MinerU | 9.7% | **0.0%** | 100% |

MinerU's 100% on receipts is absence, not misplacement — it emits no table
there at all. Every gemma checkpoint misfiles **not one** of the 184
receipt amounts.

### Bank-transaction tables

The hard case, and worth separating. 55 statements, 1,612 rows, 2,507 amounts.

| System | rows aligned | fragments | width breaks | misfiled | **width ok** |
|---|---|---|---|---|---|
| **gemma 31B 4-bit** | **1527/1612 (94.7%)** | **0** | **0** | **1.0%** | **55/55** |
| MinerU | 1042/1612 (64.6%) | 276 | 152 | 9.7% | 54/55 |
| InternVL3.5-8B | 987/1612 (61.2%) | **0** | 8 | 14.7% | 51/55 |
| gemma 12B 4-bit | 971/1612 (60.2%) | **0** | **0** | 11.7% | **55/55** |

**The 31B is the first system to read these tables properly** — 94.7% of rows
recovered against everyone else's 60–70%, zero structure broken, and 1.0% of
amounts misfiled, an order of magnitude below the next best.

**MinerU breaks structure without being paid for it.** It splits 276 rows in two
and emits 152 of the wrong width, while every gemma checkpoint breaks structure
**zero times in 1,612 rows** — and it does not recover more text for the damage.
Its content recall is 0.881, below all three 12B checkpoints (0.888, 0.917,
0.925), and its 64.6% of statement rows aligned sits between them rather than
above.

That is a **correction to an earlier reading of this table**, and the cause is
worth stating: the run that supported it was MinerU on Apple Silicon via MLX,
where content recall was 0.928 and it did lead every 12B. This study now quotes
the vLLM run on the L4s throughout, because that is the hardware production has.
Same weights, different inference stack, 41 of 165 pages different — and the
conclusion about MinerU's structural trade-off reverses. **Do not carry a
finding across a runtime change on the grounds that the checkpoint is the
same.**

What survives is narrower and still notable: MinerU **misreads** none of the
3,371 amounts. Every figure it misses is one it never emitted.

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

The 31B, which misses five amounts in 2,507 — three misread, two never emitted — recovers 94.7% of rows. Fix the
digits and the structural ceiling lifts with them.

---

## Findings

The first four turn on the exact wording of the prompt, which is reproduced in
full in the [appendix](#appendix-the-prompt) rather than quoted in fragments.

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
Transaction`. InternVL reads it as **two columns** on three of them. Every data
row inherits the extra cell, so every amount sits one column right of the
heading naming it. On a bank statement that is money out reported as money in.

| Page | amounts misfiled | normalised CER | column count |
|---|---|---|---|
| CASE033 | **38 of 38** | 0.0810 | wrong |
| CASE012 | **39 of 39** | 0.0924 | wrong |
| CASE005 | **39 of 39** | 0.1016 | wrong |
| CASE025 | 1 of 38 | 0.0977 | **right** |

Three of those pages have **every single amount** under the wrong heading and
pay 8–10% character error for it — for output that is entirely worthless.
Normalisation strips pipes as table marks, discarding exactly the delimiters
that encode the answer, so a total structural failure registers as a mild
reading day.

**CASE025 is the control the corpus supplied by accident.** Same wrapped header,
same model, same prompt — and it gets the column count right and loses one
amount instead of all of them. Its CER, 0.0977, sits *between* the three
catastrophes. So the metric cannot separate a page where the structure held from
one where it collapsed, on pages that differ in nothing else.

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
| MinerU | 99.5% | 99.8% | **0** | 16 | 7 |
| gemma 31B 4-bit | 99.9% | 99.9% | 3 | 2 | 2 |
| InternVL3.5-8B | 97.2% | 99.8% | 2 | 91 | 3 |
| gemma 12B 4-bit | 92.5% | 92.9% | 106 | 148 | 132 |

*All 165 pages, 3,371 amounts.*

The ordering is not the CER ordering. MinerU **misreads not one** of the 3,371
amounts — every miss is an omission — while paying the largest convention penalty
of any system; and the 12B gemma, first on CER and on table structure, is last.

The two failure modes are reported apart rather than summed, because they call
for different remedies. **InternVL's deficit is almost entirely omission**: 91
dropped against 2 misread, and 99.8% of what it does emit is right. **The 12B's
is corruption**: 106 misread and 132 invented, so roughly one amount in fourteen
that it prints is not on the page at all.

### An amount you cannot attribute to a row is not usable

`usable` counted an amount as good when it sat under the right heading in *any*
row. That is the right question for placement and the wrong one for usability: a
consumer reading rows as records cannot act on a figure it cannot tie to a
transaction.

MinerU forces the distinction. It emits a group's date as a row of its own and
the transaction beneath it with an empty date cell, so on `CASE015` it scores
**56 of 56 amounts placed and 0 of 56 attributable** — every figure under the
correct heading, not one of them attached to a date.

**attributable** = right value, right column, and in a row carrying the date
that identifies it. Bank statements, 2,011 amounts:

Only deployable systems are listed. The two BF16 12B checkpoints need 48 GB
and fit no card production has; they appear in finding 8, where they are the
precision control that the argument rests on, and nowhere that reads as a
shortlist.

| System | placed | **attributable** | cost |
|---|---|---|---|
| **gemma 31B 4-bit** | 99.0% | **97.7%** | 1.3 |
| gemma 12B 4-bit | 88.3% | 85.3% | 3.0 |
| InternVL3.5-8B | 85.3% | 73.0% | 12.3 |
| MinerU | 90.3% | **60.7%** | **29.6** |

**The ordering inverts.** MinerU beats both the 12B and InternVL on placement
and is *last* once the amount must identify itself. The measure flattered
exactly the systems whose failure is row segmentation.

**The cost column is a row-segmentation measure in its own right**, and a
cheaper one than table structure: it is the share of amounts a system files
correctly and then orphans. Every gemma pays 1–3 points; InternVL 12; MinerU 30.

The row key is the **date** cell and deliberately not the description as well.
Requiring both would fail an amount because a merchant name was misread, folding
reading accuracy into a placement measure — the conflation `read` and `placed`
exist to keep apart. Under the stricter form the figures are 93.3%, 59.5%,
57.0% and 55.0%; the ordering is unchanged.

Both are reported. `placed` remains the placement diagnostic; `attributable` is
what a consumer receives.

### But digits alone are not the deployment number

An amount read perfectly and filed under the wrong heading is as wrong as one
misread — downstream, neither can be told from the other. The figure a consumer
can act on is **usable**: present, with the right value, under the right
heading. That is `100% − misfiled`, since column integrity counts an amount as
misplaced when it is absent from its column for *any* reason.

Ranking by it reverses two systems:

| System | digit recall | rows aligned | width correct | **usable** |
|---|---|---|---|---|
| gemma 31B 4-bit | 99.8% | **94.7%** | 55/55 | **99.0%** |
| MinerU | **99.4%** | 64.6% | 54/55 | 90.3% |
| gemma 12B 4-bit | 90.3% | 60.2% | **55/55** | **88.3%** |
| InternVL3.5-8B | **96.9%** | 61.2% | 51/55 | **85.3%** |

*Bank statements, 2,507 amounts.*

**InternVL reads 6.6 points more digits correctly than the 12B gemma and
delivers 3 points fewer usable amounts.** It gets the numbers right and puts
them in the wrong place — 4 statements with the wrong column count against
gemma's none. MinerU loses 9 points the same way, from 152 width breaks.

So the three skills dissociate cleanly, and quoting any one of them alone
misdescribes a system:

- **gemma 12B**: structure right, digits wrong
- **InternVL3.5-8B**: digits right, structure wrong
- **gemma 31B**: both right, and an order of magnitude ahead on usable amounts
- **MinerU**: near-perfect digits, 10% misplaced, and no table at all on receipts

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

It also beats MinerU on placement, which is where MinerU had been strongest.
The two read digits almost identically — 99.8% against 99.4% — but MinerU files
one figure in eleven under the wrong heading, so **90.3% of its amounts are
usable against the 31B's 99.0%**. Reading the digits is not the same as
delivering a usable number, and the 31B is the first system here that does
both.

One qualification. The 31B is a differently trained model, not a scaled 12B, so
"capacity" is shorthand for everything that differs between the two sizes. It is
a strong test rather than the clean isolation the precision comparison was.

### 9. CER ranks systems adequately and cannot find a bad page

Every criticism of character error rate in this document has been mechanical:
it strips the pipes that encode column membership, it prices a wrong digit like
a typo. Those are arguments. This is the measurement.

**Across systems it works.** Median normalised CER against usable-amount rate,
over the six systems on bank statements: **Spearman ρ = 0.943** (p = 0.005), one
swap in six. As a way of ranking systems it is not misleading, and the case
against it should not be overstated.

**Within a system it is uncorrelated with whether a page is usable.**

| System | ρ, CER vs unusable amounts, per page | worst page |
|---|---|---|
| gemma 31B 4-bit | **−0.17** | 11% of amounts lost at CER **0.0000** |
| gemma 12B BF16 | **0.01** | 43% lost at CER 0.0653 |
| gemma 12B BF16 QAT | **0.01** | 46% lost at CER 0.0332 |
| gemma 12B 4-bit | 0.22 | 59% lost at CER 0.0472 |
| InternVL3.5-8B | 0.21 | **100% lost at CER 0.0810** |
| MinerU | 0.36 | **100% lost at CER 0.0755** |

*Worst page = the largest share of a page's amounts filed under the wrong
heading. Where several pages lose everything, the one with the lowest CER is
shown, because that is the case the metric is being asked about.*

ρ = 0.01 is noise. The best system's is **negative**: its worst page, losing 11%
of its amounts, scored a perfect 0.0000.

The most damaged pages in the corpus, by amounts lost:

| page | amounts lost | CER |
|---|---|---|
| MinerU CASE028 | **50 of 50** | 0.0755 |
| InternVL CASE005 | **39 of 39** | 0.1016 |
| InternVL CASE012 | **39 of 39** | 0.0924 |
| InternVL CASE033 | **38 of 38** | 0.0810 |

Every amount on the page in the wrong column, at a character error rate that
reads as a mildly imperfect transcription. Fourteen pages across the corpus lose
every amount this way; ten of them score below 0.11.

MinerU's 55 receipts also lose every amount, but for a different reason — it
emits no table there at all, so there is no column to be wrong about. Those are
excluded from the counts above, which would otherwise report 69.

**That split is the practical conclusion of this whole document.** Ranking
systems is something you do once. Deciding which pages a human must check is
something you do on every batch, and for that CER carries no signal at all —
ρ = 0.01 means a review queue ordered by CER is a review queue ordered at
random.

The blindness is structural rather than statistical. Normalisation removes
Markdown syntax; a table's pipes are Markdown syntax; the delimiters recording
which column an amount sits in are therefore discarded before anything is
compared. A page can be character-perfect and financially worthless, and CER
cannot represent the difference — not because it is noisy, but because the
information is gone by the time it computes.

So: report CER, because it ranks systems and it catches the catastrophes
(repetition loops, empty pages) that a field-level metric would miss entirely.
Do not triage on it, do not size a review budget from it, and do not read a
small CER as evidence that a page's numbers can be used.

---

### 10. Robustness to a bad scan is a property of the system, not of scanning

Every number above was measured on pristine renders. Production receives
scanned documents, so the figure the deployment rests on described a condition
production does not have.

The 55 statements were re-rendered through a declared degradation ladder and
read again — same prompts, same configurations. Only the images differ; the
transcripts are copied byte for byte, so any change is attributable to image
quality alone. Two ladders, because a flatbed scanner and a phone camera damage
a page differently and one severity scale would make the difference
unanswerable.

**All four systems, 2026-08-25.** Earlier versions of this finding covered the
31B and MinerU only, and said so. The two prompted models have now been run over
every tier, which is what makes the claim in the title testable rather than
asserted.

**The metric here is `attributable`, not `placed`.** An earlier version of this
section quoted 99.0% clean for the 31B, which is the share of amounts under the
right heading. This one quotes the share that are *also* in a row carrying its
identifying date — the figure the rest of this document and the deployment case
use, because a consumer reads rows as records. On clean input the two differ by
1.3 points and the choice hardly matters. **Under degradation it matters more
than anything else in this section**, and a reader comparing against an older
copy should expect every number to have moved down.

Usable — right value, right column, in an attributable row — over 55 statements:

| tier | gemma-4-31B | gemma-4-12B | InternVL3.5-8B | MinerU |
|---|---|---|---|---|
| *clean* | ***97.7%*** | ***85.3%*** | ***73.0%*** | ***60.7%*** |
| scan-light | **98.6%** | 80.9% | 70.4% | 61.4% |
| scan-moderate | **96.7%** | 88.1% | 72.5% | 47.3% |
| scan-heavy | **88.6%** | 67.1% | 59.5% | 42.8% |
| photo-light | **91.1%** | 61.8% | 69.4% | 56.6% |
| photo-moderate | **69.8%** | 39.5% | 42.5% | 45.4% |
| photo-heavy | **44.6%** | 10.9% | 18.0% | 31.4% |

**The 31B leads at every tier, and the margin widens as conditions worsen** —
17.7 points clear of the next system at scan-light, 33.7 clear at photo-heavy.
It is not merely the better system; it is differentially better where intake is
worst. That is an argument for it which a clean benchmark cannot make.

#### What degradation actually destroys is attribution, not placement

The single most useful result in this section comes from scoring both metrics
along the same ladder, for the 31B:

| tier | placed | attributable | gap |
|---|---|---|---|
| *clean* | 99.0% | 97.7% | 1.3 |
| scan-light | 99.4% | 98.6% | 0.8 |
| scan-moderate | 99.2% | 96.7% | 2.5 |
| scan-heavy | 98.6% | **88.6%** | **10.0** |
| photo-light | 98.8% | 91.1% | 7.7 |
| photo-moderate | 97.4% | **69.8%** | **27.5** |
| photo-heavy | 89.4% | **44.6%** | **44.8** |

**Placement barely moves; attribution collapses.** Across the whole ladder the
31B keeps 89–99% of amounts under the correct heading, and the share that can be
tied to a transaction falls from 97.7% to 44.6%. The mechanism is that the *date
cell* is what fails: the amount is read correctly and filed correctly, and the
row it sits in can no longer be identified, so a consumer reading rows as records
receives a number it cannot use.

This revises the previous headline. **"Scanning is free for the 31B" was true of
placement and false of attribution** — on the same images, at `scan-heavy`,
placement costs 0.4 points and attribution costs 9.1. The earlier claim was not
a measurement error; it was the wrong measurement, and it flattered the system
in exactly the direction a deployment decision is sensitive to.

#### Reading is not what fails — except for the 12B

At `photo-heavy` the 31B still reads **96.4%** of figures correctly while only
44.6% are usable. The gap is placement and attribution, not perception: the heavy
photo tier casts a shadow across part of the sheet and blurs it, so column
boundaries — which these layouts never draw — become unrecoverable while the
digits themselves survive. **Occlusion is not degradation.** No amount of model
capability recovers a column edge that is not in the image.

The 12B fails the opposite way, and this is finding 8 reappearing under load:

| tier | 31B digits | 12B digits | InternVL digits | MinerU digits |
|---|---|---|---|---|
| scan-light | **100.0%** | 88.6% | 96.4% | 99.1% |
| scan-heavy | **99.8%** | 78.5% | 95.4% | 97.0% |
| photo-moderate | **99.5%** | 63.0% | 91.9% | 94.0% |
| photo-heavy | 96.4% | **39.4%** | 74.0% | 87.3% |

The 12B stops *reading*, down to 39.4% of amounts correct, with 69.8% of what it
does read landing under the wrong heading. Capacity buys legibility under noise,
and the deficit that was visible as a few digit substitutions on clean pages
becomes the dominant failure on a bad one.

#### Photo is the harder channel, and it is where the floor is

`photo-moderate` is worse than `scan-heavy` for **every** system. If production
accepts phone photographs, that is the column to plan against, and the ladder
has found the floor: scanned documents need no quality gate this corpus can
detect, photographed ones need capture guidance or a legibility check before
parsing, and `photo-moderate` is where the cost becomes real rather than
statistical.

#### A net change is not evidence

Pages worse / better / unchanged against each system's own clean run:

| tier | gemma-4-31B | gemma-4-12B | InternVL3.5-8B | MinerU |
|---|---|---|---|---|
| scan-light | 6/10/39 noise | 21/17/17 noise | 24/17/14 noise | 3/2/50 noise |
| scan-moderate | 7/9/39 noise | 17/20/18 noise | 15/25/15 noise | **11/3/41 real** |
| scan-heavy | 9/8/38 noise | **30/15/10 real** | 18/22/15 noise | **15/1/39 real** |
| photo-light | 10/9/36 noise | **35/11/9 real** | 20/21/14 noise | 4/2/49 noise |
| photo-moderate | **19/7/29 real** | **47/2/6 real** | **33/13/9 real** | **22/4/29 real** |
| photo-heavy | **41/2/12 real** | **53/1/1 real** | **41/10/4 real** | **37/3/15 real** |

Balanced directions are the signature of the token-level perturbation that moved
31 of 165 pages between tp=1 and tp=2 on *identical* images. **A net total is not
evidence; only when the two directions stop balancing is there a signal.** The
test earns its keep by disagreeing on the same data at the same threshold — it
calls the 31B's whole scan ladder noise and MinerU's scan tail real.

**It would have been easy to report that scanning helps.** `scan-light` scores
above clean for the 31B, and the first reading of that said so. It is noise, and
counting the direction of every page's change is what shows it.

**A system must be compared against its own clean run.** An earlier version fell
back to the only baseline loaded and compared MinerU's degraded pages against the
*31B's* clean ones, reporting every MinerU tier as a real effect — including
`scan-light`, where MinerU barely moves. Borrowing a baseline is worse than
having none: it reports one system's degradation as another's, and the output
looks entirely normal. The same fault recurred in `collect()`, which read a
single clean report and so gave three of four systems no clean point at all.

#### Two things the ladder found that were not being looked for

**The transcript degrades on the scan ladder while placement does not.** Median
CER rises sharply from clean to `scan-heavy` and rows aligned falls 23 points,
while amounts under the right heading barely move. A system chosen on transcript
quality would be rejected at `scan-heavy`; the same system chosen on placement
would be kept. **This is the sharpest form of finding 9's argument** — and note
that `attributable` sits between the two, which is why it is the metric to
deploy on.

**Fragmentation falls as the image worsens.** MinerU splits 276 rows on clean
input and fewer as the picture degrades: 251 at scan-heavy, 204 at photo-heavy.
The prediction going in was the opposite, since fragmentation is a
row-segmentation failure and blur ought to attack segmentation. The obvious
artefact was ruled out first — MinerU emits the *same* number of table rows at
every severity, 1,876 clean against 1,945 at photo-heavy, on all 55 pages — so it
is not fragmenting less because it is producing less table. Blur appears to merge
the two visual lines of a wrapped description so they read as one. It falls while
everything that matters collapses.

#### Limits

**25 pages are declared unproducible and scored as total failures** — token-cap
repetition loops, each evidenced by a file kept under `_truncated/`. They fall
entirely on the 12B (16) and InternVL (9); the 31B and MinerU produced all 330.
Declaring rather than dropping keeps every system averaged over the same 55
transcripts; dropping would have averaged the two failing systems over fewer,
easier pages. The ranking does not depend on the choice — excluding them lifts
the 12B's photo-heavy figure only to roughly 12%.

Docling was never run on a degraded tier and is absent rather than shown with a
clean point alone, which would read as a system that held up perfectly.

What this does not establish: 55 statements of one document type, against
modelled degradation rather than real scans. Four systems are enough to show that
robustness differs sharply between systems and that the difference is not
predictable from clean accuracy; they are not enough to predict where a fifth
would fall.

---

### A wrong number and a missing number are not the same failure

Every measure so far counts amounts that did not arrive correctly. It does not
distinguish the two ways they fail, and downstream those could hardly be more
different:

- **silently wrong** — a plausible number reaches the consumer. A misread digit,
  an amount under the wrong heading, or one orphaned in a row that identifies
  nothing. `$1,982.56` for `$1,182.56` passes every validation a consumer is
  likely to run.
- **visibly missing** — the amount is absent. A gap can be counted, alerted on,
  and routed to a human.

Bank statements, 2,011 amounts, deployable systems only:

| System | misread | mis-column | orphaned | **silently wrong** | dropped |
|---|---|---|---|---|---|
| **gemma 31B 4-bit** | 3 | 11 | 26 | **2.0%** | 0.1% |
| gemma 12B 4-bit | **101** | 24 | 61 | 9.2% | 5.7% |
| InternVL3.5-8B | 2 | **192** | 247 | **21.9%** | 3.0% |
| MinerU | **0** | 152 | **596** | **37.2%** | 0.6% |

**Two questions, and they rank the systems differently:**

> **"Which system fabricates numbers?"**
> gemma 12B 4-bit is the worst tested — 101 misread amounts — and MinerU is
> perfect, misreading **none** of 2,507 figures.

> **"Which system delivers wrong data?"**
> MinerU is the worst by a distance at 37.2%, and the 12B 4-bit is mid-table at
> 9.2%.

Both are legitimate; they are not the same question. The second is the
deployment question, because a consumer cannot tell how a wrong number became
wrong. The first is a separate and real indictment of the 12B that the aggregate
hides, and it is what finding 8 is about: capacity takes misreads from 101 to
**3**.

**A misread digit is the most insidious failure here**, whatever the totals say.
A mis-columned amount has some chance of failing a debit/credit reconciliation
downstream; an orphaned one is missing its date and may be caught as malformed.
A wrong digit in the right field of the right row is indistinguishable from a
correct answer.

**Note what this does to MinerU's best number.** It misreads nothing at all —
the cleanest digit fidelity of any system here — and has the highest
silently-wrong rate of any system here, because it puts 596 correctly-read
amounts in rows with no date. Reading perfectly is not a defence.

---

## Limits

**Coverage.** All four gemma checkpoints produced all 165 pages. InternVL
produced 164; the one failure is scored as a total failure rather than dropped,
so every system is averaged over the same 165 transcripts. MinerU produced all
165.

**The four gemma checkpoints differ only in the checkpoint.** Every decoding and
engine setting is identical across them, including the vision budget — varying
that would confound precision or capacity with what the model can see. All six
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

**Synthetic.** Clean renders, plus a declared degradation ladder over them
(finding 10) rather than real scans. No claim is
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
appears when the metrics are kept apart. **MinerU misreads not one of the 3,371
amounts in the corpus — every miss is an omission — and still delivers fewer
usable amounts than the 31B**, because it misfiles a tenth of them.
**InternVL3.5-8B reads 6.6 points more digits correctly than the 12B gemma and
delivers 3 points fewer usable amounts**, for the same reason.

Reading a table, matching a house style and getting the digits right are three
separable skills, and no single one of them ranks these systems correctly for
anyone who cares about the money. The figure that does is **usable** — present,
right value, right heading — and it is the one to quote.

**And a measured limit on the headline metric, rather than an argued one.**
Across systems, normalised CER tracks usable-amount rate closely: Spearman
ρ = 0.943 over six systems (p = 0.005). Within a system it is uncorrelated with whether a
page's numbers survive — ρ = 0.01 for two of them, −0.17 for the best. A review
queue ordered by CER is a review queue ordered at random. Report it, because it
ranks systems and it catches the catastrophes a field-level metric would miss;
do not triage on it, and do not read a small CER as evidence that a page's
numbers can be used.

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

---

## Appendix: the prompt

Findings 1, 2, 3 and 6 are all about this text, and quoting fragments of it
makes them hard to check. It is reproduced whole, exactly as sent.

Only the part below is sent. `config/prompt.md` opens with a preamble addressed
to whoever runs the benchmark — 1,018 words in the file, **974 reach the
model** — and the runner strips everything above the `---` rule. Provenance is
recorded per run: every scored system here read the same prompt, digest
`38919c6a`, which is `config/prompt.md` at the commit this document describes.

Read it against the findings. Three things are worth noticing:

- **The headerless-table rule names its case and shows it** (finding 2). An
  earlier version stated the rule without the worked example and reached 2 of 55
  receipts; this version reaches 53. Same model, same pages.
- **The decoration rule says what to write instead of the run it removes.** That
  sentence — "do not replace an omitted run with anything else" — exists because
  a version without it led a model to substitute repeated zeros for repeated
  dots, and finish under the token cap so that nothing flagged it.
- **The date-carry rule is stated as clearly as the other two and lands on only
  one of the two models** (finding 3). Wording is not the whole story, and this
  is the evidence for that.

````text
Transcribe this document page completely, as Markdown.

Read the page top to bottom and write out every piece of text you can see, in
the order it is meant to be read. Do not summarise, do not skip repeated or
boilerplate wording, and do not add anything the page does not show.

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

**A run of repeated dots or dashes is spacing, not content — leave it out.**
Pages use runs of punctuation two ways, and both are typography rather than
text: a line of them drawn across the page as a separator, and a trail of them
padding a line out to a fixed width. Write the words and numbers at each end and
omit the run itself.

A run is **four or more** of `.` `-` `_` `=` `*` in a row. Three or fewer is
ordinary punctuation and is kept, so an ellipsis and a decimal point are written
as printed, and a hyphen inside a range or a date stays where it is.

This applies to text **on the page**. It does not apply to the `| --- |`
separator row of a pipe table, which is Markdown you are writing and must still
be there.

For example, where a statement pads a reference out to a fixed width and rules a
line beneath a section:

```
Ref: 3070829164..........................
BRIGHTWATER MUTUAL Kew AUS
--------------------------------------
```

write

```
Ref: 3070829164

BRIGHTWATER MUTUAL Kew AUS
```

The dots and the ruled line are gone; every character that carries information
is kept, including the `.` inside an amount and the digits either side of it.
Do not replace an omitted run with anything else — no substitute characters, no
placeholder.

**Labelled values go on one line as `Label: value`.** For example a page showing
"Date" beside "04/03/2025" becomes `Date: 04/03/2025`. Write the label once,
with a single colon and space, even if the page draws its own colon.

**Tables become pipe tables with a header separator row**, like this:

```
| Date | Reference | Charge |
| --- | --- | --- |
| 26/11/2019 | Sprocket Housing 6mm | $19.07 |
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
| Lanyard Clip 2pk | 19.23 |
| Gasket Ring 40mm | 19.53 |
```

Do not promote the first line of data into the heading, and do not invent column
names such as "Item" or "Price". Do not write these lines as ordinary
paragraphs — the way they line up down the page is what makes them a table.

**Where a date heads a group of rows, repeat it on every row of that group.**
A statement prints a date once and then lists that day's entries beneath it with
the date column left blank. It does this in two ways, and both are the same
thing: the date may sit on a band of its own across the table, or it may sit in
the date cell of the group's first entry. Either way, put the date in the date
cell of **every** row of that group, and do not give the date a row of its own:

```
| 14/03/2018 | Ratchet Spanner 8mm | 19.67 |
| 14/03/2018 | Torque Bar 12mm | 19.89 |
```

not a row containing only `14/03/2018` followed by rows with an empty date.
Every row should stand on its own.

Carry a date **downwards only**. Where a row's date cell is blank and no date
appears above it in the table — an opening-balance line is the usual case —
leave that cell blank rather than borrowing the date from below.

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
````

The figures in `docs/figures/prompt-*.svg`, generated by
`render_prompt_figure.py`, render one convention per image for presentation.

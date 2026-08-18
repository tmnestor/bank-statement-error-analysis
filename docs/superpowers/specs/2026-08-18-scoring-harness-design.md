# Scoring harness design

Status: proposed, 2026-08-18. Extends
`2026-08-17-document-parsing-corpus-design.md` (the corpus spec) §5, which
defines the two metrics but leaves the tool that computes them unbuilt. Read §5
before this file; where the two disagree, §5 wins and this file is wrong.

## 1. Purpose

The corpus ships a `README.md` promising its reader two numbers, and nothing
computes either. This spec covers the tool that does.

Its job is **not** to rank the four systems named in corpus spec §8.6. It is to
answer §8.6's actual question — *is the convention fair?* — by separating, for
every divergence between a prediction and the ground truth, the case where a
model misread the page from the case where it read the page correctly and wrote
it down differently. The first is the model's problem. The second is
`config/serialisation.yml`'s problem, and is fixable without re-rendering an
image.

A harness that reports only aggregate scores cannot make that separation, and so
cannot do the job it exists for.

## 2. Where it lives, and why

**A sixth `score` command in this repository**, beside `validate`, `generate`,
`serialise`, `preview` and `export`.

Corpus spec §5 says normalisation lives "in the scoring tool" so that scoring
policy can change without regenerating an image. That boundary is about
**normalisation not being baked into the corpus**, not about the tool being a
separate artifact — and it is preserved here: the generator still emits one
canonical form and never normalises.

The deciding argument is the calibration loop itself:

```
score → read the divergences → edit config/serialisation.yml
      → serialise → score again
```

Every step but the first already lives in this repo, and three of the four touch
files in `config/`. A separate repository would put a cross-repo edit in the
middle of a loop meant to be run many times in an afternoon.

The cost is real and accepted: someone holding only an exported
`parsing_YYYYMMDD/` directory cannot score without this repository. If that
becomes a genuine need, `export` can copy the harness in the same way it already
copies `prompt.md` and `serialisation.yml` — the two files that ship with the
data precisely because they are a matched pair with it.

## 3. Normalisation

Policy lives in `config/scoring.yml`. Every key is required, including keys whose
value is a no-op, so that reading the file alone answers what scoring does —
the same contract `config/serialisation.yml` already carries.

### 3.1 Keys

| Key | Value | Meaning |
|---|---|---|
| `unicode_form` | `NFKC` | Unicode normalisation form applied first |
| `fold_dashes` | `true` | En/em/figure dashes → ASCII hyphen |
| `fold_quotes` | `true` | Curly quotes and primes → ASCII `'` and `"` |
| `strip_emphasis` | `true` | Remove `**`, `__`, `*`, `_` used as emphasis |
| `strip_heading_marks` | `true` | Remove leading `#` runs |
| `strip_table_marks` | `true` | Remove cell pipes and header separator rows |
| `strip_blockquote_marks` | `true` | Remove leading `>` |
| `collapse_whitespace` | `true` | All whitespace runs → one space; trim ends |
| `fold_case` | `false` | Committed as `false`; see §3.3 |

### 3.2 Structural stripping, not character stripping

Corpus spec §5 says normalisation "strips Markdown syntax characters". Taken
literally as a character blacklist that is **wrong**, and this spec deliberately
refines it: the corpus contains real content made of the same characters.

- `Statement Period: 01/09/2023 - 23/09/2023` — a bare hyphen inside a value.
- `Delivery: Standard delivery, 5-7 business days` — likewise.
- `Credits (-)` — a Westpac column header that *is* a hyphen.
- `Payment Terms: 50% deposit, balance on completion` — a literal `%`.

Stripping `-` as "Markdown syntax" corrupts all four and would score a model
*down* for reading them correctly. So each strip is structural and positional:
a heading mark is a `#` run at the start of a line, a table mark is a `|` at a
cell boundary or a separator row of dashes and colons, an emphasis mark is a
paired delimiter. Nothing is removed for merely being a punctuation character.

### 3.3 Order is part of the policy

Normalisation steps do not commute — stripping table pipes before collapsing
whitespace leaves the double spaces the pipes were padding, and collapsing first
would make the separator row harder to recognise. The order is fixed and
specified:

1. `unicode_form`
2. `fold_dashes`, `fold_quotes`
3. `strip_emphasis`, `strip_heading_marks`, `strip_table_marks`,
   `strip_blockquote_marks`
4. `collapse_whitespace` (including trimming both ends)

`fold_case` is committed as `false` and is not a step. Corpus spec §5 is explicit
that case is deliberately not folded: reading account names and identifiers with
correct case is legitimately part of transcription. The key exists in the file
rather than being absent so that the file states the decision instead of leaving
it to be inferred from silence.

## 4. Metrics

Both metrics are computed over the same pair, differing only in whether §3 ran:

- **Normalised** — both sides normalised. Blind to wrapping, table dialect, pair
  layout and emphasis. Measures **reading**.
- **Strict** — the raw shipped forms. Measures reading plus convention
  adherence.

For each, three numbers:

| Number | Definition |
|---|---|
| Edit distance | Levenshtein, unit cost for insert/delete/substitute |
| CER | character edit distance ÷ character length of the **truth** |
| WER | word edit distance ÷ word count of the **truth**, splitting on whitespace |

Truth is the denominator, not `max(len)`: an error rate is conventionally
relative to the reference, and a system that emits nothing should score 1.0
rather than 0.5.

Empty cases are defined rather than left to a `ZeroDivisionError`: truth empty
and prediction empty is `0.0`; truth empty and prediction non-empty is `1.0`.

**Corpus aggregates are micro-averaged** — summed distances over summed truth
lengths — so a 3,128-character bank statement carries more weight than a
389-character receipt, which is the honest reading of "error rate over the
corpus". The macro average (mean of per-document rates) is also reported,
because a large gap between the two is itself a signal: it means the errors
concentrate in the short documents, and the receipts are exactly where that
would happen.

Cost, pure Python, no new dependency: **under ten minutes for four systems over
all 165 pages**. That is projected, not measured end to end — a pure-Python
Levenshtein over the largest transcript (3,128 characters) was timed at 1.67s,
scaled across the real size distribution for 165 documents and four systems
(~4.4 minutes), then doubled because each pair is scored twice, strict and
normalised. Word-level distances are negligible beside it: the median document
is 111 words against 614 characters, and the cost is quadratic. Fast enough that
the repository's five-runtime-dependency constraint holds and
`rapidfuzz`/`numpy` stay out.

## 5. Divergence classification

This is the part that answers §8.6, and its correctness rests on one idea.

**A divergence is classified by construction, not by heuristic.** For each hunk
where prediction and truth differ, normalise both sides:

- the two sides become **equal** → the difference was pure convention. By
  definition: normalisation is exactly the set of differences declared not to
  be reading errors.
- the two sides remain **different** → a genuine reading error.

There is no pattern list to maintain, nothing to tune, and no way for the
classifier to drift from the policy — it *is* the policy, applied at hunk
granularity instead of document granularity. Change `scoring.yml` and the
classification follows automatically.

### 5.1 Alignment

Hunks come from `difflib.SequenceMatcher` over whitespace-split words, whose
opcodes yield `replace` / `delete` / `insert` spans directly.

`difflib` is used **only to locate divergences, never to score them.** The
metrics in §4 come from the Levenshtein implementation, because
`SequenceMatcher` computes a longest-contiguous-match ratio, not an edit
distance, and reporting one as the other would be wrong.

### 5.2 Grouping

Grouped by exact match on the `(truth span, prediction span)` pair, counted, and
attributed to the systems exhibiting it. Exact match only — no clustering, no
fuzzy merging, no stemming. A count of 152 is then a fact about the corpus
rather than the output of a similarity threshold nobody will remember tuning.

The intended reading: a convention group appearing across **all four** systems
indicts the convention. One appearing in a single system is that system's
dialect and should not move `serialisation.yml`.

### 5.3 The quantisation caveat

Corpus spec §8.6 records that the Gemma checkpoint is 4-bit QAT. Quantisation
costs character-level fidelity first — a transposed digit in an ABN or an
amount. Such a hunk survives normalisation and is therefore classified, quite
correctly, as a reading error; the harness has no way to know it came from the
checkpoint rather than the model. Per-system attribution in §5.2 is what makes
this legible: a reading-error group unique to Gemma is a quantisation signal.
The harness reports; it does not adjudicate.

## 6. Inputs and preconditions

```
python -m generators.pipeline score \
    --corpus parsing_20260818 \
    --predictions runs/ \
    --policy config/scoring.yml \
    --report scores.json
```

`--predictions` names a directory in which **each immediate subdirectory is one
system**, named by the directory. Always — a directory of loose `.md` files is
an error, not a single anonymous system. The rule is stated rather than sniffed
so the layout cannot be guessed wrong.

```
runs/
  gemma-4-12B-it-qat-w4a16-ct/CASE001_invoices.md
  InternVL3.5-8B/CASE001_invoices.md
  docling/CASE001_invoices.md
  mineru/CASE001_invoices.md
```

Predictions pair to truth by filename stem, matching the export's own
`{case_id}_{doc_type}` convention.

Three preconditions, each a hard failure with a four-element diagnostic:

1. **Every image matches its `sha256` in `manifest.jsonl`.** The shipped README
   calls this "not ceremony": a mismatch means the predictions and the ground
   truth are different vintages, and any number computed from them is
   meaningless. Refusing beats reporting.
2. **Every transcript has a prediction, in every system.** Scoring 160 of 165
   silently understates nothing and hides a broken inference run. The error
   names the missing stems.
3. **Every prediction has a transcript.** An extra file means predictions from
   another corpus have been mixed in.

## 7. Modules

Four units, each usable and testable without the others above it:

| Module | Purpose | Depends on |
|---|---|---|
| `generators/metrics.py` | `edit_distance`, `cer`, `wer`. Pure maths — no policy, no Markdown knowledge, no I/O | nothing |
| `generators/scoring.py` | Load and validate `config/scoring.yml`; `normalise(text, policy)` | policy file |
| `generators/divergence.py` | Align, extract hunks, classify against §5, group | the two above |
| `generators/pipeline.py` | The `score` command: preconditions, orchestration, reporting | all three |

`metrics.py` knowing nothing about Markdown or policy is the boundary that
matters: it makes the numbers auditable in isolation, and it is why §5.1's
"`difflib` never scores" rule is structurally enforced rather than merely
documented.

## 8. Output

A `rich` table per system to the terminal, and the full result as JSON to
`--report`:

```json
{
  "corpus": "parsing_20260818",
  "policy": "config/scoring.yml",
  "systems": {
    "docling": {
      "normalised": {"cer": 0.014, "wer": 0.031, "distance": 1489},
      "strict":     {"cer": 0.161, "wer": 0.178, "distance": 17022},
      "macro": {"normalised_cer": 0.019, "strict_cer": 0.171},
      "documents": [{"stem": "CASE001_invoices", "normalised_cer": 0.0,
                     "strict_cer": 0.12}]
    }
  },
  "divergences": {
    "convention": [{"truth": "Total", "prediction": "**Total**",
                    "count": 152, "systems": ["docling", "mineru"]}],
    "reading":    [{"truth": "57 773 872 148", "prediction": "57 773 872 143",
                    "count": 3, "systems": ["gemma-4-12B-it-qat-w4a16-ct"]}]
  }
}
```

The JSON is the artifact of record; the terminal table is a convenience over it
and is derived from the same structure rather than computed separately.

## 9. Testing

| Layer | What it pins |
|---|---|
| `normalise` | Each policy key toggled independently, known input → known output; §3.2's four real strings survive stripping intact |
| `metrics` | Hand-computed distances; identical, empty-both, empty-truth, single-substitution |
| `divergence` | A synthetic pair carrying one emphasis hunk and one transposed digit — assert exactly one lands in each class |
| Grouping | The same hunk from two systems groups once with count 2 and both names |
| Preconditions | Each of §6's three failures raises with all four diagnostic elements |
| **End-to-end** | **Scoring the corpus against itself: CER 0, WER 0, zero hunks, both metrics** |

The end-to-end invariant is the one that catches a whole class of silent bugs. If
normalisation is asymmetric, alignment is off by one, or pairing is wrong, a
self-scored corpus stops being perfect and the test fails.

## 10. Out of scope

No charts or plots. No parallelism — §4's projected sub-ten minutes does not
justify it. No per-system configuration beyond the directory name. No fuzzy or semantic
grouping. No ranking, leaderboard, or "winner": §1 says why.

## 11. Open items

| Item | Blocked on |
|---|---|
| Whether `export` should copy the harness into `parsing_YYYYMMDD/` | Whether anyone outside this repo needs to score — see §2 |
| Whether a third `hallucination` divergence class is worth adding for spans present in the prediction and absent from the truth | Seeing whether the four systems actually produce them; today they fall under reading errors, which is defensible but coarse |

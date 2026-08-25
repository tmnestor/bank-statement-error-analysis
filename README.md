# Bank statement error analysis

How well do vision-language models and document parsers read an Australian bank
statement — and which of their errors actually matter to a system consuming the
output?

The headline result: **an amount can be read correctly, filed under the right
heading, and still be unusable**, because the row it sits in carries no date and
cannot be attributed to a transaction. Whole-page CER cannot see that, and it is
the difference between a system that looks deployable and one that is.

Full write-up: [`docs/2026-08-19-calibration-pass-findings.md`](docs/2026-08-19-calibration-pass-findings.md)
([rendered](docs/2026-08-19-calibration-pass-findings.html)).

## What is measured

| metric | question it answers |
|---|---|
| normalised / strict CER, WER | how much of the page was read, and at what convention cost |
| **column integrity** | is the amount under the right heading? |
| **attributable** | …and in a row that identifies its transaction? |
| table structure | were the rows segmented at all — fragments, width breaks |
| numeric fidelity | were the digits right, separately from where they landed |

`attributable` is the number to deploy on. The rest rank systems adequately and
miss the failure that matters.

## Quick start

```bash
conda env create -f environment.yml
conda activate docparse

python -m evaluation.cli --corpus parsing_20260820 --predictions runs \
    --report scores.json
```

The corpus is an exported directory from the
[document-parsing](https://github.com/tmnestor/document-parsing) generator. Its
manifest carries a sha256 per image, verified before anything is scored, so
scoring against the wrong vintage is impossible rather than merely detectable.

## Layout

```
evaluation/      cli.py         the score command
                 metrics.py     edit distance, CER, WER — no policy, no I/O
                 scoring.py     scoring.yml and normalisation
                 columns.py     column integrity, the error CER cannot see
                 tables.py      row segmentation and cell content
                 numerics.py    amount-level fidelity
                 divergence.py  hunk extraction, convention vs reading
                 unproduced.py  the declared-gap contract
                 eval_export.py projection into the extraction eval format
runners/         run_docling.py, run_mineru.py, run_vlm.py — one env each,
                 common.py imports NO parser, which is what lets its tests run
analysis/        degradation.py, figures.py, summary.ipynb
config/          vlm_systems.yml, scoring.yml, prompt.md and its variants
docs/            findings, the comparison-sheet reading guide
```

## Running the parsers and models

Each parser needs its own environment — their pins are mutually unsatisfiable —
and the prompted models need a GPU host. See `docs/remote-vlm-run.md`.

```bash
conda run -n docparse-docling python -m runners.run_docling --corpus <corpus> --out runs
conda run -n docparse-mineru  python -m runners.run_mineru  --corpus <corpus> --out runs --chunk 25
./run_degraded_12b_internvl.sh          # on the GPU sandbox
./score_degraded.sh                     # every tier, every system
./build_meeting_sheets.sh               # side-by-side comparison images
```

## Reading the comparison sheets

[`docs/reading-the-comparison-sheets.md`](docs/reading-the-comparison-sheets.md)
explains the colour coding. The short version: **red is wrong, blue is
elsewhere** — blue marks content the system read correctly but placed in a
structure the page does not have, which is not the same as inventing it.

## Relationship to document-parsing

This repository was split out of the corpus generator, which now lives at
[document-parsing](https://github.com/tmnestor/document-parsing). The interface
is the **exported corpus directory**, not shared code: nothing here imports a
renderer, and nothing there imports a metric. `ground_truth/bank_statements.yml`
is the one duplicated file, because `eval_export` projects it into the
extraction format.

## Quality gates

```bash
pytest tests/
ruff check --fix --ignore ARG001,ARG002,F841 .
ruff format .
mypy evaluation runners --ignore-missing-imports
```

`runners/run_docling.py` and `run_mineru.py` are structurally 0% covered: they
import a parser this environment does not have, which is the same constraint
that keeps `runners/common.py` parser-free. They are exercised by real runs, not
by pytest.

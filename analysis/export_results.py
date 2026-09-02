"""Both benchmark axes in one artifact: pages per minute and accuracy.

Teams compare systems on throughput AND on how much of the page survived. This
repository already records both, well, and in two different places:

- **Accuracy** goes into the score report, which `evaluation.cli` deliberately
  makes self-sufficient -- per-document rows are written precisely so nobody has
  to re-read the predictions, which are gitignored and machine-specific.
- **Throughput** goes into `_timing.json`, written beside the predictions by the
  runner, and aggregated across shards by `runners.common.read_timing`.

Nothing joined them. The only join that existed was a private helper inside
`analysis.figures`, whose outputs are PNGs -- so a team wanting both axes had to
glob the timing files and reimplement the shard arithmetic: images **summed**,
seconds **maxed**, because shards run concurrently and the box's clock is the
slowest of them. Reversing those misreports throughput by the shard count.

This command does the join once and writes two files:

- `results.json` -- one record per (report, system), nesting accuracy,
  throughput and the per-document rows.
- `results.csv` -- the same thing flat, one row per document, with the run-level
  fields repeated so it can be plotted or pivoted without reshaping.

Usage, once every tier is scored:

    python -m analysis.export_results scores_v2_31b_*.json --out results/

**Two columns exist to stop a misleading chart, and consumers must read them:**

`includes_model_load` -- model load is excluded from the engine-driven runners'
seconds, because it is paid once per process rather than per page and is not
what a serving cluster is sized on. A runner that cannot separate it (MinerU
shells out to a CLI that loads per invocation) sets this True, and its rate is
then a FLOOR. Plotting a floor unmarked beside load-exclusive rates understates
it silently.

`doc_type` -- `attributable` is not comparable across document types. A receipt
has one date, in its header, and attribution requires an amount to sit in a row
carrying its identifying date, which a receipt line item never does. Measured on
gemma-4-31B against a clean corpus, receipts read 184/184 amounts with 404/404
numerically correct and a median normalised CER of 0.0000, and still scored
129/184 attributable. A chart mixing document types on that axis measures the
floor, not the system.
"""

import csv
import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from analysis.tiers import read_tier
from runners.common import read_timing

REPO = Path(__file__).resolve().parent.parent

app = typer.Typer(add_completion=False, help="Join accuracy and throughput into one artifact.")

# Run-level fields repeated on every CSV row. Throughput first: it is the axis
# the score reports cannot answer alone, and the reason this command exists.
_RUN_COLUMNS = (
    "system",
    "family",
    "severity",
    "images_per_minute",
    "images_per_minute_per_card",
    "includes_model_load",
    "deployment",
    "cards",
    "images",
    "inference_seconds",
    "prompt_sha256",
    "corpus",
)


class ExportError(RuntimeError):
    """Raised when the reports and the runs cannot be joined into one table."""


def _refuse(what: str, *, where: str, expected: str, recover: str) -> NoReturn:
    """Stop with a four-element diagnostic, in this repository's shape."""
    raise ExportError(
        "Cannot export results.\n"
        f"  What:     {what}\n"
        f"  Where:    {where}\n"
        f"  Expected: {expected}\n"
        f"  Recover:  {recover}"
    )


def _doc_type(stem: str) -> str:
    """The document type a prediction stem names.

    Export filenames are `{case_id}_{doc_type}` by design -- never
    `{case_id}_{layout_id}`, so a model cannot infer the template before reading
    a pixel -- which makes the type recoverable from the stem alone.

    Args:
        stem: A prediction stem, e.g. `CASE001_bank_statements`.

    Returns:
        The document type, e.g. `bank_statements`.
    """
    _, _, doc_type = stem.partition("_")
    if not doc_type:
        _refuse(
            f"the stem {stem!r} is not `{{case_id}}_{{doc_type}}`, so its document type cannot be read.",
            where=stem,
            expected="every prediction stem to carry its type, e.g. `CASE001_bank_statements`.",
            recover="re-score against an exported corpus; `export` names files this way.",
        )
    return doc_type


def _throughput(report: Path, payload: dict, system: str, root: Path) -> dict:
    """What the box delivered on this run, from the run's own timing file.

    Args:
        report: The score report, named in diagnostics.
        payload: The parsed report, for the `predictions` directory it scored.
        system: The system whose subdirectory holds the timing file.
        root: Root that a relative `predictions` path resolves against.

    Returns:
        `read_timing`'s aggregate: rate, per-card rate, deployment and the
        `includes_model_load` flag.
    """
    recorded = payload.get("predictions")
    if not recorded:
        _refuse(
            f"{report.name} records no `predictions` directory, so its throughput cannot "
            "be found. Accuracy alone is half the deliverable.",
            where=str(report.resolve()),
            expected="every score report to name the predictions it scored, e.g.\n"
            '              {"predictions": "/path/to/runs_v2_31b/scan-heavy", ...}',
            recover="re-run `python -m evaluation.cli --corpus ... --predictions ... "
            "--report ...`, which records it. Re-scoring reads files already on disk: it "
            "needs no GPU and takes seconds.",
        )

    system_dir = root / Path(str(recorded)) / system
    timing = read_timing(system_dir)
    if timing is None:
        _refuse(
            f"{system} has predictions but no `_timing.json`, so its pages/min is unknown.",
            where=str(system_dir.resolve()),
            expected="a `_timing.json` (or `_timing.shard*.json`) written by the runner "
            "beside the predictions, e.g.\n"
            '              {"images": 189, "inference_seconds": 2100.0, "cards": 2, ...}',
            recover="run this export on the machine that holds the run -- timing files sit "
            "beside the predictions and do not travel with the report. If the predictions "
            "were made by a runner that writes no timing, re-run it so the throughput axis "
            "is measured rather than guessed.",
        )
    return timing


def export(reports: list[Path], out: Path, root: Path = REPO) -> Path:
    """Join every report to its run's timing and write the two artifacts.

    Args:
        reports: Score reports, one per tier, in the order to publish them.
        out: Directory to write `results.json` and `results.csv` into.
        root: Root that relative `corpus`/`predictions` paths resolve against.

    Returns:
        The path of the written `results.json`.

    Raises:
        ExportError: A report names no predictions, a system has no timing, or a
            corpus cannot state its tier.
    """
    runs: list[dict] = []
    rows: list[dict] = []

    for report in reports:
        payload = json.loads(report.read_text(encoding="utf-8"))
        # Unlike `analysis.degradation`, `clean` is a row here, not a dropped
        # rung: a benchmark table without its baseline has nothing to measure
        # the degraded tiers against.
        tier = read_tier(report, payload, root, _refuse)

        for system, block in payload["systems"].items():
            timing = _throughput(report, payload, system, root)
            documents = [
                {"doc_type": _doc_type(document["stem"]), **document} for document in block["documents"]
            ]
            run = {
                "system": system,
                **tier,
                "corpus": payload.get("corpus", ""),
                "report": str(report),
                "prompt_sha256": block.get("prompt_sha256"),
                "throughput": timing,
                "accuracy": {
                    key: block[key]
                    for key in ("normalised", "strict", "macro", "columns", "tables", "numbers")
                    if key in block
                },
                "documents": documents,
            }
            runs.append(run)

            shared = {
                "system": system,
                **tier,
                "images_per_minute": timing["images_per_minute"],
                "images_per_minute_per_card": timing["images_per_minute_per_card"],
                "includes_model_load": timing["includes_model_load"],
                "deployment": timing["deployment"],
                "cards": timing["cards"],
                "images": timing["images"],
                "inference_seconds": timing["inference_seconds"],
                "prompt_sha256": block.get("prompt_sha256"),
                "corpus": payload.get("corpus", ""),
            }
            rows.extend(shared | document for document in documents)

    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.json"
    results.write_text(json.dumps({"schema": 1, "runs": runs}, indent=2) + "\n", encoding="utf-8")

    # Every document field seen anywhere, in first-seen order, so a system that
    # reports a field others do not still gets a column rather than being
    # dropped. Run-level columns lead, so the file reads as a benchmark table.
    document_columns: list[str] = []
    for row in rows:
        document_columns.extend(k for k in row if k not in _RUN_COLUMNS and k not in document_columns)
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*_RUN_COLUMNS, *document_columns])
        writer.writeheader()
        writer.writerows(rows)

    return results


@app.command()
def main(
    reports: Annotated[list[Path], typer.Argument(help="Score reports, one per tier.")],
    out: Annotated[Path, typer.Option("--out", help="Directory for results.json and results.csv")],
) -> None:
    """Join every tier's accuracy to its run's throughput and write both files.

    Raises:
        typer.Exit: With code 1 when the reports and runs cannot be joined.
    """
    try:
        results = export(reports, out)
    except ExportError as err:
        typer.echo(str(err), err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Wrote {results} and {results.with_name('results.csv')}")


if __name__ == "__main__":
    app()

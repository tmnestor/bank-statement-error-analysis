"""Write the bank-statement corpora in the layout LMM_POC's extractor reads.

Mirrors `evaluation_data/synthetic_20260812` and `evaluation_data/degraded_20260812`:
a flat directory of images with `ground_truth.jsonl` and `ground_truth.csv`
beside them. No `images/` subdirectory, no manifest, no transcripts — that is
this repo's own export layout and a different consumer.

Both layouts are produced from the same render. Nothing is re-generated for the
second consumer, so the two cannot drift.

    python -m evaluation.eval_export_cli \\
        --corpus parsing_20260822 --degraded degraded --out ~/Desktop/evaluation_data
"""

import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from evaluation.eval_export import (
    EvalExportError,
    load_entries,
    project_bank_statement,
    write_set,
)

app = typer.Typer(add_completion=False)

# LMM_POC names the type in the singular and without the images/ nesting:
# CASE001_bank_statement.png. This repo uses the plural throughout because its
# filenames are {case}_{doc_type} and the type key is `bank_statements`.
_SINGULAR = {"bank_statements": "bank_statement", "invoices": "invoice", "receipts": "receipt"}


@app.command()
def export(
    corpus: Annotated[Path, typer.Option("--corpus", help="This repo's exported clean corpus.")],
    out: Annotated[Path, typer.Option("--out", help="evaluation_data root.")],
    degraded: Annotated[
        Path | None, typer.Option("--degraded", help="Directory of degraded corpora.")
    ] = None,
    ground_truth: Annotated[Path, typer.Option("--ground-truth", help="Authored YAML.")] = Path(
        "ground_truth/bank_statements.yml"
    ),
    date_stamp: Annotated[str, typer.Option("--date", help="Stamp for the directory names.")] = "",
    doc_type: Annotated[str, typer.Option("--type", help="Document type key.")] = "bank_statements",
) -> None:
    """Write a clean set and, if given, one degraded set covering every tier."""
    stamp = date_stamp or corpus.name.replace("parsing_", "")
    singular = _SINGULAR.get(doc_type, doc_type)

    try:
        entries = load_entries(ground_truth)
    except EvalExportError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None

    def emit(images: list[Path], target: Path, suffix: str = "") -> int:
        """Copy images under LMM_POC's naming and write the ground truth beside them."""
        target.mkdir(parents=True, exist_ok=True)
        records = []
        for image in sorted(images):
            case = image.stem.split("_")[0]
            if case not in entries:
                continue
            name = f"{case}_{singular}{suffix}{image.suffix}"
            shutil.copyfile(image, target / name)
            records.append({"filename": name, **project_bank_statement(entries[case])})
        if records:
            write_set(records, target)
        return len(records)

    clean_target = out / f"synthetic_{singular}_{stamp}"
    count = emit(sorted((corpus / "images").glob(f"*_{doc_type}.*")), clean_target)
    rprint(f"  [green]{clean_target.name:44}[/green] {count:4d} page(s)")

    if degraded is None or not degraded.is_dir():
        return

    # One directory for every tier together, as LMM_POC's degraded_20260812
    # holds three severities of one document type side by side. The tier suffix
    # keeps them distinguishable and matches config/degradation.yml, so a
    # filename says which ladder and severity produced it.
    degraded_target = out / f"degraded_{singular}_{stamp}"
    total = 0
    records: list[dict] = []
    for tier_dir in sorted(p for p in degraded.iterdir() if p.is_dir()):
        tier = tier_dir.name.split("_")[-1]  # e.g. "scan-light"
        suffix = "_" + tier.replace("-", "")
        degraded_target.mkdir(parents=True, exist_ok=True)
        for image in sorted((tier_dir / "images").glob(f"*_{doc_type}.*")):
            case = image.stem.split("_")[0]
            if case not in entries:
                continue
            name = f"{case}_{singular}{suffix}{image.suffix}"
            shutil.copyfile(image, degraded_target / name)
            records.append({"filename": name, **project_bank_statement(entries[case])})
            total += 1
    if records:
        write_set(records, degraded_target)
        rprint(f"  [green]{degraded_target.name:44}[/green] {total:4d} page(s)")


if __name__ == "__main__":
    app()

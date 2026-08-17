"""Corpus pipeline: validate configuration, render pages.

Usage:
    python -m generators.pipeline validate
    python -m generators.pipeline generate --type invoices --limit 3

`serialise`, `export` and `preview` (design §6) arrive with the transcript
recorder; this module deliberately carries only the two commands that need no
transcript. The predecessor's `derive` and `eval-set` commands do not cross —
both project extraction ground truth, which belongs to that repo.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from generators.bank_statement import render_bank_statement
from generators.common import FitError
from generators.content_engine import load_pools, reachable_blocked_names
from generators.invoice import render_invoice
from generators.layout_dsl.schema import LayoutSchemaError, validate_layout
from generators.loader import load_generation_config, load_ground_truth, load_layout_registry
from generators.overflow_check import build_overflow_error, check_overflow
from generators.receipt import render_receipt
from generators.schema import field_names_for, validate_entry

app = typer.Typer(add_completion=False, help="Synthetic document parsing corpus pipeline.")

_RENDERERS = {
    "bank_statements": render_bank_statement,
    "receipts": render_receipt,
    "invoices": render_invoice,
}

_DEFAULT_CONFIG = Path("config/generation_config.yml")


def _validate_layouts(layouts: dict, *, doc_type: str, layout_path: str) -> list[str]:
    """Structurally validate every DSL body in a layout registry.

    Args:
        layouts: Layout registry (layout id -> layout dict).
        doc_type: Document type, used to resolve the legal field names.
        layout_path: Path to the layout YAML, used in diagnostics.

    Returns:
        One diagnostic string per invalid layout; empty when all are well formed.
    """
    known = set(field_names_for(doc_type))
    errors: list[str] = []
    for layout_id, layout in layouts.items():
        if "body" not in layout:
            continue
        try:
            validate_layout(layout, layout_id=layout_id, layout_path=layout_path, known_fields=known)
        except LayoutSchemaError as exc:
            errors.append(str(exc))
    return errors


@app.command()
def validate(
    config: Annotated[Path, typer.Option(help="Path to generation_config.yml")] = _DEFAULT_CONFIG,
) -> None:
    """Check every ground-truth entry and layout before anything is rendered.

    Covers required fields per document type, ABN checksums, date and amount
    formats, equal item counts across parallel pipe-delimited fields, GST as one
    eleventh of a GST-inclusive total, that each `layout:` names a layout the
    registry actually holds, that the DSL bodies are well formed, that no text
    overflows its budget, and that the business-name grammar cannot emit a real
    company.

    Errors are collected rather than raised one at a time, so a single run
    reports every problem in the corpus.

    Raises:
        typer.Exit: With code 1 when any check fails.
    """
    cfg = load_generation_config(config)
    all_errors: list[str] = []

    for doc_type, doc_cfg in cfg["document_types"].items():
        gt_path = Path(doc_cfg["ground_truth"])
        if not gt_path.exists():
            all_errors.append(f"{doc_type}: ground truth not found at {gt_path}")
            continue

        gt_data = load_ground_truth(gt_path)
        layout_path = Path(doc_cfg["layouts"])
        layouts = load_layout_registry(layout_path) if layout_path.exists() else {}

        for case_id, entry in gt_data.items():
            all_errors.extend(validate_entry(str(case_id), entry))
            layout_ref = entry.get("layout", "")
            if layouts and layout_ref not in layouts:
                all_errors.append(
                    f"{case_id}: layout '{layout_ref}' not found in {layout_path}. "
                    f"Available layouts: {sorted(layouts)}"
                )

        # Every layout body is structurally validated before any rendering, so a
        # malformed primitive or unknown field reference fails here rather than
        # part-way through a several-hundred-image generate run.
        layout_errors = (
            _validate_layouts(layouts, doc_type=doc_type, layout_path=str(layout_path)) if layouts else []
        )
        all_errors.extend(layout_errors)

        # Overflow backstop: render each entry and surface any content that cannot
        # fit its box even after lossless wrap/shrink (a real design error).
        # Skipped when layout validation already failed — rendering a known-broken
        # body raises an unrelated exception instead of a useful fit diagnostic.
        renderer = _RENDERERS.get(doc_type)
        if renderer and layouts and not layout_errors:
            all_errors.extend(check_overflow(gt_data, layouts, renderer))

    # A pool edit must not make a real business name emittable. The runtime
    # blocklist in fictional_business_name() would still catch it, but only as a
    # retry during seeding; checked here it is a configuration error raised at
    # the moment someone widens business_name_parts.
    for name in reachable_blocked_names(load_pools()):
        all_errors.append(
            f"the business-name grammar can now emit '{name}', a real business on the "
            f"blocklist. A prefix in business_name_parts.surnames or .suburb_prefixes "
            f"combined with a noun in .category_nouns produces it exactly. Rename or "
            f"remove the offending part in config/data_pools.yml so no real name is "
            f"reachable."
        )

    if all_errors:
        rprint(f"[red]Validation failed with {len(all_errors)} error(s):[/red]")
        for err in all_errors:
            rprint(f"  [red]- {err}[/red]")
        raise typer.Exit(1) from None

    rprint("[green]Validation passed.[/green]")


@app.command()
def generate(
    config: Annotated[Path, typer.Option(help="Path to generation_config.yml")] = _DEFAULT_CONFIG,
    doc_type: Annotated[str | None, typer.Option("--type", help="Render only this document type.")] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Render at most N documents per type.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Override the configured output directory.")
    ] = None,
) -> None:
    """Render page images from ground truth.

    Filenames are `{case_id}_{doc_type}.png`, never `{case_id}_{layout_id}.png`
    (design §6.1): a model must not be able to infer the layout template from
    the filename before it has read a pixel.

    Raises:
        typer.Exit: With code 1 on an unknown `--type` or an invalid layout.
    """
    cfg = load_generation_config(config)
    output_dir = output if output is not None else Path(cfg["output_dir"])

    doc_types = cfg["document_types"]
    if doc_type:
        if doc_type not in doc_types:
            rprint(f"[red]Unknown document type '{doc_type}'. Available: {sorted(doc_types)}[/red]")
            raise typer.Exit(1) from None
        doc_types = {doc_type: doc_types[doc_type]}

    for dtype, doc_cfg in doc_types.items():
        renderer = _RENDERERS[dtype]
        gt_data = load_ground_truth(Path(doc_cfg["ground_truth"]))
        layout_path = Path(doc_cfg["layouts"])
        layouts = load_layout_registry(layout_path)

        layout_errors = _validate_layouts(layouts, doc_type=dtype, layout_path=str(layout_path))
        if layout_errors:
            rprint(f"[red]{dtype}: layout validation failed.[/red]")
            for err in layout_errors:
                rprint(f"[red]{err}[/red]")
            raise typer.Exit(1) from None

        target = output_dir / doc_cfg["output_subdir"] if output is None else output_dir
        target.mkdir(parents=True, exist_ok=True)

        count = 0
        for case_id, entry in gt_data.items():
            if limit is not None and count >= limit:
                break
            layout_ref = entry.get("layout", "")
            layout = layouts.get(layout_ref)
            if not layout:
                rprint(f"[yellow]Skipping {case_id}: layout '{layout_ref}' not found.[/yellow]")
                continue

            entry["case_id"] = str(case_id)
            try:
                image = renderer(entry, layout)
            except FitError as exc:
                raise build_overflow_error(
                    [f"{case_id} / {layout_ref}: {str(exc).splitlines()[0]}"]
                ) from None

            image.save(target / f"{case_id}_{dtype}.png")
            count += 1

        rprint(f"[green]{dtype}: generated {count} documents into {target}.[/green]")


if __name__ == "__main__":
    app()

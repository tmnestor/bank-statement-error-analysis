"""Corpus pipeline: validate configuration, render pages.

Usage:
    python -m generators.pipeline validate
    python -m generators.pipeline generate --type invoices --limit 3
    python -m generators.pipeline serialise
    python -m generators.pipeline preview CASE001
    python -m generators.pipeline export

The commands are ordered: `generate` renders and captures, `serialise` turns
captured events into transcripts, `export` packages what those two produced.
Each reads only the previous one's output, so a convention change re-emits every
transcript without re-rendering an image (design §6).

The predecessor's `derive` and `eval-set` commands do not cross — both project
extraction ground truth, which belongs to that repo.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.table import Table

from generators.bank_statement import render_bank_statement
from generators.columns import column_integrity
from generators.common import FitError
from generators.content_engine import load_pools, reachable_blocked_names
from generators.divergence import CONVENTION, READING, group, hunks
from generators.export import ExportError, export_corpus, sha256_of
from generators.invoice import render_invoice
from generators.layout_dsl.schema import LayoutSchemaError, validate_layout
from generators.loader import load_generation_config, load_ground_truth, load_layout_registry
from generators.metrics import cer, error_rate, wer
from generators.numerics import numeric_fidelity
from generators.overflow_check import build_overflow_error, check_overflow
from generators.receipt import render_receipt
from generators.schema import field_names_for, layout_field_names_for, validate_entry
from generators.scoring import ScoringError, load_scoring_policy, normalise
from generators.serialise import load_serialisation_policy
from generators.serialise import serialise as serialise_events
from generators.tables import table_report
from generators.unproduced import read_unproduced

app = typer.Typer(add_completion=False, help="Synthetic document parsing corpus pipeline.")

_RENDERERS = {
    "bank_statements": render_bank_statement,
    "receipts": render_receipt,
    "invoices": render_invoice,
}

_DEFAULT_CONFIG = Path("config/generation_config.yml")
_DEFAULT_POLICY = Path("config/serialisation.yml")
_DEFAULT_PROMPT = Path("config/prompt.md")
_DEFAULT_SCORING = Path("config/scoring.yml")


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
        # A layout may draw a block no other layout has, whose data is authored
        # per case under `layout_fields` rather than required of every entry of
        # the type. Those names are legal references for that layout only.
        allowed = known | set(layout_field_names_for(layout_id))
        try:
            validate_layout(layout, layout_id=layout_id, layout_path=layout_path, known_fields=allowed)
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
    derived: Annotated[
        Path | None, typer.Option("--derived", help="Override the configured derived directory.")
    ] = None,
) -> None:
    """Render page images from ground truth, capturing a transcript per page.

    Filenames are `{case_id}_{doc_type}.png`, never `{case_id}_{layout_id}.png`
    (design §6.1): a model must not be able to infer the layout template from
    the filename before it has read a pixel.

    The §8.2 coverage invariant runs here, not only under pytest: if a primitive
    puts text on the page without emitting an event, this command fails and
    names it, rather than writing a quietly incomplete transcript.

    Raises:
        typer.Exit: With code 1 on an unknown `--type` or an invalid layout.
        CoverageError: A primitive drew text without emitting an event.
    """
    cfg = load_generation_config(config)
    output_dir = output if output is not None else Path(cfg["output_dir"])
    records: list[dict] = []

    # Kept unfiltered: a partial run still has to resolve where *every* type's
    # images live, to tell a carried-over event record from a stale one.
    all_doc_types = cfg["document_types"]
    doc_types = all_doc_types
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
                image, recorder = renderer(entry, layout)
            except FitError as exc:
                raise build_overflow_error(
                    [f"{case_id} / {layout_ref}: {str(exc).splitlines()[0]}"]
                ) from None

            image_file = f"{case_id}_{dtype}.png"
            image.save(target / image_file)
            records.append(
                {
                    "case_id": str(case_id),
                    "doc_type": dtype,
                    "image_file": image_file,
                    "events": [event.as_dict() for event in recorder.events],
                }
            )
            count += 1

        rprint(f"[green]{dtype}: generated {count} documents into {target}.[/green]")

    derived_dir = derived if derived is not None else Path(cfg["derived_dir"])
    derived_dir.mkdir(parents=True, exist_ok=True)
    events_path = derived_dir / "events.jsonl"

    image_dir_for = {
        dtype: output_dir if output is not None else output_dir / doc_cfg["output_subdir"]
        for dtype, doc_cfg in all_doc_types.items()
    }
    merged, dropped = _merge_event_records(events_path, records, image_dir_for=image_dir_for)
    with events_path.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record) + "\n")

    for image_file in dropped:
        rprint(f"[yellow]Dropped stale events for {image_file}: no such image on disk.[/yellow]")
    carried = len(merged) - len(records)
    if carried:
        rprint(
            f"[green]Events written: {events_path} ({len(merged)} documents: "
            f"{len(records)} from this run, {carried} carried over from earlier runs)[/green]"
        )
    else:
        rprint(f"[green]Events written: {events_path} ({len(merged)} documents)[/green]")


def _merge_event_records(
    events_path: Path,
    fresh: list[dict],
    *,
    image_dir_for: dict[str, Path],
) -> tuple[list[dict], list[str]]:
    """Fold this run's records into whatever `events.jsonl` already holds.

    `events.jsonl` must mirror what is on disk in the output directory, and a
    partial run — `--type`, `--limit`, or both — only ever rewrites part of
    that directory. Truncating the file to this run's records therefore threw
    away the events for pages still sitting on disk: a `generate --type
    bank_statements` followed by `serialise` wrote 55 fresh transcripts beside
    110 stale ones, with nothing on either side recording that the corpus was
    now a mixture. Images already survive a partial run — they live in per-type
    subdirectories and only the regenerated ones are overwritten — so merging
    brings the event stream in line with how the images have always behaved,
    rather than inventing a new rule.

    A record is keyed by `(case_id, doc_type)`, the same granularity a partial
    run rewrites at. Fresh records replace matching ones in place, which keeps
    file order stable across runs; unmatched fresh records append. Carried-over
    records whose image has since left the disk are dropped rather than
    preserved, because a record with no page is precisely the divergence this
    file exists to prevent.

    Args:
        events_path: The `events.jsonl` this run is about to write.
        fresh: The records this run captured, in render order.
        image_dir_for: Directory holding each doc type's images, keyed by type
            — the per-type output subdirectory, or the flat override directory.

    Returns:
        The merged records to write, and the `image_file` of every carried-over
        record dropped for having no page on disk.
    """
    if not events_path.exists():
        return fresh, []

    existing = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    by_key = {(r["case_id"], r["doc_type"]): r for r in fresh}

    dropped: list[str] = []
    merged: list[dict] = []
    replaced: set[tuple[str, str]] = set()
    for record in existing:
        key = (record["case_id"], record["doc_type"])
        replacement = by_key.get(key)
        if replacement is not None:
            merged.append(replacement)
            replaced.add(key)
            continue
        directory = image_dir_for.get(record["doc_type"])
        if directory is None or not (directory / record["image_file"]).exists():
            dropped.append(record["image_file"])
            continue
        merged.append(record)

    merged.extend(r for r in fresh if (r["case_id"], r["doc_type"]) not in replaced)
    return merged, dropped


def _load_event_records(derived_dir: Path) -> list[dict]:
    """Read `events.jsonl` from a derived directory.

    Args:
        derived_dir: The directory `generate` wrote its events into.

    Returns:
        One record per rendered document.

    Raises:
        typer.Exit: With code 1 when the file is absent.
    """
    events_path = derived_dir / "events.jsonl"
    if not events_path.exists():
        rprint("[red]No captured events found.[/red]")
        rprint(f"[red]  What:     {events_path} does not exist.[/red]")
        rprint(f"[red]  Where:    {events_path.resolve()}[/red]")
        rprint("[red]  Expected: the event stream `generate` writes as it renders.[/red]")
        rprint("[red]  Recover:  run `python -m generators.pipeline generate` first.[/red]")
        raise typer.Exit(1) from None
    return [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]


class ScoreInputError(RuntimeError):
    """Raised when the corpus or the predictions cannot be trusted to score."""


def _score_input_err(what: str, *, where: str, expected: str, recover: str) -> ScoreInputError:
    """Build a four-element fail-fast diagnostic."""
    return ScoreInputError(
        "Cannot score.\n"
        f"  What:     {what}\n"
        f"  Where:    {where}\n"
        f"  Expected: {expected}\n"
        f"  Recover:  {recover}"
    )


def _verify_corpus(corpus: Path) -> list[dict]:
    """Read the manifest and check every image against its recorded hash.

    The shipped README calls this "not ceremony". A mismatch means the
    predictions and the ground truth are different vintages, and any number
    computed from them is meaningless -- so scoring refuses rather than
    reporting (scoring spec §6).

    Args:
        corpus: An exported `parsing_YYYYMMDD/` directory.

    Returns:
        The manifest rows, in file order.

    Raises:
        ScoreInputError: The manifest is absent or any image hash differs.
    """
    manifest = corpus / "manifest.jsonl"
    if not manifest.exists():
        raise _score_input_err(
            f"{manifest} does not exist.",
            where=str(manifest.resolve()),
            expected="the manifest.jsonl written by `export`, one JSON row per case, e.g.\n"
            '              {"image": "images/CASE001_invoices.png", ...}',
            recover="pass --corpus pointing at an exported parsing_YYYYMMDD/ directory, "
            "or run `python -m generators.pipeline export` to produce one.",
        )

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        image = corpus / row["image"]
        if not image.exists():
            raise _score_input_err(
                f"{row['image']} is in the manifest but not on disk.",
                where=str(image.resolve()),
                expected="every manifest row to name an image present in the corpus.",
                recover="re-export the corpus so manifest and images agree.",
            )
        if sha256_of(image) != row["sha256"]:
            raise _score_input_err(
                f"{row['image']} does not match its sha256 in the manifest.",
                where=f"{manifest.resolve()} -> {row['image']}",
                expected=f"sha256 {row['sha256']}.",
                recover="score against the corpus these predictions were produced from, or "
                "re-run the predictions against this corpus. Scoring across vintages is "
                "the failure the manifest exists to prevent.",
            )
    return rows


def _pair_predictions(corpus: Path, predictions: Path) -> dict[str, dict[str, Path]]:
    """Pair every transcript with one prediction per system, by filename stem.

    Each immediate subdirectory of `predictions` is one system, named by the
    directory. The rule is stated rather than sniffed, so a directory of loose
    `.md` files is an error rather than an anonymous system (scoring spec §6).

    Args:
        corpus: An exported `parsing_YYYYMMDD/` directory.
        predictions: Directory whose subdirectories are systems.

    Returns:
        System name -> {transcript stem -> prediction path}.

    Raises:
        ScoreInputError: No subdirectories, or any system's files do not
            exactly cover the corpus's transcripts.
    """
    if not predictions.is_dir():
        raise _score_input_err(
            f"{predictions} is not a directory.",
            where=str(predictions.resolve()),
            expected="a directory whose immediate subdirectories are systems, e.g.\n"
            "              runs/docling/CASE001_invoices.md",
            recover="create one directory per system under the --predictions path.",
        )

    systems = sorted(p for p in predictions.iterdir() if p.is_dir())
    if not systems:
        raise _score_input_err(
            f"{predictions} contains no subdirectory.",
            where=str(predictions.resolve()),
            expected="one subdirectory per system, each holding predictions named for the "
            "transcript stems, e.g.\n              runs/docling/CASE001_invoices.md",
            recover="move the prediction files into a subdirectory named for the system "
            "that produced them.",
        )

    expected_stems = {p.stem for p in (corpus / "transcripts").glob("*.md")}
    paired: dict[str, dict[str, Path]] = {}
    for system in systems:
        found = {p.stem: p for p in system.glob("*.md")}
        # A page the system declares it cannot produce is scored as a total
        # failure, not excused and not silently dropped: every system stays
        # averaged over the same transcripts, so the numbers remain comparable.
        # An UNdeclared gap is still refused — silence is indistinguishable from
        # a run that died half-way.
        declared = read_unproduced(system)
        missing = sorted(expected_stems - set(found) - declared)
        if missing:
            raise _score_input_err(
                f"system '{system.name}' has no prediction for {len(missing)} transcript(s): "
                f"{missing[:5]}{' ...' if len(missing) > 5 else ''}.",
                where=str(system.resolve()),
                expected=f"one .md per transcript stem, {len(expected_stems)} in total.",
                recover="re-run inference for the missing cases, declare them unproducible "
                f"in {system.name}/_unproduced.json, or score a corpus subset by exporting one.",
            )
        undeclared_but_present = sorted(declared & set(found))
        if undeclared_but_present:
            raise _score_input_err(
                f"system '{system.name}' declares {len(undeclared_but_present)} page(s) "
                f"unproducible but also has predictions for them: {undeclared_but_present[:5]}.",
                where=str(system / "_unproduced.json"),
                expected="a page is either produced or declared unproducible, never both.",
                recover="remove the stale declaration, or remove the prediction.",
            )
        extra = sorted(set(found) - expected_stems)
        if extra:
            raise _score_input_err(
                f"system '{system.name}' has {len(extra)} prediction(s) with no transcript: "
                f"{extra[:5]}{' ...' if len(extra) > 5 else ''}.",
                where=str(system.resolve()),
                expected="every prediction to name a transcript stem in this corpus.",
                recover="remove the extra files, or score against the corpus they came from "
                "-- an extra file usually means two corpora have been mixed.",
            )
        paired[system.name] = found
    return paired


@app.command()
def serialise(
    config: Annotated[Path, typer.Option(help="Path to generation_config.yml")] = _DEFAULT_CONFIG,
    policy: Annotated[Path, typer.Option("--policy", help="Path to serialisation.yml")] = _DEFAULT_POLICY,
    derived: Annotated[
        Path | None, typer.Option("--derived", help="Override the configured derived directory.")
    ] = None,
) -> None:
    """Turn captured events into Markdown transcripts.

    A pure function of events and policy: it renders nothing and imports no
    renderer. That is why it is a separate command (design §6) — the
    convention is the risky part of this design, so it can change and every
    transcript re-emit in seconds without re-rendering a single image.

    Raises:
        typer.Exit: With code 1 when no events have been captured.
    """
    cfg = load_generation_config(config)
    derived_dir = derived if derived is not None else Path(cfg["derived_dir"])
    records = _load_event_records(derived_dir)
    convention = load_serialisation_policy(policy)

    transcripts_dir = derived_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        target = transcripts_dir / (Path(record["image_file"]).stem + ".md")
        target.write_text(serialise_events(record["events"], convention), encoding="utf-8")

    rprint(f"[green]Serialised {len(records)} transcripts into {transcripts_dir}.[/green]")


@app.command()
def preview(
    case_id: Annotated[str, typer.Argument(help="The case to preview, e.g. CASE001")],
    config: Annotated[Path, typer.Option(help="Path to generation_config.yml")] = _DEFAULT_CONFIG,
    policy: Annotated[Path, typer.Option("--policy", help="Path to serialisation.yml")] = _DEFAULT_POLICY,
    derived: Annotated[
        Path | None, typer.Option("--derived", help="Override the configured derived directory.")
    ] = None,
) -> None:
    """Print one document's transcript beside its image path.

    Exists so the design §8.5 visual check has something to check against: a
    transcription corpus's correctness is ultimately visual, and no automated
    check catches a transcript that parses cleanly but describes the wrong page.

    Raises:
        typer.Exit: With code 1 when the case has no captured events.
    """
    cfg = load_generation_config(config)
    derived_dir = derived if derived is not None else Path(cfg["derived_dir"])
    records = _load_event_records(derived_dir)
    convention = load_serialisation_policy(policy)

    matches = [record for record in records if record["case_id"] == case_id]
    if not matches:
        rprint(f"[red]No captured events for case '{case_id}'.[/red]")
        rprint(f"[red]  Known cases: {sorted({r['case_id'] for r in records})[:8]} ...[/red]")
        raise typer.Exit(1) from None

    for record in matches:
        subdir = cfg["document_types"][record["doc_type"]]["output_subdir"]
        image_path = Path(cfg["output_dir"]) / subdir / record["image_file"]
        rprint(f"[bold]{record['case_id']}[/bold]  ({record['doc_type']})")
        rprint(f"[cyan]image:[/cyan] {image_path}")
        rprint("[cyan]transcript:[/cyan]")
        print(serialise_events(record["events"], convention))


@app.command()
def export(
    config: Annotated[Path, typer.Option(help="Path to generation_config.yml")] = _DEFAULT_CONFIG,
    policy: Annotated[Path, typer.Option("--policy", help="Path to serialisation.yml")] = _DEFAULT_POLICY,
    prompt: Annotated[Path, typer.Option("--prompt", help="Path to prompt.md")] = _DEFAULT_PROMPT,
    derived: Annotated[
        Path | None, typer.Option("--derived", help="Override the configured derived directory.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Override the configured output directory.")
    ] = None,
    target: Annotated[
        Path, typer.Option("--target", help="Directory to create the export inside.")
    ] = Path(),
    date: Annotated[
        str | None, typer.Option("--date", help="Corpus date stamp, YYYYMMDD. Defaults to today.")
    ] = None,
) -> None:
    """Assemble the dated deliverable directory (design §6.1).

    Copies images and transcripts verbatim — it never re-renders or
    re-serialises — and adds the three artifacts that make the corpus
    interpretable away from this checkout: a hashed manifest, a copy of the
    policy that produced the transcripts, and the prompt they assume.

    Raises:
        typer.Exit: With code 1 when a needed artifact is missing.
    """
    cfg = load_generation_config(config)
    derived_dir = derived if derived is not None else Path(cfg["derived_dir"])
    output_dir = output if output is not None else Path(cfg["output_dir"])
    records = _load_event_records(derived_dir)
    date_stamp = date if date is not None else datetime.now().strftime("%Y%m%d")

    try:
        root = export_corpus(
            records,
            images_root=output_dir,
            transcripts_dir=derived_dir / "transcripts",
            policy_path=policy,
            prompt_path=prompt,
            target=target,
            date_stamp=date_stamp,
        )
    except ExportError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    rprint(f"[green]Exported {len(records)} documents into {root}.[/green]")
    rprint("[cyan]Verify every image against its sha256 in manifest.jsonl before scoring.[/cyan]")


@app.command()
def score(
    corpus: Annotated[Path, typer.Option("--corpus", help="An exported parsing_YYYYMMDD/ directory.")],
    predictions: Annotated[
        Path, typer.Option("--predictions", help="Directory whose subdirectories are systems.")
    ],
    policy: Annotated[Path, typer.Option("--policy", help="Path to scoring.yml")] = _DEFAULT_SCORING,
    report: Annotated[Path, typer.Option("--report", help="Where to write the JSON report.")] = Path(
        "scores.json"
    ),
) -> None:
    """Score predictions against the corpus and classify every divergence.

    Reports two metrics per system -- normalised, which measures reading, and
    strict, which measures reading plus convention adherence -- and groups the
    divergences behind them into convention mismatches and reading errors, so
    the §8.6 calibration pass can tell which of the two it is looking at.

    Raises:
        typer.Exit: With code 1 when the corpus fails verification, the
            predictions do not cover it, or the policy is invalid.
    """
    try:
        convention = load_scoring_policy(policy)
        _verify_corpus(corpus)
        paired = _pair_predictions(corpus, predictions)
    except (ScoringError, ScoreInputError) as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    transcripts = {p.stem: p for p in (corpus / "transcripts").glob("*.md")}
    systems: dict[str, dict] = {}
    hunks_by_system: dict[str, list] = {}

    for system, files in sorted(paired.items()):
        totals = {
            "strict": {"char_distance": 0, "chars": 0, "word_distance": 0, "words": 0},
            "normalised": {"char_distance": 0, "chars": 0, "word_distance": 0, "words": 0},
        }
        per_document: list[dict] = []
        system_hunks: list = []
        # Scored separately and never folded into CER: normalisation strips the
        # pipes that encode column membership, so CER is blind to an amount
        # filed under the wrong heading -- the one error a bank statement
        # cannot tolerate.
        # Two independent questions about an extracted amount, kept apart because
        # a single rate answers neither: is the number on the page at all, and
        # was it taken from the right place. `usable` is their conjunction and is
        # what a consumer feels; it cannot rank a system on either dimension.
        columns = {
            "amounts": 0,
            "read": 0,
            "placed": 0,
            "misfiled": 0,
            "usable": 0,
            "documents_mismatched": 0,
        }
        # Table structure, scored apart from CER and apart from column
        # assignment: a system can transcribe cells near-perfectly while
        # shredding row segmentation, and one number would hide the trade.
        tables = {
            "truth_rows": 0,
            "aligned": 0,
            "fragments": 0,
            "width_breaks": 0,
            # Row-independent, so the merge layouts are not invisible: anything
            # measured over aligned rows flatters exactly the system whose rows
            # do not align.
            "recalled_cells": 0,
            "truth_cells": 0,
        }
        # Convention-blind, and the only metric here that is: amounts are read
        # out of the raw text, so a system emitting HTML tables and one
        # emitting pipe tables are compared on identical terms. It inverts the
        # CER ordering, which is the point of reporting it.
        numbers = {
            "truth_amounts": 0,
            "prediction_amounts": 0,
            "matched": 0,
            "literal": 0,
            "misread": 0,
            "dropped": 0,
            "invented": 0,
            "documents_with_an_error": 0,
        }

        # Declared-unproducible pages are scored as empty predictions, which is
        # a total failure by construction: the edit distance is the length of
        # the truth, so CER and WER are 1.0. That keeps every system averaged
        # over the same transcripts rather than quietly shrinking one's
        # denominator.
        unproduced = read_unproduced(predictions / system)

        for stem in sorted(set(files) | unproduced):
            truth = transcripts[stem].read_text(encoding="utf-8")
            prediction = "" if stem in unproduced else files[stem].read_text(encoding="utf-8")
            pairs = {
                "strict": (truth, prediction),
                "normalised": (normalise(truth, convention), normalise(prediction, convention)),
            }
            structure = table_report(truth, prediction)
            for key in tables:
                tables[key] += structure[key] or 0

            integrity = column_integrity(truth, prediction)
            columns["amounts"] += integrity["amounts"]
            columns["misfiled"] += integrity["misfiled"]
            columns["read"] += integrity["read"]
            columns["placed"] += integrity["placed"]
            # Read correctly AND filed correctly. The only one of these numbers
            # a downstream consumer can act on.
            columns["usable"] += integrity["amounts"] - integrity["misfiled"]
            columns["documents_mismatched"] += 0 if integrity["columns_match"] else 1

            figures = numeric_fidelity(truth, prediction)
            for key in figures:
                numbers[key] += figures[key]
            if figures["misread"] or figures["dropped"] or figures["invented"]:
                numbers["documents_with_an_error"] += 1

            row: dict[str, str | float | bool] = {"stem": stem}
            for metric, (left, right) in pairs.items():
                char_distance, char_rate = cer(left, right)
                word_distance, word_rate = wer(left, right)
                totals[metric]["char_distance"] += char_distance
                totals[metric]["chars"] += len(left)
                totals[metric]["word_distance"] += word_distance
                totals[metric]["words"] += len(left.split())
                row[f"{metric}_cer"] = char_rate
                row[f"{metric}_wer"] = word_rate

            # Per-document structural and numeric counts, not only the corpus
            # aggregate. Every interesting reading of this benchmark has been a
            # per-document-type one -- bank statements are the only hard tables,
            # and averaging them with near-saturated invoices hides the result --
            # so a report that carries only totals forces every such breakdown
            # to re-read the predictions. Those are gitignored and
            # machine-specific, which would tie any analysis to the machine that
            # produced them. With these fields the JSON report is sufficient on
            # its own.
            row["amounts"] = integrity["amounts"]
            row["misfiled"] = integrity["misfiled"]
            row["read"] = integrity["read"]
            row["placed"] = integrity["placed"]
            row["usable"] = integrity["amounts"] - integrity["misfiled"]
            row["columns_match"] = integrity["columns_match"]
            row["truth_rows"] = structure["truth_rows"]
            row["aligned"] = structure["aligned"]
            row["fragments"] = structure["fragments"]
            row["width_breaks"] = structure["width_breaks"]
            row["truth_amounts"] = figures["truth_amounts"]
            row["amounts_correct"] = figures["matched"]
            row["misread"] = figures["misread"]
            row["dropped"] = figures["dropped"]
            row["invented"] = figures["invented"]
            per_document.append(row)
            system_hunks.extend(hunks(truth, prediction, convention))

        systems[system] = {
            metric: {
                "cer": error_rate(totals[metric]["char_distance"], totals[metric]["chars"]),
                "wer": error_rate(totals[metric]["word_distance"], totals[metric]["words"]),
                "distance": totals[metric]["char_distance"],
            }
            for metric in ("normalised", "strict")
        }
        systems[system]["macro"] = {
            f"{metric}_{stat}": (
                sum(row[f"{metric}_{stat}"] for row in per_document) / len(per_document)
                if per_document
                else 0.0
            )
            for metric in ("normalised", "strict")
            for stat in ("cer", "wer")
        }
        systems[system]["columns"] = columns
        systems[system]["tables"] = tables
        systems[system]["numbers"] = numbers
        systems[system]["documents"] = per_document
        hunks_by_system[system] = system_hunks

    grouped = group(hunks_by_system)
    payload = {
        "corpus": str(corpus),
        "policy": str(policy),
        "systems": systems,
        "divergences": grouped,
    }
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    _print_score_report(systems, grouped, report)


def _print_score_report(systems: dict, grouped: dict, report: Path) -> None:
    """Render the JSON payload as terminal tables.

    Derived from the same structure that was written to disk rather than
    computed separately, so the terminal and the report cannot disagree.

    Args:
        systems: Per-system aggregates and per-document rows.
        grouped: Divergence groups, keyed by class.
        report: Where the JSON was written, echoed for the reader.
    """
    table = Table(title="Scores (micro-averaged; macro in the JSON report)")
    table.add_column("system")
    table.add_column("normalised CER", justify="right")
    table.add_column("normalised WER", justify="right")
    table.add_column("strict CER", justify="right")
    table.add_column("strict WER", justify="right")
    for system, scores in sorted(systems.items()):
        table.add_row(
            system,
            f"{scores['normalised']['cer']:.4f}",
            f"{scores['normalised']['wer']:.4f}",
            f"{scores['strict']['cer']:.4f}",
            f"{scores['strict']['wer']:.4f}",
        )
    rprint(table)

    _print_column_integrity(systems)
    _print_table_structure(systems)
    _print_numeric_fidelity(systems)
    _print_divergences(grouped, report)


def _print_column_integrity(systems: dict) -> None:
    """Report the two independent ways an extracted amount can be wrong.

    **1. read** — is the number on the page at all? The share of the page's
    amounts whose value appears anywhere in the prediction's tables. Position is
    discarded, so this is reading alone.

    **2. placed** — was it taken from the right position? Of the amounts that
    WERE read, the share sitting under the heading the page puts them under.
    Conditional on reading, so a system is neither credited for placing amounts
    it never produced nor charged for misplacing ones it never read.

    They are orthogonal and they rank systems differently: measured 2026-08-20
    on bank statements, InternVL3.5-8B reads 5.3 points more amounts than
    gemma-4-12B and places 8.8 points fewer of them correctly. Quoting either
    alone misdescribes both.

    **usable** is their conjunction — right value, right heading — and is what a
    downstream consumer feels, since an amount read perfectly and filed wrongly
    is indistinguishable from one misread. It is the deployment figure and the
    wrong tool for diagnosing which dimension failed.

    Reported apart from CER on purpose. The pages draw no vertical rules, so a
    model infers column membership rather than reading it; normalisation then
    strips the pipes that carry the inference, leaving CER unable to see it
    fail. An amount in the wrong column is money in reported as money out.

    Args:
        systems: Per-system aggregates carrying a `columns` block.
    """
    integrity = Table(title="Extracted amounts: is it on the page, and is it in the right place")
    integrity.add_column("system")
    integrity.add_column("amounts", justify="right")
    integrity.add_column("1. read", justify="right")
    integrity.add_column("2. placed", justify="right")
    integrity.add_column("usable", justify="right")
    integrity.add_column("docs with wrong column count", justify="right")

    for system, scores in sorted(systems.items()):
        counts = scores.get("columns")
        if not counts:
            continue
        total = counts["amounts"]
        read = (counts["read"] / total * 100) if total else 0.0
        # Conditional on having been read, which is what makes it a measure of
        # placement rather than of reading.
        placed = (counts["placed"] / counts["read"] * 100) if counts["read"] else None
        usable = (counts["usable"] / total * 100) if total else 0.0
        integrity.add_row(
            system,
            f"{total}",
            f"{read:.1f}%",
            f"{placed:.1f}%" if placed is not None else "-",
            f"{usable:.1f}%",
            f"{counts['documents_mismatched']}",
        )
    rprint(integrity)


def _print_table_structure(systems: dict) -> None:
    """Report row segmentation and cell content, apart from each other and CER.

    These pages draw no vertical rules, so a system infers table structure
    rather than reading it, and CER discards the pipes that carry the
    inference. Row segmentation and cell content are reported separately
    because they dissociate: MinerU transcribes cells at 99.5% while producing
    292 continuation fragments, gemma-4-12B produces none and misses slightly
    more characters. Averaging them would rank the two arbitrarily.

    Args:
        systems: Per-system aggregates carrying a `tables` block.
    """
    structure = Table(title="Table structure (row segmentation, then cell content)")
    structure.add_column("system")
    structure.add_column("rows aligned", justify="right")
    structure.add_column("fragments", justify="right")
    structure.add_column("width breaks", justify="right")
    structure.add_column("content recall", justify="right")

    for system, scores in sorted(systems.items()):
        counts = scores.get("tables")
        if not counts:
            continue
        rows = counts["truth_rows"]
        aligned = f"{counts['aligned']}/{rows}" if rows else "-"
        recall = f"{counts['recalled_cells'] / counts['truth_cells']:.3f}" if counts["truth_cells"] else "-"
        structure.add_row(
            system,
            aligned,
            f"{counts['fragments']}",
            f"{counts['width_breaks']}",
            recall,
        )
    rprint(structure)


def _print_numeric_fidelity(systems: dict) -> None:
    """Print amount accuracy, which no other metric in this report measures.

    CER weighs a wrong digit in a total exactly as it weighs a typo in a
    merchant's name. This separates them, and it is convention-blind: amounts
    come out of the raw text, so an HTML dialect and a pipe dialect are
    compared on the same terms.

    `misread` and `dropped` are split rather than summed because they fail
    differently — one system saw the figure and got a digit wrong, the other
    never emitted one.

    Args:
        systems: Per-system aggregates carrying a `numbers` block.
    """
    figures = Table(title="Numeric fidelity (amounts, independent of formatting)")
    figures.add_column("system")
    figures.add_column("amounts", justify="right")
    figures.add_column("correct", justify="right")
    figures.add_column("misread", justify="right")
    figures.add_column("dropped", justify="right")
    figures.add_column("invented", justify="right")
    figures.add_column("docs with an error", justify="right")

    for system, scores in sorted(systems.items()):
        counts = scores.get("numbers")
        if not counts:
            continue
        total = counts["truth_amounts"]
        correct = f"{100 * counts['matched'] / total:.1f}%" if total else "-"
        figures.add_row(
            system,
            f"{total}",
            correct,
            f"{counts['misread']}",
            f"{counts['dropped']}",
            f"{counts['invented']}",
            f"{counts['documents_with_an_error']}",
        )
    rprint(figures)


def _print_divergences(grouped: dict, report: Path) -> None:
    """Print the grouped convention and reading divergences.

    Args:
        grouped: Divergences grouped corpus-wide by class.
        report: Where the full JSON report was written.
    """
    for kind, heading in ((CONVENTION, "Convention mismatches"), (READING, "Reading errors")):
        entries = grouped[kind]
        if not entries:
            rprint(f"[green]{heading}: none.[/green]")
            continue
        divergences = Table(title=f"{heading} (top 20 of {len(entries)})")
        divergences.add_column("count", justify="right")
        divergences.add_column("transcript")
        divergences.add_column("prediction")
        divergences.add_column("systems")
        for entry in entries[:20]:
            divergences.add_row(
                str(entry["count"]),
                entry["truth"][:60],
                entry["prediction"][:60],
                ", ".join(entry["systems"]),
            )
        rprint(divergences)

    rprint(f"[green]Full report written to {report}.[/green]")


if __name__ == "__main__":
    app()

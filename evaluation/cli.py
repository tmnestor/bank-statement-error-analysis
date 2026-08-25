"""Score predictions against an exported corpus.

Usage:
    python -m evaluation.cli --corpus parsing_20260820 --predictions runs

Reports normalised and strict CER/WER, column integrity, table structure and
numeric fidelity, and classifies every divergence as a convention mismatch or a
reading error.

The corpus is verified before anything is scored: the manifest's image hashes
make scoring against the wrong vintage impossible rather than merely detectable
afterwards. A system with missing predictions is refused, because a silent gap
is indistinguishable from a run that died half-way -- declare the gap in
`_unproduced.json` and it is scored as a total failure instead.

This was `evaluation.cli` until the corpus generator moved to its own
repository. The interface between the two is the exported corpus directory, not
shared code.
"""

import hashlib
import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.table import Table

from evaluation.columns import column_integrity
from evaluation.divergence import CONVENTION, READING, group, hunks
from evaluation.metrics import cer, error_rate, wer
from evaluation.numerics import numeric_fidelity
from evaluation.scoring import ScoringError, load_scoring_policy, normalise
from evaluation.tables import table_report
from evaluation.unproduced import read_unproduced

app = typer.Typer(add_completion=False, help="Score predictions against an exported corpus.")

# Defaults the score command reads. The serialisation policy is the corpus's own
# convention and the scoring policy is how that convention is compared; both
# ship beside the data, and these paths are only the fallback when the caller
# does not point at the corpus copy.
_DEFAULT_POLICY = Path("config/serialisation.yml")
_DEFAULT_SCORING = Path("config/scoring.yml")

# THE CONTRACT WITH THE CORPUS GENERATOR, duplicated on purpose.
#
# The generator writes each image's sha256 into the manifest; this verifies it.
# The two live in separate repositories now, so the function cannot be imported
# from one into the other -- and it must not be, because the point is that two
# independent implementations agree. Chunked because a corpus image is megabytes
# and the manifest is checked for every page before anything is scored.
_CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    """Return the hex sha256 of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


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
            "or run `python -m evaluation.cli export` to produce one.",
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

    expected_stems = {p.stem for p in (corpus / "transcripts").glob("*.md") if not is_sidecar(p)}
    paired: dict[str, dict[str, Path]] = {}
    for system in systems:
        found = {p.stem: p for p in system.glob("*.md") if not is_sidecar(p)}
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

    transcripts = {p.stem: p for p in (corpus / "transcripts").glob("*.md") if not is_sidecar(p)}
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
            # Right value, right column, and in a row carrying the date that
            # identifies it. `usable` discards the row, which flatters exactly
            # the systems whose failure is row segmentation -- MinerU scores 56
            # of 56 placed and 0 attributable on CASE015, because every amount
            # sits under the correct heading in a row with an empty date.
            "attributable": 0,
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
            columns["attributable"] += integrity["attributable"]
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
            row["attributable"] = integrity["attributable"]
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
    integrity.add_column("attributable", justify="right")
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
        attributable = (counts["attributable"] / total * 100) if total else 0.0
        integrity.add_row(
            system,
            f"{total}",
            f"{read:.1f}%",
            f"{placed:.1f}%" if placed is not None else "-",
            f"{usable:.1f}%",
            f"{attributable:.1f}%",
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


def is_sidecar(path: Path) -> bool:
    """Whether a path is an OS sidecar rather than corpus content.

    macOS stores extended attributes in an AppleDouble companion named
    `._<file>` whenever it writes to a filesystem that cannot hold them,
    including the NFS share corpora are staged on. `._CASE001_invoices.md` has
    the right suffix and a plausible stem, so a 55-page corpus reads as 110 and
    every added stem is missing its image and its prediction.

    Deliberately duplicated from `runners.common._is_sidecar`: `generators/` does
    not import `runners/`, which is what lets the runner tests run where no
    parser is installed. Two copies of four lines is cheaper than that coupling.

    Args:
        path: A candidate corpus or prediction file.

    Returns:
        True when the file is an OS artefact to be ignored.
    """
    return path.name.startswith("._") or path.name == ".DS_Store"


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

if __name__ == "__main__":
    app()

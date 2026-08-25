"""Project the authored ground truth into the format LMM_POC's extractor reads.

This corpus is built for **transcription**: a page image paired with a canonical
Markdown transcript. LMM_POC does **information extraction**: a page image paired
with named field values. Both are derived from the same authored YAML, so this is
a projection rather than a second set of labels — nothing here is annotated,
estimated or re-derived from a rendered page.

That distinction matters. The predecessor repo's extraction machinery was
deliberately left behind (see CLAUDE.md), and this does not bring it back: it
reads `ground_truth/bank_statements.yml`, which this repo already authors and
validates, and writes it out in the shape a downstream consumer expects.

## The three rules that make the output match

Read against `evaluation_data/degraded_20260812/ground_truth.jsonl`, which is
what LMM_POC consumes today.

**1. Five fields, not nine.** `config/field_definitions.yaml` in LMM_POC
declares `bank_statement` as exactly `DOCUMENT_TYPE`, `STATEMENT_DATE_RANGE`,
`LINE_ITEM_DESCRIPTIONS`, `TRANSACTION_DATES`, `TRANSACTION_AMOUNTS_PAID`. This
repo authors four more — `SUPPLIER_NAME`, `TRANSACTION_AMOUNTS_RECEIVED`,
`ACCOUNT_BALANCE`, `PAYER_NAME`. They are dropped, because a field the
extractor does not declare is a field it will score as spurious.

**2. Credit rows are dropped, from every list together.** This repo keeps
parallel lists with `NOT_FOUND` placeholders so index *i* is the same
transaction in all of them. LMM_POC's records carry no `NOT_FOUND` and its three
lists are equal-length — 19 entries where the page has 27 transactions. The
difference is exactly the rows with no paid amount. So a transaction survives
only if it has one, and it is removed from all three lists at once or the
correspondence breaks.

**This is lossy and it is the consumer's choice, not ours.** A statement's
credit rows are on the page and in this repo's ground truth; they are absent
from the extraction schema. Anything scored against this file cannot be
penalised for missing them, and cannot be credited for finding them.

**3. Formatting.** ` | ` with spaces, and amounts carry a `$`. This repo stores
`328.15|219.04`; LMM_POC expects `$328.15 | $219.04`.
"""

import csv
import json
from pathlib import Path

import yaml

# LMM_POC's declared bank_statement field set, in its order. Sourced from
# config/field_definitions.yaml -> document_fields.bank_statement, and the order
# is the order its CSV header uses.
BANK_STATEMENT_FIELDS = (
    "DOCUMENT_TYPE",
    "STATEMENT_DATE_RANGE",
    "LINE_ITEM_DESCRIPTIONS",
    "TRANSACTION_DATES",
    "TRANSACTION_AMOUNTS_PAID",
)

# This repo's name -> LMM_POC's name, where they differ.
_RENAMED = {"TRANSACTION_DESCRIPTIONS": "LINE_ITEM_DESCRIPTIONS"}

# The three parallel lists, which must be filtered together.
_PARALLEL = ("TRANSACTION_DATES", "LINE_ITEM_DESCRIPTIONS", "TRANSACTION_AMOUNTS_PAID")

_ABSENT = "NOT_FOUND"
_SEPARATOR = " | "


class EvalExportError(RuntimeError):
    """Raised when the authored ground truth cannot be projected."""


def _split(value: str) -> list[str]:
    """Split one of this repo's pipe-delimited lists."""
    return [part.strip() for part in str(value).split("|")]


def _as_money(value: str) -> str:
    """Render an amount the way the consumer's records do."""
    text = value.strip()
    return text if text.startswith("$") or text == _ABSENT else f"${text}"


def project_bank_statement(fields: dict) -> dict:
    """Project one authored entry into LMM_POC's five-field record.

    Args:
        fields: The `fields:` mapping of one `ground_truth/bank_statements.yml`
            entry.

    Returns:
        The projected record, without `filename` — the caller adds that, since
        one entry yields several files once degraded.

    Raises:
        EvalExportError: The parallel lists disagree in length, which would make
            row filtering silently mis-pair dates with amounts.
    """
    renamed = {_RENAMED.get(key, key): value for key, value in fields.items()}

    lists = {name: _split(renamed.get(name, "")) for name in _PARALLEL}
    lengths = {name: len(values) for name, values in lists.items()}
    if len(set(lengths.values())) != 1:
        raise EvalExportError(
            "Cannot project this statement.\n"
            f"  What:     its parallel lists disagree in length: {lengths}. Rows are "
            "filtered by index across all three, so unequal lists would pair a date "
            "with another transaction's amount.\n"
            "  Where:    ground_truth/bank_statements.yml, this entry's fields\n"
            "  Expected: TRANSACTION_DATES, TRANSACTION_DESCRIPTIONS and "
            "TRANSACTION_AMOUNTS_PAID to hold the same number of pipe-delimited "
            "items, one per transaction on the page.\n"
            "  Recover:  `validate` checks this — run it and fix the entry it names."
        )

    # Rule 2: keep only transactions with a paid amount, and drop them from every
    # list at once.
    keep = [index for index, amount in enumerate(lists["TRANSACTION_AMOUNTS_PAID"]) if amount != _ABSENT]

    record = {"DOCUMENT_TYPE": renamed.get("DOCUMENT_TYPE", "BANK_STATEMENT")}
    record["STATEMENT_DATE_RANGE"] = renamed.get("STATEMENT_DATE_RANGE", _ABSENT)
    for name in _PARALLEL:
        values = [lists[name][index] for index in keep]
        if name == "TRANSACTION_AMOUNTS_PAID":
            values = [_as_money(v) for v in values]
        record[name] = _SEPARATOR.join(values)
    return {key: record[key] for key in BANK_STATEMENT_FIELDS}


def load_entries(path: Path) -> dict[str, dict]:
    """Read the authored ground truth, keyed by case id.

    Args:
        path: `ground_truth/bank_statements.yml`.

    Returns:
        case id -> its `fields` mapping.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    # The file is a mapping keyed by case id — `CASE001: {layout:, fields:}` —
    # so the id is the key rather than a field inside the entry. A list of
    # entries carrying `case_id:` is accepted too, since that is the shape the
    # predecessor used and the two are easy to confuse.
    if isinstance(document, dict):
        pairs = [(str(key), value) for key, value in document.items()]
    else:
        pairs = [(str(entry.get("case_id", "")), entry) for entry in document]

    keyed: dict[str, dict] = {}
    for case_id, entry in pairs:
        if not case_id or "fields" not in entry:
            raise EvalExportError(
                "Cannot project the ground truth.\n"
                f"  What:     an entry in {path} has no case id or no `fields:` block, "
                "so its record cannot be matched to a page image.\n"
                f"  Where:    {path.resolve()}\n"
                "  Expected: a mapping of case id to entry, e.g.\n"
                "              CASE001:\n"
                "                layout: cba_standard\n"
                "                fields: {DOCUMENT_TYPE: BANK_STATEMENT, ...}\n"
                "  Recover:  fix the entry, then re-run `validate`."
            )
        keyed[case_id] = entry["fields"]
    return keyed


def write_set(records: list[dict], target: Path) -> tuple[Path, Path]:
    """Write `ground_truth.jsonl` and `ground_truth.csv` beside the images.

    Both are written because the consumer ships both; its CSV names the image
    column `image_file` where the JSONL calls it `filename`, and that difference
    is theirs, not a slip here.

    Args:
        records: Projected records, each carrying `filename`.
        target: The directory holding the images.

    Returns:
        The two paths written.
    """
    target.mkdir(parents=True, exist_ok=True)
    jsonl = target / "ground_truth.jsonl"
    jsonl.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    comma = target / "ground_truth.csv"
    with comma.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_file", *BANK_STATEMENT_FIELDS])
        for record in records:
            writer.writerow([record["filename"], *(record[f] for f in BANK_STATEMENT_FIELDS)])
    return jsonl, comma

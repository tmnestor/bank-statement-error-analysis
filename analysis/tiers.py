"""Which (family, severity) a score report describes, read from the corpus.

**The manifest is the only source.** Labelling used to fall back to the corpus
directory's name and then to the report's own filename, and both are written by
whoever ran the scoring loop: renaming or copying a report relabelled the tier
silently, and a curve drawn from relabelled points is wrong in a way nothing
downstream catches.

This lives apart from its callers because two of them need the same read with
different consequences. `analysis.degradation` plots ladders, where `clean` is
not a rung and is added separately; `analysis.export_results` publishes a table,
where `clean` is the baseline row every other row is read against. Sharing the
read keeps one set of diagnostics rather than two that drift.
"""

import json
from pathlib import Path
from typing import NoReturn, Protocol


class Refuse(Protocol):
    """Builds a caller-appropriate four-element diagnostic and raises it.

    `NoReturn`, not `None`: every implementation raises, and saying so is what
    lets a caller treat the value it was checking as narrowed afterwards.
    """

    def __call__(self, what: str, *, where: str, expected: str, recover: str) -> NoReturn: ...


def read_tier(path: Path, payload: dict, root: Path, refuse: Refuse) -> dict[str, str]:
    """The tier this report scored, from the manifest of the corpus it names.

    A corpus that cannot answer stops the run. That refuses two situations which
    used to pass quietly, and both deserve to stop: a report copied back from
    the sandbox without its corpus, and a corpus exported before the generator
    stated these fields -- which is also a corpus whose images predate later
    corrections, so scoring against it is invalid regardless.

    Args:
        path: The report, named in diagnostics.
        payload: The parsed report, for the `corpus` the scorer recorded in it.
        root: Repository root that a report's `corpus` path is relative to.
        refuse: Raises the caller's own four-element diagnostic.

    Returns:
        `{"family": ..., "severity": ...}`, always -- including `clean`/`none`.
        Callers decide what a clean corpus means to them.
    """
    corpus = str(payload.get("corpus", ""))
    if not corpus:
        refuse(
            f"{path.name} records no `corpus`, so there is no manifest to read its tier from.",
            where=str(path.resolve()),
            expected="every score report to name the corpus it scored, e.g.\n"
            '              {"corpus": "degraded/parsing_20260902_scan-heavy", ...}',
            recover="re-score this run with `python -m evaluation.cli score`, which records "
            "the corpus, or remove a report that predates that field.",
        )

    manifest = root / corpus / "manifest.jsonl"
    if not manifest.exists():
        refuse(
            f"{corpus}/manifest.jsonl is not on disk, so {path.name} cannot be labelled.",
            where=str(manifest.resolve()),
            expected="the corpus a report names to be present beside this repository.",
            recover=f"copy {Path(corpus).name}/ back from the sandbox, or remove the reports "
            "scored against corpora you no longer hold.",
        )

    record: dict = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            break

    family, severity = record.get("family"), record.get("severity")
    if not family or not severity:
        refuse(
            f"{corpus}/manifest.jsonl states no `family`/`severity`, so its tier is unknown.",
            where=str(manifest.resolve()),
            expected="every manifest record to carry both, e.g.\n"
            '              {"image": "images/CASE001_bank_statements.jpg", '
            '"family": "scan", "severity": "heavy", ...}',
            recover="re-export this corpus with a generator that labels its manifest. A corpus "
            "predating those fields also predates later ladder corrections, so its images "
            "are a dead vintage and re-rendering is required in any case.",
        )

    return {"family": str(family), "severity": str(severity)}

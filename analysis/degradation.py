"""Does 99.0% usable survive a scanner?

The deployment case rests on one number measured on pristine renders: 99.0% of
bank-statement amounts usable — right value, under the right heading. Production
receives scans. This reads the per-tier score reports and answers what that
costs, per intake channel.

**The tiers are never averaged.** A mean over six severities describes an image
quality that does not exist. The output is a curve per channel, and the useful
result is where it bends: a gentle slope means the number transfers with a
margin, a cliff between two adjacent tiers locates the quality floor production
must hold above.

Usage, once the runs are scored:

    conda run -n du python -m analysis.degradation
"""

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# The order a ladder is read in. Severity is ordinal, not numeric — the tiers
# are declared points on a scale, and spacing them evenly on the axis would
# assert a linearity nothing measured.
SEVERITIES = ("clean", "light", "moderate", "heavy")


def _refuse(what: str, *, where: str, expected: str, recover: str) -> None:
    """Stop with a four-element diagnostic, in this module's existing shape."""
    raise SystemExit(
        f"Cannot label a degraded tier.\n"
        f"  What:     {what}\n"
        f"  Where:    {where}\n"
        f"  Expected: {expected}\n"
        f"  Recover:  {recover}"
    )


def _tier_of(path: Path, payload: dict, root: Path = REPO) -> dict[str, str] | None:
    """Which (family, severity) produced this report, read from the corpus's manifest.

    **The manifest is the only source.** This used to fall back to the corpus
    directory's name and then to the report's own filename, and both are written
    by whoever ran the scoring loop: renaming or copying a report relabelled the
    tier silently, and a curve drawn from relabelled points is wrong in a way
    nothing downstream catches. Three sources also meant a reader had to work
    out which one produced the label in front of them.

    So a corpus that cannot answer stops the run. That refuses two situations
    which used to pass quietly, and both deserve to stop: a report copied back
    from the sandbox without its corpus, and a corpus exported before the
    generator stated these fields -- which is also a corpus whose images predate
    later corrections, so scoring against it is invalid regardless.

    Args:
        path: The report, named in diagnostics.
        payload: The parsed report, for the `corpus` the scorer recorded in it.
        root: Repository root that a report's `corpus` path is relative to.

    Returns:
        `{"family": ..., "severity": ...}`, or None when the manifest says this
        corpus is `clean` -- a classification, not a failure to find one. The
        clean point is not a rung; `collect` adds it to both ladders separately.
    """
    corpus = str(payload.get("corpus", ""))
    if not corpus:
        _refuse(
            f"{path.name} records no `corpus`, so there is no manifest to read its tier from.",
            where=str(path.resolve()),
            expected="every score report to name the corpus it scored, e.g.\n"
            '              {"corpus": "degraded/parsing_20260902_scan-heavy", ...}',
            recover="re-score this run with `python -m evaluation.cli score`, which records "
            "the corpus, or remove a report that predates that field.",
        )

    manifest = root / corpus / "manifest.jsonl"
    if not manifest.exists():
        _refuse(
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
        _refuse(
            f"{corpus}/manifest.jsonl states no `family`/`severity`, so its tier is unknown.",
            where=str(manifest.resolve()),
            expected="every manifest record to carry both, e.g.\n"
            '              {"image": "images/CASE001_bank_statements.jpg", '
            '"family": "scan", "severity": "heavy", ...}',
            recover="re-export this corpus with a generator that labels its manifest. A corpus "
            "predating those fields also predates later ladder corrections, so its images "
            "are a dead vintage and re-rendering is required in any case.",
        )

    return None if family == "clean" else {"family": str(family), "severity": str(severity)}


def _statements(documents: list[dict]) -> list[dict]:
    """The bank-statement rows, which are the only ones this module reports on.

    The single implementation, used for the clean baseline AND every degraded
    tier. They diverged once: the clean row filtered and the degraded rows did
    not, so a table headed "Usable bank-statement amounts by image quality" had
    exactly one row that was bank-statement amounts.

    That matters because receipts carry a structural attribution floor. A
    receipt has one date, in its header, and `attributable` requires an amount
    to sit in a row carrying its identifying date -- which a receipt line item
    never does. Measured on gemma-4-31B against a clean corpus, receipts read
    184/184 amounts with 404/404 numerically correct and a median normalised CER
    of 0.0000, and still scored 129/184 attributable. Counting that floor on one
    side of a comparison and not the other makes every "points dropped" figure
    incomparable: it reported photo-heavy as 0.4799 where the bank-statement
    figure is 0.4296.

    Args:
        documents: Per-document rows from one system's block of a report.

    Returns:
        Only the bank-statement rows.
    """
    return [d for d in documents if d["stem"].endswith("_bank_statements")]


def _usable(documents: list[dict]) -> dict:
    """Reduce one system's per-document rows to the figures the case quotes.

    **`usable` here is `attributable`** — right value, right column, and in a row
    carrying its identifying date — which is what the case document ranks by and
    what the comparison sheets print. The report also carries a per-document
    `usable` field meaning `amounts - misfiled`, i.e. merely "not in the wrong
    column"; reading that made this module report InternVL at 87.1% on
    scan-light where the sheets showed 70.4%, because an amount misread or
    stranded in an undated row still counted. Two numbers under one word, and
    the more generous one was winning.
    """
    amounts = sum(d["amounts"] for d in documents)
    truth_amounts = sum(d["truth_amounts"] for d in documents)
    truth_rows = sum(d["truth_rows"] for d in documents)
    return {
        "pages": len(documents),
        "amounts": amounts,
        "usable": sum(d["attributable"] for d in documents) / amounts if amounts else None,
        "misfiled": sum(d["misfiled"] for d in documents),
        "read": sum(d["read"] for d in documents) / amounts if amounts else None,
        "digit_recall": (
            sum(d["amounts_correct"] for d in documents) / truth_amounts if truth_amounts else None
        ),
        "rows_aligned": sum(d["aligned"] for d in documents) / truth_rows if truth_rows else None,
        "width_ok": sum(1 for d in documents if d["columns_match"]),
        "median_ncer": sorted(d["normalised_cer"] for d in documents)[len(documents) // 2],
    }


def collect(reports: Path = REPO, clean: str = "") -> pd.DataFrame:
    """Gather every degraded tier's report, plus each system's clean baseline.

    The clean run is the shared origin of both ladders: the same 55 statements,
    the same system, the same prompt, differing only in image quality. Without
    it the curves have no zero and the cost cannot be read off them.

    **Every system's own clean report, never one shared origin.** The four
    systems are scored in three different clean reports, so reading only one
    left three ladders starting at their light tier and made the 12B look like
    it began the run already degraded. This is the same reasoning that
    `paired_change` records for baselines: borrowing another system's clean
    point silently reports one system's degradation as another's.

    Args:
        reports: Directory holding `scores_*.json`.
        clean: A single clean report to use instead of `CLEAN_REPORTS`. Empty
            reads them all, which is what a multi-system ladder needs.

    Returns:
        One row per (system, family, severity), ordered light to heavy.
    """
    rows: list[dict] = []
    seen: set[str] = set()

    for name in (clean,) if clean else CLEAN_REPORTS:
        baseline = reports / name
        if not baseline.exists():
            continue
        payload = json.loads(baseline.read_text(encoding="utf-8"))
        for system, block in payload["systems"].items():
            # First report wins: CLEAN_REPORTS is ordered, and a system present
            # in two of them takes the one declared for it rather than whichever
            # sorted last.
            if system in seen:
                continue
            statements = _statements(block["documents"])
            if statements:
                seen.add(system)
                # The clean point belongs to both ladders; it is duplicated
                # rather than drawn once, so each curve is readable alone.
                for family in ("scan", "photo"):
                    rows.append(
                        {"family": family, "severity": "clean", "system": system} | _usable(statements)
                    )

    for path in sorted(reports.glob("scores_*.json")):
        # Every report is parsed now, not only those whose NAME names a tier:
        # the manifest can only outrank the filename if the file is opened.
        payload = json.loads(path.read_text(encoding="utf-8"))
        tier = _tier_of(path, payload)
        if tier is None:
            continue
        for system, block in payload["systems"].items():
            # The SAME filter as the clean baseline above -- see _statements.
            rows.append({"system": system, **tier} | _usable(_statements(block["documents"])))

    if not rows:
        raise SystemExit(
            "No degraded reports found.\n"
            f"  What:     no scores_*.json in {reports} names a corpus whose manifest\n"
            "            states a degraded family and severity.\n"
            f"  Where:    {reports.resolve()}\n"
            "  Expected: one report per tier, written by the scoring loop that\n"
            "            run_degraded_31b.sh prints when it finishes, each naming a\n"
            "            corpus present on disk and labelled by its manifest.\n"
            "  Recover:  score each tier against its OWN corpus — the manifests differ\n"
            "            by construction and score refuses any other pairing."
        )

    frame = pd.DataFrame(rows)

    # A system present in a clean report but in no degraded tier has a point,
    # not a ladder -- docling is scored clean in scores_parsers.json and was
    # never run degraded. Plotting its lone clean point draws a stub that reads
    # as a system which held up perfectly, which is the opposite of true.
    laddered = set(frame.loc[frame.severity != "clean", "system"])
    dropped = sorted(set(frame.system) - laddered)
    if dropped:
        frame = frame[frame.system.isin(laddered)]

    frame["severity"] = pd.Categorical(frame.severity, categories=SEVERITIES, ordered=True)
    frame = frame.sort_values(["family", "severity"]).reset_index(drop=True)
    frame.attrs["clean_only"] = dropped
    return frame


def report(frame: pd.DataFrame) -> str:
    """Render the answer as text, per system and per channel.

    Grouped by system as well as family, because more than one system can be run
    against the same tiers and their rows would otherwise interleave under one
    heading — reading as a single system with two contradictory results.
    """
    lines = ["Usable bank-statement amounts by image quality", ""]
    for (name, family), group in frame.groupby(["system", "family"], observed=True):
        lines.append(f"  {name}  |  {family}")
        clean = group[group.severity == "clean"]
        origin = clean.usable.iloc[0] if len(clean) else None
        for _, row in group.iterrows():
            drop = "" if origin is None else f"  ({(row.usable - origin) * 100:+.1f} pts)"
            lines.append(
                f"    {row.severity:<9} usable {row.usable:.4f}{drop}"
                f"   misfiled {int(row.misfiled):4d}"
                f"   rows {row.rows_aligned:.3f}"
                f"   digits {row.digit_recall:.4f}"
                f"   nCER {row.median_ncer:.4f}"
            )
        lines.append("")
    return "\n".join(lines)


# Where each system's CLEAN run lives. A degraded tier is only interpretable
# against the same system on the same pages undegraded, so these are looked up by
# system name and never substituted for one another.
#
# The 31B appears under two names: `...-2xL4-tp2` in the degraded runs and
# `gemma-4-31B-it-qat-w4a16-ct` in its clean tp=2 report, because the system name
# is the vlm_systems.yml key and the clean re-run used the plain entry. That
# aliasing is declared rather than guessed.
CLEAN_REPORTS = ("scores_31b_tp2.json", "scores_parsers.json", "scores_v4.json")
CLEAN_ALIASES = {
    "gemma-4-31B-it-qat-w4a16-ct-2xL4-tp2": "gemma-4-31B-it-qat-w4a16-ct",
}


def paired_change(reports: Path = REPO, clean: str = "") -> pd.DataFrame:
    """Count, per tier, how many PAGES got worse, better and stayed the same.

    A net change is not evidence on its own. Token-level perturbation moves
    pages in both directions — the tp=1/tp=2 comparison on identical images
    moved 31 of 165 — so a tier where 10 pages improve and 6 worsen has told you
    nothing about degradation, whatever its total says. Only when the two
    directions stop balancing is there a signal.

    Args:
        reports: Directory holding `scores_*.json`.
        clean: The clean-corpus report to compare against.

    Returns:
        One row per tier with the three counts and the net.
    """

    def misfiled_by_stem(path: Path) -> dict[str, dict[str, int]]:
        """system -> stem -> misfiled, so two systems never merge into one."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            name: {d["stem"]: d["misfiled"] for d in _statements(block["documents"])}
            for name, block in payload["systems"].items()
        }

    baselines: dict[str, dict[str, int]] = {}
    for name in (clean,) if clean else CLEAN_REPORTS:
        path = reports / name
        if path.exists():
            baselines.update(misfiled_by_stem(path))

    rows = []
    for path in sorted(reports.glob("scores_*.json")):
        tier = _tier_of(path, json.loads(path.read_text(encoding="utf-8")))
        if tier is None:
            continue
        for name, current in misfiled_by_stem(path).items():
            # A system with no clean run has nothing to compare against, and
            # borrowing another system's baseline is worse than saying nothing:
            # it silently reports one system's degradation as another's. An
            # earlier version fell back to "the only baseline present" when just
            # one clean report was loaded, which compared MinerU's degraded
            # pages against the 31B's clean ones and called every tier a real
            # effect.
            baseline = baselines.get(name) or baselines.get(CLEAN_ALIASES.get(name, ""), {})
            if not baseline:
                continue
            deltas = [current[s] - baseline[s] for s in current if s in baseline]
            if not deltas:
                continue
            rows.append(
                {
                    "system": name,
                    **tier,
                    "worse": sum(1 for d in deltas if d > 0),
                    "better": sum(1 for d in deltas if d < 0),
                    "same": sum(1 for d in deltas if d == 0),
                    "net": sum(deltas),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["severity"] = pd.Categorical(frame.severity, categories=SEVERITIES, ordered=True)
        frame = frame.sort_values(["family", "severity"])
    return frame


def main() -> None:
    frame = collect()
    print(report(frame))

    changes = paired_change()
    if not changes.empty:
        print("Per-page change against clean — is a tier's total signal or noise?\n")
        for _, row in changes.sort_values(["system", "family", "severity"]).iterrows():
            verdict = (
                "noise: both directions, roughly balanced"
                if min(row.worse, row.better) >= max(row.worse, row.better) / 2
                else "one-directional — a real effect"
            )
            print(
                f"  {row.system[:26]:26} {row.family}-{row.severity:<9} "
                f"worse {row.worse:2d}  better {row.better:2d}"
                f"  same {row.same:2d}  net {row.net:+3d}   {verdict}"
            )
        print()

    # Where the curve bends matters more than where it ends: a cliff between two
    # adjacent tiers locates the image quality production must stay above, and
    # that is the number an intake pipeline can be specified against.
    for (name, family), group in frame.groupby(["system", "family"], observed=True):
        ordered = group.sort_values("severity")
        deltas = ordered.usable.diff().dropna()
        if len(deltas):
            worst = deltas.idxmin()
            step = ordered.loc[worst]
            previous = ordered.severity.shift().loc[worst]
            print(
                f"{name} / {family}: the largest single drop in usable amounts is "
                f"{previous} -> {step.severity}, {deltas.min() * 100:.1f} points"
            )
            rows_lost = (ordered.rows_aligned.iloc[0] - ordered.rows_aligned.iloc[-1]) * 100
            print(
                f"{name} / {family}: rows aligned falls {rows_lost:.1f} points over the same "
                "range — structure and amount placement do not degrade together"
            )


if __name__ == "__main__":
    main()

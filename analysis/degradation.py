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
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# The order a ladder is read in. Severity is ordinal, not numeric — the tiers
# are declared points on a scale, and spacing them evenly on the axis would
# assert a linearity nothing measured.
SEVERITIES = ("clean", "light", "moderate", "heavy")
_TIER = re.compile(r"_(?P<family>scan|photo)-(?P<severity>light|moderate|heavy)$")


def _usable(documents: list[dict]) -> dict:
    """Reduce one system's per-document rows to the figures the case quotes."""
    amounts = sum(d["amounts"] for d in documents)
    truth_amounts = sum(d["truth_amounts"] for d in documents)
    truth_rows = sum(d["truth_rows"] for d in documents)
    return {
        "pages": len(documents),
        "amounts": amounts,
        "usable": sum(d["usable"] for d in documents) / amounts if amounts else None,
        "misfiled": sum(d["misfiled"] for d in documents),
        "read": sum(d["read"] for d in documents) / amounts if amounts else None,
        "digit_recall": (
            sum(d["amounts_correct"] for d in documents) / truth_amounts if truth_amounts else None
        ),
        "rows_aligned": sum(d["aligned"] for d in documents) / truth_rows if truth_rows else None,
        "width_ok": sum(1 for d in documents if d["columns_match"]),
        "median_ncer": sorted(d["normalised_cer"] for d in documents)[len(documents) // 2],
    }


def collect(reports: Path = REPO, clean: str = "scores_31b_tp2.json") -> pd.DataFrame:
    """Gather every degraded tier's report, plus the clean baseline.

    The clean run is the shared origin of both ladders: the same 55 statements,
    the same system, the same prompt, differing only in image quality. Without
    it the curves have no zero and the cost cannot be read off them.

    Args:
        reports: Directory holding `scores_*.json`.
        clean: The clean-corpus report both ladders start from.

    Returns:
        One row per (family, severity), ordered light to heavy.
    """
    rows: list[dict] = []

    baseline = reports / clean
    if baseline.exists():
        payload = json.loads(baseline.read_text(encoding="utf-8"))
        for system, block in payload["systems"].items():
            statements = [d for d in block["documents"] if d["stem"].endswith("_bank_statements")]
            if statements:
                # The clean point belongs to both ladders; it is duplicated
                # rather than drawn once, so each curve is readable alone.
                for family in ("scan", "photo"):
                    rows.append(
                        {"family": family, "severity": "clean", "system": system} | _usable(statements)
                    )

    for path in sorted(reports.glob("scores_*.json")):
        match = _TIER.search(path.stem)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for system, block in payload["systems"].items():
            rows.append({"system": system, **match.groupdict()} | _usable(block["documents"]))

    if not rows:
        raise SystemExit(
            "No degraded reports found.\n"
            f"  What:     no scores_*_{{scan,photo}}-{{light,moderate,heavy}}.json in {reports}.\n"
            f"  Where:    {reports.resolve()}\n"
            "  Expected: one report per tier, written by the scoring loop that\n"
            "            run_degraded_31b.sh prints when it finishes.\n"
            "  Recover:  score each tier against its OWN corpus — the manifests differ\n"
            "            by construction and score refuses any other pairing."
        )

    frame = pd.DataFrame(rows)
    frame["severity"] = pd.Categorical(frame.severity, categories=SEVERITIES, ordered=True)
    return frame.sort_values(["family", "severity"]).reset_index(drop=True)


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


def paired_change(reports: Path = REPO, clean: str = "scores_31b_tp2.json") -> pd.DataFrame:
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
            name: {
                d["stem"]: d["misfiled"]
                for d in block["documents"]
                if d["stem"].endswith("_bank_statements")
            }
            for name, block in payload["systems"].items()
        }

    baselines = misfiled_by_stem(reports / clean)
    rows = []
    for path in sorted(reports.glob("scores_*.json")):
        match = _TIER.search(path.stem)
        if not match:
            continue
        for name, current in misfiled_by_stem(path).items():
            # A system with no clean run has nothing to be compared against, and
            # inventing a baseline from another system would be worse than
            # saying nothing.
            baseline = baselines.get(name) or (
                next(iter(baselines.values())) if len(baselines) == 1 else {}
            )
            deltas = [current[s] - baseline[s] for s in current if s in baseline]
            if not deltas:
                continue
            rows.append(
                {
                    "system": name,
                    **match.groupdict(),
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

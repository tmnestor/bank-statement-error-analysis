"""Figures for the calibration-pass summary.

Reads `scores_*.json` and nothing else. The per-document rows carry structure
and numeric counts, so every breakdown here derives from the reports alone —
predictions are gitignored and machine-specific, and an analysis that needed
them would run only on the machine that produced them.

Runs in the `du` environment, which has pandas, seaborn and matplotlib. NOT in
`docparse`: that env is deliberately five pure-Python packages, numpy was
evicted with the degradation module, and pulling it back for charts would widen
the environment the pipeline and the runner-import constraint depend on.

Every figure is written as both `.svg` and `.png`. SVG for the web and slides;
PNG because pandoc does not embed SVG into .docx — it wants EMF, and quietly
drops or rasterises SVG instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

DOC_TYPES = ("bank_statements", "invoices", "receipts")

# One accent per system, held constant across every figure so a reader can
# follow one system from chart to chart without re-reading a legend. The gemma
# variants share a hue and separate by lightness, which is the relationship
# between them: same family, different precision or size.
PALETTE = {
    "gemma 12B 4-bit": "#B2472E",
    "gemma 12B BF16 QAT": "#D98E5A",
    "gemma 12B BF16": "#E8C39E",
    "gemma 31B 4-bit": "#7C2E12",
    "InternVL3.5-8B": "#3E6B89",
    "MinerU": "#4F7A52",
    "Docling": "#8A8578",
}
INK = "#20201E"
MUTED = "#6E6A62"

SHORT = {
    "gemma-4-12B-it-qat-w4a16-ct": "gemma 12B 4-bit",
    "gemma-4-12B-it-qat-q4_0-unquantized": "gemma 12B BF16 QAT",
    "gemma-4-12B-it": "gemma 12B BF16",
    "gemma-4-31B-it-qat-w4a16-ct": "gemma 31B 4-bit",
    "InternVL3.5-8B": "InternVL3.5-8B",
    "mineru": "MinerU",
    "docling": "Docling",
}


def apply_style() -> None:
    """Set the house look once, so no figure carries its own styling."""
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.edgecolor": "#DAD6CC",
            "axes.labelcolor": MUTED,
            "grid.color": "#E6E2D8",
            "grid.linewidth": 0.8,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
        },
    )


def load(*reports: Path | str) -> pd.DataFrame:
    """Merge the reports into one tidy per-document frame.

    The prompted systems, the precision controls and the parsers were scored in
    separate runs. A later report wins on a system-name collision, so pass them
    oldest first.

    Args:
        reports: Paths to `scores_*.json` files.

    Returns:
        One row per (system, document), with a `doc_type` column.

    Raises:
        ValueError: The reports scored different corpora. Charting those
            together would compare systems against different ground truth, and
            the reason this is fatal rather than a warning is that the resulting
            figure looks entirely normal.
    """
    frames = []
    corpora = set()
    for path in reports:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        corpora.add(payload["corpus"])
        for name, block in payload["systems"].items():
            frame = pd.DataFrame(block["documents"])
            frame["system"] = SHORT.get(name, name)
            frames.append(frame)
    if len(corpora) > 1:
        raise ValueError(
            f"reports scored different corpora {sorted(corpora)}; charting them "
            "together would compare systems against different ground truth"
        )

    documents = pd.concat(frames, ignore_index=True)
    documents = documents.drop_duplicates(subset=["system", "stem"], keep="last")
    documents["doc_type"] = documents["stem"].str.split("_", n=1).str[1]
    return documents


def _rate(frame: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    """Aggregate a ratio per system, weighting documents by size.

    A mean of per-document rates would weight a three-row receipt as heavily as
    a forty-row statement, which is not the question any of these charts ask.
    """
    grouped = frame.groupby("system", observed=True)[[numerator, denominator]].sum()
    return (grouped[numerator] / grouped[denominator]).sort_values()


def _save(fig, name: str, out_dir: Path) -> list[Path]:
    """Write one figure as SVG and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, extra in ((".svg", {}), (".png", {"dpi": 200})):
        path = out_dir / f"{name}{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **extra)
        written.append(path)
    plt.close(fig)
    return written


def _title(ax, headline: str, subtitle: str) -> None:
    ax.set_title(f"{headline}\n{subtitle}", fontsize=12, color=INK, loc="left", pad=14)


def convention_cost(documents: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Normalised against strict CER, with the gap drawn between them.

    What normalisation removes is by construction formatting, so the gap IS the
    cost of the house style — the question the pass was run to answer.
    """
    medians = (
        documents.groupby("system", observed=True)[["normalised_cer", "strict_cer"]]
        .median()
        .sort_values("normalised_cer")
    )

    fig, ax = plt.subplots(figsize=(8.4, 0.55 * len(medians) + 1.6))
    for index, (system, row) in enumerate(medians.iterrows()):
        colour = PALETTE.get(system, MUTED)
        ax.plot(
            [row.normalised_cer, row.strict_cer],
            [index, index],
            color=colour,
            linewidth=7,
            alpha=0.25,
            solid_capstyle="butt",
            zorder=1,
        )
        ax.scatter([row.normalised_cer], [index], s=90, color=colour, zorder=3)
        ax.scatter([row.strict_cer], [index], s=90, color=colour, zorder=3, facecolor="white", linewidth=2)
        # Ink, not the series colour: the lighter gemma variants are pale by
        # design and their text was illegible on white. The markers already
        # identify the row.
        ax.text(
            row.strict_cer + 0.012,
            index,
            f"+{row.strict_cer - row.normalised_cer:.3f}",
            va="center",
            fontsize=9,
            color=INK,
            fontweight="bold",
        )

    # Headroom for the gap annotation, which otherwise clips at the frame.
    ax.set_xlim(right=medians.strict_cer.max() * 1.30)
    ax.set_yticks(range(len(medians)))
    ax.set_yticklabels(medians.index, fontsize=10, color=INK)
    ax.set_xlabel("median character error rate per page")
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _title(ax, "What the house style costs", "filled = reading only    hollow = reading and formatting")
    return _save(fig, "01-convention-cost", out_dir)


def numeric_fidelity(
    documents: pd.DataFrame, out_dir: Path, doc_type: str = "bank_statements"
) -> list[Path]:
    """Share of amounts reproduced exactly — the metric CER cannot express.

    Convention-blind, so it is the only measure here that puts a system emitting
    HTML tables and one emitting pipe tables on the same footing.
    """
    selected = documents[documents.doc_type == doc_type]
    accuracy = _rate(selected, "amounts_correct", "truth_amounts")

    fig, ax = plt.subplots(figsize=(8.4, 0.5 * len(accuracy) + 1.8))
    sns.barplot(
        x=accuracy.values,
        y=accuracy.index,
        hue=accuracy.index,
        palette=PALETTE,
        legend=False,
        ax=ax,
        width=0.62,
    )
    ax.axvline(1.0, color=INK, linewidth=1, linestyle=":", zorder=0)
    # Outside the bar end, in ink. Inside meant white on a pale fill for the
    # lighter gemma variants, which was unreadable.
    for index, value in enumerate(accuracy.values):
        ax.text(
            value + 0.002,
            index,
            f"{100 * value:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            color=INK,
            fontweight="bold",
        )

    ax.set_xlim(0.85, 1.012)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("amounts reproduced exactly")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", visible=False)
    _title(
        ax,
        f"Getting the numbers right — {doc_type.replace('_', ' ')}",
        "independent of formatting; the dotted line is every amount on the page",
    )
    return _save(fig, "02-numeric-fidelity", out_dir)


def structure_tradeoff(
    documents: pd.DataFrame, out_dir: Path, doc_type: str = "bank_statements"
) -> list[Path]:
    """Rows recovered against structure broken — the trade no single number shows.

    MinerU aligns the most rows and buys them with fragments and width breaks;
    gemma aligns fewer and breaks nothing. Ranking on either axis alone picks a
    winner arbitrarily.
    """
    selected = documents[documents.doc_type == doc_type].copy()
    selected["broken"] = selected.fragments + selected.width_breaks
    grouped = selected.groupby("system", observed=True).agg(
        recovered=("aligned", "sum"), rows=("truth_rows", "sum"), broken=("broken", "sum")
    )
    grouped["share"] = grouped.recovered / grouped.rows

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    sns.scatterplot(
        data=grouped.reset_index(),
        x="share",
        y="broken",
        hue="system",
        palette=PALETTE,
        s=210,
        ax=ax,
        legend=True,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )
    # A legend rather than direct labels, which read better in principle and do
    # not survive this data: every system that breaks no structure sits at y=0
    # within a few percent of the others, and the names are wider than the
    # separation between them. Vertical stagger only moved the collisions
    # around, and each system added makes it worse.
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        loc="upper left",
        handletextpad=0.4,
        borderaxespad=0.8,
        title=None,
    )

    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("share of rows recovered  →  better")
    ax.set_ylabel("rows split or of the wrong width  →  worse")
    ax.margins(0.18)
    _title(
        ax,
        f"No system wins outright — {doc_type.replace('_', ' ')}",
        "bottom right recovers the rows without breaking them",
    )
    return _save(fig, "03-structure-tradeoff", out_dir)


def by_document_type(documents: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Column integrity per document type, because the aggregate misleads.

    Bank statements are the only hard tables here. Invoices are near-saturated
    and receipts are a two-column list, so an average over the three describes
    none of them.
    """
    grouped = (
        documents.groupby(["doc_type", "system"], observed=True)[["misfiled", "amounts"]]
        .sum()
        .reset_index()
    )
    grouped["rate"] = grouped.misfiled / grouped.amounts

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=grouped, x="doc_type", y="rate", hue="system", palette=PALETTE, ax=ax, order=DOC_TYPES)
    ax.set_xticks(range(len(DOC_TYPES)))
    ax.set_xticklabels([t.replace("_", " ") for t in DOC_TYPES], fontsize=10, color=INK)
    ax.set_xlabel("")
    ax.set_ylabel("amounts under the wrong heading")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    # Headroom, or the legend sits on the 100% gridline and on the receipt bars
    # of the two parsers, which reach it.
    ax.set_ylim(top=1.28)
    ax.legend(frameon=False, fontsize=9, ncols=3, labelcolor=INK, loc="upper left")
    _title(ax, "Read it by document type", "a parser scoring 100% on receipts emits no table there at all")
    return _save(fig, "04-by-document-type", out_dir)


def all_figures(documents: pd.DataFrame, out_dir: Path = Path("docs/figures")) -> list[Path]:
    """Render every figure.

    Args:
        documents: The tidy frame from `load`.
        out_dir: Where to write; created if absent.

    Returns:
        Every path written, SVG and PNG.
    """
    apply_style()
    written: list[Path] = []
    written += convention_cost(documents, out_dir)
    written += numeric_fidelity(documents, out_dir)
    written += structure_tradeoff(documents, out_dir)
    written += by_document_type(documents, out_dir)
    return written

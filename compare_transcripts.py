"""Put a page beside two systems' transcripts, as one image.

A diff says two transcripts differ. It does not say whether a table lost a
column, whether a row was split, or whether an amount moved one cell right —
which is what this corpus is about. Rendered beside the page they came from,
those are obvious.

**Tables are drawn as tables.** The transcripts are Markdown, and a pipe table
rendered as a grid makes a structural failure a difference in *shape*: a
five-column grid beside a four-column one is visible before a single character
is read. That is finding 9's failure, and as monospace text it took squinting.

**Divergence is a filled cell, not a tinted glyph.** A cell whose content
differs from the truth is filled; a row the system invented, or never produced,
is filled whole. Fill survives being looked at from across a room, which is how
these are actually used.

Rows are aligned to the truth by `generators.tables.row_signature` — the same
matching `score` uses — so a dropped or invented row shifts nothing after it and
the cells being compared are the cells that correspond.

    python compare_transcripts.py CASE012_bank_statements \\
        --corpus parsing_20260820 \\
        --system runs_31b/gemma-4-31B-it-qat-w4a16-ct \\
        --system runs_v4/InternVL3.5-8B
"""

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw, ImageFont
from rich import print as rprint

from generators.columns import table_rows
from generators.tables import row_signature

REPO = Path(__file__).resolve().parent
MONO = REPO / "fonts" / "LiberationMono-Regular.ttf"
SANS = REPO / "fonts" / "LiberationSans-Regular.ttf"
SANS_BOLD = REPO / "fonts" / "LiberationSans-Bold.ttf"

app = typer.Typer(add_completion=False)

INK = (26, 26, 26)
MUTED = (135, 135, 135)
PAPER = (255, 255, 255)
HEADER_BG = (244, 244, 241)
GRID = (198, 198, 194)
RULE = (215, 215, 215)

# Fills, not text colours. Each says what happened to a cell or a row, and they
# are kept pale enough that the value inside stays the thing you read.
FILL_CHANGED = (253, 226, 200)  # content differs from the truth
FILL_EXTRA = (214, 233, 246)  # the system produced this and the truth has not
FILL_MISSING = (232, 232, 230)  # the truth has this and the system did not
EDGE_CHANGED = (196, 110, 44)
EDGE_EXTRA = (74, 143, 190)

_SEPARATOR = re.compile(r"^\s*\|?[\s:-]*-{2,}[\s:|-]*\|?\s*$")
_HTML_BLOCK = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)


Table = list[list[str]]


@dataclass(frozen=True)
class Block:
    """One parsed piece of a transcript.

    A tagged union rather than `(kind, payload)`: the payload is a string for
    prose and a table for a table, and a tuple of `(str, object)` made every
    downstream use an unchecked cast.

    Attributes:
        kind: "heading", "para" or "table".
        text: The text, for headings and paragraphs.
        rows: The rows, for tables.
    """

    kind: str
    text: str = ""
    rows: Table = field(default_factory=list)


def split_row(line: str) -> list[str]:
    """Split one pipe-table line into cells."""
    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return [cell.strip() for cell in trimmed.split("|")]


def parse_blocks(markdown: str) -> list[Block]:
    """Split a transcript into headings, paragraphs and tables.

    The corpus uses a deliberately small Markdown subset — one `#` heading,
    plain paragraphs, pipe tables — so this parses that subset rather than
    depending on a Markdown library the render environment would have to carry.

    Args:
        markdown: The transcript text.

    Returns:
        The blocks, in order. A table's separator row is dropped; it is Markdown
        syntax rather than content, and drawing it as a row of dashes would be a
        fourth thing to compare that means nothing.
    """
    blocks: list[Block] = []
    rows: Table = []

    def flush() -> None:
        if rows:
            blocks.append(Block("table", rows=[list(r) for r in rows]))
            rows.clear()

    # MinerU emits HTML tables rather than pipe tables on most pages, and it is
    # the system that fragments rows -- so the one output this tool most needs
    # to show is the one a pipe-only parser cannot see. Rather than write a
    # second dialect reader, hand the whole document to the scorer's own
    # , which already reads both for exactly this reason. Two
    # readers of the same dialects would be free to disagree, and the viewer
    # disagreeing with the scorer is the worst of the available bugs.
    if "<t" in markdown.lower():
        html_rows = table_rows(markdown)
        if html_rows:
            prose = _HTML_BLOCK.sub("", markdown)
            blocks = [b for b in parse_blocks(prose) if b.kind != "table"]
            blocks.append(Block("table", rows=html_rows))
            return blocks

    for line in markdown.splitlines():
        stripped = line.strip()
        if "|" in stripped and stripped.startswith("|"):
            if not _SEPARATOR.match(stripped):
                rows.append(split_row(stripped))
            continue
        flush()
        if not stripped:
            continue
        if stripped.startswith("#"):
            blocks.append(Block("heading", text=stripped.lstrip("#").strip()))
        else:
            blocks.append(Block("para", text=stripped))
    flush()
    return blocks


def align_tables(truth: Table, prediction: Table) -> list[tuple[str, list[str], list[str]]]:
    """Pair prediction rows with truth rows, keeping unmatched ones visible.

    Uses `row_signature` and `autojunk=False` for the reason
    `generators.divergence` gives: these tables are built from repeated tokens,
    and the heuristic treats exactly those as junk.

    Args:
        truth: The ground-truth table's rows.
        prediction: The system's rows.

    Returns:
        `(verdict, cells, truth_cells)` per drawn row, where verdict is
        `same`, `changed`, `extra` or `missing`.
    """
    left = [row_signature(row) for row in truth]
    right = [row_signature(row) for row in prediction]
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)

    drawn: list[tuple[str, list[str], list[str]]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                drawn.append(("same", prediction[j1 + offset], truth[i1 + offset]))
        elif tag == "replace":
            # Pair them up as far as they go so cell-level differences show;
            # the remainder is an insertion or a deletion.
            paired = min(i2 - i1, j2 - j1)
            for offset in range(paired):
                drawn.append(("changed", prediction[j1 + offset], truth[i1 + offset]))
            for offset in range(paired, j2 - j1):
                drawn.append(("extra", prediction[j1 + offset], []))
            for offset in range(paired, i2 - i1):
                drawn.append(("missing", truth[i1 + offset], truth[i1 + offset]))
        elif tag == "insert":
            for offset in range(j2 - j1):
                drawn.append(("extra", prediction[j1 + offset], []))
        else:
            for offset in range(i2 - i1):
                drawn.append(("missing", truth[i1 + offset], truth[i1 + offset]))
    return drawn


def _fit(text: str, font: ImageFont.FreeTypeFont, width: int, draw: ImageDraw.ImageDraw) -> str:
    """Truncate to fit, so a long description never overruns its cell."""
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


class Panel:
    """One column of the sheet: a heading, then rendered blocks."""

    def __init__(self, title: str, subtitle: str, width: int, scale: int) -> None:
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.scale = scale
        # Cell type is deliberately smaller than the row height: the row height
        # is set by how many rows must fit the page, while the cell font is set
        # by how many CHARACTERS must fit a column. Sizing both from one number
        # truncated every amount to '435…', which is the one thing in these
        # tables that must be read exactly.
        cell = max(9, int(scale * 0.7))
        self.mono = ImageFont.truetype(str(MONO), cell)
        self.sans = ImageFont.truetype(str(SANS), cell)
        self.head = ImageFont.truetype(str(SANS_BOLD), cell)
        self.title_font = ImageFont.truetype(str(SANS_BOLD), int(scale * 1.8))
        self.caption = ImageFont.truetype(str(MONO), int(scale * 0.92))

    def draw_table(
        self,
        draw: ImageDraw.ImageDraw,
        rows: list[tuple[str, list[str], list[str]]],
        y: int,
        pad: int,
    ) -> int:
        """Draw one table as a grid, filling cells that diverge."""
        if not rows:
            return y
        columns = max(len(cells) for _, cells, _ in rows)
        usable = self.width - 2 * pad
        # Proportional to the longest content each column holds, so a
        # description column gets the room and an amount column does not.
        widest = [1] * columns
        for _, cells, _ in rows:
            for index, cell in enumerate(cells[:columns]):
                widest[index] = max(widest[index], len(cell))
        total = sum(widest)
        col_w = [max(int(self.scale * 2.2), int(usable * w / total)) for w in widest]
        overflow = sum(col_w) - usable
        if overflow > 0:  # give it back from the widest column
            col_w[col_w.index(max(col_w))] -= overflow

        row_h = int(self.scale * 1.75)
        for position, (verdict, cells, truth_cells) in enumerate(rows):
            x = pad
            row_fill = {
                "extra": FILL_EXTRA,
                "missing": FILL_MISSING,
            }.get(verdict)
            if row_fill:
                draw.rectangle([pad, y, pad + usable, y + row_h], fill=row_fill)

            for index in range(columns):
                cell = cells[index] if index < len(cells) else ""
                expected = truth_cells[index] if index < len(truth_cells) else None
                width = col_w[index]

                # A cell is filled when it differs from the cell the truth has
                # in that position — which is only meaningful once the rows have
                # been paired, hence the alignment above.
                if verdict == "changed" and expected is not None and cell.strip() != expected.strip():
                    draw.rectangle([x, y, x + width, y + row_h], fill=FILL_CHANGED)
                    draw.rectangle([x, y, x + width, y + row_h], outline=EDGE_CHANGED, width=1)
                elif verdict == "changed" and expected is None and cell.strip():
                    # A column the truth does not have at all: the five-column
                    # failure, and the thing most worth seeing.
                    draw.rectangle([x, y, x + width, y + row_h], fill=FILL_EXTRA)
                    draw.rectangle([x, y, x + width, y + row_h], outline=EDGE_EXTRA, width=1)
                else:
                    draw.rectangle([x, y, x + width, y + row_h], outline=GRID, width=1)

                font = self.head if position == 0 else self.mono
                colour = MUTED if verdict == "missing" else INK
                draw.text(
                    (x + int(self.scale * 0.35), y + int(self.scale * 0.32)),
                    _fit(cell, font, width - int(self.scale * 0.7), draw),
                    font=font,
                    fill=colour,
                )
                x += width
            y += row_h
        return y + int(self.scale * 0.6)


def render_panel(
    title: str,
    subtitle: str,
    blocks: list[Block],
    truth_tables: list[Table],
    size: tuple[int, int],
    scale: int,
) -> Image.Image:
    """Draw a whole transcript panel: headings, paragraphs and tables."""
    width, height = size
    panel = Panel(title, subtitle, width, scale)
    image = Image.new("RGB", size, PAPER)
    draw = ImageDraw.Draw(image)
    pad = int(scale * 1.1)

    header_h = int(scale * 4.6)
    draw.rectangle([0, 0, width, header_h], fill=HEADER_BG)
    draw.line([0, header_h, width, header_h], fill=RULE, width=2)
    draw.text((pad, int(scale * 0.8)), title, font=panel.title_font, fill=INK)
    draw.text((pad, int(scale * 2.9)), subtitle, font=panel.caption, fill=MUTED)

    y = header_h + scale
    table_index = 0
    for block in blocks:
        if y > height - scale * 3:
            draw.text((pad, y), "… truncated", font=panel.caption, fill=MUTED)
            break
        if block.kind == "heading":
            y += int(scale * 0.4)
            draw.text((pad, y), block.text, font=panel.title_font, fill=INK)
            y += int(scale * 2.4)
        elif block.kind == "para":
            draw.text(
                (pad, y),
                _fit(block.text, panel.sans, width - 2 * pad, draw),
                font=panel.sans,
                fill=INK,
            )
            y += int(scale * 1.5)
        else:
            truth_rows = truth_tables[table_index] if table_index < len(truth_tables) else []
            y = panel.draw_table(draw, align_tables(truth_rows, block.rows), y, pad)
            table_index += 1
    return image


def summarise(blocks: list[Block], truth_tables: list[Table]) -> str:
    """Count table rows by verdict, so a panel states its own result."""
    tables = [block.rows for block in blocks if block.kind == "table"]
    counts = {"same": 0, "changed": 0, "extra": 0, "missing": 0}
    columns: set[int] = set()
    for index, rows in enumerate(tables):
        for row in rows:
            columns.add(len(row))
        truth_rows = truth_tables[index] if index < len(truth_tables) else []
        for verdict, _, _ in align_tables(truth_rows, rows):
            counts[verdict] += 1
    total = sum(counts.values()) or 1
    shape = f"{min(columns)}-{max(columns)}" if len(columns) > 1 else str(next(iter(columns), 0))
    return (
        f"{counts['same']}/{total} rows exact   {counts['changed']} changed   "
        f"{counts['extra']} invented   {counts['missing']} missed   |  {shape} columns"
    )


@app.command()
def compare(
    stem: Annotated[str, typer.Argument(help="Page stem, e.g. CASE001_bank_statements.")],
    corpus: Annotated[Path, typer.Option("--corpus", help="Exported corpus holding the page.")],
    system: Annotated[
        list[Path], typer.Option("--system", help="A predictions directory; repeat for each.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Where to write the comparison.")] = Path(
        "comparisons"
    ),
    scale: Annotated[
        int, typer.Option("--scale", help="Base font size in px; 0 fits it to the page height.")
    ] = 0,
) -> None:
    """Write one image: the page, the truth, then each system beside them."""
    image_path = next((corpus / "images").glob(f"{stem}.*"), None)
    transcript_path = corpus / "transcripts" / f"{stem}.md"
    if image_path is None or not transcript_path.exists():
        rprint(
            "[red]Cannot build the comparison.[/red]\n"
            f"  What:     {stem} has no image or no transcript in {corpus}.\n"
            f"  Where:    {corpus.resolve()}\n"
            f"  Expected: images/{stem}.png (or .jpg) and transcripts/{stem}.md\n"
            "  Recover:  check the stem, or point --corpus at the corpus holding it."
        )
        raise typer.Exit(1)

    truth_blocks = parse_blocks(transcript_path.read_text(encoding="utf-8"))
    truth_tables = [block.rows for block in truth_blocks if block.kind == "table"]

    collected: list[tuple[str, str, list[Block]]] = [
        ("ground truth", "authored at render time", truth_blocks)
    ]
    for directory in system:
        candidate = next(directory.rglob(f"{stem}.md"), None)
        if candidate is None:
            rprint(f"[yellow]  no prediction for {stem} in {directory}[/yellow]")
            continue
        blocks = parse_blocks(candidate.read_text(encoding="utf-8"))
        collected.append((directory.name, summarise(blocks, truth_tables), blocks))

    page = Image.open(image_path).convert("RGB")
    height = page.height

    if scale == 0:
        # Rows are the tall thing; fit the longest panel's rows to the height.
        longest = max(
            sum(len(block.rows) if block.kind == "table" else 1 for block in blocks)
            for _, _, blocks in collected
        )
        scale = max(10, min(30, int(height / ((longest + 8) * 1.75))))

    panel_width = int(scale * 52)
    panels = [page] + [
        render_panel(
            title,
            subtitle if title != "ground truth" else subtitle,
            blocks,
            truth_tables,
            (panel_width, height),
            scale,
        )
        for title, subtitle, blocks in collected
    ]

    gap = 14
    sheet = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1), height), RULE)
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap

    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{stem}_comparison.png"
    sheet.save(target)
    rprint(f"[green]{target}[/green]  {sheet.width}x{sheet.height}, {len(panels)} panel(s)")


if __name__ == "__main__":
    app()

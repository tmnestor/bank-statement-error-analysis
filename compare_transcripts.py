"""Put a page beside two systems' transcripts, as one image.

Reading two Markdown files in a diff tells you *that* they differ. It does not
tell you whether a table lost a column, whether a row was split, or whether an
amount moved one cell right — and those are the failures this corpus is about.
Rendered side by side against the page they came from, all three are obvious at
a glance.

Monospace, because a pipe table only lines up in monospace and the alignment IS
the structure being compared. Liberation Mono ships with this repo, so the
output does not depend on what fonts a machine happens to have.

Lines are tinted against the ground truth: a line the system got exactly right
is left black, one that differs is marked, and one the truth has but the system
never produced is shown as a gap. That turns "these two look similar" into
"this system dropped the fourth column on row 12".

    python compare_transcripts.py CASE001_bank_statements \\
        --corpus parsing_20260820 \\
        --system runs_31b/gemma-4-31B-it-qat-w4a16-ct \\
        --system runs_v4/InternVL3.5-8B
"""

import difflib
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw, ImageFont
from rich import print as rprint

REPO = Path(__file__).resolve().parent
MONO = REPO / "fonts" / "LiberationMono-Regular.ttf"
MONO_BOLD = REPO / "fonts" / "LiberationMono-Bold.ttf"
SANS_BOLD = REPO / "fonts" / "LiberationSans-Bold.ttf"

app = typer.Typer(add_completion=False)

# Ink for each verdict against the truth. Deliberately not red/green: the
# comparison is read by people who need to see WHERE a line differs, and a
# saturated field behind dense monospace hurts more than it helps. A tint on the
# line plus a marker in the gutter carries it.
INK = (26, 26, 26)
MUTED = (140, 140, 140)
SAME = (26, 26, 26)
CHANGED = (150, 60, 20)
EXTRA = (30, 90, 140)
MISSING = (170, 170, 170)
RULE = (215, 215, 215)
PAPER = (255, 255, 255)
HEADER_BG = (245, 245, 243)

_MARKER = {"same": " ", "changed": "~", "extra": "+", "missing": "-"}


def classify(truth: list[str], prediction: list[str]) -> list[tuple[str, str]]:
    """Label every prediction line against the truth, keeping truth's order.

    `SequenceMatcher` with `autojunk=False`, for the same reason
    `generators.divergence` disables it: these transcripts are built from
    repeated tokens — dates, `| --- |`, recurring merchants — and the heuristic
    treats anything appearing in more than 1% of a long sequence as junk, which
    is exactly the content here.

    Args:
        truth: The ground-truth transcript's lines.
        prediction: The system's lines.

    Returns:
        (verdict, text) per rendered line, where verdict is one of `same`,
        `changed`, `extra` or `missing`.
    """
    matcher = difflib.SequenceMatcher(a=truth, b=prediction, autojunk=False)
    rendered: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            rendered += [("same", line) for line in prediction[j1:j2]]
        elif tag == "replace":
            rendered += [("changed", line) for line in prediction[j1:j2]]
        elif tag == "insert":
            rendered += [("extra", line) for line in prediction[j1:j2]]
        else:  # delete — the truth had these and the system did not
            rendered += [("missing", line) for line in truth[i1:i2]]
    return rendered


def render_transcript(
    lines: list[tuple[str, str]], title: str, subtitle: str, size: tuple[int, int], scale: int
) -> Image.Image:
    """Draw one labelled, tinted transcript panel.

    Args:
        lines: Output of `classify`, or `[("same", line), ...]` for the truth.
        title: Panel heading, e.g. the system name.
        subtitle: One line of context under it, e.g. the accuracy summary.
        size: (width, height) of the panel.
        scale: Font size in pixels.

    Returns:
        The rendered panel.
    """
    width, height = size
    panel = Image.new("RGB", size, PAPER)
    draw = ImageDraw.Draw(panel)

    font = ImageFont.truetype(str(MONO), scale)
    heading = ImageFont.truetype(str(SANS_BOLD), int(scale * 1.7))
    caption = ImageFont.truetype(str(MONO), int(scale * 0.95))

    header_h = int(scale * 5)
    draw.rectangle([0, 0, width, header_h], fill=HEADER_BG)
    draw.line([0, header_h, width, header_h], fill=RULE, width=2)
    draw.text((scale, int(scale * 0.9)), title, font=heading, fill=INK)
    draw.text((scale, int(scale * 3.0)), subtitle, font=caption, fill=MUTED)

    line_h = int(scale * 1.42)
    y = header_h + scale
    for verdict, text in lines:
        if y > height - line_h:
            draw.text((scale, y), "... truncated", font=caption, fill=MUTED)
            break
        colour = {"same": SAME, "changed": CHANGED, "extra": EXTRA, "missing": MISSING}[verdict]
        # The gutter marker survives greyscale printing and photocopying, where
        # a tint does not.
        draw.text((int(scale * 0.3), y), _MARKER[verdict], font=font, fill=colour)
        shown = text if len(text) <= 120 else text[:117] + "..."
        draw.text((int(scale * 1.8), y), shown, font=font, fill=colour)
        y += line_h
    return panel


def summarise(lines: list[tuple[str, str]]) -> str:
    """One line of counts, so a panel says how it did without being read."""
    counts = {key: sum(1 for verdict, _ in lines if verdict == key) for key in _MARKER}
    total = sum(counts.values()) or 1
    return (
        f"{counts['same']}/{total} lines exact   "
        f"~{counts['changed']} changed   +{counts['extra']} extra   -{counts['missing']} missing"
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
        int, typer.Option("--scale", help="Font size in px; 0 fits the type to the page height.")
    ] = 0,
    truth: Annotated[bool, typer.Option("--truth/--no-truth", help="Include a truth panel.")] = True,
) -> None:
    """Write one image: the page, then each system's transcript beside it."""
    image_path = next((corpus / "images").glob(f"{stem}.*"), None)
    transcript_path = corpus / "transcripts" / f"{stem}.md"
    if image_path is None or not transcript_path.exists():
        rprint(
            f"[red]Cannot build the comparison.[/red]\n"
            f"  What:     {stem} has no image or no transcript in {corpus}.\n"
            f"  Where:    {corpus.resolve()}\n"
            f"  Expected: images/{stem}.png (or .jpg) and transcripts/{stem}.md\n"
            f"  Recover:  check the stem, or point --corpus at the corpus holding it."
        )
        raise typer.Exit(1)

    truth_lines = transcript_path.read_text(encoding="utf-8").splitlines()

    page = Image.open(image_path).convert("RGB")
    height = page.height

    # Collect every panel's lines first, so the type size can be chosen to fill
    # the page height. Sized at a fixed scale the transcripts occupied the top
    # third and left the rest blank, which made the text unreadable at any
    # sensible zoom -- the comparison is only useful if both sides can be read
    # at the same magnification.
    rendered: list[tuple[str, str, list[tuple[str, str]]]] = []
    if truth:
        rendered.append(
            (
                "ground truth",
                f"{len(truth_lines)} lines, authored at render time",
                [("same", line) for line in truth_lines],
            )
        )
    for directory in system:
        candidate = next(directory.rglob(f"{stem}.md"), None)
        if candidate is None:
            rprint(f"[yellow]  no prediction for {stem} in {directory}[/yellow]")
            continue
        lines = classify(truth_lines, candidate.read_text(encoding="utf-8").splitlines())
        rendered.append((directory.name, summarise(lines), lines))

    if scale == 0:
        longest = max((len(lines) for _, _, lines in rendered), default=1)
        # 6 lines of header, and 1.42 line spacing. Clamped: below 11px monospace
        # stops being legible, and above 34px a short transcript would be drawn
        # in headline type.
        scale = max(11, min(34, int(height / ((longest + 6) * 1.42))))

    panel_width = int(scale * 0.62 * 124)
    panels: list[Image.Image] = [page]
    for title, subtitle, lines in rendered:
        panels.append(render_transcript(lines, title, subtitle, (panel_width, height), scale))

    gap = 12
    total_width = sum(p.width for p in panels) + gap * (len(panels) - 1)
    sheet = Image.new("RGB", (total_width, height), RULE)
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

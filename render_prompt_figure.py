"""Render config/prompt.md as terminal-chrome SVGs, one per convention.

The findings document argues that prompt *wording* is a component under test —
three conventions were stated, measured, and adopted, and one of them moved a
model from 2/55 to 53/55 on the same pages. It never showed the prompt. These
figures are for the slide deck; the document itself carries the prompt as text,
which stays searchable and survives being published without its asset directory.

Why not the render-code skill's own entry point: it extracts fenced blocks from
a markdown file with a non-greedy `^```...^``` regex, and `prompt.md` *contains*
fenced examples. Feeding it the prompt would truncate every section at its first
worked example — silently, since a short block still renders. So the skill's
three rendering functions are reused directly and the splitting is done here,
on paragraph boundaries the prompt already has.

Usage:
    python3 render_prompt_figure.py [--out docs/figures]
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

SKILL = Path.home() / ".claude" / "skills" / "render-code" / "render_code_blocks.py"
REPO = Path(__file__).resolve().parent
PROMPT = REPO / "config" / "prompt.md"


def load_renderer():
    """Import the skill's renderer by path; it is not an installed package."""
    if not SKILL.exists():
        raise SystemExit(
            f"render-code skill not found at {SKILL}\n"
            "  Expected: the bundled render_code_blocks.py providing the Dracula\n"
            "            terminal-chrome renderer.\n"
            "  Recover:  install the render-code skill, or pass --out to a machine\n"
            "            that has it."
        )
    spec = importlib.util.spec_from_file_location("render_code_blocks", SKILL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt_body(path: Path) -> str:
    """The text the model receives: below the `---` rule, stripped.

    Mirrors `read_prompt` in runners/run_vlm.py. The preamble above the rule
    addresses whoever runs the benchmark and is not part of the prompt.
    """
    text = path.read_text(encoding="utf-8")
    _, separator, below = text.partition("\n---\n")
    return (below if separator else text).strip()


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split into (title, text), starting a section at each bolded rule.

    Every convention in the prompt opens a paragraph with `**...**`. Splitting
    there gives one figure per rule, which is the unit the document discusses
    and the unit a slide can hold. Worked examples stay with the rule they
    illustrate, which is the whole point of finding 2.
    """
    sections: list[tuple[str, str]] = []
    current: list[str] = []
    title = "the instruction"

    for paragraph in body.split("\n\n"):
        if paragraph.startswith("**") and current:
            sections.append((title, "\n\n".join(current).strip()))
            current = []
        if paragraph.startswith("**"):
            # Only the bolded span names the rule. It can wrap across lines, and
            # the prose that follows it on the same line must not come with it —
            # taking the whole first line yields titles cut mid-clause.
            bold = re.match(r"\*\*(.+?)\*\*", paragraph, re.DOTALL)
            lead = " ".join(bold.group(1).split()) if bold else paragraph.split("\n")[0]
            for tail in (", like this", ", and there is", " — "):
                lead = lead.split(tail)[0]
            title = lead.rstrip(".:— ").strip()
            if len(title) > 62:
                title = title[:59].rsplit(" ", 1)[0] + "..."
        current.append(paragraph)

    if current:
        sections.append((title, "\n\n".join(current).strip()))
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "figures")
    parser.add_argument(
        "--whole",
        action="store_true",
        help="Also render the entire prompt as one tall figure.",
    )
    args = parser.parse_args()

    renderer = load_renderer()
    args.out.mkdir(parents=True, exist_ok=True)

    body = prompt_body(PROMPT)
    sections = split_sections(body)
    print(f"{PROMPT.relative_to(REPO)}: {len(body.splitlines())} lines -> {len(sections)} figure(s)\n")

    written = []
    for index, (title, text) in enumerate(sections, 1):
        highlighted = renderer.pygmentize_svg(text, "text")
        lines = renderer.parse_pygments_svg(highlighted)
        svg = renderer.build_terminal_svg(lines, f"prompt.md — {title}")
        path = args.out / f"prompt-{index:02d}-{renderer.slugify(title)}.svg"
        path.write_text(svg, encoding="utf-8")
        written.append(path)
        print(f"  {len(text.splitlines()):3d} lines  {path.name}")

    if args.whole:
        highlighted = renderer.pygmentize_svg(body, "text")
        lines = renderer.parse_pygments_svg(highlighted)
        svg = renderer.build_terminal_svg(lines, "config/prompt.md — the whole prompt")
        path = args.out / "prompt-00-whole.svg"
        path.write_text(svg, encoding="utf-8")
        written.append(path)
        print(f"  {len(body.splitlines()):3d} lines  {path.name}")

    total = sum(p.stat().st_size for p in written) / 1024
    print(f"\n{len(written)} SVG(s), {total:.0f} KB total, in {args.out.relative_to(REPO)}/")


if __name__ == "__main__":
    sys.exit(main())

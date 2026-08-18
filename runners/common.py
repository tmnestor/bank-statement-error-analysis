"""Corpus and prediction bookkeeping shared by the parser runners.

The two parsers cannot share a conda environment — docling needs
`mlx-vlm>=0.4.3`, MinerU needs `<0.4` — so each runner is launched in its own
env. This module is the only code both of them import, and so it may use
nothing beyond the standard library: it must resolve in `docparse-docling`,
in `docparse-mineru`, and in `docparse` where the tests run.

Its job is the part that has nothing to do with parsing: which pages the
corpus contains, which of them already have a prediction, and where a
prediction is written so that `score` can pair it (scoring spec §6 —
`runs/<system>/<stem>.md`, one per transcript stem, no extras).
"""

from pathlib import Path

_ELIDE_AFTER = 5


class RunnerError(RuntimeError):
    """Raised when the corpus or the prediction directory cannot be trusted."""


def runner_error(what: str, *, where: str, expected: str, recover: str) -> RunnerError:
    """Build a four-element fail-fast diagnostic.

    Args:
        what: What is wrong.
        where: Absolute path of the thing to fix.
        expected: What a correct value looks like.
        recover: The one-line remediation step.

    Returns:
        The constructed error, for the caller to raise.
    """
    return RunnerError(
        "Cannot run the parser.\n"
        f"  What:     {what}\n"
        f"  Where:    {where}\n"
        f"  Expected: {expected}\n"
        f"  Recover:  {recover}"
    )


def _name_sample(names: list[str]) -> str:
    """Render a name list, stating how many were elided rather than trailing off.

    Args:
        names: The names to render, already sorted.

    Returns:
        A comma-separated sample, with an explicit count of any remainder.
    """
    head = ", ".join(names[:_ELIDE_AFTER])
    remainder = len(names) - _ELIDE_AFTER
    return head if remainder <= 0 else f"{head} and {remainder} more"


def corpus_stems(corpus: Path) -> list[str]:
    """List the transcript stems of an exported corpus, sorted.

    The transcripts are the authority rather than the images, because they are
    what `score` pairs predictions against.

    Args:
        corpus: An exported `parsing_YYYYMMDD/` directory.

    Returns:
        Sorted transcript stems, e.g. `["CASE001_invoices", ...]`.

    Raises:
        RunnerError: The transcripts directory is absent or empty.
    """
    transcripts = corpus / "transcripts"
    stems = sorted(p.stem for p in transcripts.glob("*.md")) if transcripts.is_dir() else []
    if not stems:
        raise runner_error(
            f"{transcripts} holds no transcript.",
            where=str(corpus.resolve()),
            expected="an exported corpus directory, e.g.\n"
            "              parsing_20260818/transcripts/CASE001_invoices.md",
            recover="pass --corpus pointing at an exported parsing_YYYYMMDD/ directory, "
            "or run `python -m generators.pipeline export` to produce one.",
        )
    return stems


def corpus_images(corpus: Path) -> dict[str, Path]:
    """Pair every transcript stem with the page image the parser must read.

    Args:
        corpus: An exported `parsing_YYYYMMDD/` directory.

    Returns:
        Stem -> image path, for every transcript stem.

    Raises:
        RunnerError: Any transcript has no matching image.
    """
    images = corpus / "images"
    paired: dict[str, Path] = {}
    missing: list[str] = []
    for stem in corpus_stems(corpus):
        image = images / f"{stem}.png"
        if image.exists():
            paired[stem] = image
        else:
            missing.append(stem)

    if missing:
        raise runner_error(
            f"{len(missing)} transcript(s) have no page image: {_name_sample(missing)}.",
            where=str(images.resolve()),
            expected="one .png per transcript stem, e.g.\n"
            "              parsing_20260818/images/CASE001_invoices.png",
            recover="re-run `python -m generators.pipeline export` to assemble a complete corpus.",
        )
    return paired


def pending(out_dir: Path, stems: list[str]) -> list[str]:
    """List the stems that still need a prediction, so a run can resume.

    A zero-byte file counts as unfinished: it is a page that died between
    creation and write, not a page that produced an empty transcription.

    Args:
        out_dir: The system's prediction directory, e.g. `runs/docling`.
        stems: Every stem the corpus expects.

    Returns:
        The stems with no non-empty prediction, in the given order.
    """
    return [
        stem for stem in stems if not (path := out_dir / f"{stem}.md").exists() or path.stat().st_size == 0
    ]


def write_prediction(out_dir: Path, stem: str, markdown: str) -> Path:
    """Write one prediction where `score` will look for it.

    Args:
        out_dir: The system's prediction directory, e.g. `runs/docling`.
        stem: The transcript stem this prediction answers.
        markdown: The parser's Markdown, written verbatim and never normalised
            — normalisation is the scoring tool's job (corpus design §5).

    Returns:
        The path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def mineru_markdown_path(raw_dir: Path, stem: str) -> Path:
    """Locate one document's Markdown inside MinerU's own output tree.

    MinerU writes a directory per document rather than a flat file, and the
    backend name is the subdirectory: `<raw>/<stem>/vlm/<stem>.md`.

    Args:
        raw_dir: The directory passed to `mineru -o`.
        stem: The document stem.

    Returns:
        The path MinerU writes its Markdown to.
    """
    return raw_dir / stem / "vlm" / f"{stem}.md"


def chunks(items: list, size: int) -> list[list]:
    """Split a list into fixed-size chunks, preserving order.

    Shared by the runners that submit work in batches — MinerU pays one model
    load per chunk, vLLM schedules a chunk concurrently — so the two cannot
    drift on an off-by-one.

    Args:
        items: The items to split.
        size: Maximum chunk length; anything below 1 is treated as 1.

    Returns:
        The chunks, in order.
    """
    step = max(1, size)
    return [items[start : start + step] for start in range(0, len(items), step)]


def keep_truncated(out_dir: Path, stem: str, markdown: str) -> Path:
    """Set a truncated generation aside for inspection, out of the way of scoring.

    A page that runs to the token cap is refused as a prediction — it is a
    repetition loop, and scoring it would blame the model's reading for the
    operator's cap. But discarding it destroys the only evidence of *what* the
    model was repeating, which is what decides whether the loop is fixable.

    The file lands in a `_truncated/` subdirectory, which `score` never sees:
    it globs `*.md` one level deep, so a subdirectory cannot masquerade as a
    prediction.

    Args:
        out_dir: The system's prediction directory.
        stem: The transcript stem this generation answers.
        markdown: The truncated text, kept verbatim.

    Returns:
        The path written.
    """
    kept = out_dir / "_truncated"
    kept.mkdir(parents=True, exist_ok=True)
    path = kept / f"{stem}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def verify_complete(out_dir: Path, stems: list[str]) -> None:
    """Refuse to report success while any page is missing a prediction.

    `score` rejects an incomplete system anyway; failing here names the gap
    while the run that caused it is still on screen.

    Args:
        out_dir: The system's prediction directory.
        stems: Every stem the corpus expects.

    Raises:
        RunnerError: Any stem has no non-empty prediction.
    """
    missing = pending(out_dir, stems)
    if missing:
        raise runner_error(
            f"{len(missing)} of {len(stems)} page(s) produced no prediction: {_name_sample(missing)}.",
            where=str(out_dir.resolve()),
            expected=f"one non-empty .md per transcript stem, {len(stems)} in total, e.g.\n"
            "              runs/docling/CASE001_invoices.md",
            recover="re-run this runner — it skips the pages already written and retries "
            "only the ones named above.",
        )

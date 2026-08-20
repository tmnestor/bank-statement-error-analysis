"""Run MinerU over an exported corpus, one prediction per page.

Runs in the `docparse-mineru` env, never in `docparse`: MinerU pins
`mlx-vlm<0.4`, which is unsatisfiable alongside docling's `>=0.4.3`.

    conda run -n docparse-mineru python -m runners.run_mineru \\
        --corpus parsing_20260818 --out runs

MinerU exposes no in-process API worth depending on here, so this shells out
to its CLI in **directory mode** — one model load per chunk instead of one per
page, which is the difference between ~20 minutes and ~2 hours over 165 pages.
Chunking rather than one 165-page invocation keeps peak memory bounded on a
16 GB machine and gives the run a resume point every `--chunk` pages.

`KMP_DUPLICATE_LIB_OK` is deliberately **not** set: this env carries exactly
one OpenMP runtime (torch's `libomp.dylib`), so the duplicate-runtime abort
that flag suppresses cannot arise, and the flag is documented as unsafe —
two runtimes sharing thread pools can return silently wrong results.
"""

import shutil
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from runners.common import (
    corpus_images,
    corpus_stems,
    mineru_markdown_path,
    pending,
    runner_error,
    shard_of,
    verify_complete,
    write_prediction,
    write_timing,
)

app = typer.Typer(add_completion=False)


def _chunks(items: list[str], size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks.

    Args:
        items: The items to split.
        size: Maximum chunk length.

    Returns:
        The chunks, in order.
    """
    return [items[start : start + size] for start in range(0, len(items), size)]


def check_workdir_outside_predictions(out: Path, workdir: Path) -> None:
    """Refuse a scratch directory that would masquerade as a scored system.

    `score` reads every immediate subdirectory of the predictions root as one
    system, so scratch living under `--out` turns into a system with no
    predictions and blocks the whole scoring run.

    Args:
        out: The predictions root passed to `--out`.
        workdir: The scratch root passed to `--workdir`.

    Raises:
        RunnerError: `workdir` is `out` or sits inside it.
    """
    resolved_out = out.resolve()
    resolved_workdir = workdir.resolve()
    if resolved_workdir == resolved_out or resolved_workdir.is_relative_to(resolved_out):
        raise runner_error(
            f"--workdir {resolved_workdir} sits inside the predictions root.",
            where=str(resolved_out),
            expected="a scratch directory beside the predictions root, not under it, e.g.\n"
            "              --out runs --workdir .mineru_work",
            recover="pass --workdir pointing somewhere outside --out; `score` reads every "
            "subdirectory of --out as a system and would score the scratch tree.",
        )


def _run_chunk(staged: Path, raw: Path, backend: str) -> subprocess.CompletedProcess:
    """Invoke the MinerU CLI over one staged directory of pages.

    Args:
        staged: Directory holding this chunk's page images.
        raw: Directory MinerU writes its own output tree into.
        backend: MinerU backend, e.g. `vlm-engine` for local MLX inference.

    Returns:
        The completed process, for the caller to inspect.
    """
    command = ["mineru", "-p", str(staged), "-o", str(raw), "-b", backend]
    return subprocess.run(command, capture_output=True, text=True, check=False)


@app.command()
def main(
    corpus: Annotated[Path, typer.Option("--corpus", help="An exported parsing_YYYYMMDD/ directory.")],
    out: Annotated[
        Path, typer.Option("--out", help="Predictions root; the system subdir is created here.")
    ] = Path("runs"),
    system: Annotated[
        str, typer.Option("--system", help="Subdirectory name, and the label `score` reports.")
    ] = "mineru",
    backend: Annotated[
        str, typer.Option("--backend", help="MinerU backend; vlm-engine is local MLX inference.")
    ] = "vlm-engine",
    chunk: Annotated[int, typer.Option("--chunk", help="Pages per MinerU invocation.")] = 25,
    shard: Annotated[
        int,
        typer.Option(
            "--shard",
            help="This process's shard index, from 0. With --shards, runs one whole "
            "replica per GPU over a disjoint slice.",
        ),
    ] = 0,
    shards: Annotated[
        int, typer.Option("--shards", help="How many processes share the work; one per GPU.")
    ] = 1,
    workdir: Annotated[
        Path,
        typer.Option(
            "--workdir",
            help="Scratch root for staged inputs and MinerU's own output tree. "
            "Must sit OUTSIDE --out: score treats every subdirectory of the "
            "predictions root as a system.",
        ),
    ] = Path(".mineru_work"),
) -> None:
    """Transcribe every corpus page with MinerU, skipping pages already done.

    Args:
        corpus: An exported corpus directory.
        out: Predictions root.
        system: Subdirectory name under `out`.
        backend: MinerU backend to select.
        chunk: Pages per invocation.
        workdir: Scratch root, kept after the run so a failure can be inspected.

    Raises:
        typer.Exit: The corpus is unusable, or a page produced no prediction.
    """
    try:
        check_workdir_outside_predictions(out, workdir)
        stems = corpus_stems(corpus)
        images = corpus_images(corpus)
    except RuntimeError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None

    out_dir = out / system
    todo = pending(out_dir, stems)
    try:
        todo = shard_of(todo, index=shard, shards=shards)
    except RuntimeError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None
    # What this process is answerable for. With several, checking every stem
    # would fail whichever finishes first, since the rest belong to other
    # processes and are still being written.
    owned = todo if shards > 1 else stems
    if shards > 1:
        rprint(f"[dim]shard {shard} of {shards}: {len(todo)} page(s) of this process[/dim]")
    rprint(f"[bold]{system}[/bold]: {len(todo)} of {len(stems)} page(s) to transcribe")
    if not todo:
        rprint("[green]nothing to do — every page already has a prediction[/green]")
        return

    raw = workdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    batches = _chunks(todo, chunk)

    for number, batch in enumerate(batches, start=1):
        staged = workdir / f"stage_{number:03d}"
        if staged.exists():
            shutil.rmtree(staged)
        staged.mkdir(parents=True)
        for stem in batch:
            shutil.copy2(images[stem], staged / images[stem].name)

        chunk_started = time.monotonic()
        completed = _run_chunk(staged, raw, backend)
        if completed.returncode != 0:
            rprint(f"[red]  chunk {number}/{len(batches)} exited {completed.returncode}[/red]")
            rprint(f"[red]  {completed.stderr.strip()[-800:]}[/red]")

        collected = 0
        for stem in batch:
            produced = mineru_markdown_path(raw, stem)
            if produced.exists():
                write_prediction(out_dir, stem, produced.read_text(encoding="utf-8"))
                collected += 1

        shutil.rmtree(staged, ignore_errors=True)
        rprint(
            f"  chunk {number}/{len(batches)}: {collected}/{len(batch)} collected "
            f"({time.monotonic() - chunk_started:.0f}s)"
        )

    elapsed = time.monotonic() - started
    rprint(f"[bold]{system}[/bold]: finished in {elapsed / 60:.1f} min")

    # Wall clock, load included, and declared as such. MinerU is invoked as a
    # subprocess per chunk and loads the model each time, so there is no
    # generate-only interval to time -- unlike run_vlm, which drives an engine
    # it loaded itself. The flag is what keeps this from being read beside the
    # engine-driven numbers as though it were measured the same way.
    if todo:
        write_timing(
            out_dir,
            system=system,
            inference_seconds=elapsed,
            pages=len(todo),
            cards=1,
            shard=shard,
            shards=shards,
            includes_model_load=True,
        )

    try:
        verify_complete(out_dir, owned)
    except RuntimeError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()

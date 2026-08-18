"""Run Docling over an exported corpus, one prediction per page.

Runs in the `docparse-docling` env, never in `docparse`: the parser and its
`mlx-vlm>=0.4.3` pin are unsatisfiable alongside MinerU's `<0.4`.

    conda run -n docparse-docling python -m runners.run_docling \\
        --corpus parsing_20260818 --out runs

Docling is one of the two systems in the §8.6 calibration pass that cannot be
told the convention. Its Markdown is written verbatim — every divergence from
the corpus transcript is signal about whether the convention is idiomatic
Markdown, so normalising here would destroy the measurement.
"""

import os
import time
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from runners.common import corpus_images, corpus_stems, pending, verify_complete, write_prediction

app = typer.Typer(add_completion=False)

_ARTIFACTS_ENV = "DOCLING_ARTIFACTS_PATH"


def _build_converter():  # noqa: ANN202 - docling types are absent in the test env
    """Build a converter pinned to the local MLX granite-docling checkpoint.

    Imports are deferred so this module stays importable in `docparse`, where
    docling is not installed and the bookkeeping tests run.

    Returns:
        A `DocumentConverter` whose IMAGE format uses the MLX VLM pipeline.

    Raises:
        RuntimeError: `DOCLING_ARTIFACTS_PATH` is unset or does not exist.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.vlm_model_specs import GRANITEDOCLING_MLX
    from docling.document_converter import DocumentConverter, ImageFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    artifacts = os.environ.get(_ARTIFACTS_ENV)
    if not artifacts or not Path(artifacts).is_dir():
        msg = (
            "Cannot run the parser.\n"
            f"  What:     {_ARTIFACTS_ENV} is unset or does not point at a directory "
            f"(got {artifacts!r}).\n"
            "  Where:    your shell environment — exported from ~/.dotfiles/zshrc\n"
            "  Expected: the docling model store, whose subdirectories are named "
            "<org>--<model>, e.g.\n"
            "              export DOCLING_ARTIFACTS_PATH=$LLM_MODELS_PATH/docling-artifacts\n"
            "  Recover:  export the variable, or `source ~/.zshrc`, before running this."
        )
        raise RuntimeError(msg)

    options = VlmPipelineOptions(vlm_options=GRANITEDOCLING_MLX)
    options.artifacts_path = artifacts
    options.enable_remote_services = False
    return DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_cls=VlmPipeline, pipeline_options=options)
        }
    )


@app.command()
def main(
    corpus: Annotated[Path, typer.Option("--corpus", help="An exported parsing_YYYYMMDD/ directory.")],
    out: Annotated[
        Path, typer.Option("--out", help="Predictions root; the system subdir is created here.")
    ] = Path("runs"),
    system: Annotated[
        str, typer.Option("--system", help="Subdirectory name, and the label `score` reports.")
    ] = "docling",
) -> None:
    """Transcribe every corpus page with Docling, skipping pages already done.

    Args:
        corpus: An exported corpus directory.
        out: Predictions root.
        system: Subdirectory name under `out`.

    Raises:
        typer.Exit: The corpus is unusable, or a page produced no prediction.
    """
    try:
        stems = corpus_stems(corpus)
        images = corpus_images(corpus)
    except RuntimeError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None

    out_dir = out / system
    todo = pending(out_dir, stems)
    rprint(f"[bold]{system}[/bold]: {len(todo)} of {len(stems)} page(s) to transcribe")
    if not todo:
        rprint("[green]nothing to do — every page already has a prediction[/green]")
        return

    try:
        converter = _build_converter()
    except RuntimeError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None

    started = time.monotonic()
    failures: list[str] = []
    for index, stem in enumerate(todo, start=1):
        page_started = time.monotonic()
        try:
            result = converter.convert(images[stem])
            write_prediction(out_dir, stem, result.document.export_to_markdown())
        except Exception as err:  # noqa: BLE001 - one bad page must not end the run
            failures.append(stem)
            rprint(f"[red]  {index}/{len(todo)} {stem} FAILED: {type(err).__name__}: {err}[/red]")
            continue
        rprint(f"  {index}/{len(todo)} {stem} ({time.monotonic() - page_started:.1f}s)")

    elapsed = time.monotonic() - started
    rprint(f"[bold]{system}[/bold]: {len(todo) - len(failures)} written in {elapsed / 60:.1f} min")

    try:
        verify_complete(out_dir, stems)
    except RuntimeError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()

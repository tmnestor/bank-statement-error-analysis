"""Run a prompted VLM over an exported corpus, one prediction per page.

This is the other half of the §8.6 calibration pass. Docling and MinerU cannot
be told the convention, so they measure whether it is idiomatic Markdown at
all; a prompted VLM reads `config/prompt.md` and so measures whether the
convention is **communicable**. Both arms write the same
`runs/<system>/<stem>.md` layout and are scored by the same command.

The prompt is sent verbatim, exactly as `prompt.md` says to. Editing the text
here rather than in `prompt.md` would silently unpair the prompt from the
transcripts it was written against.

Two transports, declared per system in `config/vlm_systems.yml`:

    local_mlx    on-device MLX checkpoint via mlx-vlm; run it in an env whose
                 mlx-vlm knows the architecture (`docparse-docling` carries
                 0.6.4, which has gemma4; `docparse-mineru`'s 0.3.9 does not).
    openai_http  an OpenAI-compatible server, which is how the remote GPU host
                 serves the two checkpoints spec §8.6 names.

HTTP goes through the standard library on purpose: the runtime dependency list
is five pure-Python packages and this is not worth a sixth.
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from evaluation.unproduced import declare_unproduced
from runners.common import (
    RunnerError,
    check_prompt_provenance,
    chunks,
    corpus_images,
    corpus_stems,
    keep_truncated,
    pending,
    runner_error,
    shard_of,
    verify_complete,
    write_prediction,
    write_settings_provenance,
    write_timing,
)
from runners.vlm_config import load_vlm_systems, resolve_base_url, system_named

app = typer.Typer(add_completion=False)

_DEFAULT_SYSTEMS = Path("config/vlm_systems.yml")
_DEFAULT_PROMPT = Path("config/prompt.md")


def apply_penalty_override(spec: dict, penalty: float | None) -> tuple[dict, bool]:
    """Override the declared repetition penalty for one invocation.

    The declared value is the benchmark's; an override is a deliberate
    experiment on the pages a model could not finish. Pages produced under one
    are not directly comparable to the rest, so the caller records which.

    Args:
        spec: A validated system spec.
        penalty: The override, or None to use the declared value.

    Returns:
        The spec to use, and whether it differs from the declaration.
    """
    if penalty is None or penalty == spec["repetition_penalty"]:
        return spec, False
    return dict(spec, repetition_penalty=penalty), True


def read_prompt(path: Path) -> str:
    """Read the shipped prompt, stripping the operator preamble.

    `prompt.md` documents itself above a `---` rule and then says "Use the text
    below verbatim". Everything above that rule is addressed to whoever runs
    the benchmark, not to the model.

    Args:
        path: Path to prompt.md.

    Returns:
        The prompt text to send.

    Raises:
        RunnerError: The file is absent or carries no text below the rule.
    """
    if not path.exists():
        raise runner_error(
            f"{path} does not exist.",
            where=str(path.resolve()),
            expected="the shipped transcription prompt, whose text below the '---' rule "
            "is sent verbatim, e.g.\n              config/prompt.md",
            recover="pass --prompt pointing at the prompt that ships with this corpus.",
        )

    text = path.read_text(encoding="utf-8")
    _, separator, below = text.partition("\n---\n")
    prompt = (below if separator else text).strip()
    if not prompt:
        raise runner_error(
            f"{path} carries no prompt text below the '---' rule.",
            where=str(path.resolve()),
            expected="the instructions to send, below a '---' line, e.g.\n"
            "              Transcribe this document page completely, as Markdown.",
            recover="restore the prompt body, or pass --prompt pointing at the shipped copy.",
        )
    return prompt


def image_data_uri(image: Path) -> str:
    """Encode a page image as a data URI for an OpenAI-compatible request.

    Args:
        image: Path to the PNG.

    Returns:
        A `data:image/png;base64,...` URI.
    """
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def message_content(spec: dict, prompt: str, image: Path) -> list[dict]:
    """Build the user message parts, in the order this system declares.

    Shared by every transport so an ordering or encoding change cannot apply to
    one and not the other. Some servers and chat templates are order-sensitive;
    the LMM_POC sandbox registers every gemma4 model with
    default_image_first=True, so the order is declared per system rather than
    assumed here.

    Args:
        spec: A validated system spec.
        prompt: The prompt text, sent verbatim.
        image: The page image.

    Returns:
        The content parts for one user message.
    """
    text_part = {"type": "text", "text": prompt}
    image_part = {"type": "image_url", "image_url": {"url": image_data_uri(image)}}
    return [image_part, text_part] if spec["image_first"] else [text_part, image_part]


def build_request(spec: dict, prompt: str, image: Path) -> tuple[str, dict, dict[str, str]]:
    """Build the chat-completions call for one page.

    Args:
        spec: A validated system spec.
        prompt: The prompt text, sent verbatim.
        image: The page image.

    Returns:
        The URL, the JSON payload, and the headers.

    Raises:
        RunnerError: The declared API-key variable is not set.
    """
    headers = {"Content-Type": "application/json"}
    key_env = str(spec["api_key_env"])
    if key_env != "none":
        key = os.environ.get(key_env)
        if not key:
            raise runner_error(
                f"the declared API-key variable {key_env} is not set.",
                where="your shell environment",
                expected=f"the server's key exported as {key_env}, e.g.\n"
                f"              export {key_env}=sk-...",
                recover=f"export {key_env}, or set api_key_env: none if the server needs no key.",
            )
        headers["Authorization"] = f"Bearer {key}"

    content = message_content(spec, prompt, image)

    payload = {
        "model": spec["model"],
        "temperature": spec["temperature"],
        "top_p": spec["top_p"],
        "max_tokens": spec["max_output_tokens"],
        "messages": [{"role": "user", "content": content}],
    }
    url = f"{resolve_base_url(spec).rstrip(chr(47))}/chat/completions"
    return url, payload, headers


def read_completion(body: str) -> str:
    """Extract the transcription from a chat-completions response.

    A truncated completion is refused rather than written: a page cut off at
    the token limit looks like a short transcription and would be scored as
    one, quietly attributing the operator's cap to the model's reading.

    Args:
        body: The raw response body.

    Returns:
        The message content.

    Raises:
        RunnerError: The body is not JSON, carries no choice, or was truncated.
    """
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as err:
        raise runner_error(
            f"the server did not return JSON: {err}",
            where=f"response body: {body.strip()[:300]}",
            expected='an OpenAI-compatible response, e.g.\n              {"choices": '
            '[{"message": {"content": "# TAX INVOICE"}}]}',
            recover="check the base_url points at the OpenAI-compatible root of the server.",
        ) from err

    choices = parsed.get("choices") or []
    if not choices:
        raise runner_error(
            "the response carries no choices.",
            where=f"response body: {body.strip()[:300]}",
            expected='at least one choice, e.g.\n              {"choices": [{"message": '
            '{"content": "..."}}]}',
            recover="check the served model name matches the declared 'model'.",
        )

    if choices[0].get("finish_reason") == "length":
        raise runner_error(
            "the completion was truncated at the token limit.",
            where="config/vlm_systems.yml -> max_output_tokens",
            expected="a limit above the longest page's transcription, e.g.\n"
            "              max_output_tokens: 8192",
            recover="raise max_output_tokens for this system and re-run; the runner "
            "retries only the pages with no prediction.",
        )

    return choices[0].get("message", {}).get("content", "")


def _transcribe_http(spec: dict, prompt: str, image: Path) -> str:
    """Transcribe one page through an OpenAI-compatible server.

    Args:
        spec: A validated system spec.
        prompt: The prompt text.
        image: The page image.

    Returns:
        The model's Markdown.

    Raises:
        RunnerError: The request fails or the response is unusable.
    """
    url, payload, headers = build_request(spec, prompt, image)
    request = urllib.request.Request(  # noqa: S310 - scheme validated at config load
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=spec["timeout_seconds"]) as response:  # noqa: S310
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        raise runner_error(
            f"the server returned HTTP {err.code}.",
            where=url,
            expected="a 200 response from an OpenAI-compatible chat-completions endpoint.",
            recover=f"check the server log; body: {err.read().decode('utf-8', 'replace')[:300]}",
        ) from err
    except urllib.error.URLError as err:
        raise runner_error(
            f"the server could not be reached: {err.reason}",
            where=url,
            expected="a reachable host, e.g.\n              base_url: http://gpu-host:8000/v1",
            recover="check the remote GPU host is serving this model and reachable from here.",
        ) from err
    return read_completion(body)


# Two architecture families, both legitimate. A tower-based model carries a
# vision encoder (gemma4: `gemma4_vision`, hidden 768, 16 layers). An
# **encoder-free** model has no tower at all and embeds patches straight into
# the language model (gemma4_unified: `gemma4_unified_vision`, no hidden_size
# and no layers, just `mm_embed_dim`/`mm_posemb_size` over 11 tensors). Looking
# only for a tower marks every encoder-free model as blind.
VISION_ATTRIBUTES: tuple[str, ...] = (
    "vision_tower",
    "vision_model",
    "visual",
    "vision_embedder",
    "embed_vision",
)


class TruncatedGeneration(RunnerError):
    """Raised when a generation stopped at the token cap.

    Carries the partial text so the caller can set it aside for inspection
    rather than destroying the only evidence of what the model repeated.
    """

    def __init__(self, message: str, text: str) -> None:
        super().__init__(message)
        self.text = text


def check_generation_complete(finish_reason: str | None, stem: str, system: str, text: str = "") -> None:
    """Refuse a generation that stopped at the token cap.

    The HTTP path refuses `finish_reason == "length"` for the same reason, and
    the local paths must not be laxer: a capped run is usually a repetition loop
    and writes a large non-empty file that satisfies every downstream check
    while scoring as a total reading failure.

    Args:
        finish_reason: The reason generation stopped, if the backend reports one.
        stem: The page being transcribed, for diagnostics.
        system: The system name, for diagnostics.
        text: The partial generation, carried on the exception for inspection.

    Raises:
        TruncatedGeneration: Generation stopped because it ran out of tokens.
    """
    if finish_reason != "length":
        return
    raise TruncatedGeneration(
        "Cannot run the parser.\n"
        f"  What:     '{system}' hit the token cap transcribing {stem}, so the page "
        "is truncated.\n"
        f"  Where:    config/vlm_systems.yml -> systems.{system}.max_output_tokens\n"
        "  Expected: generation stopping on its own before the cap, e.g.\n"
        "              max_output_tokens: 16384\n"
        "  Recover:  the partial text is kept under _truncated/ for inspection. A page "
        "that caps is a repetition loop, not a long page, so raising the cap usually "
        "just loops longer.",
        text,
    )


def present_vision_attributes(model: object) -> list[str]:
    """List which vision components a loaded model actually carries.

    Uses `hasattr`, never `dir`: an mlx `Module` is a dict subclass that serves
    its submodules through `__getattr__`, so they never appear in `dir()` and a
    `dir`-based check rejects every working model.

    Args:
        model: The loaded model.

    Returns:
        The vision attribute names present, in declaration order.
    """
    return [attribute for attribute in VISION_ATTRIBUTES if hasattr(model, attribute)]


def check_vision_available(model_attributes: list[str], *, declares_vision: bool, system: str) -> None:
    """Refuse a checkpoint that loaded text-only despite promising vision.

    This failure is silent and expensive. `gemma-4-12B-it-qat-OptiQ-4bit` keeps
    its vision weights in an `optiq/` sidecar mlx-vlm 0.6.4 does not read, so
    the model loads without a vision tower while the chat template still
    inserts an image placeholder. The model then answers "Please provide the
    image you would like me to transcribe" — a non-empty file that satisfies
    every downstream check and scores as a total reading failure, attributing
    a broken harness to the model.

    Args:
        model_attributes: Attribute names on the loaded model.
        declares_vision: Whether the checkpoint config declares a vision
            config, i.e. whether vision was promised at all.
        system: The system name, for diagnostics.

    Raises:
        RunnerError: Vision was declared but no vision tower was built.
    """
    if not declares_vision:
        return
    if any(attribute in set(model_attributes) for attribute in VISION_ATTRIBUTES):
        return
    raise runner_error(
        f"system '{system}' declares a vision config but loaded without a vision tower, "
        "so it cannot see the page.",
        where=f"config/vlm_systems.yml -> systems.{system}.model",
        expected=f"a loaded model carrying one of {list(VISION_ATTRIBUTES)}, as every "
        "usable image-text-to-text checkpoint does.",
        recover="use a checkpoint whose vision weights this mlx-vlm can load — a sidecar "
        "layout (e.g. OptiQ's optiq_vision.safetensors) needs its own runtime — or serve "
        "the model over transport: openai_http instead.",
    )


def check_weight_mismatches(mismatched: list[str], unused_towers: list[str], system: str) -> None:
    """Refuse a checkpoint whose *used* weights do not match the library.

    Loading non-strictly is what makes a version-drifted checkpoint usable at
    all, but on its own it skips every mismatch silently — including one in the
    vision tower, which would produce a plausible transcription from partly
    uninitialised weights and a benchmark number nobody could explain. So the
    skip is bounded: only the towers the operator declared unused may differ,
    and anything else stops the run.

    Args:
        mismatched: Parameter names whose checkpoint shape differs.
        unused_towers: Top-level towers this system never invokes, declared in
            `config/vlm_systems.yml`.
        system: The system name, for diagnostics.

    Raises:
        RunnerError: Any mismatch falls outside the declared unused towers.
    """
    offending = sorted(name for name in mismatched if name.split(".")[0] not in set(unused_towers))
    if offending:
        raise runner_error(
            f"system '{system}' has {len(offending)} weight(s) whose checkpoint shape "
            f"differs outside the declared unused towers: {', '.join(offending[:5])}"
            + (f" and {len(offending) - 5} more" if len(offending) > 5 else "")
            + ".",
            where=f"config/vlm_systems.yml -> systems.{system}.mlx_unused_towers",
            expected="every weight the model actually uses to match the installed "
            "mlx-vlm, e.g. for a vision-only run\n"
            "              mlx_unused_towers: [audio_tower]",
            recover="install an mlx-vlm whose implementation matches this checkpoint, or "
            "re-quantise the checkpoint — do not widen mlx_unused_towers to cover a "
            "tower the model reads.",
        )


def _load_vllm(spec: dict, system: str):  # noqa: ANN202 - vllm is absent in the test env
    """Build an in-process vLLM engine from the declared engine block.

    The sandbox that hosts these checkpoints drives vLLM through its offline
    Python API rather than an OpenAI server, so there is no endpoint to call.
    Engine arguments are declared in YAML rather than inferred: they decide
    what the model can see, and a wrong one produces a working run with quietly
    worse results.

    `soft_tokens` is written once and applied to both places vLLM reads it —
    `mm_processor_kwargs.max_soft_tokens` and
    `hf_overrides.vision_config.num_soft_tokens`. They must agree, and writing
    one value makes disagreement unrepresentable rather than merely tested.

    Args:
        spec: A validated system spec.
        system: The declared system name, for diagnostics.

    Returns:
        The constructed engine.

    Raises:
        RunnerError: vLLM is not installed in this environment.
    """
    try:
        from vllm import LLM
    except ImportError as err:
        # An ImportError here has two very different causes and they need
        # different fixes. "No module named vllm" is the wrong environment; a
        # missing symbol or shared object is vLLM being present but unloadable,
        # usually conda's libstdc++ shadowed by an older system one. Reporting
        # both as "not installed" sends the reader to the wrong repair.
        message = str(err)
        unloadable = "No module named" not in message
        raise runner_error(
            (
                f"vllm is installed but could not be loaded: {message}"
                if unloadable
                else f"vllm is not installed in this environment: {message}"
            ),
            where="the active conda environment",
            expected=(
                "a loadable vLLM. A missing CXXABI/GLIBCXX symbol means the system "
                "libstdc++ is being found before the environment's, e.g.\n"
                '              export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"'
                if unloadable
                else "an env with vLLM >= 0.23.0, the floor for the gemma4_unified "
                "architecture, e.g.\n"
                "              conda run -n vllm_env3 python -m runners.run_vlm ..."
            ),
            recover=(
                "export the library path above, or activate the environment in a clean "
                "shell — a doubled activation can leave it pointing at another prefix."
                if unloadable
                else "run this in the env that serves these checkpoints."
            ),
        ) from err

    engine = spec["vllm_engine"]

    # soft_tokens is a gemma4_unified knob: one value applied to both places
    # vLLM reads it, so the two cannot disagree. Other architectures declare
    # `none` and get neither kwarg.
    soft = engine["soft_tokens"]
    vision_kwargs: dict = {}
    if soft != "none":
        vision_kwargs = {
            "mm_processor_kwargs": {"max_soft_tokens": soft},
            "hf_overrides": {"vision_config": {"num_soft_tokens": soft}},
        }

    # InternVL3.5 ships a Mistral tokenizer whose pre-tokenizer regex mis-splits
    # runs of whitespace and digits — exactly what a whitespace-aligned,
    # digit-heavy bank statement is made of, so it corrupts amount tokenization.
    # vLLM exposes no flag for it and loads its tokenizer in the front-end AND
    # in every spawned EngineCore child, so the correction has to arrive as a
    # tokenizer directory on disk. Declare the corrected copy here; `none` uses
    # the checkpoint's own.
    tokenizer = engine["tokenizer"]
    if tokenizer != "none":
        vision_kwargs["tokenizer"] = str(tokenizer)

    return LLM(
        model=str(spec["model"]),
        tensor_parallel_size=engine["tensor_parallel_size"],
        max_model_len=engine["max_model_len"],
        gpu_memory_utilization=engine["gpu_memory_utilization"],
        max_num_seqs=engine["max_num_seqs"],
        limit_mm_per_prompt={"image": engine["limit_mm_images"]},
        enable_prefix_caching=engine["enable_prefix_caching"],
        enforce_eager=engine["enforce_eager"],
        trust_remote_code=True,
        disable_log_stats=True,
        **vision_kwargs,
    )


def transcribe_vllm_batch(
    engine, spec: dict, prompt: str, images: list[Path], system: str
) -> list[tuple[str, str, bool]]:
    """Transcribe a batch of pages with one scheduled vLLM call.

    Submitting one page per call leaves `max_num_seqs` unused — the scheduler
    never has more than one sequence in flight, so the KV cache's concurrency
    ceiling buys nothing. `engine.chat()` accepts a list of conversations and
    schedules them together, which is the only way the declared batch size and
    `max_num_seqs` act on the same run.

    Thinking is suppressed explicitly: on `gemma4_unified` it is opt-in via a
    `<|think|>` token, and this sends a bare user message with no system
    prompt, so it should already be off.

    Args:
        engine: The engine from `_load_vllm`.
        spec: A validated system spec.
        prompt: The prompt text.
        images: The page images for this batch.
        system: The system name, for diagnostics.

    Returns:
        One `(stem, text, truncated)` per page, in submission order.
    """
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=spec["temperature"],
        top_p=spec["top_p"],
        max_tokens=spec["max_output_tokens"],
        repetition_penalty=spec["repetition_penalty"],
    )
    conversations = [
        [{"role": "user", "content": message_content(spec, prompt, image)}] for image in images
    ]
    outputs = engine.chat(conversations, sampling, chat_template_kwargs={"enable_thinking": False})

    results: list[tuple[str, str, bool]] = []
    for image, output in zip(images, outputs, strict=True):
        completion = output.outputs[0]
        results.append((image.stem, completion.text, completion.finish_reason == "length"))
    return results


def _load_mlx(spec: dict, system: str):  # noqa: ANN202 - mlx types are absent in the test env
    """Load an MLX checkpoint and its processor once, for reuse across pages.

    Imported lazily so this module stays importable in `docparse`, where
    mlx-vlm is not installed and these tests run.

    Args:
        spec: A validated system spec.
        system: The declared system name, for diagnostics — the YAML key, not
            the checkpoint path, so a diagnostic points at a line that exists.

    Returns:
        The model, the processor, and the loaded config.

    Raises:
        RunnerError: mlx-vlm is absent or cannot load the checkpoint.
    """
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
    except ImportError as err:
        raise runner_error(
            f"mlx-vlm is not installed in this environment: {err}",
            where="the active conda environment",
            expected="an env whose mlx-vlm knows this architecture, e.g.\n"
            "              conda run -n docparse-docling python -m runners.run_vlm ...",
            recover="run this in docparse-docling (mlx-vlm 0.6.4, which has gemma4); "
            "docparse-mineru's 0.3.9 does not.",
        ) from err

    checkpoint = str(spec["model"])
    unused = list(spec["mlx_unused_towers"])

    # Load non-strictly only when the operator has declared which towers may
    # differ, then prove the difference stayed inside them.
    model, processor = load(checkpoint, strict=not unused)
    if unused:
        params = dict(tree_flatten(model.parameters()))
        mismatched: list[str] = []
        for shard in sorted(Path(checkpoint).glob("*.safetensors")):
            for name, weight in mx.load(str(shard)).items():
                held = params.get(name)
                if held is not None and tuple(held.shape) != tuple(weight.shape):
                    mismatched.append(name)
        check_weight_mismatches(mismatched, unused, system)

    config = load_config(checkpoint)
    declared = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
    check_vision_available(
        present_vision_attributes(model),
        declares_vision="vision_config" in declared,
        system=system,
    )
    return model, processor, config


def _transcribe_mlx(loaded: tuple, spec: dict, prompt: str, image: Path, system: str) -> str:
    """Transcribe one page with a loaded MLX model.

    Args:
        loaded: The tuple from `_load_mlx`.
        spec: A validated system spec.
        prompt: The prompt text.
        image: The page image.

    Returns:
        The model's Markdown.
    """
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = loaded
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    result = generate(
        model,
        processor,
        formatted,
        [str(image)],
        max_tokens=spec["max_output_tokens"],
        temperature=spec["temperature"],
        top_p=spec["top_p"],
        verbose=False,
    )
    check_generation_complete(getattr(result, "finish_reason", None), image.stem, system)
    return result.text if hasattr(result, "text") else str(result)


@app.command()
def main(
    corpus: Annotated[Path, typer.Option("--corpus", help="An exported parsing_YYYYMMDD/ directory.")],
    system: Annotated[str, typer.Option("--system", help="A name declared in the systems file.")],
    out: Annotated[
        Path, typer.Option("--out", help="Predictions root; the system subdir is created here.")
    ] = Path("runs"),
    systems_file: Annotated[
        Path, typer.Option("--systems", help="Where the systems are declared.")
    ] = _DEFAULT_SYSTEMS,
    prompt_path: Annotated[
        Path, typer.Option("--prompt", help="The prompt shipped with the corpus.")
    ] = _DEFAULT_PROMPT,
    shard: Annotated[
        int,
        typer.Option(
            "--shard",
            help="This process's shard index, from 0. With --shards, runs one whole "
            "engine per GPU over a disjoint slice of the pages.",
        ),
    ] = 0,
    shards: Annotated[
        int,
        typer.Option("--shards", help="How many processes share the work; one per GPU."),
    ] = 1,
    declare_unproducible: Annotated[
        bool,
        typer.Option(
            "--declare-unproducible",
            help="After the run, record any page still without a prediction as one this "
            "system cannot produce. score then counts it as a total failure rather than "
            "refusing the system. Deliberate: never inferred from a failure.",
        ),
    ] = False,
    repetition_penalty: Annotated[
        float | None,
        typer.Option(
            "--repetition-penalty",
            help="Override the declared penalty for this run only. Use to retry pages "
            "that ran to the token cap; pages produced under it are recorded.",
        ),
    ] = None,
) -> None:
    """Transcribe every corpus page with one prompted VLM, skipping pages done.

    Args:
        corpus: An exported corpus directory.
        system: The declared system to run.
        out: Predictions root.
        systems_file: Path to the systems declaration.
        prompt_path: Path to the shipped prompt.

    Raises:
        typer.Exit: Configuration is unusable, or a page produced no prediction.
    """
    try:
        spec = system_named(load_vlm_systems(systems_file), system, systems_file)
        prompt = read_prompt(prompt_path)
        stems = corpus_stems(corpus)
        images = corpus_images(corpus)
    except RunnerError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None

    spec, overridden = apply_penalty_override(spec, repetition_penalty)

    out_dir = out / system
    try:
        check_prompt_provenance(out_dir, prompt_path, prompt)
    except RunnerError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None
    todo = pending(out_dir, stems)
    try:
        todo = shard_of(todo, index=shard, shards=shards)
    except RunnerError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None
    # What this process is answerable for. With one process that is the whole
    # corpus, and finishing means every page exists. With several, it is only
    # this slice: the other shards' pages belong to other processes and are
    # still being written, so checking them would fail whichever process
    # happens to finish first — and, worse, `--declare-unproducible` would
    # write off pages another process is in the middle of transcribing.
    owned = todo if shards > 1 else stems
    if shards > 1:
        rprint(f"[dim]shard {shard} of {shards}: {len(todo)} page(s) of this process[/dim]")
    if overridden:
        rprint(
            f"[yellow]repetition_penalty overridden to {spec['repetition_penalty']} for this "
            f"run — the declared value is the benchmark's; these pages are recorded as "
            f"produced under an override.[/yellow]"
        )
        write_settings_provenance(out_dir, spec["repetition_penalty"], todo)
    rprint(
        f"[bold]{system}[/bold] ({spec['transport']}, temperature {spec['temperature']}): "
        f"{len(todo)} of {len(stems)} page(s) to transcribe"
    )
    if not todo:
        rprint("[green]nothing to do — every page already has a prediction[/green]")
        return

    loaded = None
    vllm_engine = None
    try:
        if spec["transport"] == "local_mlx":
            loaded = _load_mlx(spec, system)
        elif spec["transport"] == "vllm_offline":
            vllm_engine = _load_vllm(spec, system)
    except RunnerError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None

    started = time.monotonic()
    failures: list[str] = []

    if vllm_engine is not None:
        # Batched: the scheduler runs a whole chunk concurrently, which is what
        # max_num_seqs is for. A truncated page is set aside, not written.
        size = int(spec["vllm_engine"]["max_num_seqs"])
        done = 0
        for batch in chunks(todo, size):
            batch_started = time.monotonic()
            try:
                results = transcribe_vllm_batch(
                    vllm_engine, spec, prompt, [images[stem] for stem in batch], system
                )
            except Exception as err:  # noqa: BLE001 - one bad batch must not end the run
                failures.extend(batch)
                done += len(batch)
                rprint(f"[red]  {done}/{len(todo)} batch FAILED: {type(err).__name__}: {err}[/red]")
                continue
            for stem, markdown, truncated in results:
                done += 1
                if truncated:
                    failures.append(stem)
                    kept = keep_truncated(out_dir, stem, markdown)
                    rprint(
                        f"[red]  {done}/{len(todo)} {stem} TRUNCATED at the token cap; "
                        f"kept for inspection at {kept}[/red]"
                    )
                    continue
                write_prediction(out_dir, stem, markdown)
                rprint(f"  {done}/{len(todo)} {stem}")
            rprint(f"[dim]    batch of {len(batch)} in {time.monotonic() - batch_started:.0f}s[/dim]")
    else:
        for index, stem in enumerate(todo, start=1):
            page_started = time.monotonic()
            try:
                markdown = (
                    _transcribe_mlx(loaded, spec, prompt, images[stem], system)
                    if loaded is not None
                    else _transcribe_http(spec, prompt, images[stem])
                )
            except TruncatedGeneration as err:
                failures.append(stem)
                kept = keep_truncated(out_dir, stem, err.text)
                rprint(f"[red]  {index}/{len(todo)} {stem} TRUNCATED; kept at {kept}[/red]")
                continue
            except Exception as err:  # noqa: BLE001 - one bad page must not end the run
                failures.append(stem)
                rprint(f"[red]  {index}/{len(todo)} {stem} FAILED: {type(err).__name__}: {err}[/red]")
                continue
            write_prediction(out_dir, stem, markdown)
            rprint(f"  {index}/{len(todo)} {stem} ({time.monotonic() - page_started:.1f}s)")

    elapsed = time.monotonic() - started
    if todo:
        # cards: a tensor-parallel engine occupies tp GPUs, one per rank. Without
        # it a tp=2 run reads as twice as fast as it is per card, which is the
        # figure a cluster is sized on.
        engine = spec["vllm_engine"]
        cards = int(engine["tensor_parallel_size"]) if isinstance(engine, dict) else 1
        timing = write_timing(
            out_dir,
            system=system,
            inference_seconds=elapsed,
            pages=len(todo),
            cards=cards,
            shard=shard,
            shards=shards,
        )
        rprint(
            f"[dim]{60 * len(todo) / elapsed:.1f} images/min from this process "
            f"on {cards} card(s); timing written to {timing}[/dim]"
        )
    rprint(f"[bold]{system}[/bold]: {len(todo) - len(failures)} written in {elapsed / 60:.1f} min")

    still_missing = pending(out_dir, owned)
    if still_missing and declare_unproducible:
        declared = declare_unproduced(
            out_dir,
            still_missing,
            f"generation ran to max_output_tokens ({spec['max_output_tokens']}) at the "
            f"declared settings; no usable transcription was produced",
        )
        rprint(
            f"[yellow]declared {len(still_missing)} page(s) unproducible in {declared} — "
            f"score will count them as total failures, not skip them[/yellow]"
        )
        return

    try:
        verify_complete(out_dir, owned)
    except RunnerError as err:
        rprint(f"[red]{err}[/red]")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()

"""Declared prompted-VLM systems, loaded from YAML and validated at startup.

The §8.6 calibration pass names its systems in the spec; this file is where
that decision becomes runnable. Nothing is discovered from a checkpoint
directory or an environment variable — the same rule `generation_config.yml`
follows for document types.

Every key is required for every system, including the no-op: a local system
still writes `base_url: none`, and a remote one still writes its checkpoint
path as its served `model` id. Reading the file alone answers what each system
is and how it is reached, without consulting Python.
"""

import os
from pathlib import Path

import yaml

from runners.common import RunnerError, runner_error

TRANSPORTS: tuple[str, ...] = ("local_mlx", "openai_http", "vllm_offline")

REQUIRED_SYSTEM_KEYS: tuple[str, ...] = (
    "transport",
    "model",
    "base_url",
    "base_url_env",
    "api_key_env",
    "temperature",
    "top_p",
    "max_output_tokens",
    "timeout_seconds",
    "mlx_unused_towers",
    "image_first",
    "vllm_engine",
    "repetition_penalty",
)

# Shipped as an unmissable placeholder rather than a plausible-looking host, so
# an unconfigured endpoint fails at startup instead of posting the corpus
# somewhere unintended.
_PLACEHOLDER = "REPLACE-ME"

_NONE = "none"

# vLLM engine arguments, required whole for a vllm_offline system. Values are
# the sandbox's measured ones (LMM_POC config/run_config.yml).
VLLM_ENGINE_KEYS: tuple[str, ...] = (
    "tensor_parallel_size",
    "max_model_len",
    "gpu_memory_utilization",
    "max_num_seqs",
    "limit_mm_images",
    "enable_prefix_caching",
    "enforce_eager",
    "soft_tokens",
    "tokenizer",
)

# The checkpoint accepts only these vision budgets. 1120 and 280 were both
# measured and both regressed (LMM_POC run_config.yml), so 560 is the value
# to beat rather than a starting guess.
LEGAL_SOFT_TOKENS: tuple[int, ...] = (70, 140, 280, 560, 1120)


def load_vlm_systems(path: Path) -> dict[str, dict]:
    """Load and validate every declared system.

    Args:
        path: Path to `config/vlm_systems.yml`.

    Returns:
        System name -> validated spec mapping.

    Raises:
        RunnerError: The file is missing, unparseable, declares no system, or
            any system omits a key or declares an unusable value.
    """
    resolved = path.resolve()
    if not path.exists():
        raise runner_error(
            f"{path} does not exist.",
            where=str(resolved),
            expected="a YAML file declaring the prompted VLM systems, e.g.\n"
            "              systems:\n"
            "                gemma-4-E4B-it-qat-4bit:\n"
            "                  transport: local_mlx",
            recover="create config/vlm_systems.yml, or pass --systems pointing at it.",
        )

    try:
        declared = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise runner_error(
            f"the file is not valid YAML: {err}",
            where=str(resolved),
            expected="parseable YAML, e.g.\n              systems:\n                gemma:\n"
            "                  transport: local_mlx",
            recover="fix the syntax error at the line named above.",
        ) from err

    systems = (declared or {}).get("systems") if isinstance(declared, dict) else None
    if not isinstance(systems, dict) or not systems:
        raise runner_error(
            "no system is declared under the 'systems' key.",
            where=f"{resolved} -> systems",
            expected="a mapping of system name to spec, at least one entry, e.g.\n"
            "              systems:\n                gemma-4-E4B-it-qat-4bit:\n"
            "                  transport: local_mlx",
            recover="add one entry per system the calibration pass should run.",
        )

    for name, spec in systems.items():
        _validate_shape(name, spec, resolved)
    return systems


def _validate_shape(name: str, spec: object, path: Path) -> None:
    """Check one system declares the right keys and a known transport.

    Shape is checked for every declared system, because a typo anywhere should
    surface at startup. Whether the *target* is usable — the checkpoint on
    disk, the endpoint configured — is checked only for the system actually
    selected, so an unconfigured remote host cannot block a local run.

    Args:
        name: The system name, which is also its `runs/` subdirectory.
        spec: The declared mapping.
        path: Resolved path of the declaring file, for diagnostics.

    Raises:
        RunnerError: The spec is not a mapping, omits a key, or declares an
            unknown transport.
    """
    if not isinstance(spec, dict):
        raise runner_error(
            f"system '{name}' is {type(spec).__name__}, not a mapping.",
            where=f"{path} -> systems.{name}",
            expected=f"a mapping declaring {list(REQUIRED_SYSTEM_KEYS)}.",
            recover=f"indent the settings under 'systems.{name}:'.",
        )

    for key in REQUIRED_SYSTEM_KEYS:
        if key not in spec:
            raise runner_error(
                f"system '{name}' does not declare '{key}'.",
                where=f"{path} -> systems.{name}.{key}",
                expected=f"every key of {list(REQUIRED_SYSTEM_KEYS)} present — none has a "
                "Python default, including no-op values, e.g.\n"
                f"              {key}: none",
                recover=f"add '{key}:' under systems.{name} in {path}.",
            )

    transport = spec["transport"]
    if transport not in TRANSPORTS:
        raise runner_error(
            f"system '{name}' declares unknown transport {transport!r}.",
            where=f"{path} -> systems.{name}.transport",
            expected=f"one of {list(TRANSPORTS)}, e.g.\n              transport: local_mlx",
            recover="set transport to local_mlx for an on-device MLX checkpoint, "
            "openai_http for an OpenAI-compatible server, or vllm_offline for an "
            "in-process vLLM engine.",
        )

    if transport == "vllm_offline":
        _validate_engine(name, spec["vllm_engine"], path)


def _validate_engine(name: str, engine: object, path: Path) -> None:
    """Check a vllm_offline system declares a complete, usable engine block.

    Args:
        name: The system name.
        engine: The declared `vllm_engine` value.
        path: Resolved path of the declaring file, for diagnostics.

    Raises:
        RunnerError: The block is absent, omits a key, or names an illegal
            vision budget.
    """
    if not isinstance(engine, dict):
        raise runner_error(
            f"system '{name}' uses transport vllm_offline but declares vllm_engine: {engine!r}.",
            where=f"{path} -> systems.{name}.vllm_engine",
            expected=f"a mapping declaring {list(VLLM_ENGINE_KEYS)}, e.g.\n"
            "              vllm_engine:\n                max_model_len: 16384",
            recover="add the engine block, or switch transport to openai_http.",
        )

    for key in VLLM_ENGINE_KEYS:
        if key not in engine:
            raise runner_error(
                f"system '{name}' does not declare vllm_engine.{key}.",
                where=f"{path} -> systems.{name}.vllm_engine.{key}",
                expected=f"every key of {list(VLLM_ENGINE_KEYS)} present — engine "
                "tuning has no Python default, e.g.\n              soft_tokens: 560",
                recover=f"add '{key}:' under systems.{name}.vllm_engine in {path}.",
            )

    if engine["soft_tokens"] != _NONE and engine["soft_tokens"] not in LEGAL_SOFT_TOKENS:
        raise runner_error(
            f"system '{name}' declares soft_tokens {engine['soft_tokens']!r}, which the "
            "checkpoint does not accept.",
            where=f"{path} -> systems.{name}.vllm_engine.soft_tokens",
            expected=f"one of {list(LEGAL_SOFT_TOKENS)}, e.g.\n              soft_tokens: 560",
            recover="use 560 — 1120 and 280 were both measured on this checkpoint and "
            "both regressed, so it is the value to beat, not a starting guess.",
        )


def _validate_target(name: str, spec: dict, path: Path) -> None:
    """Check the selected system can actually be reached.

    Args:
        name: The system name.
        spec: Its validated-shape spec.
        path: Resolved path of the declaring file, for diagnostics.

    Raises:
        RunnerError: The checkpoint is absent, or the endpoint is unset or
            still carries the shipped placeholder.
    """
    transport = spec["transport"]
    if transport in ("local_mlx", "vllm_offline"):
        # Both load from disk on the machine running the transcription, so the
        # endpoint checks below do not apply; what has to exist is the weights.
        checkpoint = Path(str(spec["model"]))
        if not checkpoint.is_dir():
            raise runner_error(
                f"system '{name}' names a checkpoint that is not a directory: {checkpoint}.",
                where=f"{path} -> systems.{name}.model",
                expected="an absolute path to a checkpoint directory on THIS machine, e.g.\n"
                "              model: /home/jovyan/nfs_share/models/gemma-4-12B-it-qat-w4a16-ct",
                recover="correct the path, or run this on the host that holds the weights.",
            )
        return

    variable = str(spec["base_url_env"])
    if variable != _NONE:
        # The environment supersedes the file entirely, so `base_url` may stay
        # the shipped placeholder in a committed checkout.
        from_env = os.environ.get(variable)
        if not from_env:
            raise runner_error(
                f"system '{name}' names base_url_env {variable}, which is not set.",
                where=f"{path} -> systems.{name}.base_url_env",
                expected=f"the endpoint exported as {variable}, e.g.\n"
                f"              export {variable}=http://localhost:8000/v1",
                recover=f"export {variable}, or set base_url_env: none and put the endpoint in base_url.",
            )
        if not from_env.startswith(("http://", "https://")):
            raise runner_error(
                f"{variable} holds {from_env!r}, which has no scheme.",
                where=f"the environment variable {variable}",
                expected="an absolute URL beginning http:// or https://, e.g.\n"
                f"              export {variable}=http://localhost:8000/v1",
                recover=f"re-export {variable} with the scheme the server is served over.",
            )
        return

    base_url = str(spec["base_url"])
    if base_url == _NONE:
        raise runner_error(
            f"system '{name}' uses transport openai_http but declares base_url: none.",
            where=f"{path} -> systems.{name}.base_url",
            expected="the server's OpenAI-compatible root, e.g.\n"
            "              base_url: http://gpu-host:8000/v1",
            recover="set base_url to the serving endpoint, or switch transport to local_mlx.",
        )
    if _PLACEHOLDER in base_url:
        raise runner_error(
            f"system '{name}' still carries the shipped {_PLACEHOLDER} placeholder in base_url.",
            where=f"{path} -> systems.{name}.base_url",
            expected="the real endpoint of the remote GPU host, e.g.\n"
            "              base_url: http://gpu-host:8000/v1",
            recover=f"replace {_PLACEHOLDER} with the host actually serving this model.",
        )
    if not base_url.startswith(("http://", "https://")):
        raise runner_error(
            f"system '{name}' declares a base_url with no scheme: {base_url!r}.",
            where=f"{path} -> systems.{name}.base_url",
            expected="an absolute URL beginning http:// or https://, e.g.\n"
            "              base_url: http://gpu-host:8000/v1",
            recover="prefix the host with the scheme it is served over.",
        )


def resolve_base_url(spec: dict) -> str:
    """Return the endpoint to call, preferring the environment when one is named.

    Naming a variable in `base_url_env` lets a committed checkout keep the
    shipped placeholder while the real endpoint lives in the operator's shell —
    which is what a shared clone on a remote host wants. Validation of both
    paths happens at selection time, so this only resolves.

    Args:
        spec: A validated system spec.

    Returns:
        The endpoint URL.
    """
    variable = str(spec["base_url_env"])
    if variable != _NONE:
        return os.environ[variable]
    return str(spec["base_url"])


def system_named(systems: dict[str, dict], name: str, path: Path) -> dict:
    """Select one declared system by name.

    Args:
        systems: The validated mapping from `load_vlm_systems`.
        name: The requested system name.
        path: Path of the declaring file, for diagnostics.

    Returns:
        That system's spec.

    Raises:
        RunnerError: No system carries that name, or its target is unusable.
    """
    if name not in systems:
        raise runner_error(
            f"no system named '{name}' is declared.",
            where=f"{path.resolve()} -> systems",
            expected=f"one of the declared names: {sorted(systems)}",
            recover=f"pass --system with one of those names, or declare '{name}' in {path}.",
        )
    _validate_target(name, systems[name], path.resolve())
    return systems[name]


__all__ = [
    "REQUIRED_SYSTEM_KEYS",
    "resolve_base_url",
    "TRANSPORTS",
    "RunnerError",
    "load_vlm_systems",
    "system_named",
]

# Running the prompted VLMs on a remote GPU host

For the §8.6 calibration pass, when the VLM lives somewhere other than the
machine holding the corpus. Docling and MinerU do not need this — they run
locally (see `environment-docling.yml` / `environment-mineru.yml`).

## Two ways the model can be reached

Pick by how the host actually runs the model. vLLM offers both, and a sandbox
built on the offline API has no `/v1` to call — do not infer one from the other.

**`vllm_offline`** — an in-process vLLM engine, which is how the LMM_POC sandbox
drives these checkpoints (`from vllm import LLM` in its `models/model_loader.py`).
There is no HTTP endpoint; the runner constructs the engine itself, so it must
run **on the GPU host**, in an env with **vLLM >= 0.23.0** — the floor for the
`gemma4_unified` architecture. Anything older fails at engine load with an
unknown-architecture error. This is what `gemma-4-12B-it-qat-w4a16-ct` is
configured for.

**`openai_http`** — an OpenAI-compatible server (`vllm serve`, SGLang). The
runner is then a pure client and can run anywhere that can reach the endpoint.

## What the remote needs

Third-party dependencies are **`typer`, `rich`, `pyyaml`** and nothing else —
everything else the runner uses is standard library, and every ML import is
deferred into the function that needs it. For `vllm_offline`, add vLLM, which
the serving env already has.

Check before creating anything; an existing env usually suffices:

```bash
for e in $(conda env list | awk '/^[a-z]/ {print $1}'); do
  echo -n "$e: "
  conda run -n $e python -c 'import typer, rich, yaml; print("OK")' 2>/dev/null || echo missing
done
```

Prefer an env that is not load-bearing for serving. Do not `pip install` into a
working vLLM env unless you must — that is how those get broken.

`environment-vlm.yml` is for the **local MLX** path only and will not solve on
Linux; mlx is macOS-only. `environment.yml` builds the full generator toolchain
(Pillow, Faker, pytest) and is overkill for running predictions.

## The corpus does not travel with the repo

`parsing_*/` is gitignored, so a fresh clone has no corpus. **Copy the exported
directory across; do not regenerate it on the remote.** Rendering depends on the
installed Pillow and FreeType, so a regenerated corpus can differ pixel for pixel
from the one the transcripts and manifest describe. `score` verifies every image
against `manifest.jsonl` and refuses to score across vintages, so this surfaces
as a hard failure rather than a wrong number — but only after the whole run has
been spent.

```bash
rsync -av parsing_20260818/ remote:/path/to/doc-parsing-corpus/parsing_20260818/
```

The runner needs only `images/` and `transcripts/`; the rest matters when scoring.

## Engine arguments are declared, not inferred

For a `vllm_offline` system every engine argument lives in
`config/vlm_systems.yml` under `vllm_engine`, and all are required. They decide
what the model can **see**, so a wrong value produces a run that completes and is
quietly worse.

`soft_tokens` is the vision budget and the most consequential of them. On this
checkpoint 1120 and 280 were both measured and both regressed — median F1
0.953 → 0.369 at 1120, 0.915 → 0.882 at 280 — so 560 is the value to beat, not a
starting guess. It is written once and applied to both places vLLM reads it
(`mm_processor_kwargs.max_soft_tokens` and
`hf_overrides.vision_config.num_soft_tokens`), so the two cannot disagree.

`tensor_parallel_size: 1` means **one whole engine per GPU**, not one GPU. A
card fits a whole engine for the 12B, so a 2xL4 host runs two engines rather
than sharding one engine tp=2 across PCIe: a whole replica per card avoids an
all-reduce per layer over an interconnect these cards do not have. Where the
weights do NOT fit one card — the 31B — tp=2 is the option, and it works.

**Use both cards.** Run one process per GPU over disjoint pages:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m runners.run_vlm --corpus parsing_20260819c \
    --system gemma-4-12B-it-qat-w4a16-ct --out runs --shard 0 --shards 2 &
CUDA_VISIBLE_DEVICES=1 python -u -m runners.run_vlm --corpus parsing_20260819c \
    --system gemma-4-12B-it-qat-w4a16-ct --out runs --shard 1 --shards 2 &
wait
```

Both write to the same `--out`; the slices are disjoint and the resume logic
keeps them so. The split is strided, not contiguous, because a bank statement
takes minutes and a receipt seconds — a contiguous split would finish far apart.
Omitting `--shards` uses ONE card and leaves the other idle.

## Pointing an `openai_http` system at a server

Name a variable in `base_url_env` and export it, so the committed file stays
placeholder-only and a re-clone loses nothing:

```yaml
    base_url: http://REPLACE-ME:8000/v1   # ignored when base_url_env is named
    base_url_env: VLM_BASE_URL
```

```bash
export VLM_BASE_URL=http://localhost:8000/v1
```

A named-but-unset variable is an error, never a silent fall back to `base_url`.
`model` must match what the server advertises at `/v1/models`, not the
checkpoint path.

## Run

```bash
conda run -n <env> python -m runners.run_vlm \
    --corpus parsing_20260818 \
    --system gemma-4-12B-it-qat-w4a16-ct \
    --out runs
```

Smoke-test one page first — the runner is resumable, so it costs nothing and the
real run picks up the rest:

```bash
mkdir -p /tmp/one/images /tmp/one/transcripts
cp parsing_20260818/images/CASE001_invoices.png /tmp/one/images/
cp parsing_20260818/transcripts/CASE001_invoices.md /tmp/one/transcripts/
conda run -n <env> python -m runners.run_vlm \
    --corpus /tmp/one --system gemma-4-12B-it-qat-w4a16-ct --out /tmp/one_run
```

A re-run transcribes only stems with no non-empty prediction, so an interrupted
run costs the page in flight. A page hitting `max_output_tokens` is **refused
rather than written**: a truncated transcription is usually a repetition loop,
and writing it scores a broken generation as a catastrophic misreading.

## Send back

Only `runs/<system>/` is needed to score:

```bash
rsync -av runs/gemma-4-12B-it-qat-w4a16-ct/ local:/path/to/runs/gemma-4-12B-it-qat-w4a16-ct/
```

```bash
conda run -n docparse python -m evaluation.cli \
    --corpus parsing_20260818 --predictions runs
```

`score` reads **every** immediate subdirectory of `--out` as a system and refuses
if any lacks a prediction for every transcript stem. Do not leave a partially-run
system in `runs/`, and keep scratch directories outside it.

## Notes on these checkpoints

- Every gemma4 model in the LMM_POC sandbox registers `default_image_first=True`.
  The runner honours this through the per-system `image_first` key, which is why
  that key is declared rather than assumed.
- Thinking on `gemma4_unified` is opt-in via a `<|think|>` token at the start of
  a system prompt. The runner sends a bare user message with no system prompt,
  and passes `enable_thinking: False` as belt-and-braces.
- `gemma-4-12B-it-qat-w4a16-ct` is compressed-tensors, a vLLM/GPU format. It
  cannot be loaded by mlx-vlm at all, so any local Apple-Silicon stand-in is a
  *different quantisation* of the same base weights and must carry its own
  system name — never this one's.

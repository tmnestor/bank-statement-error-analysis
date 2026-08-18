# Running the prompted VLMs on a remote GPU host

For the §8.6 calibration pass, when the VLM is served somewhere other than the
machine holding the corpus. Docling and MinerU do not need this — they run
locally (see `environment-docling.yml` / `environment-mineru.yml`).

## What the remote needs

**No ML dependencies.** `runners/run_vlm.py` defers every mlx import into the
functions that use them, and the `openai_http` transport speaks
chat-completions over the standard library. The plain `docparse` env is enough:

```bash
conda env create -f environment.yml
conda activate docparse
```

`environment-vlm.yml` is for the **local MLX** path only and will not solve on
Linux — mlx is macOS-only. Do not reach for it here.

## The corpus does not travel with the repo

`parsing_*/` is gitignored, so a fresh clone has no corpus. **Copy the exported
directory across; do not regenerate it on the remote.** Rendering depends on the
installed Pillow and FreeType, so a regenerated corpus can differ pixel-for-pixel
from the one the transcripts and manifest describe. `score` verifies every image
against `manifest.jsonl` and refuses to score across vintages, so this surfaces
as a hard failure rather than a silent wrong number — but only after the whole
run has been spent.

```bash
rsync -av parsing_20260818/ remote:/path/to/doc-parsing-corpus/parsing_20260818/
```

The runner itself needs only `images/` and `transcripts/`; the rest of the
directory matters when scoring.

## Point the system at the server

**Preferred — keep the endpoint out of the repo.** Name a variable in
`base_url_env` and export it in the sandbox's shell. `base_url` may then stay
the shipped placeholder, so the checkout has no local edit to lose on a
re-clone and no host name in version control:

```yaml
  gemma-4-12B-it-qat-w4a16-ct:
    base_url: http://REPLACE-ME:8000/v1   # ignored when base_url_env is named
    base_url_env: VLM_BASE_URL
```

```bash
export VLM_BASE_URL=http://localhost:8000/v1
```

A named-but-unset variable is an error, never a silent fall back to `base_url` —
a misconfigured endpoint should stop the run, not quietly redirect it.

**Alternative** — put the endpoint in the file and set `base_url_env: none`. The
loader refuses the `REPLACE-ME` placeholder either way, so an unconfigured system
fails at startup rather than posting the corpus somewhere unintended.

`model` must match what the server advertises at `/v1/models`, not the
checkpoint path. If the two differ the server returns a response with no
choices, and the runner says so.

## Run

```bash
conda run -n docparse python -m runners.run_vlm \
    --corpus parsing_20260818 \
    --system gemma-4-12B-it-qat-w4a16-ct \
    --out runs
```

Resumable: a re-run transcribes only the stems with no non-empty prediction, so
an interrupted run costs the page in flight, not the run. A page that hits
`max_output_tokens` is **refused rather than written** — a truncated
transcription is usually a repetition loop, and writing it would score a broken
generation as a catastrophic misreading.

## Send back

Only `runs/<system>/` is needed to score:

```bash
rsync -av runs/gemma-4-12B-it-qat-w4a16-ct/ local:/path/to/runs/gemma-4-12B-it-qat-w4a16-ct/
```

Scoring can then happen anywhere the corpus is:

```bash
conda run -n docparse python -m generators.pipeline score \
    --corpus parsing_20260818 --predictions runs
```

`score` reads **every** immediate subdirectory of `--out` as a system and
refuses if any lacks a prediction for every transcript stem. So do not leave a
partially-run system in `runs/`, and keep scratch directories outside it.

## Serving notes

- Every gemma4 model in the LMM_POC sandbox registers `default_image_first=True`.
  The runner honours this through the per-system `image_first` key, which is why
  that key is declared rather than assumed.
- `gemma4_unified` (the encoder-free 12B) needs **vLLM >= 0.23.0**. Older
  versions fail at engine load with an unknown-architecture error.
- Thinking on `gemma4_unified` is opt-in via a `<|think|>` token at the start of
  a system prompt. The runner sends a bare user message with no system prompt,
  so thinking stays off.

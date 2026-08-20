# CoT Compression Experiment — Handoff

**Status:** Harness implemented, tested end-to-end, pushed. **No real experiment
has been run yet** — only mechanism tests on 4 toy problems.
**Branch:** `cot-compression-sidecar` @ `704b7edccf` on
`git@github.com:cjluo-nv/vllm.git` (9 commits, branched from `main` @ `aa6abec49a`)
**Code:** `examples/cot_compression/` (~1500 lines)
**Owner:** chenjiel@nvidia.com · Last updated 2026-08-20

---

## 1. The research question

> If a model's reasoning trace is compressed *before* it starts producing real
> output, how does the compressed CoT affect final output quality?

Let a reasoning model think until `</think>`, replace the trace with a compressed
version, then make it answer conditioned on the compressed trace.

## 2. What exists

| File | Purpose |
|---|---|
| `cot_sidecar.py` | The sidecar. OpenAI-compatible shim; intercepts `/v1/chat/completions` and `/v1/completions`, proxies everything else |
| `test_sidecar.py` | 4 problems x (baseline + 3 arms), mechanism checks |
| `test_completions.py` | `/v1/completions` incl. byte-exact pass-through fidelity |
| `test_edge_cases.py` | Regression tests for 4 chat-route bugs |
| `verify_conditioning.py` | Proves phase 2 is conditioned on the compressed trace |
| `run_gcp_nrt.sbatch` | Starts vLLM + sidecar + test on one GCP-NRT node |
| `README.md` | Config, arms, architecture, trace-log schema |

**No vLLM source changes.** The sidecar drives existing endpoints; it never
imports `vllm`. No new docker image needed — `fastapi`, `httpx`, `uvicorn` are
already in the `qwen38` image (`orjson` is NOT, hence stdlib `json`).

## 3. Cluster workflow (the tacit knowledge)

```
cluster   gcp-nrt   (ssh alias `gcp` = gcp-nrt-cs-001-login-001.nvidia.com)
account   coreai_numerics_nemotron
image     /lustre/fsw/portfolios/coreai/users/chenjiel/images/qwen38-x86_64-cu129.sqsh
model     /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_modelopt/hf-local/Qwen/Qwen3.8-27B-FP8
workdir   /lustre/fsw/portfolios/coreai/users/chenjiel/cot-compress/
```

- **SSH needs the sandbox disabled** from this workstation (`dangerouslyDisableSandbox: true`);
  the sandbox blocks all network. Symptom: `Network is unreachable`.
- GPU jobs: `--partition=batch --qos=short` (2h cap, better backfill) or `--qos=normal`.
  CPU-only jobs need `--qos=cpu-short` — `normal` has a `MinGRES` of 1 GPU and
  fails with `QOSMinGRES`.
- vLLM boot is **~320 s** (weights + torch.compile + CUDA graphs). Budget for it.
- Queue can be deep (300+ nodes allocated); `squeue --start` gives an estimate.

**Iterate without re-queueing** — this is the big time saver. Leave the job
running (`KEEP_ALIVE`) and exec new steps into it:

```bash
srun --overlap --jobid=<JID> --container-image=$IMG \
     --container-mounts=/lustre:/lustre --no-container-mount-home bash -lc "..."
```

Gotchas learned the hard way:
- **A sidecar started in an overlap step dies when that step exits.** Start it
  *inside* the same step that runs the test, on a fresh port.
- Code lives on lustre, so `rsync` from the local repo and the next step picks it
  up immediately. No rebuild, no re-queue.
- `set -x` echoes the script into the job log, so `grep`ing the log for a string
  that also appears in the script gives false hits. Anchor patterns (`^\[boot\]`).
- Avoid single quotes inside `bash -lc '...'` blocks; they terminate the string.

## 4. Architecture

One inbound request becomes **three GPU generations**, 8 upstream calls total
(critical path 5; three log-only detokenizes run concurrently).

```
STARTUP  C_PREFIX (compression prompt up to "<trace>\n", pre-tokenized)
         THINK_CLOSE = tok("\n</think>\n\n")

1 render          messages -> p_ids                    [LRU cached]
2 GEN 1           p_ids, stop_token_ids=[</think>]
3 tokenize        ~40-token per-request budget tail
4 GEN 2           C_PREFIX + reasoning_ids + tail      (raw token IDs)
5 GEN 3           p_ids + compressed_ids + THINK_CLOSE
6 detokenize      answer -> response
```

`</think>` plays two different roles: a **stop condition** in GEN 1 (sampling
param, nothing added to the prompt) and **prompt text** in GEN 3 (closing the
block after the compressed trace).

GEN 2 is a **fresh independent request** — new system+user turn containing the
trace, rendered with `enable_thinking: false`. For this model "disable
reasoning" literally means the template emits `<think>\n\n</think>\n\n`, a closed
empty block; there is no separate switch.

**Every arm is token-IDs -> token-IDs.** `identity` returns the trace unchanged,
`truncate` slices it, `self` returns GEN 2's output. Text is produced only for
the log. Prefix caching makes GEN 3's re-prefill of `p_ids` nearly free.

## 5. Design decisions and why

1. **Separate process, not a vLLM plugin.** Sidecar restart is ~12 s; restarting
   vLLM to reload a 27B checkpoint is ~320 s. Also lets several sidecars (one per
   arm) share one vLLM and one GPU. No engine internals are needed.
2. **Phase 1 always stops at `</think>`; reasoning is detected from the OUTPUT.**
   A stop condition that never fires cannot change the output, so arming it is
   free — and it catches the model opening a think block *on its own*, which a
   prompt-based decision misses (measured: a bare "write an essay" prompt does
   this, trace compressed 402 -> 123).
3. **Pre-tokenized compression prefix** built by rendering once with a sentinel
   and splitting on it — template-agnostic, verified against the rendered token
   IDs, with a per-request render as fallback. This is why the budget is stated
   *after* the trace: a per-request number in the prefix would make it dynamic.
4. **Pass-through must match vLLM byte for byte.** When no think block appears,
   the sidecar returns exactly what vLLM would: caller's `max_tokens`, caller's
   sampling params (omitted keys fall through to `generation_config`),
   stop-string trimmed, whitespace preserved.
5. **Logging off the event loop.** Bounded `asyncio.Queue` -> single background
   task -> `asyncio.to_thread`. One consumer means no lock is needed anywhere.
   `await put()` applies backpressure rather than dropping experiment rows; the
   lifespan drains before closing. On lustre an inline `write()` can stall every
   in-flight request for milliseconds.

## 6. Bugs found and fixed — do not reintroduce

| Bug | Symptom |
|---|---|
| Compression prompt said "preserve the original voice" | Greedy decode made verbatim copying easiest; compressor reproduced traces unchanged to `max_tokens` (ratio 0.90) |
| Compressor `max_tokens` too tight (`target*1.6`) | Compressions cut off mid-sentence, silently turning `self` into `truncate` |
| `stop` as a bare string | `list("END")` -> `['E','N','D']`; generation stopped on the letter E |
| Chat route assumed a think block always open | `enable_thinking:false` ran the compression path and ignored `max_tokens` |
| `n>1` / `logprobs` silently ignored on chat | A pass@k harness would get one choice and mis-score |
| Phase 1 used `THINK_BUDGET` not caller's `max_tokens` | `max_tokens:8` could generate thousands |
| `temperature`/`top_p` defaulted to 0.6/1.0 | Silently overrode the model's `generation_config` |
| Output `.strip()`ed and kept the stop string | `'63\n\n'` where vLLM returns `'63'` |
| `/health` returns an empty body | Any readiness probe that JSON-parses it retries forever |
| `retok_identical` measured a `rstrip`, not the round trip | Control reported drift that was our own whitespace handling |

## 7. Experimental caveats — read before running anything

These matter more than the code.

1. **The current test problems are far too easy.** Traces are 121-501 tokens and
   every arm gets every answer right — *including mechanical head/tail truncation
   at ratio 0.29*. This model restates its answer at the end of its trace, so
   truncation preserves the answer verbatim. Accuracy here measures "can the
   model copy an answer it can still see."
2. **Conditioning is difficulty-dependent.** Injecting two traces that reach the
   same correct answer by different routes gives perfect separation on a
   501-token problem (4/4 vs 0/4) but *no* route dependence on a 121-token one —
   the model just re-solves from the question. **On easy problems the compression
   arms cannot degrade, because the trace is not load-bearing.** Any real
   measurement needs AIME / GPQA-Diamond-scale problems.
3. **The compression prompt was never ablated.** Four things changed at once when
   fixing the verbatim-copy failure (dominant brevity, "never copy verbatim",
   pass/fail framing, budget repeated after the trace). Which of them matters is
   unknown.
4. **vLLM is not bit-reproducible across batch compositions.** Same seed, same
   prompt, different batching -> different output. Observed directly: an injected
   divergent trace was followed 1/1 in one run and overridden 5/5 in another. Run
   >= 3 seeds and report spread.
5. **Ratio adherence is loose** (0.28-0.39 measured against a 0.3 target). Bin by
   `achieved_ratio`, never by the requested ratio.
6. **Share phase-1 traces across arms.** Run `identity` first to populate
   `TRACE_CACHE_GLOB`; later arms reuse those exact traces. This makes the
   comparison paired, removes phase-1 sampling noise, and skips most of the GPU
   cost. Use a paired test (McNemar), not independent accuracies.
7. **Fix `THINK_BUDGET`/`ANSWER_BUDGET` across arms** or the compressed arm gets
   a larger effective answer budget purely from a shorter prompt.
8. **`/v1/completions` with a bare few-shot prompt does not exercise the model's
   intended mode** — no assistant turn, no think block. Those numbers are not
   comparable to chat-mode or published scores. Use chat, or pre-render the
   template into the prompt.

## 8. Verified behaviour (what the tests actually prove)

- Mechanism: 16/16 correct across 4 problems x (baseline + identity + truncate +
  self), all mechanism checks passing.
- Conditioning: the phase-2 prompt contains the compressed trace and **not** the
  original (`original_trace_in_prompt: false`); route injection separates 4/4 vs
  0/4 on a hard problem.
- `/v1/completions` pass-through is byte-identical to vLLM on bare prompts,
  `max_tokens` capping, and with sampling params omitted.
- Spontaneous think blocks on bare prompts are caught and compressed.
- 24 concurrent requests, no log records lost across a graceful shutdown.
- `MEASURE_RETOK=1` reports `retok_identical: true` on every record — this
  tokenizer has zero detokenize/tokenize drift. Do not assume that elsewhere.

## 9. Next steps

1. Swap in a real dataset (AIME / GPQA-Diamond). Nothing before this produces a
   meaningful number.
2. Write the replay verifier: read a trace record, rebuild
   `p_ids + compressed + </think>` from the logged IDs, re-run phase 2 at the
   logged seed, diff against `phase2.text`. Confirms the log is sufficient.
3. Populate the phase-1 trace cache with the `identity` arm.
4. Sweep arms x ratios {0.75, 0.5, 0.25, 0.1} x >= 3 seeds against the cache.
5. Ablate the compression prompt (caveat 3).
6. Analyse: McNemar paired tests, accuracy vs *achieved* ratio, segmented by
   `phase1_closed`.

Optional, discussed but not done: bucket the per-request budget tail to
pre-tokenize it (removes the last CPU call, critical path 5 -> 4); add a
`claude` arm as a quality ceiling on a subset; run a small instruct model on a
second server as the compressor.

## 10. Notes

- Research tooling, **not** a vLLM contribution. Do not propose to
  `vllm-project/vllm` (see `AGENTS.md`).
- Never use system `python3` or bare `pip` in this repo — use `uv` / `.venv`.
- Trace schema is at **3**; ratios shifted when the arms became token-level, so
  records from schema 2 and earlier are not directly comparable.
- Artifacts from the test runs are on lustre under `cot-compress/out/`.

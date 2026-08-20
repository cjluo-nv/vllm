# CoT Compression Sidecar

Experimental harness for measuring how **compressing a reasoning trace before the
model produces its answer** affects output quality.

One inbound `/v1/chat/completions` request is split into three steps:

1. **phase 1** — generate reasoning, stopping at `</think>`
2. **compress** — condense the trace (self-compression through the same vLLM
   server, with the compressor's own thinking disabled)
3. **phase 2** — resume generation with the compressed trace spliced back in

Both `/v1/chat/completions` and `/v1/completions` are intercepted; everything
else is proxied to vLLM unchanged, so an eval harness can point at the sidecar
without modification.

### /v1/completions

Same pipeline over a raw prompt. Accepts a string, a token-ID list, or a batch of
either. Two things differ from the chat endpoint:

- **A bare prompt may have no `<think>` scaffolding.** Phase 1 always stops at
  `</think>` regardless, and whether reasoning happened is read off the output:
  did phase 1 end at `</think>`? If not, that generation is the whole answer and
  is returned untouched with `no_think_block: true`, so a completions eval can
  never silently report an uncompressed run as a compressed one.

  Arming that stop costs nothing on prompts that never reason — a stop condition
  that does not fire cannot change the output — and it catches the case where
  the model opens a think block *on its own*, which a decision made from the
  prompt alone would miss. Measured: `"Write a long essay about clouds."` opens
  no think block in the prompt, yet the model emits one, and the trace is
  compressed 322 → 89 tokens.
- **Unsupported options are rejected with a 400**, not ignored: `echo`, `suffix`,
  `logprobs`, `best_of`, `prompt_logprobs`, `n > 1`, `stream`.

The caller's `stop` strings are applied to phase 1 *and* phase 2, since on a
no-think prompt phase 1 is the whole answer. `max_tokens` sets the phase-2 budget
here (unlike the chat endpoint, which uses `ANSWER_BUDGET`).

No vLLM source changes are required. The sidecar drives vLLM's existing
token-in / token-out API:

| Endpoint | Use |
|---|---|
| `POST /v1/chat/completions/render` | messages -> prompt token IDs |
| `POST /inference/v1/generate` | token IDs -> token IDs |
| `POST /tokenize` / `POST /detokenize` | text <-> token IDs |

## How it works

The sidecar is a broker: its own HTTP server that sits between the client and
vLLM, and turns one inbound request into three generations.

```
client ──POST /v1/chat/completions──► sidecar :9000
                                        │
   1. POST /v1/chat/completions/render  ├──► vLLM   messages -> prompt token IDs (p_ids)
   2. POST /inference/v1/generate       ├──► vLLM   GEN 1: p_ids, stop at </think>
      POST /detokenize                  ├──► vLLM   reasoning IDs -> text
   3. POST /v1/chat/completions/render  ├──► vLLM   NEW prompt: "compress this trace"
   4. POST /inference/v1/generate       ├──► vLLM   GEN 2: compressor, thinking OFF
      POST /tokenize                    ├──► vLLM   compressed text -> token IDs
   5. POST /inference/v1/generate       ├──► vLLM   GEN 3: p_ids + compressed + </think>
      POST /detokenize                  ├──► vLLM   answer IDs -> text
                                        │
client ◄────── OpenAI response ─────────┘
```

Three GPU generations; everything else is a CPU-side tokenizer call on the vLLM
server. The three generations arrive as three *independent* vLLM requests --
vLLM has no idea they are related and simply batches them alongside everything
else, so continuous batching is unaffected.

`</think>` appears in two mechanically different roles:

- **GEN 1** arms it as a *stop condition* (`sampling_params.stop_token_ids`).
  Nothing is added to the prompt; this only tells vLLM when to halt.
- **GEN 3** appends it as *prompt text*, closing the block after the compressed
  trace so the model knows reasoning is over.

GEN 2 is a **fresh, independent request**, not a continuation: a new system and
user turn containing the trace, rendered with `enable_thinking: false` so the
compressor answers directly instead of reasoning about how to compress.

### Why step 1 returns token IDs rather than reusing `messages`

GEN 3's prompt is `p_ids + compressed + </think>` -- the compressed trace is
spliced *inside* the assistant turn, between the think tags. There is no way to
express that by resending `messages` to `/v1/chat/completions`; the chat API
cannot prefill a partial think block. Working in token IDs also means:

- **no re-tokenization drift on the prompt.** `p_ids` is produced once and
  reused byte-for-byte in GEN 3, never detokenized and re-tokenized. This is
  what the `identity` arm's `retok_identical` measures.
- **a guaranteed prefix-cache hit.** GEN 3's prompt starts with exactly the same
  tokens as GEN 1's, so vLLM reuses that KV and only prefills the compressed
  trace -- which is the efficiency claim the whole experiment rests on.

### Why the trace is detokenized before GEN 2

It could be avoided. The static parts of the compression prompt could be
pre-tokenized at startup and the raw reasoning token IDs concatenated directly,
so GEN 1 and GEN 2 never touch text. That is the tidier design and it is
marginally more robust on tokenizers where the detokenize/tokenize round trip is
not the identity (it *is* the identity here -- measured, `retok_identical` true
on every record).

It is not done because the saving is negligible and the text is needed anyway:

- The calls it would remove are one `/detokenize` and one `/render`, both
  CPU-side, together a few milliseconds against 3-10 seconds of GPU time -- on
  the order of 0.1%.
- The trace text is written to the log regardless, since a log of raw token IDs
  would not be readable. So the detokenize is not extra work.
- The `truncate` and `identity` arms operate on text.
- Building a natural-language prompt by token concatenation puts an untested
  seam at the `<trace>\n` boundary, for no measurable gain.

The rule applied throughout: work at the token level where correctness depends
on it (the spliced GEN 3 prompt), not where it is merely tidier.

## Running

vLLM and the sidecar are two servers, so they are two processes. One shell is
fine — nothing here needs separate terminals:

```bash
# 1. vLLM, unchanged
vllm serve <model> --served-model-name m --port 8000 &

# 2. sidecar (waits for vLLM to become healthy before serving)
VLLM_URL=http://127.0.0.1:8000 MODEL=m ARM=self RATIO=0.3 \
  TRACE_LOG=traces.jsonl python3 cot_sidecar.py &

# 3. point any harness at 9000 instead of 8000
python3 test_sidecar.py --model m
```

`run_gcp_nrt.sbatch` does exactly this inside the container:

```bash
sbatch examples/cot_compression/run_gcp_nrt.sbatch
```

Keeping the sidecar out of the vLLM process is deliberate. Restarting it to
change `ARM`, `RATIO`, or the compression prompt takes ~12 s; restarting vLLM to
reload a 27B checkpoint, re-run torch.compile and recapture CUDA graphs takes
~320 s. It also lets several sidecars share one vLLM, which is how to run every
arm concurrently on a single GPU:

```bash
SIDECAR_PORT=9000 ARM=identity TRACE_LOG=identity.jsonl python3 cot_sidecar.py &
SIDECAR_PORT=9001 ARM=truncate TRACE_LOG=truncate.jsonl python3 cot_sidecar.py &
SIDECAR_PORT=9002 ARM=self     TRACE_LOG=self.jsonl     python3 cot_sidecar.py &
```

## Configuration

Environment variables, all optional except where noted:

| Var | Default | Meaning |
|---|---|---|
| `VLLM_URL` | `http://127.0.0.1:8000` | Upstream vLLM |
| `COMPRESSOR_URL` | `VLLM_URL` | Separate server for the compressor, if desired |
| `MODEL` / `COMPRESSOR_MODEL` | first served model | Served model name |
| `ARM` | `self` | `identity` \| `truncate` \| `self` |
| `RATIO` | `0.3` | Target compressed / original token ratio |
| `THINK_BUDGET` | `4096` | Max phase-1 tokens |
| `ANSWER_BUDGET` | `1024` | Max phase-2 tokens |
| `TRACE_LOG` | `traces.jsonl` | JSONL log (`.gz` supported) |
| `TRACE_CACHE_GLOB` | unset | Reuse phase-1 traces from prior runs |
| `USE_PRIORITY` | `0` | Send `priority: -1` on phase 2 (needs `--scheduling-policy priority`) |

`cot_arm` and `cot_ratio` may also be set per request in the JSON body, so one
server can serve every arm.

## Arms

| Arm | Role |
|---|---|
| `identity` | Control. Detokenize -> re-tokenize unchanged. Isolates token-boundary artifacts; `retok_identical` should be true for every record. |
| `truncate` | Non-semantic floor: keep the first and last N/2 tokens. If LLM compression does not beat this, there is no semantic story. |
| `self` | The model under test compresses its own trace with `enable_thinking: false`. |

## Trace log

One JSON object per request. Both traces are recorded — `phase1.text` /
`phase1.token_ids` for the original, `compress.text` / `compress.token_ids` for the
compressed version — because this process is the only place both exist: vLLM sees
two unrelated requests and the harness sees only the final answer.

Notable fields:

- `compress.retok_identical` — one-line identity control
- `compress.achieved_ratio` — measured, not requested; bin by this at analysis time
- `phase1.closed` — false means phase 1 exhausted `THINK_BUDGET` without emitting
  `</think>`; a different population, segment it out
- `compress.error` — a compression failure falls back to the uncompressed trace,
  which would otherwise silently inflate the compressed arm's accuracy
- `prompt_key` — join key to the harness's own results
- per-phase `ms`

## Model assumptions

The startup probe detects, and logs, whether the chat template opens the think
block in the generation prompt. For Qwen3.8 it does:

- `add_generation_prompt=true` -> `<|im_start|>assistant\n<think>\n`
- `enable_thinking=false` -> `<|im_start|>assistant\n<think>\n\n</think>\n\n`

The second form is what makes self-compression work without the compressor
burning tokens on its own reasoning. If a template ignores `enable_thinking`, the
sidecar force-closes the block at the token level instead.

## Verified

Run end-to-end on GCP-NRT (job 536366) with `Qwen/Qwen3.8-27B-FP8`, TP=1 on one
B200, vLLM `0.1.dev19754+g3a0914114`. All four required routes present; 4 problems
x (baseline + 3 arms) = 16/16 correct with all mechanism checks passing.

Startup probe output for that model:

```
[sidecar] think_start=248068 think_end=248069 te_ids=[248069] single=True
[sidecar] main prompt tail='...<|im_start|>assistant\n<think>\n' case_b=False
[sidecar] compressor tail='...<|im_start|>assistant\n<think>\n\n</think>\n\n' nothink=[]
```

Two failure modes found and fixed during that run, both worth knowing if you
adapt this to another model:

- **Verbatim copy.** A compression prompt that asks to "preserve the original
  voice" makes copying the easiest greedy continuation; the compressor
  reproduced traces unchanged until it hit `max_tokens` (ratio 0.90). Brevity
  has to be the dominant instruction.
- **Cap-as-truncation.** Too tight a `max_tokens` on the compressor cuts it off
  mid-sentence, silently turning the `self` arm into the `truncate` arm. The
  cap is now `target * 3` and `compress.truncated` is recorded.

## Is phase 2 really conditioned on the compressed trace?

Yes. `verify_conditioning.py` checks it two ways.

**Mechanically.** The sidecar records the detokenized phase-2 prompt it actually
sent (`phase2.prompt_tail`) plus `original_trace_in_prompt` /
`compressed_trace_in_prompt`. For the `self` arm those read `False` / `True` --
the original trace is absent. The prompt looks like this:

```
<|im_start|>user
A factory produces 20 robots per hour for 6 hours, ...<|im_end|>
<|im_start|>assistant
<think>
20×6 + 25×4 = 220. 85% pass: 220×0.85 = 187. ANSWER: 187
</think>
```

Only the 44-token compressed trace sits between the think tags; the 121-token
original is gone.

**Behaviourally.** Inject two traces that reach the *same correct answer* by
different routes (`cot_arm=inject`, `cot_inject=...`), so the model has no reason
to override either, and see which derivation the answer uses:

| injected route | answer used block method | answer used AP method |
|---|---|---|
| block (`5k+2`, `10k+5`) | 4/4 | 0/4 |
| arithmetic progression (`2,7,...,997`) | 0/4 | 4/4 |

Perfect separation on a 501-token-trace problem.

### Caveat: conditioning strength depends on problem difficulty

The same route-injection test on a *trivial* problem (121-token trace) shows no
route dependence at all -- the model ignores the injected derivation and re-solves
from the question. Injecting a trace that reaches a *different* answer is likewise
stochastic: observed followed 1/1 in one run and overridden 5/5 in another, at
temperature 0.6 with batch-composition nondeterminism.

This has a direct experimental consequence: **on easy problems the compression
arms cannot degrade, because the model is not leaning on the trace in the first
place.** Any real measurement needs problems hard enough that re-deriving from
scratch is expensive.

## Trace logging is off the event loop

Records go to a bounded `asyncio.Queue`; a single background task batches them
and does `json.dumps` plus the write syscall inside `asyncio.to_thread`, so the
event loop never waits on the filesystem. On lustre a `write()` can stall for
milliseconds, which would otherwise freeze every in-flight request in the
process.

- **No lock anywhere.** There is exactly one consumer, so the file handle has a
  single owner. The sidecar is single-threaded asyncio, so `threading.Lock` was
  never relevant, and the single-consumer design removes the need for the
  `asyncio.Lock` too.
- **Backpressure, not dropping.** `write_record` uses `await queue.put()`. A
  full queue slows the producing request rather than silently losing a row --
  this is an experiment log.
- **Drained on shutdown.** The lifespan pushes a sentinel and awaits the writer
  before closing the file, so the tail of a run is not lost.
- **Batched flush** every `LOG_BATCH` records or `LOG_FLUSH_SECS`, instead of a
  flush per record.

Tuning: `LOG_QUEUE_MAX` (20000), `LOG_BATCH` (64), `LOG_FLUSH_SECS` (2.0).

**Multiple processes is the one case no lock solves.** With `uvicorn --workers N`
those are separate processes; interleaved appends of multi-KB records will
corrupt lines regardless of any in-process lock. Keep one worker, or put the PID
in `TRACE_LOG`.

## Notes

This is research tooling, not a vLLM contribution. It should not be proposed as a
PR to `vllm-project/vllm`.

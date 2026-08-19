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

- **A bare prompt may have no `<think>` scaffolding**, in which case there is no
  reasoning block to compress. The sidecar detects this, returns the generation
  untouched, and sets `no_think_block: true` with `error: "no_think_block"` — so
  a completions eval can never silently report an uncompressed run as a
  compressed one. Pre-render with the chat template (prompt ending in `<think>`)
  if you want compression on this endpoint.
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

## Running

```bash
# terminal 1
vllm serve <model> --served-model-name m --port 8000

# terminal 2
VLLM_URL=http://127.0.0.1:8000 MODEL=m ARM=self RATIO=0.3 \
  TRACE_LOG=traces.jsonl python3 cot_sidecar.py

# terminal 3
python3 test_sidecar.py --model m
```

On GCP-NRT, `run_gcp_nrt.sbatch` does all three inside the container:

```bash
sbatch examples/cot_compression/run_gcp_nrt.sbatch
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

## Notes

This is research tooling, not a vLLM contribution. It should not be proposed as a
PR to `vllm-project/vllm`.

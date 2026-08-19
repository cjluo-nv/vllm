# CoT Compression Experiment — Design & Handoff

**Status:** Implemented and verified end-to-end on GCP-NRT (2026-08-19).
**Implementation:** `examples/cot_compression/` (sidecar, test, sbatch, README)
**Repo:** `/home/scratch.chenjiel_coreai/vllm` @ `aa6abec49a` (2026-08-18)
**Owner:** chenjiel@nvidia.com

## 0. Verified run

Design B (sidecar) with self-compression, on GCP-NRT job `536366`:

- **Model:** `Qwen/Qwen3.8-27B-FP8`, TP=1, one B200
- **Image:** `qwen38-x86_64-cu129.sqsh` (vLLM `0.1.dev19754+g3a0914114`)
- **Confirmed:** all four required routes present; vLLM healthy in 320 s
- **Result:** 4 problems x (baseline + 3 arms) = 16/16 correct, all mechanism
  checks passed, no compressor truncation

| prob | reasoning tok | truncate | self | achieved ratio (self) |
|---|---|---|---|---|
| 1 | 121 | 35 | 44 | 0.36 |
| 2 | 160 | 47 | 60 | 0.38 |
| 3 | 256 | 75 | 147 | 0.57 |
| 4 | 501 | 149 | 162 | 0.32 |

Representative self-compression (problem 4, 501 -> 162 tokens, `finish=stop`):

```
n²+1 ≡ 0 (mod 5) ⇒ n² ≡ 4 (mod 5) ⇒ n ≡ 2,3 (mod 5).
Sum n ≤ 1000 with n ≡ 2,3 (mod 5).
200 terms each residue.
Sum(n≡2): 200/2(2+997) = 99900.
Sum(n≡3): 200/2(3+998) = 100100.
Total = 99900 + 100100 = 200000.
ANSWER: 200000
```

Phase 2 then produced a complete, correctly formatted solution from that trace.

### What the run taught us (not predicted by the design)

1. **Verbatim-copy failure mode.** The first compression prompt said "preserve
   the original voice and formatting style". With greedy decoding that made
   copying the easiest continuation, and the compressor reproduced the trace
   unchanged until it hit `max_tokens` — problem 4 "compressed" 501 -> 450 at
   ratio 0.90. Brevity must be the *dominant* instruction, with an explicit
   "never copy verbatim" and the token budget stated twice. After the rewrite
   the same problem compressed to 0.32.
2. **Compressor `max_tokens` silently becomes truncation.** A tight cap
   (`target * 1.6`) cut compressions off mid-sentence on 2 of 4 problems,
   quietly turning the `self` arm into the `truncate` arm. Give real headroom
   (`target * 3`) and record `compress.truncated` / `compressor_finish_reason`.
3. **Zero re-tokenization drift on this tokenizer.** `retok_identical` was true
   for all 12 sidecar records — `tokenize(detokenize(ids)) == ids` held exactly.
   The identity control is clean for Qwen3.8; do not assume this for others.
   (Measure the round trip itself, not the post-`rstrip` length, or you record
   your own whitespace handling as tokenizer drift.)
4. **`/health` returns an empty body.** Any readiness probe that JSON-parses it
   will retry forever.
5. **Conditioning is real but difficulty-dependent.** Injecting two traces that
   reach the same correct answer by different routes gives perfect separation on
   the 501-token problem (block method 4/4 vs AP method 4/4, 0/4 the other way):
   the answer follows whichever derivation it was handed. On a 121-token problem
   there is no route dependence at all -- the model re-solves from the question
   and ignores the trace. So on easy problems the compression arms *cannot*
   degrade, because the trace is not load-bearing.
6. **These problems are too easy to show degradation.** Traces were 121-501
   tokens and every arm got every answer right, including mechanical truncation.
   Real signal needs harder problems (AIME / GPQA-Diamond) and longer traces.



---

## 1. The research question

> If a model's reasoning trace is compressed *before* it starts producing real
> output, how does the compressed CoT affect final output quality?

Mechanically: let a reasoning model think until it emits `</think>`, replace the
reasoning trace with a compressed version, then make the model produce its answer
conditioned on the compressed trace instead of the original.

## 2. Headline finding

**No vLLM fork is needed.** This vLLM already ships a token-in / token-out API that
provides exactly the required seam. The whole experiment is client-side orchestration
over HTTP.

| Endpoint | Purpose | Source |
|---|---|---|
| `POST /v1/chat/completions/render` | chat messages -> `{token_ids, sampling_params}` | `vllm/entrypoints/scale_out/render/api_router.py:25` |
| `POST /inference/v1/generate` | `{token_ids, sampling_params}` -> `{choices:[{token_ids, finish_reason}]}` | `vllm/entrypoints/scale_out/token_in_token_out/api_router.py:47` |
| `POST /v1/chat/completions/derender` | generate response -> `ChatCompletionResponse` (runs reasoning/tool parsers) | `vllm/entrypoints/scale_out/derender/api_router.py:31` |
| `POST /tokenize` / `POST /detokenize` | text <-> token IDs | `vllm/entrypoints/serve/tokenize/api_router.py:36,62` |

Registered automatically whenever the server supports the `generate` task
(`vllm/entrypoints/launchers/api_server/routers.py:52-55`). No flag, no plugin,
no `VLLM_PLUGINS` allowlisting required.

### Verified facts (checked against this tree)

- **Stop-at-`</think>` is exact at the token level.** `check_stop` appends the sampled
  token *before* testing it (`vllm/v1/core/sched/utils.py:108`), so a
  `stop_token_ids=[think_end_id]` hit leaves `</think>` in the returned `token_ids`
  and sets `stop_reason` to that ID. No string matching needed.
- **Prefix caching is on by default** (`vllm/config/cache.py:107`), so phase 2's
  re-prefill of the original prompt is nearly free.
- `/v1/completions` also accepts a token-ID prompt
  (`vllm/entrypoints/openai/completion/protocol.py:50`) with `return_token_ids`
  (line 168) and `include_stop_str_in_output` (line 80) — a viable fallback path
  that avoids the scale-out routes entirely.
- `--scheduling-policy priority` exists (`vllm/engine/arg_utils.py:1541`,
  `vllm/config/scheduler.py:99`); `GenerateRequest.priority` is honored, lower = earlier.
  Errors if the server was not started with priority scheduling.
- `max_num_seqs` defaults to **128** (`vllm/config/scheduler.py:44`).
- `chat_template_kwargs` is supported on `ChatCompletionRequest`
  (`vllm/entrypoints/openai/chat_completion/protocol.py:358`), and
  `reasoning_effort: "none"` maps to `enable_thinking: False` (lines 585-588).
- `/tokenize` returns `{count, max_model_len, tokens, token_strs}`;
  `/detokenize` takes `{tokens}` and returns `{prompt}`
  (`vllm/entrypoints/serve/tokenize/protocol.py:161-185`).
- `httpx.AsyncClient` defaults to `max_connections=100, max_keepalive_connections=20`
  — **below** vLLM's default batch size. Must be raised.
- `tests/evals/gsm8k/gsm8k_eval.py` already talks to a vLLM endpoint and does answer
  extraction; cheapest source of scoring/dataset code to lift.

### Rejected approach: patching the engine

Do **not** hook `vllm/v1/engine/output_processor.py`, a `LogitsProcessor`, or similar.
Logits processors run inside the batched forward loop, so a blocking external call
there stalls every request in the batch; and V1 has no API to mutate a running
request's token sequence mid-flight. The two-phase client loop gets identical
semantics with zero engine risk.

---

## 3. Two architectures

### Design A — offline three-pass (recommended when you control the driver)

1. Generate **all** reasoning traces in one saturated vLLM run; persist to JSONL.
2. Compress the whole set.
3. Resume **all** items in a second saturated vLLM run.

Pros: GPU saturated throughout, no duty-cycle loss, arms trivially share phase-1
traces (paired comparison), compression cache falls out for free.
Cons: requires writing the driver loop.

### Design B — sidecar (recommended when the eval harness must stay unmodified)

A ~150-line FastAPI process impersonating vLLM. The harness sends one
`/v1/chat/completions` request and gets one normal response; the sidecar does the
two-phase dance inside it.

```
before:   lm-eval  ──────────────────────────────────►  vLLM :8000
after:    lm-eval  ──►  sidecar :9000  ──(N calls)───►  vLLM :8000
```

Pros: harness is untouched; the sidecar is the only place both traces coexist, so
it is the natural instrumentation point.
Cons: inherently duty-cycled if compression is an external API (~10-25% lower GPU
utilization even after tuning).

**Choose B only if the harness matters.** Otherwise A.

---

## 4. Prompt-shape probe (do this first, both designs)

Reasoning templates differ in whether the generation prompt already opens the think
block. Qwen3 / DeepSeek-R1 emit `<think>\n` as part of the generation prompt; other
models make the model emit `<think>` itself. Determine which at startup by rendering
a dummy request and detokenizing the tail:

- **Case A** (template opens it): `think_prefix = []`
- **Case B** (model opens it): `think_prefix = tokenize("<think>\n")`, and strip a
  leading `<think>` token from phase-1 output before compressing.

Everything downstream depends on getting this right. Eyeball the detokenized tail
manually once rather than trusting the heuristic.

---

## 5. Reference implementation — sidecar (final, all fixes folded in)

```python
# cot_sidecar.py
#   uv pip install fastapi uvicorn httpx orjson anthropic
#   ARM=self RATIO=0.25 TRACE_LOG=arm_self_r25.jsonl.gz \
#     uvicorn cot_sidecar:app --port 9000 --workers 1
import asyncio, glob, gzip, hashlib, json, os, time, uuid
from contextlib import asynccontextmanager

import httpx, orjson
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

VLLM            = os.environ.get("VLLM_URL", "http://localhost:8000")
COMPRESSOR_URL  = os.environ.get("COMPRESSOR_URL", VLLM)   # 2nd server for small-model arm
MODEL           = os.environ.get("MODEL", "")
COMPRESSOR_MODEL= os.environ.get("COMPRESSOR_MODEL", MODEL)
ARM             = os.environ.get("ARM", "identity")        # identity|truncate|self|claude
RATIO           = float(os.environ.get("RATIO", "0.5"))
THINK_BUDGET    = int(os.environ.get("THINK_BUDGET", "8192"))
ANSWER_BUDGET   = int(os.environ.get("ANSWER_BUDGET", "2048"))
TRACE_LOG       = os.environ.get("TRACE_LOG", "traces.jsonl.gz")
TRACE_CACHE     = os.environ.get("TRACE_CACHE_GLOB", "")   # reuse phase-1 across arms
USE_PRIORITY    = os.environ.get("USE_PRIORITY", "0") == "1"
CLAUDE_CONC     = int(os.environ.get("CLAUDE_CONCURRENCY", "16"))
COMPRESS_PROMPT_VERSION = "v3"

JSON_HDR = {"content-type": "application/json"}
LIMITS   = httpx.Limits(max_connections=512, max_keepalive_connections=512)

up   = httpx.AsyncClient(base_url=VLLM,           timeout=3600.0, limits=LIMITS)
cup  = httpx.AsyncClient(base_url=COMPRESSOR_URL, timeout=3600.0, limits=LIMITS)
SEM  = asyncio.Semaphore(CLAUDE_CONC)
LOCK = asyncio.Lock()
S    = {}          # runtime state: think ids, prefixes, log handle, phase-1 cache


async def _post(client, path, payload):
    r = await client.post(path, content=orjson.dumps(payload), headers=JSON_HDR)
    r.raise_for_status()
    return orjson.loads(r.content)

post  = lambda p, b: _post(up,  p, b)
cpost = lambda p, b: _post(cup, p, b)


def prompt_key(messages) -> str:
    canon = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


@asynccontextmanager
async def lifespan(app):
    S["think_end"]   = (await post("/tokenize", {"prompt": "</think>",
                                    "add_special_tokens": False}))["tokens"][-1]
    S["think_start"] = (await post("/tokenize", {"prompt": "<think>",
                                    "add_special_tokens": False}))["tokens"][-1]

    # --- prompt-shape probe (section 4). VERIFY THIS BY EYE ONCE. ---
    probe = await post("/v1/chat/completions/render",
                       {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1})
    tail = (await post("/detokenize", {"tokens": probe["token_ids"][-16:]}))["prompt"]
    S["case_b"] = "<think>" not in tail
    S["think_prefix"] = (await post("/tokenize", {"prompt": "<think>\n",
                          "add_special_tokens": False}))["tokens"] if S["case_b"] else []
    print(f"[sidecar] prompt tail={tail!r} case_b={S['case_b']}")

    # compressor-side probe: force an empty think block if the template opens one
    cprobe = await cpost("/v1/chat/completions/render",
                         {"model": COMPRESSOR_MODEL,
                          "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,
                          "chat_template_kwargs": {"enable_thinking": False}})
    ctail = (await cpost("/detokenize", {"tokens": cprobe["token_ids"][-16:]}))["prompt"]
    S["nothink"] = [] if ("<think>" not in ctail or "</think>" in ctail) else [S["think_end"]]

    S["cache"] = load_trace_cache(TRACE_CACHE) if TRACE_CACHE else {}
    S["log"] = (gzip.open(TRACE_LOG, "at") if TRACE_LOG.endswith(".gz")
                else open(TRACE_LOG, "a"))
    try:
        yield
    finally:
        S["log"].close(); await up.aclose(); await cup.aclose()


def load_trace_cache(pattern):
    """(prompt_key, seed) -> reasoning token ids, from prior runs' logs."""
    cache = {}
    for path in glob.glob(pattern):
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt") as f:
            for line in f:
                try:
                    r = orjson.loads(line)
                    cache[(r["prompt_key"], r["sampling"]["seed"])] = (
                        r["phase1"]["token_ids"], r["phase1"]["closed"])
                except Exception:
                    continue
    print(f"[sidecar] phase-1 cache: {len(cache)} traces")
    return cache


app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    if body.get("stream"):
        return JSONResponse({"error": {"message": "sidecar is non-streaming"}}, 400)

    te, tp = S["think_end"], S["think_prefix"]
    temp, seed = body.get("temperature", 0.6), body.get("seed")
    common = {"temperature": temp, "top_p": body.get("top_p", 1.0), "seed": seed}
    pkey = prompt_key(body["messages"])

    t0 = time.perf_counter()

    # (1) messages -> prompt token ids
    rendered = await post("/v1/chat/completions/render",
                          {**body, "stream": False, "max_tokens": 1})
    P = rendered["token_ids"]

    # (2) phase 1: think until </think>  (cache hit skips the GPU entirely)
    cached = S["cache"].get((pkey, seed))
    if cached is not None:
        (reasoning_ids, closed), fr1 = cached, "cached"
    else:
        c1 = (await post("/inference/v1/generate", {
            "token_ids": P,
            "sampling_params": {**common, "max_tokens": THINK_BUDGET,
                                "stop_token_ids": [te]},
        }))["choices"][0]
        gen, fr1 = c1["token_ids"], c1["finish_reason"]
        if S["case_b"] and gen and gen[0] == S["think_start"]:
            gen = gen[1:]
        closed = bool(gen) and gen[-1] == te
        reasoning_ids = gen[:-1] if closed else gen
    t1 = time.perf_counter()

    raw = (await post("/detokenize", {"tokens": reasoning_ids}))["prompt"]

    # (3) compress out of band; never fail the request, but record the failure
    target = max(1, int(len(reasoning_ids) * RATIO))
    try:
        short = raw if not closed else await compress(raw, reasoning_ids, ARM, target)
        cerr = None
    except Exception as e:
        short, cerr = raw, f"{type(e).__name__}: {e}"
    C = (await post("/tokenize", {"prompt": short + "\n",
                                  "add_special_tokens": False}))["tokens"]
    t2 = time.perf_counter()

    # (4) phase 2: resume after a force-closed think block
    p2 = {"token_ids": P + tp + C + [te],
          "sampling_params": {**common, "max_tokens": ANSWER_BUDGET}}
    if USE_PRIORITY:
        p2["priority"] = -1
    c2 = (await post("/inference/v1/generate", p2))["choices"][0]
    answer = (await post("/detokenize", {"tokens": c2["token_ids"]}))["prompt"]
    answer = strip_specials(answer)
    t3 = time.perf_counter()

    rid = f"chatcmpl-{uuid.uuid4().hex}"
    record = {
        "schema": 1, "id": rid, "ts": time.time(),
        "arm": ARM, "ratio_requested": RATIO,
        "compress_prompt_version": COMPRESS_PROMPT_VERSION,
        "prompt_key": pkey,
        "client_request_id": req.headers.get("x-request-id"),
        "messages": body["messages"],
        "sampling": {"temperature": temp, "top_p": body.get("top_p", 1.0), "seed": seed,
                     "think_budget": THINK_BUDGET, "answer_budget": ANSWER_BUDGET},
        "n_prompt_tokens": len(P),
        "phase1": {"text": raw, "token_ids": reasoning_ids, "n_tokens": len(reasoning_ids),
                   "finish_reason": fr1, "closed": closed,
                   "cache_hit": cached is not None,
                   "ms": round((t1 - t0) * 1000)},
        "compress": {"text": short, "token_ids": C, "n_tokens": len(C),
                     "achieved_ratio": len(C) / max(len(reasoning_ids), 1),
                     "retok_identical": C == reasoning_ids,
                     "error": cerr, "ms": round((t2 - t1) * 1000)},
        "phase2": {"text": answer, "n_tokens": len(c2["token_ids"]),
                   "finish_reason": c2["finish_reason"], "ms": round((t3 - t2) * 1000)},
    }
    async with LOCK:
        S["log"].write(orjson.dumps(record).decode() + "\n")

    return {
        "id": rid, "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model", MODEL),
        "choices": [{"index": 0, "finish_reason": c2["finish_reason"],
                     "message": {"role": "assistant", "content": answer,
                                 "reasoning_content": short}}],
        "usage": {"prompt_tokens": len(P) + len(tp) + len(C) + 1,
                  "completion_tokens": len(c2["token_ids"]),
                  "total_tokens": len(P) + len(tp) + len(C) + 1 + len(c2["token_ids"])},
    }


# NOTE: must be registered AFTER /v1/chat/completions so the specific route wins.
# Harnesses call /v1/models to resolve the model name and /health to wait for readiness;
# without this they fail before sending a single completion.
@app.api_route("/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, req: Request):
    r = await up.request(req.method, "/" + path, content=await req.body(),
                         headers=JSON_HDR)
    return Response(r.content, r.status_code,
                    media_type=r.headers.get("content-type"))
```

### Compression arms

```python
COMPRESS_SYS = (
    "You rewrite reasoning traces. Given a trace, produce a condensed version "
    "that preserves every intermediate result, constraint, and decision needed "
    "to finish the problem. Preserve the original voice and formatting style. "
    "Output only the condensed trace, no preamble."
)

async def compress(raw, ids, arm, target):
    if arm == "identity":  return raw                       # re-tokenization control
    if arm == "truncate":  return await truncate_head_tail(ids, target)
    if arm == "self":      return await compress_llm(raw, target)
    if arm == "claude":    return await compress_claude(raw, target)
    raise ValueError(arm)

async def truncate_head_tail(ids, target):
    h = target // 2
    keep = ids[:h] + ids[len(ids) - (target - h):] if len(ids) > target else ids
    return (await post("/detokenize", {"tokens": keep}))["prompt"]

async def compress_llm(raw, target):
    """Self-compression, or a small instruct model on COMPRESSOR_URL."""
    msgs = [{"role": "system", "content": COMPRESS_SYS},
            {"role": "user",
             "content": f"Condense to about {target} tokens.\n\n<trace>\n{raw}\n</trace>"}]
    rendered = await cpost("/v1/chat/completions/render",
                           {"model": COMPRESSOR_MODEL, "messages": msgs, "max_tokens": 1,
                            "chat_template_kwargs": {"enable_thinking": False}})
    Pc = rendered["token_ids"] + S["nothink"]      # belt-and-braces: force-close think
    out = await cpost("/inference/v1/generate", {
        "token_ids": Pc,
        "sampling_params": {"temperature": 0.0, "max_tokens": int(target * 1.6)},
    })
    text = (await cpost("/detokenize",
                        {"tokens": out["choices"][0]["token_ids"]}))["prompt"]
    return strip_specials(text).strip()

async def compress_claude(raw, target):
    import anthropic
    client = anthropic.AsyncAnthropic()          # ANTHROPIC_API_KEY from env
    async with SEM:
        m = await client.messages.create(
            model="claude-opus-5", max_tokens=int(target * 1.6),
            system=COMPRESS_SYS,
            messages=[{"role": "user",
                       "content": f"Condense to about {target} tokens.\n\n"
                                  f"<trace>\n{raw}\n</trace>"}])
    return m.content[0].text.strip()
```

`strip_specials` must remove the tokenizer's special tokens (e.g. `<|im_end|>`) —
`/detokenize` round-trips them, and you do not want `<|im_end|>` spliced into the
model's think block.

---

## 6. Suppressing the compressor's own reasoning (self-compression)

The model under test is a reasoning model; asked to compress, it will *think about
how to compress*, burning thousands of tokens. Three levers, most robust first:

1. **Force-close the think block at the token level** — append `think_end_id` to the
   rendered compression prompt. Works regardless of template support. (`S["nothink"]`)
2. `chat_template_kwargs: {"enable_thinking": false}` — clean when honored (Qwen3),
   silently ignored otherwise.
3. `reasoning_effort: "none"` — vLLM maps it to `enable_thinking: False`.

Use 2 **and** 1: ask the template nicely, then verify at the token level.

### Why self-compression is attractive

It turns compression into another vLLM request in the same continuous batch. The
duty-cycle gap disappears; so does the Anthropic rate limit, the data egress, and
the prefix-cache eviction risk during the gap.

### What it costs

The confound flips rather than vanishing: a drop could be compression loss *or*
"this model is bad at compressing." Mitigate by keeping `truncate` as the floor and
one `claude` arm on a subset as the ceiling — the gap between `self` and `claude`
at matched *achieved* ratio is the compressor-quality term, measured not assumed.

**Middle ground:** a second vLLM server running a small instruct model
(e.g. Qwen3-4B-Instruct, no reasoning) on a spare GPU. No rate limits, no egress,
better length-target adherence than a reasoning model, and it batches independently
so it doesn't compete for the KV cache of the model under test. Point
`COMPRESSOR_URL`/`COMPRESSOR_MODEL` at it; nothing else changes.

---

## 7. Experimental design

### Arms

| Arm | Role |
|---|---|
| `identity` | **Critical control.** Detokenize -> re-tokenize unchanged. Isolates token-boundary artifacts. If this arm moves accuracy, the measurement is contaminated. |
| `truncate` | Non-semantic floor. If LLM compression doesn't beat "keep first/last N tokens," there is no semantic story. |
| `self` | Primary arm (see §6). |
| `claude` | Ceiling, on a subset (~few hundred items). |

Sweep `ratio in {0.75, 0.5, 0.25, 0.1}` to get a curve, not a point.

### Non-negotiables

- **Share phase-1 traces across arms.** Run `identity` first to populate the trace
  cache, then every later arm reuses those exact traces. Makes the comparison paired,
  removes phase-1 sampling noise, and skips most of the GPU cost. This is the single
  most important design decision.
- **Fix `THINK_BUDGET` and `ANSWER_BUDGET` across all arms.** Otherwise the compressed
  arm gets a larger effective answer budget purely because its prompt is shorter.
- **Use a paired test** (McNemar on correct/incorrect flips), not independent-accuracy
  comparison. Much more powerful at n ~ 200-500.
- **Bin by `achieved_ratio`, not requested ratio.** Length-target adherence is poor;
  the requested ratio is fiction.
- **>= 3 seeds per arm.** Fixing `seed` and `temperature` does *not* make vLLM
  bit-reproducible — kernel reduction order varies with batch composition.

### Datasets

GSM8K is a reasonable smoke test but likely too easy to show degradation. Add AIME
or GPQA-Diamond where reasoning length actually matters. Lift scoring/answer
extraction from `tests/evals/gsm8k/gsm8k_eval.py`.

### Metrics per (sample, arm, seed)

`correct`, `reasoning_tokens`, `compressed_tokens`, `achieved_ratio`, `answer_tokens`,
`phase1_closed`, `retok_identical`, `answer_text`.

---

## 8. Logging schema

The sidecar is the **only** place both traces coexist — vLLM sees two unrelated
requests, the harness sees only the final answer. The JSONL record in §5 is the
entire experimental record. Key fields and why they exist:

- `phase1.token_ids` + `compress.token_ids` — text alone cannot reveal
  re-tokenization drift; `detokenize(tokenize(x))` can be byte-identical while the
  token sequence differs.
- `compress.retok_identical` — one-line identity control. Should be `True` for every
  record on the `identity` arm; anywhere it isn't is a token-boundary artifact.
- `prompt_key` — join key between the sidecar log and the harness's own results.
  The harness records by dataset index, the sidecar only sees `messages`.
- `compress.error` — a compression failure silently falling back to the uncompressed
  trace would otherwise inflate the compressed arm's accuracy.
- `compress_prompt_version` — traces from different compression prompts are not
  comparable, and you *will* iterate on that prompt.
- `phase1.closed` — traces that exhausted `THINK_BUDGET` without emitting `</think>`
  are a different population. Segment them out; do not mix silently.
- per-phase `ms` — needed the first time GPU utilization looks wrong.

**Operational:** gzip (`.jsonl.gz`, ~5x); one writer only (`--workers 1`, or put the
PID in the filename — interleaved appends from separate processes corrupt lines).

**Before the real sweep,** write a ten-line replay script: read a record, rebuild
`P + think_prefix + C + [think_end]` from the logged IDs, re-run phase 2 at the logged
seed, diff against `phase2.text`. If replay doesn't reproduce, something needed is
missing from the log — find that out on 20 samples, not after a three-day sweep.

---

## 9. Throughput notes (sidecar)

Continuous batching is never off — it is an engine property, and the sidecar's calls
are ordinary requests that vLLM batches alongside everything else, freely mixing
phase-1 and phase-2 requests from different harness calls. The sidecar can only
*starve* the batch:

- **Duty cycle (external-API arms only).** With `N` concurrent harness workers,
  in-flight sequences ~= `N * (t_p1 + t_p2) / (t_p1 + t_compress + t_p2)`.
  Fix by raising harness concurrency by `1/duty_cycle` and by caching compressions
  (key: `sha256(trace) + prompt_version + ratio`). Self-compression eliminates this
  entirely.
- **httpx pool.** Defaults (`max_connections=100`, `max_keepalive=20`) sit *below*
  vLLM's `max_num_seqs=128`, and one harness request maps to ~4-6 upstream calls.
  Raised to 512/512 in §5.
- **Blocking work in the event loop.** No per-request `open()`; use an async lock and
  a persistent handle. Use `orjson` — JSON-serializing 10k-element int arrays with
  stdlib `json` becomes the bottleneck at high concurrency.
- **Prefix-cache eviction during the gap.** `P`'s blocks sit unused for the whole
  compression call and can be LRU-evicted, costing a full re-prefill. Cost, not
  correctness. Compression caching fixes it; check hit rate on `/metrics` if phase-2
  prefill times look like full prompt reprocessing.
- **Priority scheduling.** `--scheduling-policy priority` + `priority: -1` on phase 2
  drains in-flight items instead of accumulating half-done ones holding KV.

Streaming is rejected with a 400, not faked: compression is inherently a barrier.
Most eval harnesses default to non-streaming; a 400 is a clearer failure than a hang.

---

## 10. Open decisions

- [ ] Model under test (drives the §4 probe and the reasoning-parser choice)
- [ ] Dataset(s) and n per arm
- [ ] Design A (offline three-pass) vs B (sidecar) — depends on whether the harness
      must stay unmodified
- [ ] Compressor for the primary arm: self, or a small instruct model on a 2nd server
- [ ] Ratio grid and seed count
- [ ] `THINK_BUDGET` / `ANSWER_BUDGET` values

## 11. Next steps

Steps 1-2 are done (see §0). Remaining:

1. ~~Prompt-shape probe~~ — done: `case_b=False`, `think_end=248069` (single token).
2. ~~Implement and smoke-test~~ — done: `examples/cot_compression/`.
3. Write and run the replay verifier (§8).
4. Swap in a harder dataset (AIME / GPQA-Diamond); the current problems are too
   easy to show any degradation.
5. Populate the phase-1 trace cache with the `identity` arm.
6. Sweep arms x ratios x seeds against the cache.
7. Analyze: McNemar paired tests, accuracy vs achieved-ratio curves, segmented by
   `phase1_closed`.

---

## Notes

- Per `AGENTS.md`: this is experimental research tooling, **not** a vLLM contribution.
  Nothing here should be proposed as a PR to `vllm-project/vllm`.
- Never use system `python3` or bare `pip` in this repo — use `uv` and `.venv/bin/python`.
- Reasoning traces sent to the Anthropic API leave the network. Check that against
  whatever dataset is used. The `self` arm avoids this entirely.

# Design: online (streaming) reasoning compression

**Status:** design for review. Nothing implemented, nothing run.
**Relationship to `cot_sidecar.py`:** new sibling module. `cot_sidecar.py` is not
modified — it is the reference implementation behind the published GPQA 90.37 /
HLE 29.19 numbers and should stay byte-identical to what produced them.

## Question

`cot_sidecar.py` compresses the reasoning trace *after* the model stops thinking.
That answered "is the trace load-bearing?" — on GPQA, no (90.37 vs a 90.50 identity
control while discarding 97.6% of the trace); on HLE, somewhat (-2.88).

It did **not** answer whether reasoning can *continue* from a compressed state.
Those are different capabilities: the first is a read, the second requires the
compressed state to be in-distribution as a prefix the model then writes from.

This design tests the second, by compressing every 4096 tokens and continuing.

Secondary consequence: peak context stops tracking total thinking length, so the
`1.9n + 240 <= 262144` constraint that pinned `THINK_BUDGET` at 196,608 no longer
applies and the arm can be budget-matched to the **no-sidecar baseline** for the
first time.

## Prior art this follows

| | carryover | training | we borrow |
|---|---|---|---|
| [InftyThink](https://arxiv.org/abs/2503.06692) (ICLR'26) | generated summary, replaces prior | SFT on relabeled OpenR1-Math | the reason-summarize-restart loop |
| [Delethink](https://arxiv.org/abs/2510.06557) (ICLR'26) | last m tokens verbatim | GRPO | self-authored state; the zero-shot evidence |
| [LightThinker](https://arxiv.org/abs/2502.15589) (EMNLP'25) | 9 gist-token KV slots | SFT + attention mask | the segmentation ablation; the numeric-loss failure mode |

All three train. The relevant evidence that an untrained model can do this is
Delethink §7.1 ("Delethink Tracing"): off-the-shelf **GPT-OSS 120B and Qwen3 30B
reach near-full performance zero-shot at 16K chunks**, and R1-Distill-1.5B recovers
most of LongCoT at 8K. Qwen3 30B is the closest published neighbour to
Qwen3.8-27B-FP8.

This design differs from all three in one way that matters: **summaries accumulate
rather than replace.** Each chunk is compressed exactly once and never re-compressed,
which removes the compounding-loss failure mode that recursive summarization dies of.
The cost is that state is O(N/8) rather than O(1) — still ~35K against a 262K window,
so it does not bind.

## Flow

```
head_0 = P + <think>
R_0    = gen(head_0, <=4096, stop=</think>)
S_0    = gen(head_0 + R_0 + SUMMARY_HINT, <=512, T=0.3)

head_1 = P + <think> + "Notes...[1] S_0" + CONTINUE_HINT
R_1    = gen(head_1, <=4096, stop=</think>)
S_1    = gen(head_1 + R_1 + SUMMARY_HINT, <=512, T=0.3)

head_2 = P + <think> + "Notes...[1] S_0 [2] S_1" + CONTINUE_HINT
...
```

`R_0 .. R_(n-1)` are discarded once summarized. `R_n` is kept **verbatim** — it holds
the conclusion, and compressing it is the one guaranteed-harmful compression available.

There are exactly two exits, and **neither manufactures an answer** (see
"Out-of-budget behaviour"):

- model emits `</think>` — answer generated from state + `R_n` verbatim
- budget exhausted — return what stock vLLM returns for an unclosed block

Peak context is `|P| + |state| + 4096`, independent of total thinking.

### Loop

```python
async def run(q_ids):
    state, gen, R = [], Counters(), None

    for i in range(MAX_CHUNKS):
        prefix = state_ids(state) + CONT_IDS if state else []
        head = q_ids + THINK_OPEN + prefix
        if len(head) + C_GEN > MAX_MODEL_LEN - CTX_SAFETY:
            break

        R = await generate(head, max_tokens=C_GEN,
                           stop_token_ids=[THINK_END_ID], temperature=1.0)
        gen.thinking += len(R.ids)

        if R.stop_reason == THINK_END_ID:              # done reasoning
            ans = await generate(head + R.ids + [THINK_END_ID],
                                 max_tokens=answer_room(gen), temperature=1.0)
            return finish(state, R.ids, ans, gen, closed=True)

        S = await generate(head + R.ids + SUMM_IDS,    # still inside <think>
                           max_tokens=SUMM_CAP, temperature=T_SUMM,
                           stop_token_ids=[THINK_END_ID])
        gen.summary += len(S.ids)
        state.append(sanitize(S.ids))

    # Budget exhausted, or the guard fired before any chunk ran (R is None).
    # Do NOT inject </think> and answer: stock vLLM returns an unclosed block
    # with empty content, and manufacturing an answer here would score points
    # the baseline did not get. See "Out-of-budget behaviour".
    return unclosed(state, R, gen)
```

Four deliberate choices:

1. **The summary is a continuation of the same sequence.** `head + R.ids + SUMM_IDS`
   shares its whole prefix with the call that just ran, so it is a prefix-cache hit
   when it lands on the same worker (see "Cost" for the DP caveat). More importantly
   it keeps the summary **self-authored, first-person, inside `<think>`** — the shape
   Delethink's RL discovers on its own, rather than the third-person summarization
   task a separate compression call produces.
2. **`SUMMARY_HINT` does not close `</think>`.** Closing it first would make the model
   write a user-facing response, which is a different distribution: reader-oriented
   summary prose instead of a note-to-self.
3. **No answer is generated on the budget-exhausted path.** `</think>` means done and
   one follow-up call finishes; running out of budget returns an unclosed block, as
   vLLM does.
4. **`stop_token_ids` rather than generate-then-truncate** — no paying for discarded
   tokens, and `stop_reason` distinguishes "done thinking" from "chunk full".

## Prompt surfaces

These are the design. Everything else is plumbing.

```python
CONTINUE_HINT = "\n\nPicking up where I left off:\n"

SUMMARY_HINT = (
    "\n\nI'm out of space, so I'll stop mid-thought here and note what I need "
    "to resume: exact values I've established, what I was partway through, "
    "what I've ruled out, and what's still open. Only what's new since my "
    "last note.\n\n"
)

def render_state(blocks):
    body = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(blocks))
    return f"Notes from my earlier work on this problem:\n\n{body}\n"
```

Each clause of `SUMMARY_HINT` targets a specific measured failure:

| clause | failure it targets |
|---|---|
| "stop mid-thought here" | model summarizing as if a truncated line of reasoning had concluded, then never returning to it |
| "exact values I've established" | LightThinker's documented numeric-loss bug — *"not sufficiently sensitive to numerical values"*, and *"such errors occur frequently"* |
| "what I was partway through" | losing the in-flight step at the cut |
| "what I've ruled out" | re-exploration, which silently eats the savings |
| "what's still open" | the resumption target |
| "only what's new since my last note" | state bloat — the model can see its prior notes and will otherwise restate them |

Halved from the first draft (~90 words to ~44) per review. It is re-tokenized into
every summary call, and a long instruction at the seam both costs attention and biases
the model toward list-shaped output rather than a natural note.

The output is deliberately **free-form prose**, not a schema. Named fields would make
the items separately inspectable, but JSON or headed sections inside a reasoning trace
are off-distribution for an untrained model, and format errors become a new failure
mode. Naming the targets in the *instruction* while leaving the *output* prose keeps
most of the benefit.

## Parameters

Budget is matched to the **no-sidecar baseline**, which used `MAX_NEW = 245760`
covering reasoning and answer in one stream:

```
245,760  R blocks = 60 x 4096   (exact)   <- reasoning may use the whole pool
```

`max_tokens` in the baseline is a **single pool** shared by reasoning and answer, so
the arm mirrors that rather than pre-splitting it:

```python
def answer_room(gen):        # what baseline would have left for the answer
    return min(ANSWER_BUDGET, 245_760 - gen.thinking)
```

Reasoning can therefore consume the entire 245,760 — exactly as baseline can — and
when it does, there is nothing left for an answer and nothing is generated.

| knob | value | rationale |
|---|---|---|
| `C_GEN` | 4096 | small chunks are safe because accumulation means no compounding; also keeps per-step compression at ~14x, where the compressor's habitual ~300-token output is a natural length |
| `THINK_TOTAL` | 245,760 | R blocks only; == baseline `MAX_NEW` |
| `MAX_CHUNKS` | 60 | exact |
| `SUMM_CAP` | 512 | compressor writes ~300 regardless |
| `T_SUMM` | 0.3 | state serialization wants fidelity, not diversity; reasoning stays at 1.0. **Unmeasured — my call, flagged for review.** |
| `ANSWER_BUDGET` | 8192 | ceiling only; actual room is `245760 - thinking` |
| `MAX_MODEL_LEN` / `CTX_SAFETY` | 262144 / 256 | reused |
| `request_timeout` (NEL) | **7200** | must change, see Cost |

**Summaries sit outside the budget**, consistent with `cot_sidecar.py` where
`THINK_BUDGET` covered GEN 1 and the compressor's ~300 tokens were separate. Counting
them would handicap the arm against the baseline it is compared to. They still appear
in `completion_tokens` for verbosity, tracked on a separate line.

Worst-case state: 60 x 512 = 30,720. Peak context ~36K against a 262K window.

## Cost

Worst-case decode per request: 245,760 + ~18,000 summaries ~= **264K tokens**,
versus the offline arm's ~205K. At the ~57 tok/s the offline run implies, that is
~4,600s against the current 3600s `request_timeout` — hence 7200.

This is inherent, not an artifact: matching baseline means generating baseline's token
count plus the method's overhead, inside one HTTP request.

**The prefix-cache claim needs a caveat.** `head + R.ids + SUMM_IDS` is a cache hit only
if it lands on the worker holding `head + R.ids`. Under `DP=8` there is no routing
affinity, so a summary call can land elsewhere and re-prefill 4096 tokens. Correct
either way; free only when it lands right.

**Where the benefit is.** Weights are read once per batch step and amortized; KV is read
per sequence, so KV dominates at batch. Estimating ~64 KB/token (16 full-attention
layers, ~1024 KV dims, bf16):

| | KV / sequence | seqs in ~150 GB |
|---|---|---|
| baseline @ 262K ctx | ~16.8 GB | ~8 |
| chunked @ 35K ctx | ~2.2 GB | ~68 |

At full depth the baseline cannot fill `max_num_seqs: 16` — it is KV-capacity-limited
to roughly half of it. Note this is an estimate from assumed head dims, not read off
the config.

Applied to the observed distributions:

| | median chunks | benefit |
|---|---|---|
| GPQA (p50 7,136) | 2 | ~none; overhead slightly negative |
| HLE (p50 32,754) | 8 | modest |
| HLE tail (p95 116K, max 191K) | 29-47 | large |

Overhead is roughly constant per chunk; benefit grows with depth. The deep requests
both benefit most and set total run time.

**This is not primarily a perf play** and should not be sold as one. The reasons to run
it are the accuracy question and the budget-matching. Throughput on the long tail is a
bonus to report, not a claim to defend.

## Out-of-budget behaviour

**Requirement (review):** when reasoning runs out of budget, return exactly what the
stock vLLM server returns. This is not cosmetic — it is a validity requirement.

`cot_sidecar.py` already records the measured baseline behaviour:

> *"vLLM ... puts an unclosed block in reasoning_content and leaves content empty.
> Measured on the uncompressed xhigh baseline: every 245760-token non-terminating
> generation came back with `generation=""` and `reasoning_content=`the whole trace,
> so the grader saw an empty answer and simply marked it wrong."*

So a baseline request that never closes `</think>` is **scored wrong**. If this arm
instead injected `</think>` and generated an answer for those same requests, it would
collect points the baseline never had a chance at — a confound in the arm's favour,
concentrated on exactly the hardest problems.

The first draft of this design did that. It is removed.

```python
def unclosed(state, R, gen):
    reasoning = state_text(state) + (detok(R.ids) if R else "")
    return {"content": "",                  # grader sees an empty answer
            "reasoning_content": reasoning, # unclosed block, as vLLM does
            "finish_reason": "length",
            "completion_tokens": gen.thinking + gen.summary}
```

Three properties, each matching vLLM or the offline arm:

| property | value | why |
|---|---|---|
| `content` | `""` | vLLM leaves it empty; the grader marks it wrong, as for baseline |
| `reasoning_content` | state + `R_n` | vLLM's reasoning parser puts the unclosed block here |
| `finish_reason` | `"length"` | matches vLLM |
| `completion_tokens` | tokens actually generated | vLLM reports real generation; `cot_sidecar.py` keeps `answer_ids = gen` for the same reason |

Reusing `cot_sidecar.py`'s `no_think_block` return shape keeps both compressed arms
byte-comparable at this boundary.

`RC_MAX_CHARS` stays at 0 (unbounded) as in the offline arm — the baseline judged 905
rows carrying >128K chars of `reasoning_content` without incident, so capping it is
unnecessary and would diverge from what was measured.

This also subsumes edge case **G2**: when the context guard fires before any chunk
runs, `R` is `None`, `reasoning` is just the (empty) state, and the same return fires.
No special case, no `None` deref.

## Arms

**Compress arm only**, by decision. Consequence to record: a delta of any size is
ambiguous between compression loss and the cost of the interrupt/re-prompt machinery.

Two controls are one flag each if that ambiguity matters later:

- **lossless-chunked** — same loop, pass full prior chunks instead of summaries.
  Isolates the machinery from the compression. The direct analogue of the identity
  control that made the offline +0.72 defensible.
- **truncate / Delethink Tracing** — `state = R[-m:]`. The arm with published zero-shot
  evidence. If it matches compress, GEN 2 is doing no work.

## Metrics

`think_closed` is **primary, not diagnostic**. Re-seeding with a compact state removes
the growing-context cue that tells a model it has been at this a while; if the
`</think>` rate collapses versus baseline, that is the finding.

Also: `n_chunks`, `thinking_tokens`, `summary_tokens`, `answer_tokens`, `state_tokens`,
`peak_context`, `summary_truncated` (hit `SUMM_CAP`), `summary_empty`,
`summary_fallbacks`.

Report `thinking_tokens` alongside `completion_tokens` — the latter now includes
summaries, so "verbosity" no longer means what it meant in the offline study.

## Edge cases

Enumerated systematically. Honest status: this list found eight gaps in the first
draft of the loop; the ones marked **must** are in scope for v1. One (`R is None`) was
since resolved by the out-of-budget rework, leaving four.

### Control flow

| case | handling |
|---|---|
| `</think>` in chunk 0 | happy path; `head_0` is byte-identical to a baseline generation, so short problems run exactly as baseline |
| `</think>` never emitted across 60 chunks | `unclosed()`; `think_closed=False`; content empty, as vLLM |
| `</think>` mid-summary | `stop_token_ids`; keep what preceded |
| summary hits `SUMM_CAP` | used as-is, counted via `summary_truncated` |
| **EOS instead of `</think>`** | **must** — branch on `stop_reason == THINK_END_ID`, not `finish_reason == "stop"`. `finish_reason` is `"stop"` for both, and treating EOS as a `</think>` would inject one and ask for an answer against a sequence the model considers finished |

### Length

| case | handling |
|---|---|
| state growth (58 x 512 + P + 4096 ~= 35K) | guard; never fires at these sizes |
| large `P`, guard fires at `i=0` | **resolved by the design change** — `unclosed()` handles `R is None`; no separate path |
| answer prompt exceeding the window | `answer_room()` clamps to the remaining pool; also clamp to window room, as `compress_llm_ids` already does for the compressor cap |

### Token and state integrity

| case | handling |
|---|---|
| `<think>` double-injection | **resolved** — `prompt_opens_think()` already exists in `cot_sidecar.py`; `THINK_OPEN = []` when the template already opens it |
| stop token included in `output.ids` | verify once against a live server; if included, the explicit `+ [THINK_END_ID]` would double it |
| **empty summary** | **must** — if `</think>` is the first summary token, `S.ids` is empty and we append `""`, rendering a bare `[3]` into the notes. Skip empty; on repeat fall back to `R[-m:]` |
| **reserved tokens inside a summary** | **must** — nothing stops the model writing `<think>` inside `S`. Re-feeding creates a nested open block. Sanitize via the existing `_strip_trailing_specials` / `strip_specials` helpers |
| **detok -> retok round trip on state** | **must** — keep `S` as token IDs and splice between pre-tokenized scaffolding. The round trip can change token counts and, on partial-UTF-8 boundaries, content |

### Degradation

Bounded retry per sub-call; on permanent summary failure fall back to `R[-m:]` rather
than failing the request. That is Delethink Tracing as a degradation path — the loop
survives and `summary_fallbacks` records how often it fired.

### Content-level, no clean fix

Not bugs; these are what the experiment measures. Visible only in the metrics.

- model re-explores despite the ruled-out clause — shows up as elevated `n_chunks`
- summary says "nothing new", accumulating dead blocks
- model answers inside a summary, wasting remaining chunks
- degenerate repetition — pre-existing risk; chunking may amplify or damp it, unknown

## Open questions for review

1. **`T_SUMM = 0.3`** is unmeasured. Reasoning runs at 1.0. Fidelity over diversity is
   the argument, but it interacts with the compressor's fixed ~300-token habit.
2. **`CONTINUE_HINT` does a lot of work for four words.** It is the seam where the model
   either picks up smoothly or notices something is off, and the cheapest thing to ablate.
3. **No progress signal in the state.** Left out to keep notes clean, but that is exactly
   the cue whose removal threatens `think_closed`. Adding `[segment 12, ~180K left]` to
   `render_state` is one line — better decided now than diagnosed after a run.
4. **Hard cut at 4096, no `\n\n` boundary.** LightThinker measured -6.2% for fixed-count
   cutting, but at ~56-token segments where a mid-sentence cut destroys a large fraction
   of the segment; at 4096 it is proportionally far smaller, and `SUMMARY_HINT` now warns
   the model explicitly. Cheapest thing to add back if summaries look like they are losing
   the in-flight step.

## Suggested run ladder

1. **Single checkpoint** — cut once at 50%, continue, answer. GPQA only. Isolates
   "can reasoning continue from a compressed state" with one variable. If this fails,
   58 checkpoints certainly fail.
2. Full loop, GPQA.
3. HLE, where the trace is load-bearing and the answer actually matters.

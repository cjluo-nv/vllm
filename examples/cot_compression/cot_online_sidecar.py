# SPDX-License-Identifier: Apache-2.0
"""Online (streaming) reasoning compression sidecar.

Sibling of ``cot_sidecar.py``, which compresses the reasoning trace *after* the
model stops thinking. This one compresses *during* generation: reason C_GEN
tokens, write a short note, drop the reasoning, continue from the notes.

See DESIGN-online-compression.md for the rationale. The three properties that
matter and are easy to break:

1. The note is generated as a *continuation of the same sequence*, inside the
   open ``<think>`` block. Not a separate compression call. That keeps it
   self-authored and first-person, and makes it a prefix-cache hit.
2. ``SUMMARY_HINT`` must not close ``</think>``. Closing it first would make the
   model write a user-facing response instead of a note to itself.
3. Running out of budget returns what stock vLLM returns for an unclosed block:
   ``content=""``, the trace in ``reasoning_content``, ``finish_reason=length``.
   It does NOT inject ``</think>`` and manufacture an answer -- the baseline
   scores those requests wrong, so answering them would collect points the
   baseline never had a chance at.

Notes accumulate rather than replace, so each chunk is compressed exactly once
and recursive-summarisation compounding loss does not arise.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

# ------------------------------- knobs -------------------------------

VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL", "")

C_GEN = int(os.environ.get("C_GEN", "4096"))
SUMM_CAP = int(os.environ.get("SUMM_CAP", "512"))
T_SUMM = float(os.environ.get("T_SUMM", "0.3"))
# R blocks only; summaries are method overhead and sit outside it, as
# THINK_BUDGET did in cot_sidecar.py. Default == baseline MAX_NEW.
THINK_TOTAL = int(os.environ.get("THINK_TOTAL", "245760"))
ANSWER_BUDGET = int(os.environ.get("ANSWER_BUDGET", "8192"))
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "60"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "262144"))
CTX_SAFETY = int(os.environ.get("CTX_SAFETY", "256"))
# Delethink-style fallback used only when a summary cannot be produced.
FALLBACK_TAIL = int(os.environ.get("FALLBACK_TAIL", "512"))
RC_MAX_CHARS = int(os.environ.get("RC_MAX_CHARS", "0"))

TRACE_LOG = os.environ.get("TRACE_LOG", "online_traces.jsonl")
LOG_QUEUE_MAX = int(os.environ.get("LOG_QUEUE_MAX", "20000"))
LOG_BATCH = int(os.environ.get("LOG_BATCH", "64"))
LOG_FLUSH_SECS = float(os.environ.get("LOG_FLUSH_SECS", "2.0"))
RENDER_CACHE_MAX = int(os.environ.get("RENDER_CACHE_MAX", "4096"))

JSON_HDR = {"content-type": "application/json"}
_SAMPLING_KEYS = ("temperature", "top_p", "seed", "presence_penalty",
                  "frequency_penalty", "repetition_penalty", "min_p", "top_k")

CONTINUE_HINT = "\n\nPicking up where I left off:\n"

# Halved from the first draft per review. Every clause targets a measured
# failure; see the clause table in DESIGN-online-compression.md.
SUMMARY_HINT = (
    "\n\nI'm out of space, so I'll stop mid-thought here and note what I need "
    "to resume: exact values I've established, what I was partway through, "
    "what I've ruled out, and what's still open. Only what's new since my "
    "last note.\n\n"
)
NOTES_HEAD = "Notes from my earlier work on this problem:\n\n"

S: dict = {"render_cache": {}, "http_calls": 0, "requests": 0}
up = httpx.AsyncClient(base_url=VLLM, timeout=httpx.Timeout(None))


# ------------------------------ plumbing ------------------------------

async def post(path: str, payload: dict) -> dict:
    S["http_calls"] += 1
    r = await up.post(path, content=json.dumps(payload).encode(),
                      headers=JSON_HDR)
    if r.status_code >= 400:
        raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:400]}")
    return r.json()


async def tok(text: str) -> list[int]:
    if not text:
        return []
    return (await post("/tokenize", {"model": MODEL, "prompt": text,
                                     "add_special_tokens": False}))["tokens"]


async def detok(ids: list[int]) -> str:
    if not ids:
        return ""
    return (await post("/detokenize", {"model": MODEL,
                                       "tokens": list(ids)}))["prompt"]


async def generate(ids: list[int], *, max_tokens: int, sp: dict,
                   stop_think: bool) -> dict:
    params = {**sp, "max_tokens": max_tokens}
    if stop_think:
        params["stop_token_ids"] = [S["think_end"]]
    out = await post("/inference/v1/generate",
                     {"token_ids": list(ids), "sampling_params": params})
    return out["choices"][0]


def sampling_common(body: dict) -> dict:
    """Forward only what the caller set, so vLLM resolves its own defaults."""
    return {k: body[k] for k in _SAMPLING_KEYS if body.get(k) is not None}


def render_key(body: dict) -> str:
    return hashlib.sha256(json.dumps({
        "messages": body.get("messages"), "ctk": body.get("chat_template_kwargs"),
        "tools": body.get("tools"), "tpl": body.get("chat_template"),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prompt_key(messages) -> str:
    canon = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def prompt_opens_think(p_ids: list[int]) -> bool:
    """True if the prompt ends inside an unclosed <think> block."""
    ts, te = S["think_start"], S["think_end"]
    last_open = last_close = -1
    for i, t in enumerate(p_ids):
        if t == ts:
            last_open = i
        elif t == te:
            last_close = i
    return last_open > last_close


def strip_specials(text: str) -> str:
    for t in S.get("special_strs", ()):
        text = text.replace(t, "")
    return text


def sanitize_ids(ids: list[int]) -> list[int]:
    """Drop reserved scaffolding tokens from a generated note.

    Nothing stops the model writing <think> or </think> inside its own note;
    splicing that back would open or close a block we do not control.
    """
    banned = S["banned_ids"]
    return [t for t in ids if t not in banned]


def find_think_end(ids: list[int]) -> int | None:
    """Index of </think>, or None.

    Membership rather than a last-token check, so it is correct whether or not
    vLLM includes the stop token in the output.
    """
    te = S["think_end"]
    for i, t in enumerate(ids):
        if t == te:
            return i
    return None


# --------------------------- state rendering ---------------------------

async def label_ids(i: int) -> list[int]:
    cache = S["label_cache"]
    if i not in cache:
        cache[i] = await tok(f"[{i + 1}] ")
    return cache[i]


async def state_ids(blocks: list[list[int]]) -> list[int]:
    """Notes block + continue hint, spliced from pre-tokenized scaffolding.

    Blocks stay as token IDs end to end. Detokenising each note and
    re-tokenising the joined text would change token counts and, on partial
    UTF-8 boundaries, content.
    """
    if not blocks:
        return []
    out = list(S["notes_head_ids"])
    for i, b in enumerate(blocks):
        if i:
            out += S["sep_ids"]
        out += await label_ids(i)
        out += b
    return out + S["cont_ids"]


async def state_text(blocks: list[list[int]]) -> str:
    if not blocks:
        return ""
    parts = [f"[{i + 1}] {await detok(b)}" for i, b in enumerate(blocks)]
    return NOTES_HEAD + "\n\n".join(parts)


# ------------------------------- the loop -------------------------------

class Counters:
    __slots__ = ("thinking", "summary", "answer", "chunks", "peak_ctx",
                 "summary_empty", "summary_truncated", "summary_fallbacks")

    def __init__(self):
        self.thinking = self.summary = self.answer = self.chunks = 0
        self.peak_ctx = 0
        self.summary_empty = self.summary_truncated = self.summary_fallbacks = 0

    def as_dict(self, state_tokens: int) -> dict:
        return {"n_chunks": self.chunks, "thinking_tokens": self.thinking,
                "summary_tokens": self.summary, "answer_tokens": self.answer,
                "state_tokens": state_tokens, "peak_context": self.peak_ctx,
                "summary_empty": self.summary_empty,
                "summary_truncated": self.summary_truncated,
                "summary_fallbacks": self.summary_fallbacks}


def answer_room(gen: Counters) -> int:
    """What the baseline's single max_tokens pool would have left.

    Baseline shares one budget between reasoning and answer, so the arm mirrors
    that instead of pre-splitting it.
    """
    return max(0, min(ANSWER_BUDGET, THINK_TOTAL - gen.thinking))


async def run_online(p_ids: list[int], *, sp: dict, think_open: list[int]):
    """Returns (reasoning_content, content, finish_reason, gen, closed)."""
    state: list[list[int]] = []
    gen = Counters()
    R: list[int] | None = None
    term_fr = "length"          # why reasoning stopped, when it never closed

    for _ in range(MAX_CHUNKS):
        head = p_ids + think_open + await state_ids(state)
        room = MAX_MODEL_LEN - CTX_SAFETY - len(head)
        if room < 256:
            break                       # cannot fit a useful chunk
        budget = min(C_GEN, THINK_TOTAL - gen.thinking, room)
        if budget <= 0:
            break
        gen.peak_ctx = max(gen.peak_ctx, len(head) + budget)

        c = await generate(head, max_tokens=budget, sp=sp, stop_think=True)
        R = c["token_ids"]
        te_at = find_think_end(R)

        if te_at is not None:                       # model finished reasoning
            R = R[:te_at]
            gen.thinking += len(R)
            gen.chunks += 1
            a = await generate(head + R + S["think_end_ids"],
                               max_tokens=answer_room(gen), sp=sp,
                               stop_think=False)
            gen.answer = len(a["token_ids"])
            answer = strip_specials(await detok(a["token_ids"])).strip()
            reasoning = await state_text(state) + await detok(R)
            return reasoning, answer, a["finish_reason"], gen, True

        gen.thinking += len(R)
        gen.chunks += 1

        if c["finish_reason"] != "length":
            # EOS (or a caller stop) rather than </think>. finish_reason is
            # "stop" for both, so branching on it alone would mistake this for
            # a finished think block and answer against a dead sequence.
            # Carry the real reason out: vLLM reports "stop" for an EOS that
            # lands inside an unclosed block, not "length".
            term_fr = c["finish_reason"]
            break

        if gen.thinking >= THINK_TOTAL:
            break

        s = await generate(head + R + S["summ_ids"], max_tokens=SUMM_CAP,
                           sp={**sp, "temperature": T_SUMM}, stop_think=True)
        note = sanitize_ids(s["token_ids"])
        te_at = find_think_end(s["token_ids"])
        if te_at is not None:
            note = sanitize_ids(s["token_ids"][:te_at])
        if s["finish_reason"] == "length":
            gen.summary_truncated += 1

        if not note:
            # The model closed the block or produced nothing. Fall back to
            # Delethink-style truncation so the loop survives.
            gen.summary_empty += 1
            note = R[-FALLBACK_TAIL:]
            gen.summary_fallbacks += 1
        gen.summary += len(note)
        state.append(note)

    # Out of budget, EOS mid-reasoning, or the guard fired before any chunk
    # (R is None). vLLM returns the unclosed block with empty content; the
    # grader marks it wrong, exactly as it does for the baseline.
    reasoning = await state_text(state) + (await detok(R) if R else "")
    return reasoning, "", term_fr, gen, False


# ------------------------------ logging ------------------------------

async def write_record(rec: dict):
    q = S.get("logq")
    if q is None:
        return
    try:
        q.put_nowait(rec)
    except asyncio.QueueFull:
        S["log_dropped"] = S.get("log_dropped", 0) + 1


async def log_writer():
    q, buf = S["logq"], []
    with open(TRACE_LOG, "a") as fh:
        while True:
            try:
                buf.append(await asyncio.wait_for(q.get(), LOG_FLUSH_SECS))
            except asyncio.TimeoutError:
                pass
            if buf and (len(buf) >= LOG_BATCH or q.empty()):
                fh.write("".join(json.dumps(r) + "\n" for r in buf))
                fh.flush()
                buf.clear()


# ------------------------------ lifespan ------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    for _ in range(120):
        try:
            if (await up.get("/health")).status_code == 200:
                break
        except Exception:
            pass
        await asyncio.sleep(5)

    if not MODEL:
        globals()["MODEL"] = (await up.get("/v1/models")).json()["data"][0]["id"]

    te_ids = await tok("</think>")
    S["think_end_ids"] = te_ids
    S["think_end"] = te_ids[-1]
    S["think_start"] = (await tok("<think>"))[-1]
    S["banned_ids"] = {S["think_start"], S["think_end"]}
    S["special_strs"] = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")

    S["summ_ids"] = await tok(SUMMARY_HINT)
    S["cont_ids"] = await tok(CONTINUE_HINT)
    S["notes_head_ids"] = await tok(NOTES_HEAD)
    S["sep_ids"] = await tok("\n\n")
    S["label_cache"] = {}
    S["think_open_ids"] = await tok("<think>")

    S["logq"] = asyncio.Queue(maxsize=LOG_QUEUE_MAX)
    S["writer"] = asyncio.create_task(log_writer())
    yield
    S["writer"].cancel()


app = FastAPI(lifespan=lifespan)


# ------------------------------ chat route ------------------------------

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    for k in ("logprobs", "top_logprobs"):
        if body.get(k):
            return JSONResponse(
                {"error": {"message": f"{k} unsupported by the online sidecar"}},
                400)
    if body.get("stream"):
        return JSONResponse(
            {"error": {"message": "stream unsupported by the online sidecar"}},
            400)

    t0 = time.perf_counter()
    sp = sampling_common(body)
    pkey = prompt_key(body["messages"])

    rk = render_key(body)
    p_ids = S["render_cache"].get(rk)
    if p_ids is None:
        p_ids = (await post("/v1/chat/completions/render",
                            {**body, "stream": False,
                             "max_tokens": 1}))["token_ids"]
        if len(S["render_cache"]) < RENDER_CACHE_MAX:
            S["render_cache"][rk] = p_ids

    opens = prompt_opens_think(p_ids)
    if not opens:
        # Thinking disabled (or a plain prompt): nothing to compress, so behave
        # exactly like vLLM rather than forcing a think block open.
        r = await post("/v1/chat/completions", {**body, "stream": False})
        return r

    think_open: list[int] = []          # the template already opened it
    reasoning, answer, fr, gen, closed = await run_online(
        p_ids, sp=sp, think_open=think_open)

    S["requests"] += 1
    rid = f"chatcmpl-{uuid.uuid4().hex}"
    state_tokens = max(0, gen.summary)
    meta = {**gen.as_dict(state_tokens), "think_closed": closed,
            "ms": round((time.perf_counter() - t0) * 1000)}
    await write_record({"id": rid, "prompt_key": pkey, "n_prompt": len(p_ids),
                        "sampling": {**sp, "c_gen": C_GEN,
                                     "think_total": THINK_TOTAL,
                                     "summ_cap": SUMM_CAP, "t_summ": T_SUMM},
                        "finish_reason": fr, "online": meta,
                        "reasoning_content": reasoning, "content": answer})

    # completion_tokens counts every token actually generated -- reasoning,
    # notes and answer -- as vLLM reports real generation. thinking_tokens is
    # carried separately in cot_online so verbosity stays comparable.
    total_out = gen.thinking + gen.summary + gen.answer
    return {
        "id": rid, "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model", MODEL),
        "choices": [{"index": 0, "finish_reason": fr,
                     "message": {"role": "assistant", "content": answer,
                                 "reasoning_content":
                                     reasoning[:RC_MAX_CHARS]
                                     if RC_MAX_CHARS else reasoning}}],
        "usage": {"prompt_tokens": len(p_ids), "completion_tokens": total_out,
                  "total_tokens": len(p_ids) + total_out},
        "cot_online": meta,
    }


@app.get("/sidecar/stats")
async def stats():
    return {"requests": S["requests"], "http_calls": S["http_calls"],
            "log_dropped": S.get("log_dropped", 0),
            "render_cache": len(S["render_cache"])}


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, req: Request):
    body = await req.body()
    if req.method == "GET":
        r = await up.get("/" + path)
    else:
        r = await up.post("/" + path, content=body,
                          headers={"content-type": req.headers.get(
                              "content-type", "application/json")})
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",
                port=int(os.environ.get("SIDECAR_PORT", "9000")),
                log_level="warning")

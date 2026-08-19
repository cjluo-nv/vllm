#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CoT-compression sidecar: an OpenAI-compatible shim that sits in front of vLLM.

One inbound /v1/chat/completions request is split into:

  phase 1   generate reasoning, stopping at ``</think>``
  compress  condense the trace (self-compression through the same vLLM server,
            with the compressor's own thinking disabled)
  phase 2   resume generation with the compressed trace spliced back in

Everything else is proxied straight through to vLLM, so an eval harness can
point at this process unmodified.
"""

import asyncio
import glob
import gzip
import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
COMPRESSOR_URL = os.environ.get("COMPRESSOR_URL", VLLM)
MODEL = os.environ.get("MODEL", "")
COMPRESSOR_MODEL = os.environ.get("COMPRESSOR_MODEL", "") or MODEL
ARM = os.environ.get("ARM", "self")  # identity | truncate | self
RATIO = float(os.environ.get("RATIO", "0.3"))
THINK_BUDGET = int(os.environ.get("THINK_BUDGET", "4096"))
ANSWER_BUDGET = int(os.environ.get("ANSWER_BUDGET", "1024"))
TRACE_LOG = os.environ.get("TRACE_LOG", "traces.jsonl")
TRACE_CACHE_GLOB = os.environ.get("TRACE_CACHE_GLOB", "")
USE_PRIORITY = os.environ.get("USE_PRIORITY", "0") == "1"
COMPRESS_PROMPT_VERSION = "v1"

JSON_HDR = {"content-type": "application/json"}
LIMITS = httpx.Limits(max_connections=512, max_keepalive_connections=512)

up = httpx.AsyncClient(base_url=VLLM, timeout=3600.0, limits=LIMITS)
cup = httpx.AsyncClient(base_url=COMPRESSOR_URL, timeout=3600.0, limits=LIMITS)
LOCK = asyncio.Lock()
S: dict = {}

# NOTE: an earlier version of this prompt said "preserve the original voice and
# formatting style". Combined with greedy decoding that made verbatim copying the
# easiest continuation, and the compressor reproduced the trace unchanged until it
# hit max_tokens. Brevity has to be the dominant instruction.
COMPRESS_SYS = (
    "You compress reasoning traces. You are given a model's intermediate "
    "reasoning. Rewrite it so it is SUBSTANTIALLY SHORTER than the input while "
    "preserving everything needed to finish the problem: intermediate results, "
    "constraints, and the decisions that follow from them.\n"
    "Rules:\n"
    "- Never copy the trace verbatim. If your output is not much shorter than "
    "the input, you have failed the task.\n"
    "- Delete false starts, dead ends, restatements of the question, "
    "self-checks, and repeated calculations.\n"
    "- Keep every numeric result and the chain that produces it.\n"
    "- Use terse notation. Full sentences are not required.\n"
    "- Output only the compressed trace: no preamble, commentary, or headings."
)


async def _post(client: httpx.AsyncClient, path: str, payload: dict) -> dict:
    r = await client.post(path, content=json.dumps(payload).encode(), headers=JSON_HDR)
    if r.status_code >= 400:
        raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:400]}")
    return r.json()


async def post(path: str, payload: dict) -> dict:
    return await _post(up, path, payload)


async def cpost(path: str, payload: dict) -> dict:
    return await _post(cup, path, payload)


def prompt_key(messages) -> str:
    canon = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def strip_specials(text: str) -> str:
    for tok in S.get("special_strs", ()):
        text = text.replace(tok, "")
    return text


def load_trace_cache(pattern: str) -> dict:
    cache = {}
    for path in glob.glob(pattern):
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        key = (r["prompt_key"], r["sampling"]["seed"])
                        cache[key] = (r["phase1"]["token_ids"], r["phase1"]["closed"])
                    except Exception:
                        continue
        except OSError:
            continue
    return cache


async def tok(text: str) -> list[int]:
    return (await post("/tokenize", {"model": MODEL, "prompt": text,
                                     "add_special_tokens": False}))["tokens"]


async def detok(ids: list[int]) -> str:
    if not ids:
        return ""
    return (await post("/detokenize", {"model": MODEL, "tokens": ids}))["prompt"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for _ in range(120):  # wait for vLLM to come up
        try:
            r = await up.get("/health")
            if r.status_code == 200:
                break
        except Exception:
            pass
        await asyncio.sleep(5)

    if not MODEL:
        models = (await up.get("/v1/models")).json()
        S["model"] = models["data"][0]["id"]
        globals()["MODEL"] = S["model"]
        if not COMPRESSOR_MODEL:
            globals()["COMPRESSOR_MODEL"] = S["model"]

    te_ids = await tok("</think>")
    S["think_end_ids"] = te_ids
    S["think_end"] = te_ids[-1]
    S["te_single"] = len(te_ids) == 1
    S["think_start"] = (await tok("<think>"))[-1]
    S["special_strs"] = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")

    # ---- prompt-shape probe: does the generation prompt open <think>? ----
    probe = await post("/v1/chat/completions/render",
                       {"model": MODEL, "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}]})
    tail = await detok(probe["token_ids"][-24:])
    S["case_b"] = "<think>" not in tail
    S["think_prefix"] = await tok("<think>\n") if S["case_b"] else []

    # ---- compressor-side probe: is its think block already closed? ----
    cprobe = await cpost("/v1/chat/completions/render",
                         {"model": COMPRESSOR_MODEL, "max_tokens": 1,
                          "messages": [{"role": "user", "content": "hi"}],
                          "chat_template_kwargs": {"enable_thinking": False}})
    ctail = await detok(cprobe["token_ids"][-24:])
    S["nothink"] = ([] if ("<think>" not in ctail or "</think>" in ctail)
                    else [S["think_end"]])

    S["cache"] = load_trace_cache(TRACE_CACHE_GLOB) if TRACE_CACHE_GLOB else {}
    S["log"] = (gzip.open(TRACE_LOG, "at") if TRACE_LOG.endswith(".gz")
                else open(TRACE_LOG, "a"))

    print(f"[sidecar] model={MODEL}", flush=True)
    print(f"[sidecar] think_start={S['think_start']} think_end={S['think_end']} "
          f"te_ids={te_ids} single={S['te_single']}", flush=True)
    if not S["te_single"]:
        print("[sidecar] WARNING: </think> is multi-token; falling back to a "
              "string stop and re-tokenizing the trace body.", flush=True)
    print(f"[sidecar] main prompt tail={tail!r} case_b={S['case_b']}", flush=True)
    print(f"[sidecar] compressor tail={ctail!r} nothink={S['nothink']}", flush=True)
    print(f"[sidecar] arm={ARM} ratio={RATIO} cache={len(S['cache'])}", flush=True)
    try:
        yield
    finally:
        S["log"].close()
        await up.aclose()
        await cup.aclose()


app = FastAPI(lifespan=lifespan)


# --------------------------- compression arms ---------------------------

async def truncate_head_tail(ids: list[int], target: int) -> str:
    if len(ids) <= target:
        return await detok(ids)
    head = target // 2
    keep = ids[:head] + ids[len(ids) - (target - head):]
    return await detok(keep)


async def compress_llm(raw: str, target: int, meta: dict, n_input: int) -> str:
    """Self-compression: same vLLM server, compressor's own thinking disabled."""
    msgs = [
        {"role": "system", "content": COMPRESS_SYS},
        {"role": "user",
         "content": (
             f"Compress the trace below to at most {target} tokens "
             f"(roughly {max(8, int(target * 0.75))} words). The input is "
             f"{n_input} tokens, so your output must be far shorter.\n\n"
             f"<trace>\n{raw}\n</trace>\n\n"
             f"Remember: at most {target} tokens, and never copy verbatim.")},
    ]
    rendered = await cpost("/v1/chat/completions/render",
                           {"model": COMPRESSOR_MODEL, "messages": msgs,
                            "max_tokens": 1,
                            "chat_template_kwargs": {"enable_thinking": False}})
    pc = rendered["token_ids"] + S["nothink"]
    # Headroom matters: if the compressor is cut off at max_tokens the result is
    # a truncated compression, which silently turns this arm into the truncate
    # arm. Give it room, then measure what it actually produced.
    cap = max(128, int(target * 3))
    out = await cpost("/inference/v1/generate", {
        "token_ids": pc,
        "sampling_params": {"temperature": 0.0, "max_tokens": cap},
    })
    choice = out["choices"][0]
    meta["compressor_finish_reason"] = choice["finish_reason"]
    meta["compressor_cap"] = cap
    meta["truncated"] = choice["finish_reason"] == "length"
    text = await detok(choice["token_ids"])
    return strip_specials(text).strip()


async def compress(raw: str, ids: list[int], arm: str, target: int,
                   meta: dict) -> str:
    if arm == "identity":
        return raw
    if arm == "truncate":
        return await truncate_head_tail(ids, target)
    if arm == "self":
        return await compress_llm(raw, target, meta, len(ids))
    raise ValueError(f"unknown arm {arm!r}")


# ------------------------------ main route ------------------------------

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    if body.get("stream"):
        return JSONResponse({"error": {"message": "sidecar is non-streaming"}}, 400)

    # Per-request overrides so one server can serve every arm.
    arm = body.pop("cot_arm", ARM)
    ratio = float(body.pop("cot_ratio", RATIO))

    te = S["think_end"]
    tp = S["think_prefix"]
    temp = body.get("temperature", 0.6)
    seed = body.get("seed")
    common = {"temperature": temp, "top_p": body.get("top_p", 1.0), "seed": seed}
    pkey = prompt_key(body["messages"])

    t0 = time.perf_counter()

    # (1) messages -> prompt token ids
    rendered = await post("/v1/chat/completions/render",
                          {**body, "stream": False, "max_tokens": 1})
    p_ids = rendered["token_ids"]

    # (2) phase 1: think until </think>
    cached = S["cache"].get((pkey, seed))
    if cached is not None:
        reasoning_ids, closed = cached
        fr1 = "cached"
    else:
        sp = {**common, "max_tokens": THINK_BUDGET}
        if S["te_single"]:
            sp["stop_token_ids"] = [te]
        else:
            sp["stop"] = ["</think>"]
            sp["include_stop_str_in_output"] = True
        c1 = (await post("/inference/v1/generate",
                         {"token_ids": p_ids, "sampling_params": sp}))["choices"][0]
        gen = c1["token_ids"]
        fr1 = c1["finish_reason"]
        if S["case_b"] and gen and gen[0] == S["think_start"]:
            gen = gen[1:]
        if S["te_single"]:
            closed = bool(gen) and gen[-1] == te
            reasoning_ids = gen[:-1] if closed else gen
        else:
            gen_text = await detok(gen)
            closed = gen_text.rstrip().endswith("</think>")
            reasoning_ids = (await tok(gen_text.rstrip()[:-len("</think>")])
                             if closed else gen)
    t1 = time.perf_counter()

    raw = await detok(reasoning_ids)

    # (3) compress; never fail the request, but record the failure
    target = max(16, int(len(reasoning_ids) * ratio))
    cerr = None
    cmeta: dict = {}
    if closed:
        try:
            short = await compress(raw, reasoning_ids, arm, target, cmeta)
        except Exception as e:  # noqa: BLE001
            short, cerr = raw, f"{type(e).__name__}: {e}"
    else:
        short, cerr = raw, "phase1_not_closed"
    short = short.rstrip()

    c_ids = await tok(short)                       # measured compressed length
    splice = await tok(short + "\n</think>\n\n")   # exact phase-2 continuation
    # Pure detokenize->tokenize round trip of the ORIGINAL trace. This is the
    # token-boundary artifact the identity arm exists to detect; it must not be
    # conflated with the rstrip applied above for splicing.
    roundtrip_ids = await tok(raw)
    t2 = time.perf_counter()

    # (4) phase 2: resume after the closed think block
    p2 = {"token_ids": p_ids + tp + splice,
          "sampling_params": {**common, "max_tokens": ANSWER_BUDGET}}
    if USE_PRIORITY:
        p2["priority"] = -1
    c2 = (await post("/inference/v1/generate", p2))["choices"][0]
    answer = strip_specials(await detok(c2["token_ids"])).strip()
    t3 = time.perf_counter()

    rid = f"chatcmpl-{uuid.uuid4().hex}"
    record = {
        "schema": 1, "id": rid, "ts": time.time(),
        "arm": arm, "ratio_requested": ratio,
        "compress_prompt_version": COMPRESS_PROMPT_VERSION,
        "prompt_key": pkey,
        "client_request_id": req.headers.get("x-request-id"),
        "messages": body["messages"],
        "sampling": {"temperature": temp, "top_p": body.get("top_p", 1.0),
                     "seed": seed, "think_budget": THINK_BUDGET,
                     "answer_budget": ANSWER_BUDGET},
        "n_prompt_tokens": len(p_ids),
        "phase1": {"text": raw, "token_ids": reasoning_ids,
                   "n_tokens": len(reasoning_ids), "finish_reason": fr1,
                   "closed": closed, "cache_hit": cached is not None,
                   "ms": round((t1 - t0) * 1000)},
        "compress": {"text": short, "token_ids": c_ids, "n_tokens": len(c_ids),
                     "achieved_ratio": len(c_ids) / max(len(reasoning_ids), 1),
                     "retok_identical": roundtrip_ids == reasoning_ids,
                     "roundtrip_n_tokens": len(roundtrip_ids),
                     "error": cerr, "ms": round((t2 - t1) * 1000),
                     "target_tokens": target, **cmeta},
        "phase2": {"text": answer, "n_tokens": len(c2["token_ids"]),
                   "finish_reason": c2["finish_reason"],
                   "ms": round((t3 - t2) * 1000)},
    }
    async with LOCK:
        S["log"].write(json.dumps(record) + "\n")
        S["log"].flush()

    n_prompt = len(p_ids) + len(tp) + len(splice)
    return {
        "id": rid, "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model", MODEL),
        "choices": [{"index": 0, "finish_reason": c2["finish_reason"],
                     "message": {"role": "assistant", "content": answer,
                                 "reasoning_content": short}}],
        "usage": {"prompt_tokens": n_prompt,
                  "completion_tokens": len(c2["token_ids"]),
                  "total_tokens": n_prompt + len(c2["token_ids"])},
        "cot_compression": {
            "arm": arm,
            "original_reasoning": raw,
            "compressed_reasoning": short,
            "reasoning_tokens": len(reasoning_ids),
            "compressed_tokens": len(c_ids),
            "achieved_ratio": round(len(c_ids) / max(len(reasoning_ids), 1), 4),
            "phase1_closed": closed,
            "compressor_truncated": cmeta.get("truncated"),
            "error": cerr,
            "ms": {"phase1": round((t1 - t0) * 1000),
                   "compress": round((t2 - t1) * 1000),
                   "phase2": round((t3 - t2) * 1000)},
        },
    }


# Registered last so the specific route above wins. Harnesses need /v1/models
# and /health to reach the real server.
@app.api_route("/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, req: Request):
    body = await req.body()
    headers = JSON_HDR if body else {}
    r = await up.request(req.method, "/" + path, content=body, headers=headers)
    return Response(r.content, r.status_code,
                    media_type=r.headers.get("content-type"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("SIDECAR_HOST", "0.0.0.0"),
                port=int(os.environ.get("SIDECAR_PORT", "9000")),
                log_level=os.environ.get("SIDECAR_LOG_LEVEL", "info"))

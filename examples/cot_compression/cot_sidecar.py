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
LOG_QUEUE_MAX = int(os.environ.get("LOG_QUEUE_MAX", "20000"))
LOG_BATCH = int(os.environ.get("LOG_BATCH", "64"))
LOG_FLUSH_SECS = float(os.environ.get("LOG_FLUSH_SECS", "2.0"))
COMPRESS_PROMPT_VERSION = "v1"

JSON_HDR = {"content-type": "application/json"}
LIMITS = httpx.Limits(max_connections=512, max_keepalive_connections=512)

up = httpx.AsyncClient(base_url=VLLM, timeout=3600.0, limits=LIMITS)
cup = httpx.AsyncClient(base_url=COMPRESSOR_URL, timeout=3600.0, limits=LIMITS)
# One producer per request, exactly ONE consumer (log_writer). A single
# consumer is what makes this safe without any lock: nothing else touches the
# file handle. See write_record() for why an asyncio.Lock was not enough.
LOG_Q: asyncio.Queue = asyncio.Queue(maxsize=LOG_QUEUE_MAX)
_LOG_STOP = object()
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
    # Used by phase 2 whenever the prompt itself did not open the block.
    S["think_prefix_ids"] = await tok("<think>\n")

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
    writer = asyncio.create_task(log_writer())
    try:
        yield
    finally:
        # Drain before closing or the tail of the run is lost.
        await LOG_Q.put(_LOG_STOP)
        await writer
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
                   meta: dict, inject: str | None = None) -> str:
    if arm == "inject":
        if inject is None:
            raise ValueError("arm 'inject' requires cot_inject in the request")
        return inject
    if arm == "identity":
        return raw
    if arm == "truncate":
        return await truncate_head_tail(ids, target)
    if arm == "self":
        return await compress_llm(raw, target, meta, len(ids))
    raise ValueError(f"unknown arm {arm!r}")


# --------------------------- shared pipeline ----------------------------

_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "seed",
                  "presence_penalty", "frequency_penalty", "repetition_penalty")


def sampling_common(body: dict) -> dict:
    """Forward only the sampling params the caller actually set.

    Injecting our own defaults (e.g. temperature=0.6) would silently override
    the model's generation_config, so a pass-through request would not match
    what vLLM returns for the same body. Omitted keys let vLLM resolve its own
    defaults.
    """
    return {k: body[k] for k in _SAMPLING_KEYS if body.get(k) is not None}


async def prompt_opens_think(p_ids: list[int]) -> bool:
    """True if the prompt ends inside an unclosed <think> block."""
    tail = await detok(p_ids[-64:])
    return tail.rfind("<think>") > tail.rfind("</think>")


async def cot_pipeline(p_ids, *, common, arm, ratio, inject, client_stop,
                       answer_budget, cached=None, prompt_opens=True,
                       include_stop=False):
    """think -> compress -> resume, shared by both endpoints.

    Phase 1 ALWAYS stops at ``</think>``, whether or not the prompt opened a
    think block. A stop condition that never fires does not change the output,
    so arming it costs nothing on prompts that never reason -- and it catches
    the case where the model opens a think block on its own, which a decision
    made from the prompt alone would miss.

    Whether reasoning happened is then read off one fact: did phase 1 end at
    ``</think>``? If not, phase 1 is the whole answer and is returned verbatim
    with ``no_think`` set, so callers never report an uncompressed generation as
    a compressed one.

    ``prompt_opens`` says whether the prompt already contains the opening
    ``<think>``. It only decides the phase-1 budget and whether phase 2 has to
    re-insert the opening tag.
    """
    te = S["think_end"]
    tp = [] if prompt_opens else S["think_prefix_ids"]
    t0 = time.perf_counter()

    if cached is not None:
        reasoning_ids, closed = cached
        fr1, gen = "cached", None
    else:
        # A prompt that already opened <think> needs room to reason; one that
        # did not is most likely an ordinary completion, so respect the
        # caller's budget rather than burning THINK_BUDGET to find out.
        sp = {**common,
              "max_tokens": THINK_BUDGET if prompt_opens else answer_budget}
        stops = list(client_stop or [])
        if S["te_single"]:
            sp["stop_token_ids"] = [te]
        else:
            stops.append("</think>")
            sp["include_stop_str_in_output"] = True
        if stops:
            sp["stop"] = stops
        c1 = (await post("/inference/v1/generate",
                         {"token_ids": p_ids, "sampling_params": sp}))["choices"][0]
        gen = c1["token_ids"]
        fr1 = c1["finish_reason"]
        if S["te_single"]:
            closed = bool(gen) and gen[-1] == te
        else:
            closed = (await detok(gen)).rstrip().endswith("</think>")

        if not closed:
            # Phase 1 never reached </think>, so it is the whole answer.
            # Verbatim: strip_specials() stands in for skip_special_tokens=True
            # and trim_at_stop() for stop-string removal; no .strip(), since
            # vLLM preserves leading and trailing whitespace.
            text = trim_at_stop(strip_specials(await detok(gen)),
                                client_stop, include_stop)
            ms = round((time.perf_counter() - t0) * 1000)
            return {"no_think": True, "closed": False, "fr1": fr1,
                    "reasoning_ids": [], "raw": "", "short": "", "c_ids": [],
                    "cmeta": {}, "cerr": "no_think_block", "answer": text,
                    "answer_ids": gen, "fr2": fr1, "p2_text": "",
                    "n_prompt": len(p_ids), "target": 0, "ms": (0, 0, ms)}

        # The model may have emitted the opening <think> itself; that tag
        # belongs to the scaffolding, not to the trace.
        if gen and gen[0] == S["think_start"]:
            gen = gen[1:]
        if S["te_single"]:
            reasoning_ids = gen[:-1]
        else:
            gtext = (await detok(gen)).rstrip()
            reasoning_ids = await tok(gtext[: -len("</think>")])
    t1 = time.perf_counter()

    raw = await detok(reasoning_ids)

    # No </think> and phase 1 ended on its own terms: there was no reasoning
    # block to compress and this generation is already the complete answer.
    # Hand it back untouched rather than force-closing a block that never opened.
    no_think = (not closed) and fr1 in ("stop", "cached")
    if no_think:
        return {"no_think": True, "closed": False, "fr1": fr1,
                "reasoning_ids": reasoning_ids, "raw": raw, "short": raw,
                "c_ids": reasoning_ids, "cmeta": {}, "cerr": "no_think_block",
                "answer": strip_specials(raw).strip(),
                "answer_ids": reasoning_ids, "fr2": fr1, "p2_text": "",
                "n_prompt": len(p_ids), "target": 0,
                "ms": (round((t1 - t0) * 1000), 0, 0)}

    target = max(16, int(len(reasoning_ids) * ratio))
    cerr, cmeta = None, {}
    if closed:
        try:
            short = await compress(raw, reasoning_ids, arm, target, cmeta, inject)
        except Exception as e:  # noqa: BLE001
            short, cerr = raw, f"{type(e).__name__}: {e}"
    else:
        short, cerr = raw, "phase1_not_closed"
    short = short.rstrip()

    c_ids = await tok(short)
    splice = await tok(short + "\n</think>\n\n")
    roundtrip_ids = await tok(raw)
    t2 = time.perf_counter()

    p2_ids = p_ids + tp + splice
    p2_text = await detok(p2_ids)
    p2 = {"token_ids": p2_ids,
          "sampling_params": {**common, "max_tokens": answer_budget}}
    if client_stop:
        p2["sampling_params"]["stop"] = list(client_stop)
    if USE_PRIORITY:
        p2["priority"] = -1
    c2 = (await post("/inference/v1/generate", p2))["choices"][0]
    answer = trim_at_stop(strip_specials(await detok(c2["token_ids"])),
                          client_stop, include_stop).strip()
    t3 = time.perf_counter()

    return {"no_think": False, "closed": closed, "fr1": fr1,
            "reasoning_ids": reasoning_ids, "raw": raw, "short": short,
            "c_ids": c_ids, "roundtrip_ids": roundtrip_ids, "cmeta": cmeta,
            "cerr": cerr, "answer": answer, "answer_ids": c2["token_ids"],
            "fr2": c2["finish_reason"], "p2_text": p2_text,
            "n_prompt": len(p2_ids), "target": target,
            "ms": (round((t1 - t0) * 1000), round((t2 - t1) * 1000),
                   round((t3 - t2) * 1000))}


def build_record(res, *, arm, ratio, pkey, extra):
    m1, mc, m2 = res["ms"]
    rec = {
        "schema": 2, "ts": time.time(), "arm": arm, "ratio_requested": ratio,
        "compress_prompt_version": COMPRESS_PROMPT_VERSION, "prompt_key": pkey,
        "phase1": {"text": res["raw"], "token_ids": res["reasoning_ids"],
                   "n_tokens": len(res["reasoning_ids"]),
                   "finish_reason": res["fr1"], "closed": res["closed"],
                   "no_think_block": res["no_think"], "ms": m1},
        "compress": {"text": res["short"], "token_ids": res["c_ids"],
                     "n_tokens": len(res["c_ids"]),
                     "achieved_ratio": len(res["c_ids"])
                     / max(len(res["reasoning_ids"]), 1),
                     "retok_identical": res.get("roundtrip_ids")
                     == res["reasoning_ids"],
                     "roundtrip_n_tokens": len(res.get("roundtrip_ids") or []),
                     "error": res["cerr"], "ms": mc,
                     "target_tokens": res["target"], **res["cmeta"]},
        "phase2": {"text": res["answer"], "n_tokens": len(res["answer_ids"]),
                   "finish_reason": res["fr2"], "ms": m2,
                   "prompt_tail": res["p2_text"][-1500:],
                   "n_prompt_tokens": res["n_prompt"],
                   "original_trace_in_prompt":
                       None if res["no_think"]
                       else res["raw"].strip() in res["p2_text"],
                   "compressed_trace_in_prompt":
                       None if res["no_think"]
                       else res["short"].strip() in res["p2_text"]},
    }
    rec.update(extra)
    return rec


def normalize_stop(stop):
    """OpenAI allows `stop` to be a bare string. Without this, `list("END")`
    yields ['E','N','D'] and generation stops on the first letter E."""
    if stop is None or stop == []:
        return None
    return [stop] if isinstance(stop, str) else list(stop)


def reject_unsupported(body, unsupported):
    """Options the two-phase pipeline cannot honour. Returning a plausible
    looking response for these would be worse than failing."""
    if body.get("stream"):
        return "sidecar is non-streaming"
    bad = [k for k in unsupported if body.get(k)]
    if bad:
        return f"cot sidecar does not support {bad}"
    if int(body.get("n", 1) or 1) != 1:
        return "cot sidecar supports only n=1"
    return None


def trim_at_stop(text: str, stops, include: bool) -> str:
    """Replicate vLLM's stop-string handling.

    /v1/completions removes the matched stop string from the returned text when
    include_stop_str_in_output is false (the default), but /inference/v1/generate
    returns token ids that still contain it. Without this the pass-through path
    returns '63\\n\\n' where vLLM returns '63'.
    """
    if include or not stops:
        return text
    cut = len(text)
    for st in stops:
        i = text.find(st)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def _write_batch(fh, recs):
    """Runs in a worker thread. json.dumps plus the write syscall both happen
    off the event loop; on a network filesystem like lustre a write() can stall
    for milliseconds, which would otherwise freeze every in-flight request."""
    fh.write("".join(json.dumps(r) + "\n" for r in recs))
    fh.flush()


async def write_record(rec):
    """Hand the record to the writer task.

    Uses ``await put`` rather than ``put_nowait`` so a full queue applies
    backpressure instead of dropping records -- this is an experiment log and a
    silently missing row is worse than a slow one. The queue is sized so that
    only a pathological stall could fill it.
    """
    await LOG_Q.put(rec)


async def log_writer():
    """Single consumer. Batches whatever has accumulated, writes it in a thread.

    Because there is exactly one of these, no lock is needed anywhere: the file
    handle has a single owner. Flushes at most every LOG_FLUSH_SECS so a crash
    loses at most that much, rather than flushing on every record.
    """
    fh = S["log"]
    stopping = False
    while not stopping:
        try:
            rec = await asyncio.wait_for(LOG_Q.get(), timeout=LOG_FLUSH_SECS)
        except asyncio.TimeoutError:
            continue
        batch = []
        if rec is _LOG_STOP:
            stopping = True
        else:
            batch.append(rec)
        while len(batch) < LOG_BATCH and not LOG_Q.empty():
            nxt = LOG_Q.get_nowait()
            if nxt is _LOG_STOP:
                stopping = True
                break
            batch.append(nxt)
        if batch:
            await asyncio.to_thread(_write_batch, fh, batch)


def cot_meta(res, arm):
    m1, mc, m2 = res["ms"]
    return {
        "arm": arm,
        "original_reasoning": res["raw"],
        "compressed_reasoning": res["short"],
        "reasoning_tokens": len(res["reasoning_ids"]),
        "compressed_tokens": len(res["c_ids"]),
        "achieved_ratio": round(len(res["c_ids"])
                                / max(len(res["reasoning_ids"]), 1), 4),
        "phase1_closed": res["closed"],
        "no_think_block": res["no_think"],
        "original_trace_in_prompt":
            None if res["no_think"] else res["raw"].strip() in res["p2_text"],
        "compressed_trace_in_prompt":
            None if res["no_think"] else res["short"].strip() in res["p2_text"],
        "compressor_truncated": res["cmeta"].get("truncated"),
        "error": res["cerr"],
        "ms": {"phase1": m1, "compress": mc, "phase2": m2},
    }


# ------------------------------ chat route ------------------------------

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    err = reject_unsupported(body, ("logprobs", "top_logprobs"))
    if err:
        return JSONResponse({"error": {"message": err}}, 400)

    arm = body.pop("cot_arm", ARM)
    ratio = float(body.pop("cot_ratio", RATIO))
    inject = body.pop("cot_inject", None)

    seed = body.get("seed")
    common = sampling_common(body)
    stop = normalize_stop(body.get("stop"))
    pkey = prompt_key(body["messages"])

    rendered = await post("/v1/chat/completions/render",
                          {**body, "stream": False, "max_tokens": 1})
    p_ids = rendered["token_ids"]

    # chat_template_kwargs={"enable_thinking": false} renders a *closed* empty
    # think block, so there is nothing to compress. Probe rather than assume.
    opens = await prompt_opens_think(p_ids)
    # Compression arms must share a fixed budget or the comparison is invalid,
    # so ANSWER_BUDGET wins there. When the prompt opens no think block the
    # caller's max_tokens has to be honoured for the response to match vLLM.
    budget = ANSWER_BUDGET if opens else int(body.get("max_tokens")
                                             or ANSWER_BUDGET)

    res = await cot_pipeline(p_ids, common=common, arm=arm, ratio=ratio,
                             inject=inject, client_stop=stop,
                             answer_budget=budget, prompt_opens=opens,
                             include_stop=bool(body.get(
                                 "include_stop_str_in_output", False)),
                             cached=S["cache"].get((pkey, seed)))

    rid = f"chatcmpl-{uuid.uuid4().hex}"
    await write_record(build_record(res, arm=arm, ratio=ratio, pkey=pkey, extra={
        "id": rid, "endpoint": "chat", "messages": body["messages"],
        "client_request_id": req.headers.get("x-request-id"),
        "sampling": {**common, "think_budget": THINK_BUDGET,
                     "answer_budget": budget},
        "n_prompt_tokens": len(p_ids)}))

    return {
        "id": rid, "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model", MODEL),
        "choices": [{"index": 0, "finish_reason": res["fr2"],
                     "message": {"role": "assistant", "content": res["answer"],
                                 "reasoning_content": res["short"]}}],
        "usage": {"prompt_tokens": res["n_prompt"],
                  "completion_tokens": len(res["answer_ids"]),
                  "total_tokens": res["n_prompt"] + len(res["answer_ids"])},
        "cot_compression": cot_meta(res, arm),
    }


# --------------------------- completions route ---------------------------

_UNSUPPORTED = ("echo", "suffix", "logprobs", "best_of", "prompt_logprobs")


@app.post("/v1/completions")
async def completions(req: Request):
    """Same pipeline over a raw prompt.

    Rejects options the pipeline cannot honour rather than silently ignoring
    them -- a completions eval that quietly bypassed compression would report a
    baseline number that looks like a result.
    """
    body = await req.json()
    err = reject_unsupported(body, _UNSUPPORTED)
    if err:
        return JSONResponse({"error": {"message": err}}, 400)

    arm = body.pop("cot_arm", ARM)
    ratio = float(body.pop("cot_ratio", RATIO))
    inject = body.pop("cot_inject", None)

    raw_prompt = body.get("prompt")
    add_special = body.get("add_special_tokens", True)
    if isinstance(raw_prompt, str):
        prompts = [raw_prompt]
    elif isinstance(raw_prompt, list) and raw_prompt and isinstance(raw_prompt[0], int):
        prompts = [raw_prompt]
    elif isinstance(raw_prompt, list):
        prompts = list(raw_prompt)
    else:
        return JSONResponse({"error": {"message": "invalid prompt"}}, 400)

    async def tokenize_prompt(p):
        if isinstance(p, str):
            return (await post("/tokenize", {"model": MODEL, "prompt": p,
                                             "add_special_tokens": add_special}))["tokens"]
        return list(p)

    seed = body.get("seed")
    common = sampling_common(body)
    stop = normalize_stop(body.get("stop"))
    budget = int(body.get("max_tokens") or ANSWER_BUDGET)
    include_stop = bool(body.get("include_stop_str_in_output", False))

    async def one(idx, p):
        p_ids = await tokenize_prompt(p)
        pkey = prompt_key(p if isinstance(p, str) else p_ids)
        opens = await prompt_opens_think(p_ids)
        res = await cot_pipeline(p_ids, common=common, arm=arm, ratio=ratio,
                                 inject=inject, client_stop=stop,
                                 answer_budget=budget, prompt_opens=opens,
                                 include_stop=include_stop,
                                 cached=S["cache"].get((pkey, seed)))
        await write_record(build_record(res, arm=arm, ratio=ratio, pkey=pkey,
                                        extra={
            "id": f"cmpl-{uuid.uuid4().hex}", "endpoint": "completions",
            "prompt": p if isinstance(p, str) else None,
            "client_request_id": req.headers.get("x-request-id"),
            "sampling": {**common, "think_budget": THINK_BUDGET,
                         "answer_budget": budget},
            "n_prompt_tokens": len(p_ids)}))
        return idx, res

    results = [r for _, r in sorted(await asyncio.gather(
        *(one(i, p) for i, p in enumerate(prompts))), key=lambda x: x[0])]

    n_prompt = sum(r["n_prompt"] for r in results)
    n_out = sum(len(r["answer_ids"]) for r in results)
    return {
        "id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion",
        "created": int(time.time()), "model": body.get("model", MODEL),
        "choices": [{"index": i, "text": r["answer"], "logprobs": None,
                     "finish_reason": r["fr2"]}
                    for i, r in enumerate(results)],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                  "total_tokens": n_prompt + n_out},
        "cot_compression": [cot_meta(r, arm) for r in results],
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

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for cot_online_sidecar's loop, against a scripted backend.

No GPU and no vLLM. `post()` is replaced with a fake that serves /tokenize,
/detokenize and a queue of canned /inference/v1/generate responses, so every
control-flow branch is driven deterministically.

Covers the four v1 must-fix gaps from DESIGN-online-compression.md plus the
out-of-budget behaviour required by review:

  1  EOS vs </think>            -- finish_reason is "stop" for both
  2  empty summary              -- falls back to Delethink-style tail
  3  reserved tokens in a note  -- sanitised before splicing back
  4  detok/retok round trip     -- notes stay token IDs end to end
  5  out of budget              -- content="", trace in reasoning_content
  6  guard fires at chunk 0     -- R is None, no deref
"""
import asyncio
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cot_online_sidecar as M          # noqa: E402

TS, TE = 1000001, 1000002               # <think>, </think>


def enc(text):
    out, i = [], 0
    while i < len(text):
        if text.startswith("<think>", i):
            out.append(TS); i += 7
        elif text.startswith("</think>", i):
            out.append(TE); i += 8
        else:
            out.append(ord(text[i])); i += 1
    return out


def dec(ids):
    return "".join("<think>" if i == TS else "</think>" if i == TE else chr(i)
                   for i in ids)


SCRIPT = []          # queue of (token_ids, finish_reason)
CALLS = []           # (prompt_ids, max_tokens, sampling_params)


async def fake_post(path, payload):
    if path == "/tokenize":
        return {"tokens": enc(payload["prompt"])}
    if path == "/detokenize":
        return {"prompt": dec(payload["tokens"])}
    if path == "/inference/v1/generate":
        sp = payload["sampling_params"]
        CALLS.append((payload["token_ids"], sp["max_tokens"], sp))
        ids, fr = SCRIPT.pop(0)
        return {"choices": [{"token_ids": list(ids), "finish_reason": fr}]}
    raise AssertionError(f"unexpected path {path}")


async def setup():
    M.post = fake_post
    M.MODEL = "fake"
    S = M.S
    S.update({"render_cache": {}, "http_calls": 0, "requests": 0})
    S["think_end_ids"] = [TE]
    S["think_end"] = TE
    S["think_start"] = TS
    S["banned_ids"] = {TS, TE}
    S["special_strs"] = ("<|im_end|>",)
    S["summ_ids"] = enc(M.SUMMARY_HINT)
    S["cont_ids"] = enc(M.CONTINUE_HINT)
    S["notes_head_ids"] = enc(M.NOTES_HEAD)
    S["sep_ids"] = enc("\n\n")
    S["label_cache"] = {}
    S["logq"] = None


def script(*items):
    SCRIPT.clear(); CALLS.clear()
    SCRIPT.extend(items)


def run(p_ids, **kw):
    return asyncio.get_event_loop().run_until_complete(
        M.run_online(p_ids, sp={"temperature": 1.0}, think_open=[], **kw))


FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    asyncio.get_event_loop().run_until_complete(setup())
    P = enc("Q? <think>")

    print("=" * 74)
    print("1. </think> inside chunk 0 -> answer generated, one follow-up call")
    print("=" * 74)
    M.THINK_TOTAL, M.MAX_CHUNKS, M.C_GEN = 245760, 60, 4096
    script((enc("reasoning here") + [TE], "stop"), (enc("The answer."), "stop"))
    reasoning, answer, fr, gen, closed = run(P)
    check(closed is True, "closed=True")
    check(answer == "The answer.", f"content={answer!r}")
    check(fr == "stop", f"finish_reason={fr!r}")
    check(gen.chunks == 1 and gen.summary == 0, "1 chunk, no summary written")
    check(reasoning == "reasoning here", f"reasoning_content={reasoning!r}")
    check(len(CALLS) == 2, f"exactly 2 generate calls, got {len(CALLS)}")
    check(TE not in CALLS[0][0], "chunk call prompt has no stray </think>")
    check(CALLS[1][0][-1] == TE, "answer prompt ends with </think>")

    print("\n" + "=" * 74)
    print("2. two chunks then close -> note accumulates into the next prompt")
    print("=" * 74)
    script((enc("A" * 20), "length"), (enc("note one"), "stop"),
           (enc("more") + [TE], "stop"), (enc("Final."), "stop"))
    M.C_GEN = 20
    reasoning, answer, fr, gen, closed = run(P)
    check(gen.chunks == 2, f"2 chunks, got {gen.chunks}")
    check(gen.summary == len(enc("note one")), "note counted")
    check("Notes from my earlier work" in reasoning, "notes head in reasoning")
    check("[1] note one" in reasoning, "labelled note in reasoning")
    check(dec(CALLS[2][0]).endswith(M.CONTINUE_HINT),
          "chunk 2 prompt ends with CONTINUE_HINT")
    check("[1] note one" in dec(CALLS[2][0]), "chunk 2 prompt carries the note")

    print("\n" + "=" * 74)
    print("3. OUT OF BUDGET -> content empty, trace in reasoning_content")
    print("=" * 74)
    M.C_GEN, M.THINK_TOTAL, M.MAX_CHUNKS = 10, 30, 60
    script((enc("A" * 10), "length"), (enc("n1"), "stop"),
           (enc("B" * 10), "length"), (enc("n2"), "stop"),
           (enc("C" * 10), "length"))
    reasoning, answer, fr, gen, closed = run(P)
    check(answer == "", f"content is empty, got {answer!r}")
    check(fr == "length", f"finish_reason={fr!r} (vLLM reports length)")
    check(closed is False, "think_closed=False")
    check(gen.thinking == 30, f"thinking==THINK_TOTAL, got {gen.thinking}")
    check(reasoning.endswith("C" * 10), "final chunk verbatim in reasoning")
    check("[1] n1" in reasoning and "[2] n2" in reasoning, "notes retained")
    check(not SCRIPT, "no answer call was made")

    print("\n" + "=" * 74)
    print("4. EOS inside <think> -> model FINISHED, close the tag and answer")
    print("=" * 74)
    # Measured on run fdc7b2c8356e1bb6: 5.8% of GPQA and 22.1% of HLE requests
    # ended this way at 2-16 chunks, every one carrying a complete answer that
    # was being discarded. Distinct from out-of-budget (case 3), which must keep
    # returning empty content.
    M.C_GEN, M.THINK_TOTAL = 4096, 245760
    script((enc("...so the answer is 42.") + [ord("\u0001")], "stop"),
           (enc("Answer: 42"), "stop"))
    M.S["special_ids"] = {1}
    reasoning, answer, fr, gen, closed = run(P)
    check(closed is True, "treated as finished, not as out-of-budget")
    check(answer == "Answer: 42", f"answer extracted (got {answer!r})")
    check(gen.eos_in_think == 1, f"eos_in_think counted ({gen.eos_in_think})")
    check(1 not in CALLS[1][0], "trailing special stripped before splicing")
    check(CALLS[1][0][-1] == TE, "</think> injected before the answer call")
    check("42" in reasoning, "reasoning carried out")

    print("\n" + "=" * 74)
    print("4b. out-of-budget is UNCHANGED by the 4 fix (no answer)")
    print("=" * 74)
    M.C_GEN, M.THINK_TOTAL = 10, 20
    script((enc("A" * 10), "length"), (enc("n1"), "stop"),
           (enc("B" * 10), "length"))
    reasoning, answer, fr, gen, closed = run(P)
    check(answer == "" and fr == "length" and closed is False,
          "budget path still returns empty content with length")
    check(gen.eos_in_think == 0, "not counted as eos_in_think")

    print("\n" + "=" * 74)
    print("5. empty note -> Delethink tail fallback, loop survives")
    print("=" * 74)
    M.C_GEN, M.THINK_TOTAL, M.FALLBACK_TAIL = 10, 20, 4
    script((enc("ABCDEFGHIJ"), "length"), ([TE], "stop"),
           (enc("KLMNOPQRST"), "length"))
    reasoning, answer, fr, gen, closed = run(P)
    check(gen.summary_empty == 1, f"summary_empty={gen.summary_empty}")
    check(gen.summary_fallbacks == 1, f"fallbacks={gen.summary_fallbacks}")
    check("[1] GHIJ" in reasoning, f"note is the 4-token tail; got {reasoning!r}")

    print("\n" + "=" * 74)
    print("6. reserved tokens inside a note are stripped before splicing")
    print("=" * 74)
    M.C_GEN, M.THINK_TOTAL = 10, 20
    script((enc("ABCDEFGHIJ"), "length"),
           ([ord("x"), TS, ord("y")], "stop"),
           (enc("KLMNOPQRST"), "length"))
    reasoning, answer, fr, gen, closed = run(P)
    p2 = CALLS[2][0]
    # P itself ends with <think>, so one occurrence is expected; two would mean
    # the note's <think> survived into the re-fed prompt.
    check(p2.count(TS) == 1,
          f"<think> stripped from note (expect 1 from P, got {p2.count(TS)})")
    check(p2.count(TE) == 0, "</think> stripped from the note before re-feeding")
    check("[1] xy" in reasoning, f"note sanitised to 'xy'; got {reasoning!r}")

    print("\n" + "=" * 74)
    print("7. guard fires before any chunk (R is None) -> no deref, no crash")
    print("=" * 74)
    M.MAX_MODEL_LEN, M.CTX_SAFETY = 300, 256
    script()
    reasoning, answer, fr, gen, closed = run(enc("x" * 280))
    check(gen.chunks == 0, "no chunks ran")
    check(answer == "" and reasoning == "", "empty response, no exception")
    check(fr == "length", f"finish_reason={fr!r}")
    M.MAX_MODEL_LEN = 262144

    print("\n" + "=" * 74)
    print("8. answer_room mirrors the baseline's single shared pool")
    print("=" * 74)
    M.THINK_TOTAL, M.ANSWER_BUDGET = 245760, 8192
    g = M.Counters()
    g.thinking = 1000
    check(M.answer_room(g) == 8192, "plenty left -> ANSWER_BUDGET")
    g.thinking = 245760 - 100
    check(M.answer_room(g) == 100, "nearly exhausted -> remainder only")
    g.thinking = 245760
    check(M.answer_room(g) == 0, "exhausted -> nothing left for an answer")

    print("\n" + "=" * 74)
    print(f"{len(FAILS)} failure(s)" if FAILS else "ALL PASS")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

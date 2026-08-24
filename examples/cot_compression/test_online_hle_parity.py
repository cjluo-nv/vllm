#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parity test: out-of-budget must look exactly like the no-sidecar baseline.

Ground truth is `hle_truncated_baseline.json`, extracted from the uncompressed
xhigh HLE run (2158 rows). 11 rows hit `finish_reason: length`, and every one of
them came back:

    generation           = ""            (0 chars, all 11)
    reasoning_content    = 253K-858K chars (the whole unclosed trace)
    num_generated_tokens = 245760        (exactly the budget, all 11)
    symbolic_correct     = False         (all 11)

So the grader saw an empty answer and marked every one wrong. If the online arm
injected </think> and generated an answer for those same requests it would be
competing for 11 points the baseline had no chance at -- 0.51 HLE points, about
18% of the -2.88 delta measured for offline compression. That is why this is a
validity requirement and not a cosmetic one.

Run: python3 test_online_hle_parity.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cot_online_sidecar as M                      # noqa: E402
from test_online_sidecar import TE, dec, enc, fake_post, setup   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "hle_truncated_baseline.json")

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


class StubRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def main():
    rows = json.load(open(BASELINE))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(setup())

    print("=" * 74)
    print(f"baseline ground truth ({len(rows)} truncated HLE rows)")
    print("=" * 74)
    check(all(r["gen_chars"] == 0 for r in rows), "every generation is empty")
    check(all(r["ntok"] == 245760 for r in rows), "every ntok == 245760")
    check(all(r["correct"] is False for r in rows), "every row scored wrong")
    print(f"  reasoning_content spans {min(r['rc_chars'] for r in rows):,}"
          f" - {max(r['rc_chars'] for r in rows):,} chars")

    # A non-terminating trace: every chunk fills, </think> never appears.
    # Scaled down 1000x so the test runs instantly; the branch is identical.
    C, TOTAL = 16, 160
    M.C_GEN, M.THINK_TOTAL, M.MAX_CHUNKS = C, TOTAL, 60
    M.MAX_MODEL_LEN, M.CTX_SAFETY, M.ANSWER_BUDGET = 262144, 256, 8192
    M.RC_MAX_CHARS = 0

    import test_online_sidecar as T
    T.SCRIPT.clear(); T.CALLS.clear()
    n_chunks = TOTAL // C
    for i in range(n_chunks):
        T.SCRIPT.append((enc(chr(65 + i) * C), "length"))       # reasoning
        if i < n_chunks - 1:
            # No note after the last chunk: it is kept verbatim, so
            # summarising it would be wasted work.
            T.SCRIPT.append((enc(f"note{i}"), "stop"))
    M.post = fake_post

    print("\n" + "=" * 74)
    print("online arm on a non-terminating trace")
    print("=" * 74)
    body = {"model": "fake", "messages": [{"role": "user", "content": "Q?"}],
            "temperature": 1.0}
    # render + prompt_opens_think are served by the fake backend
    M.S["render_cache"].clear()
    M.S["render_cache"][M.render_key(body)] = enc("Q? <think>")
    resp = loop.run_until_complete(M.chat(StubRequest(body)))

    ch = resp["choices"][0]
    msg = ch["message"]
    meta = resp["cot_online"]

    check(msg["content"] == "", f"content is empty (got {msg['content']!r})")
    check(ch["finish_reason"] == "length",
          f"finish_reason == 'length' (got {ch['finish_reason']!r})")
    check(meta["think_closed"] is False, "think_closed is False")
    check(len(msg["reasoning_content"]) > 0,
          "reasoning_content carries the unclosed trace")
    check(meta["thinking_tokens"] == TOTAL,
          f"thinking == budget ({meta['thinking_tokens']} vs {TOTAL})")

    # completion_tokens must report tokens actually generated, as vLLM does.
    expect = meta["thinking_tokens"] + meta["summary_tokens"] + meta["answer_tokens"]
    check(resp["usage"]["completion_tokens"] == expect,
          f"completion_tokens counts real generation ({expect})")
    check(meta["answer_tokens"] == 0, "no answer tokens were generated")
    check(not T.SCRIPT, "backend script fully consumed (no extra answer call)")
    check(len(T.CALLS) == 2 * n_chunks - 1,
          f"{2 * n_chunks - 1} calls: {n_chunks} chunks + {n_chunks - 1} notes,"
          f" got {len(T.CALLS)}")
    check(meta["n_chunks"] == n_chunks, f"n_chunks == {n_chunks}")

    print("\n" + "=" * 74)
    print("shape parity with the 11 baseline rows")
    print("=" * 74)
    for r in rows[:3]:
        same = (msg["content"] == "" and r["gen_chars"] == 0
                and ch["finish_reason"] == "length"
                and len(msg["reasoning_content"]) > 0 and r["rc_chars"] > 0)
        check(same, f"{r['id']}: content empty + trace present + length")
    check(all(r["correct"] is False for r in rows),
          "baseline scored all 11 wrong -> arm forfeits the same 11, as it must")

    print("\n" + "=" * 74)
    print(f"{len(FAILS)} failure(s)" if FAILS else "ALL PASS")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

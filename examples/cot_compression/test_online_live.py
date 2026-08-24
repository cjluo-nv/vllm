#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live smoke test for cot_online_sidecar against a real vLLM server.

Checks the things a scripted backend cannot: real stop_token_ids semantics,
whether the chat template already opens <think>, and that the loop actually
chunks and resumes on a real model.

  usage: test_online_live.py <sidecar_url> <vllm_url> <model>
"""
import json
import sys
import urllib.request

FAILS = []


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def main():
    side, vllm, model = sys.argv[1], sys.argv[2], sys.argv[3]

    print("=" * 74)
    print("A. backend probe: template shape and stop_token_ids semantics")
    print("=" * 74)
    te = post(f"{vllm}/tokenize",
              {"model": model, "prompt": "</think>",
               "add_special_tokens": False})["tokens"]
    print(f"  </think> -> {te}")
    check(len(te) >= 1, "</think> tokenises")
    te_id = te[-1]

    rend = post(f"{vllm}/v1/chat/completions/render",
                {"model": model, "max_tokens": 1,
                 "messages": [{"role": "user", "content": "hi"}]})
    tail = post(f"{vllm}/detokenize",
                {"model": model, "tokens": rend["token_ids"][-24:]})["prompt"]
    opens = "<think>" in tail
    print(f"  rendered tail: {tail!r}")
    check(opens, "template already opens <think> (so think_open must be [])")

    # Does /inference/v1/generate include the stop token in token_ids?
    g = post(f"{vllm}/inference/v1/generate",
             {"token_ids": rend["token_ids"],
              "sampling_params": {"temperature": 0.0, "max_tokens": 2048,
                                  "stop_token_ids": [te_id]}})["choices"][0]
    included = te_id in g["token_ids"]
    print(f"  stop_token_ids: finish_reason={g['finish_reason']!r} "
          f"len={len(g['token_ids'])} includes_stop={included}")
    check(True, f"stop token {'IS' if included else 'is NOT'} included "
                "(loop uses membership, so either is handled)")

    print("\n" + "=" * 74)
    print("B. easy question closes </think> and produces an answer")
    print("=" * 74)
    r = post(f"{side}/v1/chat/completions",
             {"model": model, "temperature": 0.0,
              "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}]})
    ch, meta = r["choices"][0], r["cot_online"]
    print("  meta:", json.dumps(meta))
    print(f"  content={ch['message']['content'][:120]!r}")
    check(meta["think_closed"] is True, "think_closed=True")
    check(ch["message"]["content"].strip() != "", "answer is non-empty")
    check(meta["answer_tokens"] > 0, "answer tokens generated")
    check(r["usage"]["completion_tokens"] ==
          meta["thinking_tokens"] + meta["summary_tokens"] + meta["answer_tokens"],
          "completion_tokens == thinking + summary + answer")

    print("\n" + "=" * 74)
    print("C. multi-chunk: notes accumulate and reasoning resumes")
    print("=" * 74)
    hard = ("Prove that for every positive integer n, the number "
            "n^5 - n is divisible by 30. Then generalise as far as you can, "
            "and carefully verify each step.")
    r = post(f"{side}/v1/chat/completions",
             {"model": model, "temperature": 0.0,
              "messages": [{"role": "user", "content": hard}]})
    ch, meta = r["choices"][0], r["cot_online"]
    print("  meta:", json.dumps(meta))
    check(meta["n_chunks"] >= 2, f"chunked at least twice ({meta['n_chunks']})")
    check(meta["summary_tokens"] > 0, "notes were written")
    rc = ch["message"]["reasoning_content"]
    check("Notes from my earlier work" in rc, "notes head present in trace")
    check("[1] " in rc, "labelled note present in trace")
    i = rc.find("[1] ")
    print("  ---- first note ----")
    print("  " + rc[i:i + 600].replace("\n", "\n  "))
    print("  --------------------")
    check(meta["summary_fallbacks"] == 0,
          f"no tail fallbacks ({meta['summary_fallbacks']})")

    print("\n" + "=" * 74)
    print("D. out of budget -> empty content, trace in reasoning_content")
    print("=" * 74)
    # Server is started with a small THINK_TOTAL so this path is reachable.
    r = post(f"{side}/v1/chat/completions",
             {"model": model, "temperature": 1.0,
              "messages": [{"role": "user", "content":
                            "Enumerate every 4-digit prime and for each one "
                            "verify primality by trial division, showing all work."}]})
    ch, meta = r["choices"][0], r["cot_online"]
    print("  meta:", json.dumps(meta))
    print(f"  finish_reason={ch['finish_reason']!r} "
          f"content={ch['message']['content'][:60]!r}")
    if meta["think_closed"]:
        print("  (model closed the block; budget path not exercised here)")
    else:
        check(ch["message"]["content"] == "", "content empty, as vLLM does")
        check(ch["finish_reason"] == "length", "finish_reason=length")
        check(len(ch["message"]["reasoning_content"]) > 0,
              "trace in reasoning_content")
        check(meta["answer_tokens"] == 0, "no answer manufactured")

    print("\n" + "=" * 74)
    print("E. enable_thinking=false passes through to vLLM unchanged")
    print("=" * 74)
    body = {"model": model, "temperature": 0.0, "max_tokens": 64,
            "messages": [{"role": "user", "content": "Name three colours."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    a = post(f"{side}/v1/chat/completions", body)
    b = post(f"{vllm}/v1/chat/completions", body)
    ta = a["choices"][0]["message"]["content"]
    tb = b["choices"][0]["message"]["content"]
    print(f"  sidecar: {ta[:70]!r}")
    print(f"  vllm   : {tb[:70]!r}")
    check(ta == tb, "sidecar matches vLLM byte for byte")
    check("cot_online" not in a, "no online metadata on the passthrough path")

    print("\n" + "=" * 74)
    print(f"{len(FAILS)} failure(s)" if FAILS else "ALL PASS")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

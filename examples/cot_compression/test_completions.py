#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test /v1/completions support in the CoT-compression sidecar.

Covers the two prompt shapes that behave differently:

  1. A pre-rendered chat prompt ending in `<think>` -- the shape a harness
     produces with --apply_chat_template. Compression should happen.
  2. A bare few-shot text prompt with no think scaffolding. There is no
     reasoning block to compress; the sidecar must say so (no_think_block)
     rather than report an uncompressed generation as a compressed one.

Also checks batch prompts, token-id prompts, and that unsupported options are
rejected instead of silently ignored.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

Q = ("Let S be the set of all positive integers n with n <= 1000 such that "
     "n^2 + 1 is divisible by 5. What is the sum of all elements of S? "
     "End your response with 'ANSWER: <integer>'.")

CHAT_PROMPT = (f"<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n<think>\n")

FEWSHOT = (
    "Question: What is 12 * 4?\nAnswer: 48\n\n"
    "Question: What is 15 + 27?\nAnswer: 42\n\n"
    "Question: What is 100 - 37?\nAnswer: "
)


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", default="http://127.0.0.1:9000")
    ap.add_argument("--model", default="qwen38")
    a = ap.parse_args()
    url = f"{a.sidecar}/v1/completions"
    fails = []

    def show(cc, label):
        print(f"  [{label}] reasoning={cc['reasoning_tokens']} -> "
              f"compressed={cc['compressed_tokens']} "
              f"ratio={cc['achieved_ratio']} "
              f"no_think={cc['no_think_block']} "
              f"orig_in_p2={cc['original_trace_in_prompt']} "
              f"comp_in_p2={cc['compressed_trace_in_prompt']}")

    print("=" * 74)
    print("1. pre-rendered chat prompt (has <think>) -- should COMPRESS")
    print("=" * 74)
    r = post(url, {"model": a.model, "prompt": CHAT_PROMPT,
                   "add_special_tokens": False, "temperature": 0.6,
                   "top_p": 0.95, "seed": 1234, "max_tokens": 1024,
                   "cot_arm": "self", "cot_ratio": 0.3})
    cc = r["cot_compression"][0]
    show(cc, "self")
    print(f"  text: {r['choices'][0]['text'][:220]!r}")
    if cc["no_think_block"]:
        fails.append("1: no think block detected on a prompt that ends in <think>")
    if cc["compressed_tokens"] >= cc["reasoning_tokens"]:
        fails.append("1: trace did not shrink")
    if cc["original_trace_in_prompt"]:
        fails.append("1: original trace leaked into the phase-2 prompt")
    if not cc["compressed_trace_in_prompt"]:
        fails.append("1: compressed trace absent from the phase-2 prompt")

    print("\n" + "=" * 74)
    print("2. bare few-shot prompt (no <think>) -- should report NO COMPRESSION")
    print("=" * 74)
    r = post(url, {"model": a.model, "prompt": FEWSHOT, "temperature": 0.0,
                   "max_tokens": 64, "stop": ["\n\n", "Question:"],
                   "cot_arm": "self", "cot_ratio": 0.3})
    cc = r["cot_compression"][0]
    show(cc, "self")
    print(f"  text: {r['choices'][0]['text'][:160]!r}")
    if not cc["no_think_block"]:
        fails.append("2: bare prompt should have been flagged no_think_block")
    if cc["original_trace_in_prompt"] is not None:
        fails.append("2: provenance flags should be null when nothing was spliced")
    if cc["error"] != "no_think_block":
        fails.append(f"2: expected error='no_think_block', got {cc['error']!r}")

    print("\n" + "=" * 74)
    print("3. batch prompts + token-id prompt")
    print("=" * 74)
    r = post(url, {"model": a.model, "prompt": [CHAT_PROMPT, CHAT_PROMPT],
                   "add_special_tokens": False, "temperature": 0.6,
                   "top_p": 0.95, "seed": 7, "max_tokens": 512,
                   "cot_arm": "truncate", "cot_ratio": 0.3})
    print(f"  choices returned: {len(r['choices'])}")
    for i, cc in enumerate(r["cot_compression"]):
        show(cc, f"batch[{i}]")
    if len(r["choices"]) != 2:
        fails.append("3: batch of 2 prompts did not return 2 choices")

    ids = post(f"{a.sidecar}/tokenize",
               {"model": a.model, "prompt": CHAT_PROMPT,
                "add_special_tokens": False})["tokens"]
    r = post(url, {"model": a.model, "prompt": ids, "temperature": 0.6,
                   "top_p": 0.95, "seed": 1234, "max_tokens": 512,
                   "cot_arm": "self", "cot_ratio": 0.3})
    cc = r["cot_compression"][0]
    show(cc, "token-ids")
    if cc["no_think_block"]:
        fails.append("4: token-id prompt was not recognised as having a think block")

    print("\n" + "=" * 74)
    print("4. unsupported options are rejected, not ignored")
    print("=" * 74)
    for extra in ({"echo": True}, {"n": 2}, {"logprobs": 5}):
        try:
            post(url, {"model": a.model, "prompt": CHAT_PROMPT,
                       "max_tokens": 16, **extra})
            fails.append(f"5: {extra} was accepted but should be rejected")
            print(f"  {extra}: ACCEPTED (bad)")
        except urllib.error.HTTPError as e:
            print(f"  {extra}: rejected {e.code} - "
                  f"{json.loads(e.read())['error']['message'][:70]}")

    print("\n" + "=" * 74)
    print("5. pass-through fidelity: sidecar output == vLLM output")
    print("=" * 74)
    vllm_url = a.sidecar.rsplit(":", 1)[0] + ":8000/v1/completions"

    # (a) identical generation for a bare prompt
    req = {"model": a.model, "prompt": FEWSHOT, "temperature": 0.0,
           "max_tokens": 32, "seed": 99, "stop": ["\n\n", "Question:"]}
    direct = post(vllm_url, dict(req))["choices"][0]
    viaside = post(url, dict(req))["choices"][0]
    same = direct["text"] == viaside["text"]
    print(f"  vLLM direct : {direct['text']!r} ({direct['finish_reason']})")
    print(f"  via sidecar : {viaside['text']!r} ({viaside['finish_reason']})")
    print(f"  identical   : {same}")
    if not same:
        fails.append("5a: pass-through text differs from vLLM direct")
    if direct["finish_reason"] != viaside["finish_reason"]:
        fails.append("5b: pass-through finish_reason differs from vLLM direct")

    # (b) caller max_tokens is honoured, not replaced by THINK_BUDGET
    req = {"model": a.model, "prompt": "Write a long essay about clouds.",
           "temperature": 0.0, "max_tokens": 8, "seed": 99}
    direct = post(vllm_url, dict(req))["choices"][0]
    viaside = post(url, dict(req))["choices"][0]
    print(f"\n  max_tokens=8  vLLM direct : {direct['text']!r} "
          f"({direct['finish_reason']})")
    print(f"  max_tokens=8  via sidecar : {viaside['text']!r} "
          f"({viaside['finish_reason']})")
    if viaside["finish_reason"] != "length":
        fails.append(f"5c: max_tokens=8 gave finish_reason="
                     f"{viaside['finish_reason']!r}, expected 'length'")
    if direct["text"] != viaside["text"]:
        fails.append("5d: max_tokens-capped text differs from vLLM direct")

    # (c) omitted sampling params must not be overridden by sidecar defaults
    req = {"model": a.model, "prompt": FEWSHOT, "max_tokens": 16, "seed": 5,
           "stop": ["\n\n"]}
    direct = post(vllm_url, dict(req))["choices"][0]["text"]
    viaside = post(url, dict(req))["choices"][0]["text"]
    print(f"\n  no temp/top_p  vLLM direct : {direct!r}")
    print(f"  no temp/top_p  via sidecar : {viaside!r}")
    print("  (both should use the model's generation_config defaults)")

    print("\n" + "=" * 74)
    print("6. spontaneous <think> on a bare prompt is caught and compressed")
    print("=" * 74)
    # The prompt opens no think block, but this model emits one on its own for
    # open-ended requests. Because phase 1 always stops at </think>, that is now
    # detected from the OUTPUT rather than assumed absent from the prompt.
    r = post(url, {"model": a.model, "prompt": "Write a long essay about clouds.",
                   "temperature": 0.6, "top_p": 0.95, "seed": 99,
                   "max_tokens": 2048, "cot_arm": "self", "cot_ratio": 0.3})
    cc = r["cot_compression"][0]
    show(cc, "spontaneous")
    print(f"  text: {r['choices'][0]['text'][:150]!r}")
    if cc["no_think_block"]:
        print("  NOTE: model did not open a think block on this run; "
              "nothing to catch (sampling-dependent, not a failure)")
    else:
        if cc["compressed_tokens"] >= cc["reasoning_tokens"]:
            fails.append("6: spontaneous trace was not compressed")
        if cc["original_trace_in_prompt"]:
            fails.append("6: original trace leaked into the phase-2 prompt")
        if not cc["compressed_trace_in_prompt"]:
            fails.append("6: compressed trace absent from the phase-2 prompt")

    print("\n" + "=" * 74)
    for f in fails:
        print(f"FAIL  {f}")
    if not fails:
        print("ALL /v1/completions CHECKS PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

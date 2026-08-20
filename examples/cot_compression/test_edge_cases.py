#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for chat-route bugs found by audit.

  1. `stop` as a bare string was passed to list(), exploding "END" into
     ['E','N','D'] so generation stopped on the first letter E.
  2. enable_thinking=false renders a CLOSED empty think block; the route
     assumed a think block was always open and ran the compression path.
  3/4. n>1 and logprobs were silently ignored on chat (completions rejected
     them), so a pass@k harness would get one choice and mis-score.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

Q = ("Count from 1 to 8, one number per line. Then write the word END on its "
     "own line, then write MORE.")


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
    url = f"{a.sidecar}/v1/chat/completions"
    fails = []

    print("=" * 74)
    print("1. `stop` as a bare string must behave like the one-element list")
    print("=" * 74)
    base = {"model": a.model, "messages": [{"role": "user", "content": Q}],
            "temperature": 0.0, "seed": 3, "cot_arm": "identity"}
    as_str = post(url, {**base, "stop": "END"})
    as_list = post(url, {**base, "stop": ["END"]})
    t_str = as_str["choices"][0]["message"]["content"]
    t_list = as_list["choices"][0]["message"]["content"]
    print(f"  stop='END'   -> {t_str[:80]!r} ({len(t_str)} chars)")
    print(f"  stop=['END'] -> {t_list[:80]!r} ({len(t_list)} chars)")
    if t_str != t_list:
        fails.append("1: bare-string stop differs from single-element list")
    if len(t_str) < 5:
        fails.append("1: bare-string stop truncated almost immediately "
                     "(likely exploded into characters)")

    print("\n" + "=" * 74)
    print("2. enable_thinking=false -> no think block, caller max_tokens honoured")
    print("=" * 74)
    r = post(url, {"model": a.model,
                   "messages": [{"role": "user", "content": "Name three colours."}],
                   "temperature": 0.0, "seed": 3, "max_tokens": 12,
                   "chat_template_kwargs": {"enable_thinking": False}})
    cc = r["cot_compression"]
    n_out = r["usage"]["completion_tokens"]
    print(f"  no_think_block={cc['no_think_block']} "
          f"completion_tokens={n_out} finish={r['choices'][0]['finish_reason']}")
    print(f"  text: {r['choices'][0]['message']['content']!r}")
    if not cc["no_think_block"]:
        fails.append("2: enable_thinking=false was not detected as no-think")
    if n_out > 12:
        fails.append(f"2: max_tokens=12 ignored, generated {n_out} tokens")

    print("\n" + "=" * 74)
    print("3/4. unsupported chat options are rejected, not ignored")
    print("=" * 74)
    for extra in ({"n": 2}, {"logprobs": True}, {"stream": True}):
        try:
            post(url, {**base, "max_tokens": 8, **extra})
            fails.append(f"3/4: {extra} accepted but should be rejected")
            print(f"  {extra}: ACCEPTED (bad)")
        except urllib.error.HTTPError as e:
            msg = json.loads(e.read())["error"]["message"]
            print(f"  {extra}: rejected {e.code} - {msg[:60]}")

    print("\n" + "=" * 74)
    for f in fails:
        print(f"FAIL  {f}")
    if not fails:
        print("ALL EDGE-CASE CHECKS PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

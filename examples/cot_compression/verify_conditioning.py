#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove that phase 2 is conditioned on the COMPRESSED trace, not the original.

Three independent checks:

  A. Provenance   the phase-2 prompt contains the compressed trace and does NOT
                  contain the original one (read back from the sidecar, which
                  detokenizes the prompt it actually sent).
  B. Route        ASSERTED. Inject two traces that both reach the CORRECT answer
                  (187) by different routes -- multiply by 0.85, or subtract the
                  33 failures. Both are right, so the model has no reason to
                  override either; if the answer's derivation tracks whichever
                  route was injected, it was generated from the injected text.
                  Run n times because vLLM is not bit-reproducible across batch
                  compositions.
  C. Divergent    REPORTED, NOT ASSERTED. Inject a trace reaching a different
                  answer (107) via a fact absent from the question. Whether the
                  model follows or overrides is stochastic -- observed both ways
                  across runs at temperature 0.6 -- so this is a rate, not a
                  pass/fail.
"""

import argparse
import json
import re
import sys
import urllib.request

Q = ("A factory produces 20 robots per hour for 6 hours, then 25 robots per "
     "hour for 4 hours. Exactly 15% of all robots produced fail inspection. "
     "How many robots pass inspection? End your response with "
     "'ANSWER: <integer>'.")

ROUTE_MUL = (
    "Total produced = 20*6 + 25*4 = 220.\n"
    "Pass rate is 85%, so multiply: 220 * 0.85 = 187.\n"
    "ANSWER: 187"
)

ROUTE_SUB = (
    "Total produced = 20*6 + 25*4 = 220.\n"
    "Failures are 15% of 220 = 33.\n"
    "Subtract the failures: 220 - 33 = 187.\n"
    "ANSWER: 187"
)

DIVERGENT = (
    "Total produced = 20*6 + 25*4 = 220.\n"
    "Of these, 220 * 0.85 = 187 pass inspection.\n"
    "But 80 of the passing units were withdrawn before shipment.\n"
    "Net passing = 187 - 80 = 107.\n"
    "ANSWER: 107"
)


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(sidecar, model, **extra):
    return post(f"{sidecar}/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": Q}],
                 "temperature": 0.6, "top_p": 0.95, "seed": 1234, **extra})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", default="http://127.0.0.1:9000")
    ap.add_argument("--model", default="qwen38")
    a = ap.parse_args()
    fails = []

    print("=" * 74)
    print("CHECK A - provenance of the phase-2 prompt")
    print("=" * 74)
    for arm in ("identity", "self"):
        r = ask(a.sidecar, a.model, cot_arm=arm, cot_ratio=0.3)
        cc = r["cot_compression"]
        orig, comp = cc["original_trace_in_prompt"], cc["compressed_trace_in_prompt"]
        print(f"\n[{arm}] reasoning {cc['reasoning_tokens']} -> "
              f"{cc['compressed_tokens']} tokens")
        print(f"  original trace present in phase-2 prompt   : {orig}")
        print(f"  compressed trace present in phase-2 prompt : {comp}")
        if not comp:
            fails.append(f"A/{arm}: compressed trace absent from phase-2 prompt")
        if arm == "self" and orig:
            fails.append("A/self: ORIGINAL trace leaked into the phase-2 prompt")
        if arm == "identity" and not orig:
            fails.append("A/identity: identity arm should preserve the original")

    print("\n" + "=" * 74)
    print("CHECK B - route dependence (both injected traces are CORRECT)")
    print("=" * 74)
    N = 3
    for label, trace, marker in (("multiply", ROUTE_MUL, "33"),
                                 ("subtract", ROUTE_SUB, "33")):
        hits = 0
        for _ in range(N):
            r = ask(a.sidecar, a.model, cot_arm="inject", cot_inject=trace)
            ans = r["choices"][0]["message"]["content"]
            if marker in ans:
                hits += 1
        print(f"\n[{label} route] '33' (the subtraction step) appears in "
              f"{hits}/{N} answers")
        if label == "subtract" and hits == 0:
            fails.append("B: subtract-route trace did not produce the "
                         "subtraction step in any answer")
        if label == "multiply" and hits == N:
            fails.append("B: multiply-route trace still produced the "
                         "subtraction step every time")

    print("\n" + "=" * 74)
    print("CHECK C - divergent trace, reported as a rate")
    print("=" * 74)
    M, follow = 5, 0
    for _ in range(M):
        r = ask(a.sidecar, a.model, cot_arm="inject", cot_inject=DIVERGENT)
        ans = r["choices"][0]["message"]["content"]
        m = re.findall(r"ANSWER:\s*(-?[\d,]+)", ans)
        if m and int(m[-1].replace(",", "")) == 107:
            follow += 1
    print(f"\n  followed the injected trace (107): {follow}/{M}")
    print(f"  overrode it and recomputed  (187): {M - follow}/{M}")
    print("  (not asserted -- this is a behavioural rate, not a mechanism check)")

    print("\n" + "=" * 74)
    for f in fails:
        print(f"FAIL  {f}")
    if not fails:
        print("ALL CONDITIONING CHECKS PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

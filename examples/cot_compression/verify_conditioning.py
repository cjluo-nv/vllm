#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove that phase 2 is conditioned on the COMPRESSED trace, not the original.

Three independent checks:

  A. Provenance   the phase-2 prompt contains the compressed trace and does NOT
                  contain the original one (read back from the sidecar, which
                  detokenizes the prompt it actually sent).
  B. Route        ASSERTED. On a HARD problem (501-token trace), inject two
                  traces that both reach the CORRECT answer by different routes
                  -- block summation, or arithmetic progressions. Both are
                  right, so the model has no reason to override either; if the
                  derivation tracks whichever route was injected, the answer was
                  generated from the injected text. Run n times because vLLM is
                  not bit-reproducible across batch compositions.
                  NOTE: this test only discriminates on hard problems. On the
                  easy factory problem the model re-solves from the question and
                  ignores the injected route entirely (measured 0/3 both ways),
                  which is itself a finding: on easy problems the trace is not
                  load-bearing and compression cannot degrade anything.
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

HARD_Q = ("Let S be the set of all positive integers n with n <= 1000 such that "
          "n^2 + 1 is divisible by 5. What is the sum of all elements of S? "
          "End your response with 'ANSWER: <integer>'.")

ROUTE_BLOCK = (
    "n^2+1 = 0 mod 5 => n^2 = 4 mod 5 => n = 2,3 mod 5.\n"
    "Index k=0..199 over blocks 5k+1..5k+5. Valid: 5k+2 and 5k+3.\n"
    "Sum per block: (5k+2)+(5k+3) = 10k+5.\n"
    "Total = sum_{k=0}^{199} (10k+5) = 10*19900 + 1000 = 200000.\n"
    "ANSWER: 200000"
)

ROUTE_AP = (
    "n^2+1 = 0 mod 5 => n^2 = 4 mod 5 => n = 2,3 mod 5.\n"
    "Residue 2: the AP 2,7,...,997 has 200 terms.\n"
    "Residue 3: the AP 3,8,...,998 has 200 terms.\n"
    "Sums: 200/2*(2+997)=99900 and 200/2*(3+998)=100100. Total 200000.\n"
    "ANSWER: 200000"
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


def ask(sidecar, model, question=Q, **extra):
    return post(f"{sidecar}/v1/chat/completions",
                {"model": model,
                 "messages": [{"role": "user", "content": question}],
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
    print("CHECK B - route dependence on a HARD problem (both routes CORRECT)")
    print("=" * 74)
    N = 4
    for label, trace in (("block", ROUTE_BLOCK), (" ap  ", ROUTE_AP)):
        nb = na = 0
        for _ in range(N):
            r = ask(a.sidecar, a.model, question=HARD_Q,
                    cot_arm="inject", cot_inject=trace)
            t = r["choices"][0]["message"]["content"].replace(" ", "")
            if "10k" in t or "5k+2" in t:
                nb += 1
            if "997" in t:
                na += 1
        print(f"  injected {label} route -> answer used block {nb}/{N}, "
              f"AP {na}/{N}")
        if label == "block" and nb < N:
            fails.append(f"B: block route only tracked {nb}/{N}")
        if label.strip() == "ap" and na < N:
            fails.append(f"B: AP route only tracked {na}/{N}")

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

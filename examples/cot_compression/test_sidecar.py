#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for the CoT-compression sidecar.

Checks the mechanism (think block closes, trace shrinks, generation resumes)
and whether the model still reaches the known answer with a compressed trace.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

PROBLEMS = [
    {
        "q": "A factory produces 20 robots per hour for 6 hours, then 25 robots "
             "per hour for 4 hours. Exactly 15% of all robots produced fail "
             "inspection. How many robots pass inspection? End your response "
             "with 'ANSWER: <integer>'.",
        "a": 187,
    },
    {
        "q": "What is the remainder when 2^100 is divided by 125? End your "
             "response with 'ANSWER: <integer>'.",
        "a": 1,
    },
    {
        "q": "How many positive integers less than 1000 are divisible by 3 or 5 "
             "but not by 15? End your response with 'ANSWER: <integer>'.",
        "a": 400,
    },
    {
        "q": "Let S be the set of all positive integers n with n <= 1000 such "
             "that n^2 + 1 is divisible by 5. What is the sum of all elements "
             "of S? End your response with 'ANSWER: <integer>'.",
        "a": 200000,
    },
]


def http(url, payload=None, timeout=3600, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ping(url, timeout=10):
    """GET that only checks the status code. /health returns an empty body,
    so it must not be JSON-parsed."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def wait_for(url, label, tries=180):
    for i in range(tries):
        try:
            ping(url, timeout=10)
            print(f"[test] {label} is up", flush=True)
            return True
        except Exception:
            if i % 12 == 0:
                print(f"[test] waiting for {label} ({i * 5}s)", flush=True)
            time.sleep(5)
    print(f"[test] TIMEOUT waiting for {label}", flush=True)
    return False


def extract(text):
    m = re.findall(r"ANSWER:\s*(-?[\d,]+)", text)
    if not m:
        m = re.findall(r"(-?\d[\d,]*)", text)
    if not m:
        return None
    try:
        return int(m[-1].replace(",", ""))
    except ValueError:
        return None


def show(title, text, limit=700):
    print(f"\n----- {title} ({len(text)} chars) -----", flush=True)
    body = text if len(text) <= limit else text[:limit // 2] + \
        f"\n  ...[{len(text) - limit} chars elided]...\n" + text[-limit // 2:]
    print(body, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", default="http://127.0.0.1:9000")
    ap.add_argument("--vllm", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--arms", default="identity,truncate,self")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n", type=int, default=len(PROBLEMS))
    args = ap.parse_args()

    if not wait_for(f"{args.vllm}/health", "vllm"):
        return 1
    if not wait_for(f"{args.sidecar}/health", "sidecar"):
        return 1

    model = args.model or http(f"{args.vllm}/v1/models")["data"][0]["id"]
    print(f"[test] model={model}", flush=True)

    results = []
    for pi, prob in enumerate(PROBLEMS[:args.n]):
        print("\n" + "=" * 78, flush=True)
        print(f"PROBLEM {pi + 1}: {prob['q'][:90]}...", flush=True)
        print(f"EXPECTED: {prob['a']}", flush=True)
        print("=" * 78, flush=True)
        msgs = [{"role": "user", "content": prob["q"]}]

        # ---- baseline: straight to vLLM, uncompressed reasoning ----
        t0 = time.time()
        base = http(f"{args.vllm}/v1/chat/completions",
                    {"model": model, "messages": msgs, "temperature": 0.6,
                     "top_p": 0.95, "seed": args.seed, "max_tokens": 5120})
        bmsg = base["choices"][0]["message"]
        bans = extract(bmsg.get("content") or "")
        breason = bmsg.get("reasoning_content") or ""
        print(f"\n### BASELINE (no compression)  {time.time() - t0:.1f}s", flush=True)
        print(f"  reasoning chars={len(breason)}  answer={bans}  "
              f"correct={bans == prob['a']}", flush=True)
        show("BASELINE reasoning", breason)
        show("BASELINE answer", bmsg.get("content") or "")
        results.append({"problem": pi + 1, "arm": "baseline", "correct": bans == prob["a"],
                        "answer": bans})

        # ---- sidecar arms ----
        for arm in args.arms.split(","):
            arm = arm.strip()
            t0 = time.time()
            try:
                resp = http(f"{args.sidecar}/v1/chat/completions",
                            {"model": model, "messages": msgs, "temperature": 0.6,
                             "top_p": 0.95, "seed": args.seed,
                             "cot_arm": arm, "cot_ratio": args.ratio})
            except urllib.error.HTTPError as e:
                print(f"\n### ARM {arm}: HTTP {e.code} {e.read()[:400]}", flush=True)
                results.append({"problem": pi + 1, "arm": arm, "correct": False,
                                "error": f"HTTP {e.code}"})
                continue

            msg = resp["choices"][0]["message"]
            cc = resp["cot_compression"]
            ans = extract(msg.get("content") or "")
            ok = ans == prob["a"]
            print(f"\n### ARM {arm}  {time.time() - t0:.1f}s", flush=True)
            print(f"  reasoning_tokens={cc['reasoning_tokens']} -> "
                  f"compressed_tokens={cc['compressed_tokens']} "
                  f"(achieved_ratio={cc['achieved_ratio']})", flush=True)
            print(f"  phase1_closed={cc['phase1_closed']} "
                  f"compressor_truncated={cc.get('compressor_truncated')} "
                  f"error={cc['error']}", flush=True)
            print(f"  ms={cc['ms']}", flush=True)
            print(f"  answer={ans}  correct={ok}", flush=True)
            show(f"{arm} ORIGINAL reasoning (phase 1 output)",
                 cc.get("original_reasoning") or "")
            show(f"{arm} COMPRESSED reasoning (spliced back in)",
                 cc.get("compressed_reasoning") or "")
            show(f"{arm} FINAL answer (generated from compressed trace)",
                 msg.get("content") or "")
            results.append({"problem": pi + 1, "arm": arm, "correct": ok,
                            "answer": ans,
                            "reasoning_tokens": cc["reasoning_tokens"],
                            "compressed_tokens": cc["compressed_tokens"],
                            "achieved_ratio": cc["achieved_ratio"],
                            "phase1_closed": cc["phase1_closed"],
                            "compressor_truncated": cc.get("compressor_truncated"),
                            "error": cc["error"]})

    print("\n" + "=" * 78, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 78, flush=True)
    hdr = f"{'prob':<5}{'arm':<11}{'ok':<6}{'ans':<8}{'reason':<9}{'compr':<8}{'ratio':<8}"
    print(hdr, flush=True)
    for r in results:
        print(f"{r['problem']:<5}{r['arm']:<11}{str(r['correct']):<6}"
              f"{str(r.get('answer')):<8}{str(r.get('reasoning_tokens', '-')):<9}"
              f"{str(r.get('compressed_tokens', '-')):<8}"
              f"{str(r.get('achieved_ratio', '-')):<8}", flush=True)

    # ---- mechanism checks (independent of whether the answer is right) ----
    print("\nMECHANISM CHECKS", flush=True)
    fails = []
    arms = [r for r in results if r["arm"] != "baseline"
            and "reasoning_tokens" in r]
    if not arms:
        fails.append("no sidecar arm produced a result")
    for r in arms:
        tag = f"p{r['problem']}/{r['arm']}"
        if not r["phase1_closed"]:
            fails.append(f"{tag}: phase 1 never emitted </think>")
        if r["error"]:
            fails.append(f"{tag}: compression error {r['error']}")
        if r.get("compressor_truncated"):
            fails.append(f"{tag}: compressor hit max_tokens; the compression is "
                         f"truncated, not complete")
        if r["arm"] in ("truncate", "self") and \
                r["compressed_tokens"] >= r["reasoning_tokens"]:
            fails.append(f"{tag}: trace did not shrink "
                         f"({r['reasoning_tokens']} -> {r['compressed_tokens']})")
        if r["answer"] is None:
            fails.append(f"{tag}: phase 2 produced no parseable answer")
    for f in fails:
        print(f"  FAIL  {f}", flush=True)
    if not fails:
        print("  all mechanism checks passed", flush=True)

    n_ok = sum(1 for r in results if r["correct"])
    print(f"\naccuracy: {n_ok}/{len(results)} across all arms", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

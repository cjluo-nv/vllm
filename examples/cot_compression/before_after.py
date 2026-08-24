#!/usr/bin/env python3
"""Pair the online arm's compressed trace against the uncompressed baseline.

The sidecar discards each raw chunk once it has been summarised, so a
chunk-aligned "before" is not recoverable from a run. What IS comparable is the
same PROBLEM under both arms: the baseline's full trace versus the online arm's
accumulated notes plus final verbatim chunk.

  usage: before_after.py <online_output.jsonl> <baseline_output.jsonl> [n]
"""
import json
import re
import sys

ON, BASE = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 6

online, rows = {}, 0
for line in open(ON):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    rows += 1
    rc = r.get("reasoning_content") or ""
    if "Notes from my earlier work" in rc:
        online[r.get("id")] = {
            "rc": rc, "gen": r.get("generation") or "",
            "ntok": r.get("num_generated_tokens"),
            "fr": r.get("finish_reason"),
            "correct": r.get("symbolic_correct"),
            "notes": len(re.findall(r"^\[\d+\] ", rc, re.M)),
            "problem": r.get("problem") or "",
        }

print(f"online rows so far : {rows}")
print(f"multi-chunk (compressed) : {len(online)}")
if not online:
    sys.exit(0)

ids = sorted(online, key=lambda k: -online[k]["notes"])[:60]
want = set(ids)
base = {}
for line in open(BASE):
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("id") in want:
        base[r["id"]] = {"rc": r.get("reasoning_content") or "",
                         "gen": r.get("generation") or "",
                         "ntok": r.get("num_generated_tokens"),
                         "correct": r.get("symbolic_correct")}
        if len(base) == len(want):
            break

pairs = [{"id": i, "on": online[i], "bs": base[i]} for i in ids if i in base]
print(f"matched to baseline : {len(pairs)}\n")

hdr = f"{'id':26} {'notes':>5} {'before':>10} {'after':>9} {'ratio':>7}  base/online"
print(hdr)
print("-" * len(hdr))
for e in pairs[:20]:
    o, b = e["on"], e["bs"]
    ratio = len(b["rc"]) / max(len(o["rc"]), 1)
    print(f"{e['id']:26} {o['notes']:5d} {len(b['rc']):9,}c {len(o['rc']):8,}c "
          f"{ratio:6.1f}x  {b['correct']}/{o['correct']}")

json.dump(pairs[:N], open("/tmp/ba_out.json", "w"))
print(f"\nwrote {min(N, len(pairs))} full pairs to /tmp/ba_out.json")

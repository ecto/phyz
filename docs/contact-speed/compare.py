"""Compare two contact_speed row files: hash equality, min/median us, stage us.

usage: python compare.py before.jsonl after.jsonl
"""
import json, sys
a = {json.loads(l)["scene"]: json.loads(l) for l in open(sys.argv[1])}
b = {json.loads(l)["scene"]: json.loads(l) for l in open(sys.argv[2])}
for k, r in b.items():
    o = a.get(k)
    if not o:
        continue
    same = "EXACT" if o["state_hash"] == r["state_hash"] else f"MOVED {o['state_hash']}->{r['state_hash']}"
    st = " ".join(f"{s}={o['stages_us_share_allocs'][s][0]:.1f}->{v[0]:.1f}" for s, v in r["stages_us_share_allocs"].items()
                  if abs(o['stages_us_share_allocs'][s][0] - v[0]) > 0.5)
    print(f"{k:11s} min {o['us_min']:7.1f} -> {r['us_min']:7.1f} ({o['us_min']/r['us_min']:.2f}x)  med {o['us_per_step']:7.1f} -> {r['us_per_step']:7.1f}"
          f"  allocs {o['allocs_per_step']:.0f}->{r['allocs_per_step']:.0f}  {same}\n            {st}")

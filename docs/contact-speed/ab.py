"""Interleaved A/B of contact_speed binaries on a shared machine.

Rows benched minutes apart swing +-15 % with the other lanes' load even in
thread CPU time; a small change needs its before/after measured *alternately*.
Each round runs every binary once (CS_REPS runs each, the binary reports its
own min), in rotating order; the table is the min and median over rounds of
those per-round mins. Also checks every binary reports the same state_hash
per scene.

usage: python ab.py <filter> <rounds> name=path [name=path ...]
"""
import json, os, statistics, subprocess, sys

filt, rounds = sys.argv[1], int(sys.argv[2])
bins = [a.split("=", 1) for a in sys.argv[3:]]
env = dict(os.environ, CS_REPS=os.environ.get("CS_REPS", "3"), CS_NO_ANATOMY="1")
res = {}   # (bin, scene) -> [us_min per round]
hashes = {}
for r in range(rounds):
    order = bins[r % len(bins):] + bins[:r % len(bins)]
    for name, path in order:
        out = subprocess.run([path, filt], env=env, capture_output=True, text=True).stdout
        for line in out.splitlines():
            row = json.loads(line)
            res.setdefault((name, row["scene"]), []).append(row["us_min"])
            hashes.setdefault(row["scene"], {})[name] = row["state_hash"]
scenes = sorted({s for _, s in res})
names = [n for n, _ in bins]
print("scene".ljust(12) + "".join(f"{n:>22s}" for n in names) + "   hashes")
for s in scenes:
    cells = []
    for n in names:
        v = res.get((n, s), [])
        cells.append(f"{min(v):9.1f} / {statistics.median(v):7.1f}  " if v else " " * 22)
    same = "same" if len(set(hashes[s].values())) == 1 else f"DIFFER {hashes[s]}"
    print(s.ljust(12) + "".join(cells) + "   " + same)
base = names[0]
for n in names[1:]:
    for s in scenes:
        a, b = res.get((base, s)), res.get((n, s))
        if a and b:
            print(f"{s:12s} {base}->{n}: min {min(a)/min(b):.3f}x  median {statistics.median(a)/statistics.median(b):.3f}x")

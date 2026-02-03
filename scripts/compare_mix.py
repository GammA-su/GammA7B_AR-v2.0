import json, os, subprocess, math
from collections import Counter

pack_out = os.environ["PACK_OUT"]
targets = {
  "filtered_web": 0.30,
  "books_longform": 0.18,
  "reference": 0.08,
  "forums_qa": 0.10,
  "code_docs": 0.18,
  "math": 0.12,
  "tutorials_notebooks": 0.02,
  "paper_teasers": 0.02,
}

# count
cmd = f"zstdcat '{pack_out}/index.jsonl.zst' | jq -r '.domain'"
out = subprocess.check_output(cmd, shell=True, text=True)
doms = out.splitlines()
c = Counter(doms)
n = sum(c.values())

print(f"rows={n}")
for k in sorted(targets):
    obs = c.get(k, 0) / n
    tgt = targets[k]
    print(f"{k:20s} obs={obs:6.3f}  tgt={tgt:6.3f}  delta={obs-tgt:+6.3f}")

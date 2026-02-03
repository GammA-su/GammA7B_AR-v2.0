#!/usr/bin/env python3
"""Estimate per-domain yield from split jsonl.zst files.

Yield proxy: average text length per doc (chars) for each domain.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import zstandard as zstd

try:
    import orjson as _orjson  # type: ignore

    def loads(b: bytes):
        return _orjson.loads(b)

except Exception:
    def loads(b: bytes):
        return json.loads(b)


def iter_lines_zst_bytes(path: str, chunk_size: int = 8 << 20):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        with dctx.stream_reader(f) as r:
            buf = b""
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                buf += chunk
                lines = buf.split(b"\n")
                buf = lines.pop()
                for line in lines:
                    if line:
                        yield line
            if buf.strip():
                yield buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True, help="dir containing <domain>.jsonl.zst")
    ap.add_argument("--out", required=True, help="write yield json here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-lines-per-domain", type=int, default=20000)
    ap.add_argument("--min-docs", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=5000)
    args = ap.parse_args()

    per_dom_doc = defaultdict(lambda: defaultdict(int))

    files = []
    for fn in os.listdir(args.split_dir):
        if fn.endswith(".jsonl.zst"):
            dom = fn[: -len(".jsonl.zst")]
            files.append((dom, os.path.join(args.split_dir, fn)))
    files.sort()

    for dom, path in files:
        docs = per_dom_doc[dom]
        for i, line in enumerate(iter_lines_zst_bytes(path), start=1):
            if i > args.max_lines_per_domain:
                break
            try:
                obj = loads(line)
            except Exception:
                continue
            doc_id = obj.get("doc_id") or obj.get("id") or obj.get("_id") or f"line_{i}"
            if "text" in obj and isinstance(obj["text"], str):
                toks_proxy = len(obj["text"])
            else:
                toks_proxy = len(line)
            docs[doc_id] += toks_proxy
            if i % args.log_every == 0:
                print(f"[{dom}] lines={i} docs={len(docs)}", file=sys.stderr)
            if len(docs) >= args.min_docs:
                break

    out = {}
    for dom, doc_map in per_dom_doc.items():
        vals = list(doc_map.values())
        if not vals:
            out[dom] = {"docs": 0, "avg_yield": 0.0}
            continue
        out[dom] = {"docs": len(vals), "avg_yield": sum(vals) / len(vals)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"Wrote: {args.out}")
    for dom in sorted(out):
        print(f"{dom:20s} docs={out[dom]['docs']:6d} avg_yield={out[dom]['avg_yield']:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()

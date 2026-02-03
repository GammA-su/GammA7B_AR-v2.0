#!/usr/bin/env python3
"""
Split a (potentially huge) deduped.jsonl.zst into per-domain jsonl.zst files.

Fast path:
- Operate on bytes throughout (no TextIOWrapper decode/encode).
- Parse domain via orjson.loads(bytes) when available; write original bytes line.

Optional ultra-fast path (less safe):
- --skip-parse-regex extracts "domain":"..." via regex on bytes, without JSON parsing.

Designed for TB-scale streaming.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, Optional, Tuple

import zstandard as zstd

try:
    import orjson as _orjson  # type: ignore

    def loads(b: bytes):
        return _orjson.loads(b)

except Exception:
    import json as _json

    def loads(b: bytes):
        # json.loads can accept bytes; it will decode as UTF-8 internally
        return _json.loads(b)

_DOMAIN_RE = re.compile(rb'"domain"\s*:\s*"([^"]+)"')


def _safe_name(s: str) -> str:
    # filesystem-safe; keeps alnum, dot, dash, underscore; collapses others to "_"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _open_writer(
    out_dir: str,
    domain: str,
    level: int,
    threads: int,
    write_size: int,
    use_tmp: bool,
) -> Tuple[object, object, str]:
    dom = _safe_name(domain) or "unknown"
    final_path = os.path.join(out_dir, f"{dom}.jsonl.zst")
    path = final_path + ".tmp" if use_tmp else final_path

    # Buffered binary file handle; OS buffering + zstd internal buffering.
    f = open(path, "wb")

    # threads: -1 means "all cores" in zstandard python bindings
    cctx = zstd.ZstdCompressor(level=level, threads=threads)
    # write_size increases internal buffer to reduce syscall overhead
    w = cctx.stream_writer(f, write_size=write_size)
    return f, w, final_path


def _log_progress(seen: int, bad: int, started: float, comp_pos: int, total_comp_bytes: int, counts: Counter):
    elapsed = max(1e-9, time.time() - started)
    pct = (comp_pos / total_comp_bytes) * 100.0 if total_comp_bytes else 0.0
    rate = seen / elapsed
    eta = (elapsed * (100.0 / max(pct, 1e-6))) - elapsed if pct > 0 else 0.0
    top = counts.most_common(3)
    print(
        f"PROGRESS seen={seen} bad={bad} pct~={pct:.1f}% rate={rate:.1f}/s ETA~={eta:.0f}s top={top}",
        file=sys.stderr,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup-zst", required=True, help="Path to deduped.jsonl.zst")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-domain files")
    ap.add_argument("--log-every", type=int, default=200000, help="Log progress every N lines")
    ap.add_argument("--compression-level", type=int, default=1, help="zstd compression level for outputs (speed: 1)")
    ap.add_argument("--threads", type=int, default=-1, help="zstd compressor threads (-1 = all cores)")
    ap.add_argument("--write-size", type=int, default=1 << 20, help="zstd stream_writer buffer size (bytes)")
    ap.add_argument("--domains", default="", help="Comma-separated whitelist of domains to split (others skipped)")
    ap.add_argument(
        "--skip-parse-regex",
        action="store_true",
        help='Unsafe-fast: extract domain using regex on bytes without JSON parsing',
    )
    ap.add_argument(
        "--use-tmp",
        action="store_true",
        help="Write to *.tmp then rename to final to reduce partial-file risk on crash",
    )
    args = ap.parse_args()

    dedup_zst = args.dedup_zst
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    total_comp_bytes = os.path.getsize(dedup_zst)

    allow = None
    if args.domains.strip():
        allow = {d.strip() for d in args.domains.split(",") if d.strip()}

    writers: Dict[str, Tuple[object, object, str]] = {}
    counts: Counter = Counter()
    seen = 0
    bad = 0
    started = time.time()

    def get_writer(domain: str):
        if domain not in writers:
            writers[domain] = _open_writer(
                out_dir=out_dir,
                domain=domain,
                level=args.compression_level,
                threads=args.threads,
                write_size=args.write_size,
                use_tmp=args.use_tmp,
            )
        return writers[domain][1]  # stream writer

    dctx = zstd.ZstdDecompressor()
    with open(dedup_zst, "rb") as f:
        with dctx.stream_reader(f) as r:
            buf = b""
            # Larger chunks reduce overhead; tune as needed.
            while True:
                chunk = r.read(8 << 20)  # 8 MiB
                if not chunk:
                    break
                buf += chunk
                # Split by newline; keep last partial line in buf.
                lines = buf.split(b"\n")
                buf = lines.pop()
                for line in lines:
                    if not line:
                        continue
                    seen += 1
                    domain: Optional[str] = None

                    if args.skip_parse_regex:
                        m = _DOMAIN_RE.search(line)
                        if m:
                            try:
                                domain = m.group(1).decode("utf-8", errors="replace")
                            except Exception:
                                domain = "unknown"
                        else:
                            bad += 1
                            continue
                    else:
                        try:
                            obj = loads(line)
                            domain = obj.get("domain") if isinstance(obj, dict) else None
                            if domain is None:
                                domain = "unknown"
                        except Exception:
                            bad += 1
                            if bad <= 5:
                                print("WARN: bad json line", file=sys.stderr)
                            continue

                    if allow is not None and domain not in allow:
                        continue

                    w = get_writer(domain)
                    # write original bytes line (+ newline) to avoid re-encoding costs
                    w.write(line + b"\n")
                    counts[domain] += 1

                    if args.log_every > 0 and (seen % args.log_every == 0):
                        _log_progress(seen, bad, started, f.tell(), total_comp_bytes, counts)

            # Handle any remainder (last line w/o newline)
            if buf.strip():
                seen += 1
                line = buf
                domain = None
                if args.skip_parse_regex:
                    m = _DOMAIN_RE.search(line)
                    if m:
                        domain = m.group(1).decode("utf-8", errors="replace")
                    else:
                        bad += 1
                        domain = None
                else:
                    try:
                        obj = loads(line)
                        domain = obj.get("domain") if isinstance(obj, dict) else None
                    except Exception:
                        bad += 1
                        domain = None
                if domain is None:
                    domain = "unknown"
                if allow is None or domain in allow:
                    w = get_writer(domain)
                    w.write(line + b"\n")
                    counts[domain] += 1

    # Close writers and optionally rename tmp -> final
    for domain, (fh, w, final_path) in writers.items():
        try:
            w.flush(zstd.FLUSH_FRAME)
        except Exception:
            pass
        try:
            w.close()
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass
        if args.use_tmp:
            tmp_path = final_path + ".tmp"
            if os.path.exists(tmp_path):
                os.replace(tmp_path, final_path)

    top = counts.most_common(50)
    print("DONE. Top domains by count:", file=sys.stderr)
    for dom, n in top:
        print(f"{n:10d}  {dom}")
    print(f"TOTAL seen={seen} bad={bad} unique_domains={len(counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()

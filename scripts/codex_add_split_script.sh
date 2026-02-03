set -euo pipefail

REPO="/home/alpahos/1to/GammA7B_AR-v2.0"
cd "$REPO"

mkdir -p scripts

cat <<'PY' > scripts/32_split_dedup_by_domain.py
#!/usr/bin/env python3
"""
Split a monolithic deduped.jsonl.zst into per-domain jsonl.zst streams.

Design goals:
- streaming (O(1) RAM aside from writer cache)
- fast: bytes pipeline (no utf-8 decode/encode), zstd threads, big output buffers
- robust: fast-path regex extraction of "domain", safe fallback to orjson/json if needed
- optional allowlist of domains
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, Optional, Tuple

import zstandard as zstd

try:
    import orjson as _orjson  # type: ignore
    def _loads(b: bytes):
        return _orjson.loads(b)
except Exception:
    import json as _json
    def _loads(b: bytes):
        return _json.loads(b)

# Fast-path: extract domain from bytes without full JSON parse.
# Matches: "domain":"books_longform" (no whitespace assumptions beyond permissive \s*)
_DOMAIN_RE = re.compile(rb'"domain"\s*:\s*"([^"]+)"')

def _extract_domain_fast(line: bytes) -> Optional[str]:
    m = _DOMAIN_RE.search(line)
    if not m:
        return None
    try:
        return m.group(1).decode("utf-8", errors="replace")
    except Exception:
        return None


def _human(n: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}PiB"


class DomainWriters:
    def __init__(self, out_dir: str, level: int, threads: int, buf_bytes: int):
        self.out_dir = out_dir
        self.level = level
        self.threads = threads
        self.buf_bytes = buf_bytes
        self._writers: Dict[str, Tuple[io.BufferedWriter, zstd.ZstdCompressionWriter]] = {}

    def get(self, domain: str) -> zstd.ZstdCompressionWriter:
        w = self._writers.get(domain)
        if w is not None:
            return w[1]

        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{domain}.jsonl.zst")

        # Fresh file is faster; for safety use "wb" (caller can choose to reuse directory).
        raw_f = open(path, "wb")

        # Big buffered writer reduces syscall overhead significantly.
        buf_f = io.BufferedWriter(raw_f, buffer_size=self.buf_bytes)

        cctx = zstd.ZstdCompressor(level=self.level, threads=self.threads)
        zw = cctx.stream_writer(buf_f)

        self._writers[domain] = (buf_f, zw)
        return zw

    def close(self) -> None:
        for dom, (buf_f, zw) in self._writers.items():
            try:
                zw.flush(zstd.FLUSH_FRAME)
            except Exception:
                pass
            try:
                zw.close()
            except Exception:
                pass
            try:
                buf_f.flush()
            except Exception:
                pass
            try:
                buf_f.close()
            except Exception:
                pass
        self._writers.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup-zst", required=True, help="Path to deduped.jsonl.zst")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-domain .jsonl.zst files")
    ap.add_argument("--log-every", type=int, default=200000, help="Log progress every N lines")
    ap.add_argument("--level", type=int, default=6, help="Zstd compression level for outputs")
    ap.add_argument("--threads", type=int, default=-1, help="Zstd threads for output compression (-1 = all cores)")
    ap.add_argument("--buf-bytes", type=int, default=8 * 1024 * 1024, help="Buffered writer size per domain")
    ap.add_argument("--domains", default="", help="Comma-separated allowlist of domains to keep (empty = all)")
    ap.add_argument("--fast-domain", action="store_true", help="Use regex fast-path for domain extraction (default on)")
    ap.add_argument("--no-fast-domain", dest="fast_domain", action="store_false", help="Disable regex fast-path")
    ap.set_defaults(fast_domain=True)
    args = ap.parse_args()

    dedup_zst = args.dedup_zst
    out_dir = args.out_dir
    log_every = max(1, int(args.log_every))

    allow = None
    if args.domains.strip():
        allow = set([d.strip() for d in args.domains.split(",") if d.strip()])

    total_comp_bytes = os.path.getsize(dedup_zst)

    writers = DomainWriters(out_dir=out_dir, level=int(args.level), threads=int(args.threads), buf_bytes=int(args.buf_bytes))
    counts = Counter()

    seen = 0
    bad = 0
    started = time.time()
    last_log = started

    dctx = zstd.ZstdDecompressor()
    # Streaming bytes reader; we iterate over lines in bytes via a buffered wrapper.
    with open(dedup_zst, "rb") as f:
        with dctx.stream_reader(f) as r:
            br = io.BufferedReader(r, buffer_size=8 * 1024 * 1024)
            while True:
                line = br.readline()
                if not line:
                    break
                if line == b"\n":
                    continue

                seen += 1
                # Strip trailing newline once; keep bytes otherwise.
                if line.endswith(b"\n"):
                    line = line[:-1]

                dom = None
                if args.fast_domain:
                    dom = _extract_domain_fast(line)

                if dom is None:
                    # Safe fallback: parse JSON (slower).
                    try:
                        obj = _loads(line)
                        dom = obj.get("domain") or "unknown"
                    except Exception:
                        bad += 1
                        if bad <= 5:
                            print("WARN: bad json line", file=sys.stderr)
                        continue

                if allow is not None and dom not in allow:
                    continue

                try:
                    writers.get(dom).write(line + b"\n")
                except Exception as e:
                    raise RuntimeError(f"write failed for domain={dom}: {e}") from e

                counts[dom] += 1

                if seen % log_every == 0:
                    now = time.time()
                    elapsed = max(1e-9, now - started)
                    rate = seen / elapsed

                    # Approximate compressed position via underlying file handle.
                    # For zstd stream_reader, f.tell() is a reasonable proxy for compressed bytes consumed.
                    comp_pos = f.tell()
                    pct = (comp_pos / total_comp_bytes) * 100.0 if total_comp_bytes else 0.0
                    eta = (elapsed * (100.0 / max(pct, 1e-6))) - elapsed if pct > 0 else float("inf")

                    top = counts.most_common(5)
                    dt = now - last_log
                    last_log = now
                    print(
                        f"PROGRESS seen={seen} bad={bad} "
                        f"comp={_human(comp_pos)}/{_human(total_comp_bytes)} pct~={pct:.1f}% "
                        f"rate={rate:.1f}/s (dt={dt:.1f}s) ETA~={eta:.0f}s top={top}",
                        file=sys.stderr,
                    )

    writers.close()

    print("DONE. Top domains by count:", file=sys.stderr)
    for dom, n in counts.most_common(50):
        print(f"{n:10d}  {dom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

chmod +x scripts/32_split_dedup_by_domain.py

echo "OK: wrote scripts/32_split_dedup_by_domain.py"

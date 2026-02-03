#!/usr/bin/env python3
import argparse
import io
import json
from collections import Counter

import zstandard as zstd

DOMAIN_ALIASES = {
    "code": "code_docs",
    "the_stack_v2": "code_docs",
    "the-stack": "code_docs",
    "stack": "code_docs",
    "wiki": "reference",
    "wikipedia": "reference",
    "encyclopedia": "reference",
    "books": "books_longform",
    "longform": "books_longform",
    "web": "filtered_web",
    "filteredweb": "filtered_web",
    "papers": "paper_teasers",
    "paper": "paper_teasers",
    "paper_preview": "paper_teasers",
    "tutorials": "tutorials_notebooks",
    "notebooks": "tutorials_notebooks",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup-zst", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--log-every", type=int, default=200000)
    ap.add_argument("--max-logs", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    raw_cnt = Counter()
    raw_bytes = Counter()
    canon_cnt = Counter()
    canon_bytes = Counter()

    dctx = zstd.ZstdDecompressor()
    seen = 0
    skipped = 0
    print("START", args.dedup_zst)
    print("LOG_EVERY", args.log_every)
    if args.max_logs:
        print("MAX_LOGS", args.max_logs)
    if args.limit:
        print("LIMIT", args.limit)
    with open(args.dedup_zst, "rb") as f, dctx.stream_reader(f) as r:
        t = io.TextIOWrapper(r, encoding="utf-8")
        for i, line in enumerate(t):
            if args.limit and i >= args.limit:
                break
            try:
                obj = json.loads(line)
            except Exception:
                skipped += 1
                continue
            dom = (obj.get("domain") or "").strip()
            txt = obj.get("text") or ""
            b = len(txt.encode("utf-8", "replace"))
            raw_cnt[dom] += 1
            raw_bytes[dom] += b
            cdom = DOMAIN_ALIASES.get(dom, dom)
            canon_cnt[cdom] += 1
            canon_bytes[cdom] += b
            seen += 1
            if args.log_every and seen % args.log_every == 0:
                top_raw = raw_bytes.most_common(5)
                top_canon = canon_bytes.most_common(5)
                try:
                    print(f"PROGRESS seen={seen} skipped={skipped}")
                    print("TOP_RAW_BYTES", top_raw)
                    print("TOP_CANON_BYTES", top_canon)
                except BrokenPipeError:
                    return
                if args.max_logs:
                    args.max_logs -= 1
                    if args.max_logs <= 0:
                        return

    try:
        print("DONE seen", seen, "skipped", skipped)
    except BrokenPipeError:
        return
    print("== raw domains by bytes ==")
    for d, b in raw_bytes.most_common(50):
        try:
            print(f"{b:12d}  {raw_cnt[d]:9d}  {d}")
        except BrokenPipeError:
            return

    print("\n== canonical domains by bytes ==")
    for d, b in canon_bytes.most_common(50):
        try:
            print(f"{b:12d}  {canon_cnt[d]:9d}  {d}")
        except BrokenPipeError:
            return


if __name__ == "__main__":
    main()

import argparse, os, json, hashlib, time
from datetime import datetime, timezone

import zstandard as zstd
from datasets import load_dataset

def zst_writer(path: str, level: int, threads: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "wb")
    cctx = zstd.ZstdCompressor(level=level, threads=threads)
    w = cctx.stream_writer(f)
    return w, f

def main():
    ap = argparse.ArgumentParser(description="Export wikimedia/wikipedia HF -> normalized jsonl.zst shards (repo-style).")
    ap.add_argument("--config", required=True, help="e.g. 20231101.en")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", default="data/normalized/reference/wikipedia")
    ap.add_argument("--max-docs", type=int, default=200000)
    ap.add_argument("--part-size", type=int, default=20000)
    ap.add_argument("--zstd-level", type=int, default=3)
    ap.add_argument("--zstd-threads", type=int, default=0, help="0=auto/all cores")
    ap.add_argument("--log-every", type=int, default=2000)
    args = ap.parse_args()

    ds = load_dataset(
        "wikimedia/wikipedia",
        args.config,
        split=args.split,
        streaming=True,
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    part = 0
    wrote_in_part = 0
    total = 0
    started = time.time()

    out_path = os.path.join(out_dir, f"part_{part:05d}.jsonl.zst")
    w, f = zst_writer(out_path, args.zstd_level, args.zstd_threads)

    for row in ds:
        if total >= args.max_docs:
            break

        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        rid = row.get("id", None)
        if rid is None:
            rid = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

        obj = {
            "text": text,
            "id": str(rid),
            "doc_id": str(rid),
            "source": "wikipedia",
            "domain": "reference",
            "created_at": now_iso,
            "meta": {
                "title": row.get("title", ""),
                "url": row.get("url", ""),
            },
        }

        w.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        total += 1
        wrote_in_part += 1

        if wrote_in_part >= args.part_size:
            w.flush(zstd.FLUSH_FRAME)
            w.close()
            f.close()
            part += 1
            wrote_in_part = 0
            out_path = os.path.join(out_dir, f"part_{part:05d}.jsonl.zst")
            w, f = zst_writer(out_path, args.zstd_level, args.zstd_threads)

        if total % args.log_every == 0:
            dt = max(1e-9, time.time() - started)
            print(f"[wikipedia] total={total} part={part} rate={total/dt:.1f}/s out={out_path}")

    w.flush(zstd.FLUSH_FRAME)
    w.close()
    f.close()
    dt = max(1e-9, time.time() - started)
    print(f"[wikipedia] done total={total} rate={total/dt:.1f}/s out_dir={out_dir}")

if __name__ == "__main__":
    main()

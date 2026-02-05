import argparse, os, json, hashlib, time
from datetime import datetime, timezone

import zstandard as zstd
from datasets import load_dataset, get_dataset_split_names

DATASET_NAME = "lvwerra/stack-exchange-paired"
SPLIT_PRIORITY = ["train", "test", "validation"]


def _safe_get_split_names(dataset_name: str):
    try:
        splits = get_dataset_split_names(dataset_name)
    except Exception:
        return []
    if not splits:
        return []
    return list(splits)


def _probe_priority_splits(dataset_name: str):
    available = []
    for split in SPLIT_PRIORITY:
        try:
            load_dataset(dataset_name, split=split, streaming=True)
        except Exception:
            continue
        available.append(split)
    return available


def _auto_select_split(dataset_name: str):
    available = _safe_get_split_names(dataset_name)

    for split in SPLIT_PRIORITY:
        try:
            load_dataset(dataset_name, split=split, streaming=True)
        except Exception:
            continue
        return split, available or [split]

    if available:
        for split in available:
            try:
                load_dataset(dataset_name, split=split, streaming=True)
            except Exception:
                continue
            return split, available

    raise SystemExit(f"[stackexchange] no available splits found for {dataset_name}")


def _validate_split(dataset_name: str, split: str):
    available = _safe_get_split_names(dataset_name)
    if available:
        if split not in available:
            print(f"[stackexchange] available splits: {available}")
            raise SystemExit(f"[stackexchange] split '{split}' not available for {dataset_name}")
    try:
        load_dataset(dataset_name, split=split, streaming=True)
    except Exception:
        if not available:
            available = _probe_priority_splits(dataset_name)
        print(f"[stackexchange] available splits: {available}")
        raise SystemExit(f"[stackexchange] split '{split}' not available for {dataset_name}")
    return available

def zst_writer(path: str, level: int, threads: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "wb")
    cctx = zstd.ZstdCompressor(level=level, threads=threads)
    w = cctx.stream_writer(f)
    return w, f

def main():
    ap = argparse.ArgumentParser(description="Export lvwerra/stack-exchange-paired HF -> normalized jsonl.zst shards.")
    ap.add_argument("--split", default="auto")
    ap.add_argument("--out-dir", default="data/normalized/forums_qa/stackexchange")
    ap.add_argument("--max-docs", type=int, default=400000)
    ap.add_argument("--part-size", type=int, default=20000)
    ap.add_argument("--zstd-level", type=int, default=3)
    ap.add_argument("--zstd-threads", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--use-k", action="store_true", help="Use response_k instead of response_j (optional second pass).")
    args = ap.parse_args()

    if args.split == "auto":
        split, available = _auto_select_split(DATASET_NAME)
    else:
        split = args.split
        available = _validate_split(DATASET_NAME, split)

    print(f"[stackexchange] using split={split} available={available}")

    ds = load_dataset(
        DATASET_NAME,
        split=split,
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

    ans_key = "response_k" if args.use_k else "response_j"

    for row in ds:
        if total >= args.max_docs:
            break

        q = row.get("question", "")
        a = row.get(ans_key, "")
        if not isinstance(q, str) or not q.strip():
            continue
        if not isinstance(a, str) or not a.strip():
            continue

        text = f"Question:\n{q}\n\nAnswer:\n{a}\n"

        qid = row.get("qid", None)
        if qid is None:
            qid = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

        obj = {
            "text": text,
            "id": str(qid),
            "doc_id": str(qid),
            "source": "stackexchange",
            "domain": "forums_qa",
            "created_at": now_iso,
            "meta": {
                "site": row.get("site", ""),
                "question_id": row.get("question_id", ""),
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
            print(f"[stackexchange] total={total} part={part} rate={total/dt:.1f}/s out={out_path}")

    w.flush(zstd.FLUSH_FRAME)
    w.close()
    f.close()
    dt = max(1e-9, time.time() - started)
    print(f"[stackexchange] done total={total} rate={total/dt:.1f}/s out_dir={out_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, glob, io, json, os, sys
import zstandard as zstd

def iter_jsonl_zst(path: str):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f, dctx.stream_reader(f) as r:
        t = io.TextIOWrapper(r, encoding="utf-8", errors="replace")
        for line in t:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl_zst(path: str, rows, level: int = 6):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cctx = zstd.ZstdCompressor(level=level)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        with cctx.stream_writer(f) as w:
            for obj in rows:
                s = json.dumps(obj, ensure_ascii=False)
                w.write((s + "\n").encode("utf-8"))
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--glob", default="part_*.jsonl.zst")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--max-chars", type=int, required=True)
    ap.add_argument("--zstd-level", type=int, default=6)
    args = ap.parse_args()

    in_paths = sorted(glob.glob(os.path.join(args.in_dir, args.glob)))
    if not in_paths:
        raise SystemExit(f"No inputs matched: {os.path.join(args.in_dir, args.glob)}")

    for inp in in_paths:
        base = os.path.basename(inp)
        outp = os.path.join(args.out_dir, base)

        def gen():
            n = 0
            for obj in iter_jsonl_zst(inp):
                txt = obj.get(args.text_key)
                if isinstance(txt, str) and len(txt) > args.max_chars:
                    obj[args.text_key] = txt[: args.max_chars]
                yield obj
                n += 1
            if n == 0:
                # still write a valid empty file
                return

        write_jsonl_zst(outp, gen(), level=args.zstd_level)
        print(f"[ok] {inp} -> {outp}", file=sys.stderr)

if __name__ == "__main__":
    main()

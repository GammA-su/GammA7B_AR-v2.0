import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple

import datasets

from gamma7b_data.schema import NormalizedDocument
from gamma7b_data.utils import ensure_dir, open_zst_writer, stable_hash


def _chunk_text(text: str, chunk_chars: int, overlap: int) -> Iterator[Tuple[int, int, str]]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_chars")
    step = chunk_chars - overlap
    idx = 0
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_chars]
        if not chunk:
            break
        yield idx, start, chunk
        idx += 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream HF dataset to local normalized jsonl.zst")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", dest="config_name")
    parser.add_argument("--config-name", dest="config_name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--domain", default="filtered_web")
    parser.add_argument("--source", default=None)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--chunk-chars", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.max_chars is not None and args.max_chars < 0:
        parser.error("--max-chars must be >= 0")
    if args.chunk_chars is not None:
        if args.chunk_chars <= 0:
            parser.error("--chunk-chars must be > 0")
        if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_chars:
            parser.error("--chunk-overlap must be >= 0 and < chunk-chars")

    out_path = Path(args.out_path)
    ensure_dir(out_path.parent)

    source = args.source or args.dataset.split("/")[-1]

    def _load(split: str):
        if args.config_name:
            return datasets.load_dataset(args.dataset, args.config_name, split=split, streaming=True)
        return datasets.load_dataset(args.dataset, split=split, streaming=True)

    try:
        ds = _load(args.split)
    except Exception:
        if args.config_name:
            splits = datasets.get_dataset_split_names(args.dataset, args.config_name)
        else:
            splits = datasets.get_dataset_split_names(args.dataset)
        fallback = None
        for candidate in ("train", "test"):
            if candidate in splits:
                fallback = candidate
                break
        if fallback is None:
            fallback = splits[0] if splits else args.split
        if fallback != args.split:
            print(
                f"[hf_to_local_normalized] Split '{args.split}' not found; using '{fallback}'.",
                file=sys.stderr,
            )
        ds = _load(fallback)

    with open_zst_writer(out_path) as writer:
        for idx, row in enumerate(ds):
            text = row.get(args.text_field) or ""
            if args.max_chars is not None:
                text = text[: args.max_chars]
            base_id = row.get(args.id_field) or stable_hash(f"{args.dataset}:{idx}:{text}")
            created_at = row.get("timestamp") or row.get("date")

            if args.chunk_chars:
                for chunk_idx, start, chunk in _chunk_text(text, args.chunk_chars, args.chunk_overlap):
                    doc = NormalizedDocument(
                        text=chunk,
                        source=source,
                        domain=args.domain,
                        doc_id=f"{base_id}::chunk{chunk_idx}",
                        created_at=created_at,
                        meta={"chunk_index": chunk_idx, "chunk_start": start, "chunk_overlap": args.chunk_overlap},
                    )
                    writer.write((json.dumps(doc.to_json(), ensure_ascii=False) + "\n").encode("utf-8"))
            else:
                doc = NormalizedDocument(
                    text=text,
                    source=source,
                    domain=args.domain,
                    doc_id=str(base_id),
                    created_at=created_at,
                )
                writer.write((json.dumps(doc.to_json(), ensure_ascii=False) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()

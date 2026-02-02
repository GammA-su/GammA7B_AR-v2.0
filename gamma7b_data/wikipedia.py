import json
from pathlib import Path
from typing import Iterator, Optional

from .schema import NormalizedDocument
from .utils import open_zst_writer, read_jsonl, stable_hash


def _iter_wikiextractor_lines(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield payload


def _iter_input(path: Path) -> Iterator[dict]:
    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.is_dir():
                continue
            yield from _iter_wikiextractor_lines(file)
    else:
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.zst"):
            yield from read_jsonl(path)
        else:
            yield from _iter_wikiextractor_lines(path)


def ingest_wikipedia(
    input_path: Path,
    out_path: Path,
    limit: Optional[int] = None,
    domain: str = "reference",
    source: str = "wikipedia",
    license_tag: str = "CC-BY-SA",
    logger=None,
    log_every: int = 300,
) -> None:
    count = 0
    if logger:
        logger.info("Wikipedia ingest config: input=%s limit=%s", input_path, limit)
    with open_zst_writer(out_path) as writer:
        for record in _iter_input(input_path):
            text = record.get("text") or ""
            title = record.get("title") or ""
            if title and text:
                text = f"{title}\n\n{text}"
            doc_id = record.get("id") or record.get("doc_id") or stable_hash(f"{source}:{title}:{text}")
            created_at = record.get("created_at") or record.get("timestamp")
            meta = {"title": title} if title else None
            doc = NormalizedDocument(
                text=text,
                source=source,
                domain=domain,
                doc_id=str(doc_id),
                license_tag=license_tag,
                created_at=created_at,
                meta=meta,
            )
            writer.write((json.dumps(doc.to_json(), ensure_ascii=False) + "\n").encode("utf-8"))
            count += 1
            if logger and count % log_every == 0:
                logger.info("Wikipedia ingest progress: docs=%s", count)
            if limit is not None and count >= limit:
                break
    if logger:
        logger.info("Wikipedia ingest done: docs=%s out=%s", count, out_path)

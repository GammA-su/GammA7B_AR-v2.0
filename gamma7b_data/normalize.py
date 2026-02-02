from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .filters import apply_filters, build_default_filters
from .manifest import ShardManifest
from .schema import NormalizedDocument
from .utils import estimate_tokens, iter_files, open_zst_writer, read_jsonl, stable_hash, stable_shard_name


def _stable_doc_id(source: str, text: str, fallback: str = "") -> str:
    seed = f"{source}:{fallback}:{text}"
    return stable_hash(seed)


def normalize_record(
    record: Dict,
    domain: str,
    source: str,
    license_tag: Optional[str] = None,
) -> NormalizedDocument:
    text = record.get("text") or record.get("content") or ""
    if not text and "question" in record:
        q = record.get("question", {})
        a = record.get("answer", {})
        q_title = q.get("title", "")
        q_body = q.get("body", "")
        a_body = a.get("body", "")
        text = f"Q: {q_title}\n{q_body}\nA: {a_body}".strip()

    doc_id = record.get("doc_id") or record.get("id") or _stable_doc_id(source, text)
    created_at = record.get("created_at") or record.get("timestamp") or record.get("creation_date")
    if not created_at and "question" in record:
        created_at = record.get("question", {}).get("creation_date")

    resolved_license = record.get("license_tag") or record.get("license") or license_tag
    meta = record.get("meta") or {}

    for key in ("title", "url", "tags", "language"):
        if key in record and key not in meta:
            meta[key] = record[key]
    if "question" in record and "tags" in record["question"] and "tags" not in meta:
        meta["tags"] = record["question"]["tags"]

    return NormalizedDocument(
        text=text,
        source=source,
        domain=domain,
        doc_id=str(doc_id),
        license_tag=resolved_license,
        created_at=created_at,
        meta=meta or None,
    )


def normalize_files(
    inputs: List[Path],
    out_dir: Path,
    domain: str,
    source: str,
    seed: int,
    shard_size: int = 5000,
    license_tag: Optional[str] = None,
    use_language_heuristic: bool = True,
    chars_per_token: float = 4.0,
    logger=None,
    log_every: int = 300,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    filters = build_default_filters(use_language_heuristic=use_language_heuristic)

    shard_paths: List[Path] = []
    shard_idx = 0
    buffer: List[str] = []
    manifest = ShardManifest(
        shard_path=str(out_dir / stable_shard_name("shard", shard_idx)),
        domain=domain,
        source=source,
        seed=seed,
    )

    def flush() -> None:
        nonlocal shard_idx, buffer, manifest
        if not buffer:
            return
        shard_path = out_dir / stable_shard_name("shard", shard_idx)
        manifest.shard_path = str(shard_path)
        with open_zst_writer(shard_path) as writer:
            for item in buffer:
                writer.write((item + "\n").encode("utf-8"))
        manifest_path = shard_path.with_suffix(".manifest.json")
        manifest.write(manifest_path)
        shard_paths.append(shard_path)
        buffer = []
        shard_idx += 1
        manifest = ShardManifest(
            shard_path=str(out_dir / stable_shard_name("shard", shard_idx)),
            domain=domain,
            source=source,
            seed=seed,
        )
        if logger:
            logger.info("Normalized shard written: %s", shard_path)

    processed = 0
    kept = 0
    dropped = 0
    if logger:
        logger.info("Normalize config: shard_size=%s chars_per_token=%.2f", shard_size, chars_per_token)
    for path in iter_files(inputs):
        if logger:
            logger.info("Normalize reading: %s", path)
        file_processed = 0
        file_kept = 0
        file_dropped = 0
        for record in read_jsonl(path):
            doc = normalize_record(record, domain=domain, source=source, license_tag=license_tag)
            keep, reason = apply_filters(doc.text, filters)
            if not keep:
                manifest.add_drop(reason or "filtered")
                processed += 1
                dropped += 1
                file_processed += 1
                file_dropped += 1
                if logger and processed % log_every == 0:
                    logger.info(
                        "Normalize progress: processed=%s kept=%s dropped=%s",
                        processed,
                        kept,
                        dropped,
                    )
                continue
            est_tokens = estimate_tokens(doc.text, chars_per_token=chars_per_token)
            manifest.add_doc(doc.text, est_tokens)
            buffer.append(json_dumps(doc.to_json()))
            processed += 1
            kept += 1
            file_processed += 1
            file_kept += 1
            if logger and processed % log_every == 0:
                logger.info(
                    "Normalize progress: processed=%s kept=%s dropped=%s",
                    processed,
                    kept,
                    dropped,
                )
            if len(buffer) >= shard_size:
                flush()
        if logger:
            logger.info(
                "Normalize file done: %s processed=%s kept=%s dropped=%s",
                path,
                file_processed,
                file_kept,
                file_dropped,
            )
    flush()
    if logger:
        logger.info("Normalize complete: processed=%s kept=%s dropped=%s shards=%s", processed, kept, dropped, len(shard_paths))
    return shard_paths


def json_dumps(payload: Dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)

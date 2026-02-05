import datetime
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .filters import apply_filters, build_default_filters
from .manifest import ShardManifest
from .schema import NormalizedDocument
from .utils import estimate_tokens, iter_files, open_zst_writer, read_jsonl, stable_hash, stable_shard_name

try:
    import orjson  # type: ignore
except Exception:
    orjson = None


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


def _normalize_single_file(
    path: Path,
    out_dir: Path,
    domain: str,
    source: str,
    seed: int,
    shard_size: int,
    license_tag: Optional[str],
    use_language_heuristic: bool,
    chars_per_token: float,
    file_index: int,
) -> Tuple[List[Path], int, int, int]:
    filters = build_default_filters(use_language_heuristic=use_language_heuristic)
    shard_paths: List[Path] = []
    shard_idx = 0
    buffer: List[bytes] = []
    manifest = ShardManifest(
        shard_path=str(out_dir / stable_shard_name("shard", file_index * 1_000_000 + shard_idx)),
        domain=domain,
        source=source,
        seed=seed,
    )

    def flush() -> None:
        nonlocal shard_idx, buffer, manifest
        if not buffer:
            return
        shard_path = out_dir / stable_shard_name("shard", file_index * 1_000_000 + shard_idx)
        manifest.shard_path = str(shard_path)
        with open_zst_writer(shard_path) as writer:
            for item in buffer:
                writer.write(item + b"\n")
        manifest_path = shard_path.with_suffix(".manifest.json")
        manifest.write(manifest_path)
        shard_paths.append(shard_path)
        buffer = []
        shard_idx += 1
        manifest = ShardManifest(
            shard_path=str(out_dir / stable_shard_name("shard", file_index * 1_000_000 + shard_idx)),
            domain=domain,
            source=source,
            seed=seed,
        )

    processed = 0
    kept = 0
    dropped = 0
    for record in read_jsonl(path):
        doc = normalize_record(record, domain=domain, source=source, license_tag=license_tag)
        keep, reason = apply_filters(doc.text, filters)
        if not keep:
            manifest.add_drop(reason or "filtered")
            processed += 1
            dropped += 1
            continue
        est_tokens = estimate_tokens(doc.text, chars_per_token=chars_per_token)
        manifest.add_doc(doc.text, est_tokens)
        buffer.append(json_dumps(doc.to_json()))
        processed += 1
        kept += 1
        if len(buffer) >= shard_size:
            flush()
    flush()
    return shard_paths, processed, kept, dropped


def _normalize_records_chunk(
    records: List[Dict],
    domain: str,
    source: str,
    license_tag: Optional[str],
    use_language_heuristic: bool,
    chars_per_token: float,
) -> Tuple[List[Tuple[bytes, int, int]], int, int, int, Dict[str, int], int]:
    filters = build_default_filters(use_language_heuristic=use_language_heuristic)
    kept_rows: List[Tuple[bytes, int, int]] = []
    processed = 0
    kept = 0
    dropped = 0
    drops: Dict[str, int] = {}
    total_text_bytes = 0
    for record in records:
        doc = normalize_record(record, domain=domain, source=source, license_tag=license_tag)
        keep, reason = apply_filters(doc.text, filters)
        processed += 1
        if not keep:
            dropped += 1
            key = reason or "filtered"
            drops[key] = drops.get(key, 0) + 1
            continue
        est_tokens = estimate_tokens(doc.text, chars_per_token=chars_per_token)
        text_bytes = len(doc.text.encode("utf-8", errors="replace"))
        kept_rows.append((json_dumps(doc.to_json()), text_bytes, est_tokens))
        total_text_bytes += text_bytes
        kept += 1
    return kept_rows, processed, kept, dropped, drops, total_text_bytes


def _normalize_single_file_parallel(
    path: Path,
    out_dir: Path,
    domain: str,
    source: str,
    seed: int,
    shard_size: int,
    license_tag: Optional[str],
    use_language_heuristic: bool,
    chars_per_token: float,
    file_index: int,
    workers: int,
    chunk_size: int,
    logger=None,
    log_every: int = 300,
) -> Tuple[List[Path], int, int, int]:
    shard_paths: List[Path] = []
    shard_idx = 0
    buffer: List[bytes] = []
    manifest = ShardManifest(
        shard_path=str(out_dir / stable_shard_name("shard", file_index * 1_000_000 + shard_idx)),
        domain=domain,
        source=source,
        seed=seed,
    )

    def flush() -> None:
        nonlocal shard_idx, buffer, manifest
        if not buffer:
            return
        shard_path = out_dir / stable_shard_name("shard", file_index * 1_000_000 + shard_idx)
        manifest.shard_path = str(shard_path)
        with open_zst_writer(shard_path) as writer:
            for item in buffer:
                writer.write(item + b"\n")
        manifest_path = shard_path.with_suffix(".manifest.json")
        manifest.write(manifest_path)
        shard_paths.append(shard_path)
        buffer = []
        shard_idx += 1
        manifest = ShardManifest(
            shard_path=str(out_dir / stable_shard_name("shard", file_index * 1_000_000 + shard_idx)),
            domain=domain,
            source=source,
            seed=seed,
        )
        if logger:
            logger.info("Normalized shard written: %s", shard_path)

    def consume_result(result) -> None:
        nonlocal processed, kept, dropped, total_text_bytes, last_logged, last_ts
        rows, p, k, d, drops, text_bytes = result
        processed += p
        kept += k
        dropped += d
        total_text_bytes += text_bytes
        for reason, count in drops.items():
            manifest.drops[reason] = manifest.drops.get(reason, 0) + count
        for item, bytes_len, est_tokens in rows:
            manifest.counts["docs"] += 1
            manifest.counts["bytes"] += bytes_len
            manifest.counts["est_tokens"] += est_tokens
            buffer.append(item)
            if len(buffer) >= shard_size:
                flush()
        if logger and processed % log_every == 0:
            now = time.time()
            avg_bytes_per_doc = total_text_bytes / max(1, processed)
            est_total_docs = int((total_bytes * 2.0) / max(1.0, avg_bytes_per_doc))
            est_total_docs = max(processed, est_total_docs)
            rate = (processed - last_logged) / max(1e-6, now - last_ts)
            remaining = max(0, est_total_docs - processed)
            eta = int(remaining / max(1e-6, rate))
            pct = (processed / max(1, est_total_docs)) * 100.0
            last_logged = processed
            last_ts = now
            logger.info(
                "Normalize progress: processed=%s kept=%s dropped=%s pct~=%.1f%% ETA~=%s",
                processed,
                kept,
                dropped,
                pct,
                str(datetime.timedelta(seconds=eta)),
            )

    processed = 0
    kept = 0
    dropped = 0
    total_text_bytes = 0
    last_logged = 0
    last_ts = time.time()
    start_ts = time.time()

    total_bytes = path.stat().st_size
    if logger:
        logger.info(
            "Normalize single-file parallel: %s workers=%s chunk_size=%s shard_size=%s",
            path,
            workers,
            chunk_size,
            shard_size,
        )

    from concurrent.futures import ProcessPoolExecutor, as_completed

    futures = []
    pending_records: List[Dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for record in read_jsonl(path):
            pending_records.append(record)
            if len(pending_records) >= chunk_size:
                futures.append(
                    ex.submit(
                        _normalize_records_chunk,
                        pending_records,
                        domain,
                        source,
                        license_tag,
                        use_language_heuristic,
                        chars_per_token,
                    )
                )
                pending_records = []
                if len(futures) >= workers * 3:
                    done = next(as_completed(futures))
                    futures.remove(done)
                    consume_result(done.result())
        if pending_records:
            futures.append(
                ex.submit(
                    _normalize_records_chunk,
                    pending_records,
                    domain,
                    source,
                    license_tag,
                    use_language_heuristic,
                    chars_per_token,
                )
            )
        for fut in as_completed(futures):
            consume_result(fut.result())

    flush()
    if logger:
        logger.info(
            "Normalize complete (parallel single file): processed=%s kept=%s dropped=%s shards=%s",
            processed,
            kept,
            dropped,
            len(shard_paths),
        )
    return shard_paths, processed, kept, dropped


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
    workers: int = 16,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    filters = build_default_filters(use_language_heuristic=use_language_heuristic)

    shard_paths: List[Path] = []
    files = list(iter_files(inputs))
    total_bytes = sum(p.stat().st_size for p in files)
    bytes_done = 0
    start_ts = time.time()

    if workers and workers > 1 and len(files) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if logger:
            logger.info("Normalize parallel: workers=%s files=%s", workers, len(files))
        processed = 0
        kept = 0
        dropped = 0
        futures = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for file_index, path in enumerate(files):
                futures.append(
                    ex.submit(
                        _normalize_single_file,
                        path,
                        out_dir,
                        domain,
                        source,
                        seed,
                        shard_size,
                        license_tag,
                        use_language_heuristic,
                        chars_per_token,
                        file_index,
                    )
                )
            done_files = 0
            for fut in as_completed(futures):
                paths_out, p, k, d = fut.result()
                shard_paths.extend(paths_out)
                processed += p
                kept += k
                dropped += d
                done_files += 1
                if logger:
                    logger.info(
                        "Normalize file done (parallel): %s/%s processed=%s kept=%s dropped=%s",
                        done_files,
                        len(futures),
                        p,
                        k,
                        d,
                    )
                bytes_done = sum(p.stat().st_size for p in files[:done_files])
                if total_bytes > 0 and logger:
                    pct = (bytes_done / total_bytes) * 100.0
                    elapsed = max(1e-6, time.time() - start_ts)
                    eta = (elapsed / max(bytes_done, 1)) * (total_bytes - bytes_done)
                    logger.info(
                        "Normalize progress (bytes): %.1f%% ETA=%s",
                        pct,
                        str(datetime.timedelta(seconds=int(eta))),
                    )
        if logger:
            logger.info(
                "Normalize complete: processed=%s kept=%s dropped=%s shards=%s",
                processed,
                kept,
                dropped,
                len(shard_paths),
            )
        return shard_paths
    if workers and workers > 1 and len(files) == 1:
        shard_paths, processed, kept, dropped = _normalize_single_file_parallel(
            path=files[0],
            out_dir=out_dir,
            domain=domain,
            source=source,
            seed=seed,
            shard_size=shard_size,
            license_tag=license_tag,
            use_language_heuristic=use_language_heuristic,
            chars_per_token=chars_per_token,
            file_index=0,
            workers=workers,
            chunk_size=max(1000, shard_size),
            logger=logger,
            log_every=log_every,
        )
        if logger:
            logger.info(
                "Normalize complete: processed=%s kept=%s dropped=%s shards=%s",
                processed,
                kept,
                dropped,
                len(shard_paths),
            )
        return shard_paths
    shard_idx = 0
    buffer: List[bytes] = []
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
                writer.write(item + b"\n")
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
    total_text_bytes = 0
    last_logged = 0
    last_ts = time.time()
    if logger:
        logger.info("Normalize config: shard_size=%s chars_per_token=%.2f", shard_size, chars_per_token)
    for path in files:
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
            total_text_bytes += len(doc.text.encode("utf-8", errors="replace"))
            processed += 1
            kept += 1
            file_processed += 1
            file_kept += 1
            if logger and processed % log_every == 0:
                now = time.time()
                elapsed = max(1e-6, now - start_ts)
                avg_bytes_per_doc = total_text_bytes / max(1, processed)
                est_total_docs = int((total_bytes * 2.0) / max(1.0, avg_bytes_per_doc))
                est_total_docs = max(processed, est_total_docs)
                rate = (processed - last_logged) / max(1e-6, now - last_ts)
                remaining = max(0, est_total_docs - processed)
                eta = int(remaining / max(1e-6, rate))
                pct = (processed / max(1, est_total_docs)) * 100.0
                last_logged = processed
                last_ts = now
                logger.info(
                    "Normalize progress: processed=%s kept=%s dropped=%s pct~=%.1f%% ETA~=%s",
                    processed,
                    kept,
                    dropped,
                    pct,
                    str(datetime.timedelta(seconds=eta)),
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
        bytes_done += path.stat().st_size
        if total_bytes > 0 and logger:
            pct = (bytes_done / total_bytes) * 100.0
            elapsed = max(1e-6, time.time() - start_ts)
            eta = (elapsed / max(bytes_done, 1)) * (total_bytes - bytes_done)
            logger.info(
                "Normalize progress (bytes): %.1f%% ETA=%s",
                pct,
                str(datetime.timedelta(seconds=int(eta))),
            )
    flush()
    if logger:
        logger.info("Normalize complete: processed=%s kept=%s dropped=%s shards=%s", processed, kept, dropped, len(shard_paths))
    return shard_paths


def json_dumps(payload: Dict) -> bytes:
    import json

    if orjson is not None:
        return orjson.dumps(payload)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")

import datetime
import json
import os
import pickle
import shutil
import time
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import typer

from .config import expand_source_paths, load_manifest
from .dedup import Deduper, SimhashClusterer, exact_hash, normalize_for_simhash, simhash_from_normalized
from .hf_configs import filter_hf_configs
from .hf_stream import HF_SOURCE_MAP, stream_hf_dataset_to_normalized, stream_hf_to_normalized
from .manifest import PackManifest
from .normalize import normalize_files
from .packing import PackBuilder, default_created_at, load_tokenizer
from .sampling import TokenAwareSampler
from .schema import NormalizedDocument
from .stackexchange import ingest_stackexchange
from .repair_jsonl import repair_jsonl_zst
from .utils import (
    ensure_dir,
    estimate_tokens,
    get_log_every,
    initialize_runtime,
    iter_files,
    open_zst_reader,
    open_zst_writer,
    quality_score,
    read_jsonl,
    resolve_inputs,
    setup_logger,
    stable_hash,
)
from .wikipedia import ingest_wikipedia

app = typer.Typer(add_completion=False)


def _simhash_worker(args):
    text, hash_bits = args
    normalized = normalize_for_simhash(text)
    fp = simhash_from_normalized(normalized, hash_bits)
    token_count = len(normalized.split())
    return fp, len(normalized), token_count


def _compute_fps(texts: List[str], hash_bits: int, executor=None, chunksize: int = 128) -> List[tuple]:
    if executor:
        return list(executor.map(_simhash_worker, [(t, hash_bits) for t in texts], chunksize=chunksize))
    return [_simhash_worker((t, hash_bits)) for t in texts]


def _iter_records_with_fp(
    path: Path,
    hash_bits: int,
    executor=None,
    batch_size: int = 2048,
    chunksize: int = 128,
):
    records: List[Dict[str, object]] = []
    texts: List[str] = []
    for record in read_jsonl(path):
        if "doc_id" not in record and "id" in record:
            record["doc_id"] = str(record.pop("id"))
        records.append(record)
        texts.append(record.get("text", ""))
        if len(records) >= batch_size:
            fps = _compute_fps(texts, hash_bits, executor=executor, chunksize=chunksize)
            for rec, fp in zip(records, fps):
                yield rec, fp
            records = []
            texts = []
    if records:
        fps = _compute_fps(texts, hash_bits, executor=executor, chunksize=chunksize)
        for rec, fp in zip(records, fps):
            yield rec, fp


def _dedup_params_hash(params: Dict[str, object]) -> str:
    payload = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_pickle(path: Path, payload: Dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _load_pickle(path: Path) -> Dict[str, object]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _stat_jsonl_file(path: Path) -> Dict[str, object]:
    import io

    total_chars = 0
    n_docs = 0
    bytes_size = path.stat().st_size if path.exists() else 0
    empty = True
    if str(path).endswith(".zst"):
        with open_zst_reader(path) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                if not line:
                    continue
                empty = False
                n_docs += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_chars += len(str(obj.get("text", "")))
    else:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line:
                    continue
                empty = False
                n_docs += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_chars += len(str(obj.get("text", "")))
    return {"n_docs": n_docs, "total_chars": total_chars, "total_bytes": bytes_size, "empty": empty}


def _repair_worker(args):
    path, inplace, backup, zstd_level, max_bad, validate, log_every, threads = args
    out_path = path if inplace else path.with_name(path.name + ".repaired")
    return repair_jsonl_zst(
        path,
        out_path,
        inplace=inplace,
        backup=backup,
        zstd_level=zstd_level,
        max_bad=max_bad,
        validate=validate,
        log_every=log_every,
        logger=None,
        threads=threads,
    )


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose"),
    cpu_threads: int = typer.Option(16, "--cpu-threads"),
    faiss_gpu: int = typer.Option(1, "--faiss-gpu"),
    log_every: int = typer.Option(300, "--log-every"),
) -> None:
    import os

    os.environ["GAMMA7B_LOG_EVERY"] = str(max(1, int(log_every)))
    logger = setup_logger(verbose=verbose)
    initialize_runtime(logger, cpu_threads=cpu_threads, faiss_gpu_device=faiss_gpu)


@app.command("hf-configs")
def hf_configs_cmd(
    dataset: str = typer.Option(..., "--dataset"),
    lang: Optional[str] = typer.Option(None, "--lang", help="ISO 639-3 code, e.g. eng"),
    script: Optional[str] = typer.Option(None, "--script", help="Script code, e.g. Latn"),
    contains: Optional[str] = typer.Option(None, "--contains", help="Substring filter"),
    filter: Optional[str] = typer.Option(None, "--filter", help="Deprecated: use --contains"),
) -> None:
    from datasets import get_dataset_config_names

    configs = get_dataset_config_names(dataset)
    if contains is None and filter is not None:
        contains = filter
    configs = filter_hf_configs(configs, lang=lang, script=script, contains=contains)
    for cfg in configs:
        typer.echo(cfg)
    typer.echo(f"Total configs: {len(configs)}")


@app.command("hf-stream")
def hf_stream(
    source: List[str] = typer.Option(..., "--source", help="HF source key (repeatable)"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    domain: Optional[str] = typer.Option(None, "--domain"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    split: Optional[str] = typer.Option(None, "--split"),
    config_name: Optional[str] = typer.Option(None, "--config-name"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    _ = seed
    logger = setup_logger()
    log_every = get_log_every()
    logger.info("HF stream start: sources=%s log_every=%s", ",".join(source), log_every)
    for name in source:
        cfg = HF_SOURCE_MAP.get(name)
        if cfg is None:
            raise typer.BadParameter(f"Unknown HF source: {name}")
        resolved_domain = domain or cfg.get("domain", "filtered_web")
        out_path = out_dir / resolved_domain / name / "part_00000.jsonl.zst"
        ensure_dir(out_path.parent)
        logger.info("Streaming %s -> %s", name, out_path)
        stream_hf_to_normalized(
            source_name=name,
            out_path=out_path,
            domain=resolved_domain,
            source=name,
            limit=limit,
            split=split,
            config_name=config_name,
            logger=logger,
            log_every=log_every,
        )
    logger.info("HF stream done")


@app.command("repair-jsonl-zst")
def repair_jsonl_zst_cmd(
    in_path: List[Path] = typer.Option([], "--in"),
    in_glob: List[str] = typer.Option([], "--in-glob"),
    inplace: bool = typer.Option(False, "--inplace/--no-inplace"),
    backup: bool = typer.Option(True, "--backup/--no-backup"),
    zstd_level: int = typer.Option(3, "--zstd-level"),
    max_bad: int = typer.Option(1000, "--max-bad"),
    validate: bool = typer.Option(True, "--validate/--no-validate"),
    log_every: int = typer.Option(0, "--log-every"),
    threads: int = typer.Option(16, "--threads"),
    workers: int = typer.Option(16, "--workers"),
) -> None:
    import glob

    logger = setup_logger()
    targets: List[Path] = []
    for pattern in in_glob:
        targets.extend(Path(p) for p in glob.glob(pattern))
    targets.extend(in_path)
    if not targets:
        raise typer.BadParameter("No input paths provided.")
    for path in targets:
        if not path.exists():
            raise typer.BadParameter(f"Missing input path: {path}")
    if workers and workers > 1 and len(targets) > 1:
        from concurrent.futures import ProcessPoolExecutor

        max_workers = min(workers, len(targets))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            args_iter = [
                (path, inplace, backup, zstd_level, max_bad, validate, log_every, threads)
                for path in targets
            ]
            for path, result in zip(targets, executor.map(_repair_worker, args_iter)):
                logger.info(
                    "Repair %s: ok=%s bad=%s bytes=%s backup=%s",
                    path,
                    result["records_ok"],
                    result["records_bad"],
                    result["wrote_bytes"],
                    result["backup_path"],
                )
    else:
        for path in targets:
            out_path = path if inplace else path.with_name(path.name + ".repaired")
            result = repair_jsonl_zst(
                path,
                out_path,
                inplace=inplace,
                backup=backup,
                zstd_level=zstd_level,
                max_bad=max_bad,
                validate=validate,
                log_every=log_every,
                logger=logger,
                threads=threads,
            )
            logger.info(
                "Repair %s: ok=%s bad=%s bytes=%s backup=%s",
                path,
                result["records_ok"],
                result["records_bad"],
                result["wrote_bytes"],
                result["backup_path"],
            )


@app.command("ingest-wikipedia")
def ingest_wikipedia_cmd(
    input_path: Path = typer.Option(..., "--input"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    domain: str = typer.Option("reference", "--domain"),
    source: str = typer.Option("wikipedia", "--source"),
    license_tag: str = typer.Option("CC-BY-SA", "--license-tag"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    _ = seed
    logger = setup_logger()
    log_every = get_log_every()
    logger.info("Wikipedia ingest start: input=%s log_every=%s", input_path, log_every)
    out_path = out_dir / domain / source / "part_00000.jsonl.zst"
    ensure_dir(out_path.parent)
    ingest_wikipedia(
        input_path,
        out_path,
        limit=limit,
        domain=domain,
        source=source,
        license_tag=license_tag,
        logger=logger,
        log_every=log_every,
    )
    logger.info("Wikipedia ingest done: %s", out_path)


@app.command("ingest-stackexchange")
def ingest_stackexchange_cmd(
    input_path: Path = typer.Option(..., "--input"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    domain: str = typer.Option("forums_qa", "--domain"),
    source: str = typer.Option("stackexchange", "--source"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    _ = seed
    logger = setup_logger()
    log_every = get_log_every()
    logger.info("StackExchange ingest start: input=%s log_every=%s", input_path, log_every)
    out_path = out_dir / domain / source / "part_00000.jsonl.zst"
    ensure_dir(out_path.parent)
    ingest_stackexchange(
        input_path,
        out_path,
        limit=limit,
        domain=domain,
        source=source,
        logger=logger,
        log_every=log_every,
    )
    logger.info("StackExchange ingest done: %s", out_path)


@app.command("normalize")
def normalize_cmd(
    input: List[str] = typer.Option(..., "--input"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    domain: str = typer.Option(..., "--domain"),
    source: str = typer.Option(..., "--source"),
    seed: int = typer.Option(0, "--seed"),
    shard_size: int = typer.Option(5000, "--shard-size"),
    workers: int = typer.Option(16, "--workers"),
    license_tag: Optional[str] = typer.Option(None, "--license-tag"),
    no_lang_heuristic: bool = typer.Option(False, "--no-lang-heuristic"),
    chars_per_token: float = typer.Option(4.0, "--chars-per-token"),
) -> None:
    logger = setup_logger()
    log_every = get_log_every()
    logger.info("Normalize start: inputs=%s log_every=%s", len(input), log_every)
    inputs = resolve_inputs(input)
    out_path = out_dir / domain / source
    ensure_dir(out_path)
    normalize_files(
        inputs=inputs,
        out_dir=out_path,
        domain=domain,
        source=source,
        seed=seed,
        shard_size=shard_size,
        workers=workers,
        license_tag=license_tag,
        use_language_heuristic=not no_lang_heuristic,
        chars_per_token=chars_per_token,
        logger=logger,
        log_every=log_every,
    )
    logger.info("Normalize done: out_dir=%s", out_path)


@app.command("dedup")
def dedup_cmd(
    input: List[str] = typer.Option(..., "--input"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    mode: str = typer.Option("both", "--mode"),
    scope: str = typer.Option(
        "per-source",
        "--scope",
        help="Dedup only within each source; cross-source dupes remain.",
    ),
    simhash_threshold: int = typer.Option(3, "--simhash-threshold"),
    hash_bits: int = typer.Option(64, "--hash-bits"),
    seed: int = typer.Option(0, "--seed"),
    workers: int = typer.Option(16, "--workers"),
    log_every: int = typer.Option(0, "--log-every"),
    near_selection: str = typer.Option(
        "best",
        "--near-selection",
        help="Near-dup selection: best does two-pass best-in-cluster; first is single-pass and faster.",
    ),
    no_near: bool = typer.Option(False, "--no-near", help="Disable near-dup detection; exact-only dedup."),
    resume: bool = typer.Option(False, "--resume"),
    checkpoint_every_files: int = typer.Option(1, "--checkpoint-every-files"),
    checkpoint_path: Optional[Path] = typer.Option(None, "--checkpoint-path"),
) -> None:
    from typer.models import OptionInfo

    if isinstance(workers, OptionInfo):
        workers = 16
    if isinstance(log_every, OptionInfo):
        log_every = 0
    if isinstance(near_selection, OptionInfo):
        near_selection = "best"
    if isinstance(no_near, OptionInfo):
        no_near = False
    if isinstance(resume, OptionInfo):
        resume = False
    if isinstance(checkpoint_every_files, OptionInfo):
        checkpoint_every_files = 1
    if isinstance(checkpoint_path, OptionInfo):
        checkpoint_path = None
    _ = seed
    if log_every and log_every > 0:
        os.environ["GAMMA7B_LOG_EVERY"] = str(max(1, int(log_every)))
    logger = setup_logger()
    log_every = get_log_every()
    logger.info(
        "Dedup start: inputs=%s log_every=%s scope=%s near_selection=%s no_near=%s",
        len(input),
        log_every,
        scope,
        near_selection,
        no_near,
    )
    inputs = sorted(resolve_inputs(input), key=lambda p: str(p))
    filtered_inputs = []
    for p in inputs:
        if p.is_dir():
            filtered_inputs.append(p)
            continue
        name = p.name
        if name.endswith(".manifest.json") or ".manifest." in name:
            continue
        if name.endswith(".jsonl") or name.endswith(".jsonl.zst"):
            filtered_inputs.append(p)
    inputs = filtered_inputs
    if not inputs:
        raise typer.BadParameter("No valid input files found after filtering (.jsonl/.jsonl.zst).")
    bytes_done = 0
    start_ts = time.time()
    ensure_dir(out_dir)
    out_path = out_dir / "deduped.jsonl.zst"
    log_path = out_dir / "decisions.jsonl.zst"
    out_dir_resolved = out_dir.resolve()

    file_list = []
    for p in iter_files(inputs):
        name = p.name
        if name.endswith(".manifest.json") or ".manifest." in name:
            continue
        if not (name.endswith(".jsonl") or name.endswith(".jsonl.zst")):
            continue
        try:
            if p.resolve().is_relative_to(out_dir_resolved):
                continue
        except Exception:
            if str(p.resolve()).startswith(str(out_dir_resolved)):
                continue
        file_list.append(p)
    if not file_list:
        raise typer.BadParameter("No valid input files found after filtering (.jsonl/.jsonl.zst).")
    total_bytes = sum(p.stat().st_size for p in file_list)

    use_near = mode in {"near", "both"}
    use_exact = mode in {"exact", "both"}
    if mode not in {"near", "both", "exact"}:
        raise typer.BadParameter("mode must be exact, near, or both")
    if no_near:
        use_near = False
        use_exact = True
    if near_selection not in {"best", "first"}:
        raise typer.BadParameter("near-selection must be best or first")
    if scope not in {"global", "per-source"}:
        raise typer.BadParameter("scope must be global or per-source")

    deduper_global = Deduper(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
    deduper_by_source = {}

    kept = 0
    dropped = 0
    processed = 0
    if checkpoint_path is None:
        checkpoint_path = out_dir / "dedup_state.pkl"

    effective_mode = "exact" if no_near else mode
    params = {
        "mode": effective_mode,
        "scope": scope,
        "simhash_threshold": simhash_threshold,
        "hash_bits": hash_bits,
        "seed": seed,
        "near_selection": near_selection,
        "no_near": no_near,
        "inputs": [str(p) for p in file_list],
    }
    params_hash = _dedup_params_hash(params)
    stop_after = os.getenv("GAMMA7B_DEDUP_STOP_AFTER_FILES")
    stop_after = int(stop_after) if stop_after else None

    state = None
    if resume:
        if not checkpoint_path.exists():
            raise typer.BadParameter(f"Resume requested but checkpoint not found: {checkpoint_path}")
        state = _load_pickle(checkpoint_path)
        if state.get("params_hash") != params_hash:
            raise typer.BadParameter("Checkpoint params mismatch; refusing to resume.")
        logger.info("Dedup resume: phase=%s checkpoint=%s", state.get("phase"), checkpoint_path)

    if use_near and near_selection == "best":
        logger.info("Dedup mode near/both: computing best-in-cluster candidates")
        cluster_best: Dict[str, str] = {}
        cluster_best_rank: Dict[str, tuple] = {}
        cluster_counts: Dict[str, int] = {}
        exact_seen: Dict[str, str] = {}
        clusterers: Dict[str, SimhashClusterer] = {}
        pre_processed = 0
        completed_pass1 = set()

        if state and state.get("phase") in {"pass1", "pass2"}:
            cluster_best = state.get("cluster_best", {})
            cluster_best_rank = state.get("cluster_best_rank", {})
            cluster_counts = state.get("cluster_counts", {})
            exact_seen = state.get("exact_seen", {})
            clusterers = state.get("clusterers", {})
            completed_pass1 = set(state.get("completed_pass1", []))
            if state.get("phase") == "pass2":
                logger.info("Dedup resume: pass1 already complete, skipping.")
            pre_processed = state.get("pre_processed", 0)

        executor = None
        if workers and workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            executor = ProcessPoolExecutor(max_workers=workers)
        try:
            for path in file_list:
                if state and state.get("phase") == "pass2":
                    break
                if path.as_posix() in completed_pass1:
                    continue
                for record, (fp, norm_len, token_count) in _iter_records_with_fp(
                    path, hash_bits, executor=executor
                ):
                    doc = NormalizedDocument(**record)
                    if use_exact:
                        content_hash = exact_hash(doc.text)
                        if content_hash in exact_seen:
                            continue
                        exact_seen[content_hash] = doc.doc_id
                    if scope == "per-source":
                        source_key = f"{doc.domain}:{doc.source}"
                        clusterer = clusterers.setdefault(
                            source_key, SimhashClusterer(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
                        )
                        cluster_key_prefix = f"{source_key}:"
                    else:
                        clusterer = clusterers.setdefault(
                            "__global__", SimhashClusterer(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
                        )
                        cluster_key_prefix = ""
                    match = clusterer.assign_fp(doc.doc_id, fp, normalized_len=norm_len)
                    cluster_key = f"{cluster_key_prefix}{match.cluster_id}"
                    score = quality_score(doc.text, doc.meta)
                    rank = (score, token_count, str(doc.doc_id))
                    cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
                    best_rank = cluster_best_rank.get(cluster_key)
                    if best_rank is None or rank > best_rank:
                        cluster_best_rank[cluster_key] = rank
                        cluster_best[cluster_key] = doc.doc_id
                    pre_processed += 1
                    if pre_processed % log_every == 0:
                        logger.info(
                            "Dedup prepass: processed=%s clusters=%s",
                            pre_processed,
                            len(cluster_best),
                        )
                completed_pass1.add(path.as_posix())
                if checkpoint_every_files and len(completed_pass1) % checkpoint_every_files == 0:
                    _atomic_write_pickle(
                        checkpoint_path,
                        {
                            "phase": "pass1",
                            "params_hash": params_hash,
                            "cluster_best": cluster_best,
                            "cluster_best_rank": cluster_best_rank,
                            "cluster_counts": cluster_counts,
                            "exact_seen": exact_seen,
                            "clusterers": clusterers,
                            "completed_pass1": sorted(completed_pass1),
                            "pre_processed": pre_processed,
                        },
                    )
                if stop_after and len(completed_pass1) >= stop_after:
                    raise SystemExit("GAMMA7B_DEDUP_STOP_AFTER_FILES triggered (pass1).")
                bytes_done += path.stat().st_size
                if total_bytes > 0:
                    pct = (bytes_done / total_bytes) * 100.0
                    elapsed = max(1e-6, time.time() - start_ts)
                    eta = (elapsed / max(bytes_done, 1)) * (total_bytes - bytes_done)
                    logger.info(
                        "Dedup prepass progress: %.1f%% ETA=%s",
                        pct,
                        str(datetime.timedelta(seconds=int(eta))),
                    )
        finally:
            if executor:
                executor.shutdown()

        exact_seen = {}
        clusterers = {}
        completed_pass2 = set()
        if state and state.get("phase") == "pass2":
            completed_pass2 = set(state.get("completed_pass2", []))
            if completed_pass2:
                logger.info("Dedup resume: restarting pass2 from scratch for safe output.")
                completed_pass2 = set()
        _atomic_write_pickle(
            checkpoint_path,
            {
                "phase": "pass2",
                "params_hash": params_hash,
                "cluster_best": cluster_best,
                "cluster_counts": cluster_counts,
                "completed_pass1": sorted(completed_pass1),
                "completed_pass2": sorted(completed_pass2),
            },
        )
        out_tmp = out_path.with_name(out_path.name + ".tmp")
        log_tmp = log_path.with_name(log_path.name + ".tmp")
        with open_zst_writer(out_tmp) as out_writer, open_zst_writer(log_tmp) as log_writer:
            executor = None
            if workers and workers > 1:
                from concurrent.futures import ProcessPoolExecutor

                executor = ProcessPoolExecutor(max_workers=workers)
            bytes_done = 0
            start_ts = time.time()
            for path in file_list:
                logger.info("Dedup reading: %s", path)
                file_processed = 0
                file_kept = 0
                file_dropped = 0
                if path.as_posix() in completed_pass2:
                    continue
                for record, (fp, norm_len, _) in _iter_records_with_fp(
                    path, hash_bits, executor=executor
                ):
                    doc = NormalizedDocument(**record)
                    score = quality_score(doc.text, doc.meta)
                    cluster_id = None
                    cluster_key = None
                    cluster_size = None
                    match_id = None
                    distance = None
                    if use_exact:
                        content_hash = exact_hash(doc.text)
                        if content_hash in exact_seen:
                            decision_keep = False
                            reason = "exact_dup"
                            dup_of = exact_seen[content_hash]
                        else:
                            exact_seen[content_hash] = doc.doc_id
                            decision_keep = None
                            reason = None
                            dup_of = None
                    else:
                        decision_keep = None
                        reason = None
                        dup_of = None

                    if decision_keep is None:
                        if scope == "per-source":
                            source_key = f"{doc.domain}:{doc.source}"
                            clusterer = clusterers.setdefault(
                                source_key, SimhashClusterer(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
                            )
                            cluster_key_prefix = f"{source_key}:"
                        else:
                            clusterer = clusterers.setdefault(
                                "__global__", SimhashClusterer(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
                            )
                            cluster_key_prefix = ""
                        match = clusterer.assign_fp(doc.doc_id, fp, normalized_len=norm_len)
                        cluster_id = match.cluster_id
                        cluster_key = f"{cluster_key_prefix}{cluster_id}"
                        cluster_size = cluster_counts.get(cluster_key, 1)
                        match_id = match.match_id
                        distance = match.distance
                        best_id = cluster_best.get(cluster_key)
                        decision_keep = best_id == doc.doc_id
                        if decision_keep:
                            reason = "keep_best" if cluster_size and cluster_size > 1 else "keep"
                        else:
                            reason = "near_dup"
                            dup_of = best_id

                    if decision_keep:
                        out_writer.write((json.dumps(doc.to_json(), ensure_ascii=False) + "\n").encode("utf-8"))
                        kept += 1
                        file_kept += 1
                    else:
                        dropped += 1
                        file_dropped += 1
                    processed += 1
                    file_processed += 1
                    log_payload = {
                        "doc_id": doc.doc_id,
                        "source": doc.source,
                        "domain": doc.domain,
                        "keep": decision_keep,
                        "reason": reason,
                        "dup_of": dup_of,
                        "score": score,
                        "cluster_id": cluster_id,
                        "cluster_size": cluster_size,
                        "match_id": match_id,
                        "distance": distance,
                    }
                    log_writer.write((json.dumps(log_payload, ensure_ascii=False) + "\n").encode("utf-8"))
                    if processed % log_every == 0:
                        logger.info("Dedup progress: processed=%s kept=%s dropped=%s", processed, kept, dropped)
                logger.info(
                    "Dedup file done: %s processed=%s kept=%s dropped=%s",
                    path,
                    file_processed,
                    file_kept,
                    file_dropped,
                )
                completed_pass2.add(path.as_posix())
                if checkpoint_every_files and len(completed_pass2) % checkpoint_every_files == 0:
                    _atomic_write_pickle(
                        checkpoint_path,
                        {
                            "phase": "pass2",
                            "params_hash": params_hash,
                            "cluster_best": cluster_best,
                            "cluster_counts": cluster_counts,
                            "completed_pass1": sorted(completed_pass1),
                            "completed_pass2": sorted(completed_pass2),
                        },
                    )
                if stop_after and len(completed_pass2) >= stop_after:
                    raise SystemExit("GAMMA7B_DEDUP_STOP_AFTER_FILES triggered (pass2).")
                bytes_done += path.stat().st_size
                if total_bytes > 0:
                    pct = (bytes_done / total_bytes) * 100.0
                    elapsed = max(1e-6, time.time() - start_ts)
                    eta = (elapsed / max(bytes_done, 1)) * (total_bytes - bytes_done)
                    logger.info(
                        "Dedup progress (bytes): %.1f%% ETA=%s",
                        pct,
                        str(datetime.timedelta(seconds=int(eta))),
                    )
            if executor:
                executor.shutdown()
        os.replace(out_tmp, out_path)
        os.replace(log_tmp, log_path)
    elif use_near and near_selection == "first":
        completed = set()
        if state and state.get("phase") == "single":
            completed = set(state.get("completed_single", []))
        with open_zst_writer(out_path) as out_writer, open_zst_writer(log_path) as log_writer:
            clusterers: Dict[str, SimhashClusterer] = {}
            exact_seen: Dict[str, str] = {}
            cluster_counts: Dict[str, int] = {}
            bytes_done = 0
            start_ts = time.time()
            executor = None
            if workers and workers > 1:
                from concurrent.futures import ProcessPoolExecutor

                executor = ProcessPoolExecutor(max_workers=workers)
            for path in file_list:
                if path.as_posix() in completed:
                    continue
                logger.info("Dedup reading: %s", path)
                file_processed = 0
                file_kept = 0
                file_dropped = 0
                for record, (fp, norm_len, _) in _iter_records_with_fp(
                    path, hash_bits, executor=executor
                ):
                    doc = NormalizedDocument(**record)
                    score = quality_score(doc.text, doc.meta)
                    cluster_id = None
                    cluster_key = None
                    cluster_size = None
                    match_id = None
                    distance = None
                    if use_exact:
                        content_hash = exact_hash(doc.text)
                        if content_hash in exact_seen:
                            decision_keep = False
                            reason = "exact_dup"
                            dup_of = exact_seen[content_hash]
                        else:
                            exact_seen[content_hash] = doc.doc_id
                            decision_keep = None
                            reason = None
                            dup_of = None
                    else:
                        decision_keep = None
                        reason = None
                        dup_of = None

                    if decision_keep is None:
                        if scope == "per-source":
                            source_key = f"{doc.domain}:{doc.source}"
                            clusterer = clusterers.setdefault(
                                source_key, SimhashClusterer(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
                            )
                            cluster_key_prefix = f"{source_key}:"
                        else:
                            clusterer = clusterers.setdefault(
                                "__global__", SimhashClusterer(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
                            )
                            cluster_key_prefix = ""
                        match = clusterer.assign_fp(doc.doc_id, fp, normalized_len=norm_len)
                        cluster_id = match.cluster_id
                        cluster_key = f"{cluster_key_prefix}{cluster_id}"
                        match_id = match.match_id
                        distance = match.distance
                        cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
                        cluster_size = cluster_counts.get(cluster_key, 1)
                        if match_id is None and cluster_id == doc.doc_id:
                            decision_keep = True
                            reason = "keep"
                            dup_of = None
                        else:
                            decision_keep = False
                            reason = "near_dup"
                            dup_of = cluster_id

                    if decision_keep:
                        out_writer.write((json.dumps(doc.to_json(), ensure_ascii=False) + "\n").encode("utf-8"))
                        kept += 1
                        file_kept += 1
                    else:
                        dropped += 1
                        file_dropped += 1
                    processed += 1
                    file_processed += 1
                    log_payload = {
                        "doc_id": doc.doc_id,
                        "source": doc.source,
                        "domain": doc.domain,
                        "keep": decision_keep,
                        "reason": reason,
                        "dup_of": dup_of,
                        "score": score,
                        "cluster_id": cluster_id,
                        "cluster_size": cluster_size,
                        "match_id": match_id,
                        "distance": distance,
                    }
                    log_writer.write((json.dumps(log_payload, ensure_ascii=False) + "\n").encode("utf-8"))
                    if processed % log_every == 0:
                        logger.info("Dedup progress: processed=%s kept=%s dropped=%s", processed, kept, dropped)
                logger.info(
                    "Dedup file done: %s processed=%s kept=%s dropped=%s",
                    path,
                    file_processed,
                    file_kept,
                    file_dropped,
                )
                completed.add(path.as_posix())
                if checkpoint_every_files and len(completed) % checkpoint_every_files == 0:
                    _atomic_write_pickle(
                        checkpoint_path,
                        {
                            "phase": "single",
                            "params_hash": params_hash,
                            "completed_single": sorted(completed),
                        },
                    )
                if stop_after and len(completed) >= stop_after:
                    raise SystemExit("GAMMA7B_DEDUP_STOP_AFTER_FILES triggered (single).")
                bytes_done += path.stat().st_size
                if total_bytes > 0:
                    pct = (bytes_done / total_bytes) * 100.0
                    elapsed = max(1e-6, time.time() - start_ts)
                    eta = (elapsed / max(bytes_done, 1)) * (total_bytes - bytes_done)
                    logger.info(
                        "Dedup progress (bytes): %.1f%% ETA=%s",
                        pct,
                        str(datetime.timedelta(seconds=int(eta))),
                    )
            if executor:
                executor.shutdown()
    else:
        completed = set()
        if state and state.get("phase") == "single":
            completed = set(state.get("completed_single", []))
        with open_zst_writer(out_path) as out_writer, open_zst_writer(log_path) as log_writer:
            bytes_done = 0
            start_ts = time.time()
            for path in file_list:
                if path.as_posix() in completed:
                    continue
                logger.info("Dedup reading: %s", path)
                file_processed = 0
                file_kept = 0
                file_dropped = 0
                for record in read_jsonl(path):
                    if "doc_id" not in record and "id" in record:
                        record["doc_id"] = str(record.pop("id"))
                    doc = NormalizedDocument(**record)
                    if scope == "per-source":
                        source_key = f"{doc.domain}:{doc.source}"
                        deduper = deduper_by_source.setdefault(
                            source_key, Deduper(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
                        )
                    else:
                        deduper = deduper_global
                    decision = deduper.check(doc, use_near=use_near)
                    if decision.keep:
                        out_writer.write((json.dumps(doc.to_json(), ensure_ascii=False) + "\n").encode("utf-8"))
                        kept += 1
                        file_kept += 1
                    else:
                        dropped += 1
                        file_dropped += 1
                    processed += 1
                    file_processed += 1
                    log_payload = {
                        "doc_id": doc.doc_id,
                        "source": doc.source,
                        "domain": doc.domain,
                        "keep": decision.keep,
                        "reason": decision.reason,
                        "dup_of": decision.dup_of,
                        "score": decision.score,
                        "meta": decision.meta,
                    }
                    log_writer.write((json.dumps(log_payload, ensure_ascii=False) + "\n").encode("utf-8"))
                    if processed % log_every == 0:
                        logger.info("Dedup progress: processed=%s kept=%s dropped=%s", processed, kept, dropped)
                logger.info(
                    "Dedup file done: %s processed=%s kept=%s dropped=%s",
                    path,
                    file_processed,
                    file_kept,
                    file_dropped,
                )
                bytes_done += path.stat().st_size
                if total_bytes > 0:
                    pct = (bytes_done / total_bytes) * 100.0
                    elapsed = max(1e-6, time.time() - start_ts)
                    eta = (elapsed / max(bytes_done, 1)) * (total_bytes - bytes_done)
                    logger.info(
                        "Dedup progress (bytes): %.1f%% ETA=%s",
                        pct,
                        str(datetime.timedelta(seconds=int(eta))),
                    )
                completed.add(path.as_posix())
                if checkpoint_every_files and len(completed) % checkpoint_every_files == 0:
                    _atomic_write_pickle(
                        checkpoint_path,
                        {
                            "phase": "single",
                            "params_hash": params_hash,
                            "completed_single": sorted(completed),
                        },
                    )
                if stop_after and len(completed) >= stop_after:
                    raise SystemExit("GAMMA7B_DEDUP_STOP_AFTER_FILES triggered (single).")

    stats = {
        "kept": kept,
        "dropped": dropped,
        "mode": mode,
        "scope": scope,
        "simhash_threshold": simhash_threshold,
        "hash_bits": hash_bits,
    }
    (out_dir / "dedup_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    logger.info("Dedup done: kept=%s dropped=%s out=%s", kept, dropped, out_path)


def _iter_docs_for_source(
    paths: List[str],
    shuffle_buffer_docs: int = 0,
    seed: int = 0,
    repeat: bool = False,
    source_key: str = "",
):
    epoch = 0
    while True:
        if shuffle_buffer_docs <= 0:
            for path in resolve_inputs(paths):
                for record in read_jsonl(path):
                    yield record
        else:
            seed_str = f"{seed}:{source_key}:{epoch}"
            rng_seed = int(stable_hash(seed_str), 16) % (2**31 - 1)
            rng = random.Random(rng_seed)
            buffer: List[dict] = []
            for path in resolve_inputs(paths):
                for record in read_jsonl(path):
                    buffer.append(record)
                    if len(buffer) >= shuffle_buffer_docs:
                        idx = rng.randrange(len(buffer))
                        yield buffer.pop(idx)
            while buffer:
                idx = rng.randrange(len(buffer))
                yield buffer.pop(idx)

        epoch += 1
        if not repeat:
            break


def _git_sha() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _resolve_data_root(data_root: Optional[Path]) -> Optional[str]:
    from typer.models import OptionInfo

    if isinstance(data_root, OptionInfo):
        data_root = None
    if data_root is None:
        env_root = os.getenv("GAMMA7B_DATA_ROOT")
        if env_root:
            return env_root
        env_root = os.getenv("GAMMA_DATA_ROOT")
        if env_root:
            return env_root
        env_root = os.getenv("DATA_ROOT")
        if env_root:
            return env_root
        return None
    return str(data_root)

MAX_PROBE_LINE_BYTES = 8 * 1024 * 1024


def _read_first_jsonl_line(path: Path) -> Dict[str, object]:
    import io

    with open_zst_reader(path) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        line = text_stream.readline(MAX_PROBE_LINE_BYTES + 1)
    if not line:
        return {"status": "empty"}
    if len(line) > MAX_PROBE_LINE_BYTES:
        return {"status": "too_long"}
    return {"status": "ok", "line": line}


def _audit_manifest_local(cfg, strict: bool, logger=None, data_root: Optional[str] = None) -> Dict[str, object]:
    report = {
        "sources": [],
        "total_files": 0,
        "total_bytes": 0,
        "missing": [],
        "placeholders": [],
        "probe_errors": [],
        "probe_skips": [],
    }
    for domain_name in sorted(cfg.domains.keys()):
        domain = cfg.domains[domain_name]
        for source_name in sorted(domain.sources.keys()):
            source = domain.sources[source_name]
            if source.type not in {"local_dir", "local_jsonl"}:
                continue
            if source.weight <= 0:
                continue
            paths = expand_source_paths(source.paths, data_root=data_root)
            files = [path for path in paths if Path(path).is_file()]
            total_bytes = sum(Path(path).stat().st_size for path in files)
            entry = {
                "domain": domain.name,
                "source": source.name,
                "type": source.type,
                "weight": source.weight,
                "n_files": len(files),
                "total_bytes": total_bytes,
                "example_paths": files[:3],
            }
            if files:
                try:
                    first_path = files[0]
                    probe = _read_first_jsonl_line(Path(first_path))
                    if probe["status"] == "too_long":
                        report["probe_skips"].append(
                            {
                                "domain": domain.name,
                                "source": source.name,
                                "path": first_path,
                                "reason": "probe_line_too_long",
                            }
                        )
                    elif probe["status"] == "ok":
                        obj = json.loads(probe["line"])
                        placeholder_reason = None
                        if obj.get("license_tag") == "placeholder":
                            placeholder_reason = "license_tag"
                        elif obj.get("meta", {}).get("note") == "placeholder":
                            placeholder_reason = "meta.note"
                        elif "placeholder" in str(obj.get("text", "")).lower():
                            placeholder_reason = "text"
                        if placeholder_reason:
                            report["placeholders"].append(
                                {
                                    "domain": domain.name,
                                    "source": source.name,
                                    "path": first_path,
                                    "reason": placeholder_reason,
                                }
                            )
                except Exception as exc:
                    report["probe_errors"].append(
                        {
                            "domain": domain.name,
                            "source": source.name,
                            "path": first_path,
                            "error": str(exc),
                        }
                    )
            if not files:
                report["missing"].append(f"{domain.name}/{source.name}")
                if logger:
                    logger.warning("Missing local source %s/%s", domain.name, source.name)
                if strict:
                    raise typer.BadParameter(
                        f"Missing local source {domain.name}/{source.name} (no files matched)."
                    )
            report["sources"].append(entry)
            report["total_files"] += len(files)
            report["total_bytes"] += total_bytes
    return report


@app.command("manifest-audit")
def manifest_audit_cmd(
    manifest: Path = typer.Option(..., "--manifest"),
    strict: bool = typer.Option(False, "--strict"),
    out: Optional[Path] = typer.Option(None, "--out"),
    data_root: Optional[Path] = typer.Option(None, "--data-root"),
) -> None:
    logger = setup_logger()
    cfg = load_manifest(manifest)
    try:
        import yaml

        raw_payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        raw_domains = raw_payload.get("domains", {}) if isinstance(raw_payload, dict) else {}
        raw_domain_weights = {k: float(v.get("weight", 0.0)) for k, v in raw_domains.items() if isinstance(v, dict)}
    except Exception:
        raw_domain_weights = {}
    resolved_root = _resolve_data_root(data_root)
    report = _audit_manifest_local(cfg, strict=False, logger=logger, data_root=resolved_root)
    manifest_text = manifest.read_text(encoding="utf-8")
    report["manifest_hash"] = stable_hash(manifest_text)
    report["manifest_path"] = str(manifest)
    payload = json.dumps(report, indent=2)
    if out:
        ensure_dir(out.parent)
        out.write_text(payload + "\n", encoding="utf-8")
    else:
        typer.echo(payload)
    if report["probe_errors"]:
        logger.warning("Manifest audit probe errors: %s", json.dumps(report["probe_errors"], ensure_ascii=False))
    if strict:
        if report["missing"]:
            raise typer.BadParameter(f"Missing local sources: {', '.join(report['missing'])}")
        if report["placeholders"]:
            raise typer.BadParameter("Placeholder records detected in local sources.")
        if report["probe_errors"]:
            snippet = report["probe_errors"][:10]
            raise typer.BadParameter(
                "Probe errors detected in local sources (first 10): "
                + json.dumps(snippet, ensure_ascii=False)
            )


@app.command("mix-report")
def mix_report_cmd(
    manifest: Path = typer.Option(..., "--manifest"),
    max_docs_per_source: int = typer.Option(20000, "--max-docs-per-source"),
    sample_hf_stream_docs: int = typer.Option(0, "--sample-hf-stream-docs"),
    max_sample_chars_per_doc: int = typer.Option(20000, "--max-sample-chars-per-doc"),
    out: Optional[Path] = typer.Option(None, "--out"),
    data_root: Optional[Path] = typer.Option(None, "--data-root"),
    strict: bool = typer.Option(False, "--strict"),
    workers: int = typer.Option(16, "--workers"),
) -> None:
    import io
    from typer.models import OptionInfo

    logger = setup_logger()
    cfg = load_manifest(manifest)
    resolved_root = _resolve_data_root(data_root)
    if isinstance(sample_hf_stream_docs, OptionInfo):
        sample_hf_stream_docs = 0
    if isinstance(max_sample_chars_per_doc, OptionInfo):
        max_sample_chars_per_doc = 20000
    if isinstance(workers, OptionInfo):
        workers = 16
    report = {
        "manifest_path": str(manifest),
        "manifest_hash": stable_hash(manifest.read_text(encoding="utf-8")),
        "max_docs_per_source": max_docs_per_source,
        "sample_hf_stream_docs": sample_hf_stream_docs,
        "max_sample_chars_per_doc": max_sample_chars_per_doc,
        "sources": [],
        "totals": {"measured_docs": 0, "measured_chars": 0, "measured_bytes": 0},
        "domain_shares": {},
        "empty_zst_files": [],
        "sample_errors": [],
    }
    domain_totals_chars: Dict[str, int] = {}
    domain_totals_bytes: Dict[str, int] = {}
    domain_totals_chars_all: Dict[str, int] = {}
    domain_totals_bytes_all: Dict[str, int] = {}

    def _sample_hf_stream(
        dataset_name: str,
        config_name: Optional[str],
        split: str,
        text_field: str,
        revision: Optional[str],
    ) -> Dict[str, int]:
        if sample_hf_stream_docs <= 0:
            return {"n_docs": 0, "total_chars": 0, "total_bytes": 0}
        import datasets

        if config_name:
            ds = datasets.load_dataset(
                dataset_name, config_name, split=split, streaming=True, revision=revision
            )
        else:
            ds = datasets.load_dataset(dataset_name, split=split, streaming=True, revision=revision)
        n_docs = 0
        total_chars = 0
        total_bytes = 0
        for row in ds:
            text = row.get(text_field) or ""
            clipped = text[:max_sample_chars_per_doc]
            n_docs += 1
            total_chars += len(clipped)
            total_bytes += len(clipped.encode("utf-8", errors="ignore"))
            if n_docs >= sample_hf_stream_docs:
                break
        return {"n_docs": n_docs, "total_chars": total_chars, "total_bytes": total_bytes}

    for domain_name in sorted(cfg.domains.keys()):
        domain = cfg.domains[domain_name]
        for source_name in sorted(domain.sources.keys()):
            source = domain.sources[source_name]
            entry = {
                "domain": domain.name,
                "source": source.name,
                "type": source.type,
                "target_weight_domain": domain.weight,
                "target_weight_source": source.weight,
            }
            if source.type == "local_dir":
                paths = expand_source_paths(source.paths, data_root=resolved_root)
                n_docs = 0
                total_chars = 0
                total_bytes = 0
                n_empty_files = 0
                capped = False
                file_paths = [Path(p) for p in paths if Path(p).is_file()]
                if workers and workers > 1:
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = [executor.submit(_stat_jsonl_file, p) for p in file_paths]
                        for file_path, fut in zip(file_paths, futures):
                            stats = fut.result()
                            n_docs += stats["n_docs"]
                            total_chars += stats["total_chars"]
                            total_bytes += stats["total_bytes"]
                            if stats["empty"]:
                                n_empty_files += 1
                                if str(file_path).endswith(".zst"):
                                    report["empty_zst_files"].append(str(file_path))
                            if n_docs >= max_docs_per_source:
                                capped = True
                                break
                else:
                    for file_path in file_paths:
                        stats = _stat_jsonl_file(file_path)
                        n_docs += stats["n_docs"]
                        total_chars += stats["total_chars"]
                        total_bytes += stats["total_bytes"]
                        if stats["empty"]:
                            n_empty_files += 1
                            if str(file_path).endswith(".zst"):
                                report["empty_zst_files"].append(str(file_path))
                        if n_docs >= max_docs_per_source:
                            capped = True
                            break
                entry.update(
                    {
                        "n_docs": n_docs,
                        "total_chars": total_chars,
                        "total_bytes": total_bytes,
                        "n_empty_files": n_empty_files,
                        "capped": capped,
                        "paths": paths,
                    }
                )
                report["totals"]["measured_docs"] += n_docs
                report["totals"]["measured_chars"] += total_chars
                report["totals"]["measured_bytes"] += total_bytes
                domain_totals_chars[domain.name] = domain_totals_chars.get(domain.name, 0) + total_chars
                domain_totals_bytes[domain.name] = domain_totals_bytes.get(domain.name, 0) + total_bytes
                domain_totals_chars_all[domain.name] = domain_totals_chars_all.get(domain.name, 0) + total_chars
                domain_totals_bytes_all[domain.name] = domain_totals_bytes_all.get(domain.name, 0) + total_bytes
            elif source.type == "hf_stream" and sample_hf_stream_docs > 0:
                params = source.params or {}
                cfg_map = HF_SOURCE_MAP.get(source.name, {})
                dataset_name = params.get("dataset") or cfg_map.get("dataset")
                if not dataset_name:
                    entry.update({"unmeasured": True, "params": source.params})
                    report["sources"].append(entry)
                    continue
                config_name = params.get("config_name") or params.get("config") or params.get("subset")
                if config_name is None:
                    config_name = cfg_map.get("config_name")
                split = params.get("split") or cfg_map.get("split", "train")
                text_field = params.get("text_field") or cfg_map.get("text_field", "text")
                revision = params.get("revision") or cfg_map.get("revision")
                try:
                    sampled = _sample_hf_stream(
                        dataset_name=dataset_name,
                        config_name=config_name,
                        split=split,
                        text_field=text_field,
                        revision=revision,
                    )
                    entry.update(
                        {
                            "n_docs_sampled": sampled["n_docs"],
                            "total_chars_sampled": sampled["total_chars"],
                            "total_bytes_sampled": sampled["total_bytes"],
                        }
                    )
                    domain_totals_chars_all[domain.name] = domain_totals_chars_all.get(
                        domain.name, 0
                    ) + sampled["total_chars"]
                    domain_totals_bytes_all[domain.name] = domain_totals_bytes_all.get(
                        domain.name, 0
                    ) + sampled["total_bytes"]
                except Exception as exc:
                    entry.update(
                        {
                            "sample_error": str(exc),
                            "n_docs_sampled": 0,
                            "total_chars_sampled": 0,
                            "total_bytes_sampled": 0,
                        }
                    )
                    report["sample_errors"].append(
                        {"domain": domain.name, "source": source.name, "error": str(exc)}
                    )
            else:
                entry.update({"unmeasured": True, "params": source.params})
            report["sources"].append(entry)

    total_chars = report["totals"]["measured_chars"]
    total_bytes = report["totals"]["measured_bytes"]
    for domain_name in sorted(domain_totals_chars.keys()):
        report["domain_shares"][domain_name] = {
            "chars": domain_totals_chars[domain_name],
            "bytes": domain_totals_bytes.get(domain_name, 0),
            "share_chars": (domain_totals_chars[domain_name] / total_chars) if total_chars else 0.0,
            "share_bytes": (domain_totals_bytes.get(domain_name, 0) / total_bytes) if total_bytes else 0.0,
        }
    report["domain_shares_local_only"] = report["domain_shares"]
    total_chars_all = sum(domain_totals_chars_all.values())
    total_bytes_all = sum(domain_totals_bytes_all.values())
    report["domain_shares_estimated_all_sources"] = {}
    for domain_name in sorted(domain_totals_chars_all.keys()):
        report["domain_shares_estimated_all_sources"][domain_name] = {
            "chars": domain_totals_chars_all[domain_name],
            "bytes": domain_totals_bytes_all.get(domain_name, 0),
            "share_chars": (domain_totals_chars_all[domain_name] / total_chars_all) if total_chars_all else 0.0,
            "share_bytes": (domain_totals_bytes_all.get(domain_name, 0) / total_bytes_all) if total_bytes_all else 0.0,
        }

    payload = json.dumps(report, indent=2)
    if out:
        ensure_dir(out.parent)
        out.write_text(payload + "\n", encoding="utf-8")
        logger.info("Mix report written: %s", out)
    else:
        typer.echo(payload)
    if not isinstance(strict, OptionInfo) and strict and report["sample_errors"]:
        raise typer.BadParameter("HF sampling errors detected in mix-report.")


@app.command("ingest-manifest")
def ingest_manifest_cmd(
    manifest: Path = typer.Option(..., "--manifest"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    max_docs_per_source: Optional[int] = typer.Option(None, "--max-docs-per-source"),
    fail_on_missing_local: bool = typer.Option(True, "--fail-on-missing-local/--no-fail-on-missing-local"),
    data_root: Optional[Path] = typer.Option(None, "--data-root"),
) -> None:
    logger = setup_logger()
    log_every = get_log_every()
    cfg = load_manifest(manifest)
    ensure_dir(out_dir)

    manifest_text = manifest.read_text(encoding="utf-8")
    manifest_hash = stable_hash(manifest_text)
    git_sha = _git_sha()
    hf_sources = []

    resolved_root = _resolve_data_root(data_root)
    audit_report = _audit_manifest_local(cfg, strict=fail_on_missing_local, logger=logger, data_root=resolved_root)
    audit_report["manifest_hash"] = manifest_hash
    audit_report["manifest_path"] = str(manifest)
    audit_path = out_dir / "manifest_audit.json"
    audit_path.write_text(json.dumps(audit_report, indent=2) + "\n", encoding="utf-8")

    logger.info("Ingest manifest start: %s log_every=%s", manifest, log_every)
    for domain_name in sorted(cfg.domains.keys()):
        domain = cfg.domains[domain_name]
        for source_name in sorted(domain.sources.keys()):
            source = domain.sources[source_name]
            if source.type != "hf_stream":
                logger.info("Skip source %s/%s type=%s", domain.name, source.name, source.type)
                continue
            params = source.params or {}
            cfg_map = HF_SOURCE_MAP.get(source.name, {})
            dataset_name = params.get("dataset") or cfg_map.get("dataset")
            if not dataset_name:
                raise typer.BadParameter(f"Missing dataset for source {domain.name}/{source.name}")
            config_name = params.get("config_name") or params.get("config") or params.get("subset")
            if config_name is None:
                config_name = cfg_map.get("config_name")
            split = params.get("split") or cfg_map.get("split", "train")
            text_field = params.get("text_field") or cfg_map.get("text_field", "text")
            id_field = params.get("id_field") or cfg_map.get("id_field", "id")
            meta_fields = params.get("meta_fields") or cfg_map.get("meta_fields", [])
            license_tag = params.get("license_tag")
            revision = params.get("revision") or cfg_map.get("revision")

            out_path = out_dir / domain.name / source.name / "part_00000.jsonl.zst"
            ensure_dir(out_path.parent)
            logger.info("Ingest %s/%s -> %s", domain.name, source.name, out_path)
            stream_hf_dataset_to_normalized(
                dataset_name=dataset_name,
                out_path=out_path,
                domain=domain.name,
                source=source.name,
                limit=max_docs_per_source,
                split=split,
                config_name=config_name,
                text_field=text_field,
                id_field=id_field,
                meta_fields=meta_fields,
                license_tag=license_tag,
                revision=revision,
                logger=logger,
                log_every=log_every,
            )
            hf_sources.append(
                {
                    "dataset": dataset_name,
                    "config": config_name,
                    "revision": revision,
                }
            )

    logger.info("Ingest manifest done")
    logger.info("Repro footer: manifest_hash=%s git_sha=%s", manifest_hash, git_sha)
    if hf_sources:
        logger.info("HF sources: %s", json.dumps(hf_sources, ensure_ascii=False))


@app.command("pack")
def pack_cmd(
    manifest: Path = typer.Option(..., "--manifest"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir"),
    seq_len: Optional[int] = typer.Option(None, "--seq-len"),
    num_seqs: Optional[int] = typer.Option(None, "--num-seqs"),
    tokenizer_path: Optional[str] = typer.Option(None, "--tokenizer"),
    eos_token: Optional[str] = typer.Option(None, "--eos-token"),
    eos_id: Optional[int] = typer.Option(None, "--eos-id"),
    emit_index: bool = typer.Option(True, "--emit-index/--no-emit-index"),
    created_at: Optional[str] = typer.Option(None, "--created-at"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    chars_per_token: Optional[float] = typer.Option(None, "--chars-per-token"),
    data_root: Optional[Path] = typer.Option(None, "--data-root"),
    shuffle_buffer_docs: int = typer.Option(0, "--shuffle-buffer-docs"),
    repeat_sources: bool = typer.Option(
        False,
        "--repeat-sources",
        help="If a source hits EOF, reopen and continue (useful for small corpora / smoke packs).",
    ),
) -> None:
    from typer.models import OptionInfo

    logger = setup_logger()
    log_every = get_log_every()
    cfg = load_manifest(manifest)
    seed = seed if seed is not None else cfg.seed
    seq_len = seq_len or int(cfg.packing.get("seq_len", 4096))
    num_seqs = num_seqs or cfg.packing.get("num_seqs")
    if num_seqs is None:
        raise typer.BadParameter("num_seqs must be provided or set in manifest.packing")
    num_seqs = int(num_seqs)
    tokenizer_path = tokenizer_path or cfg.packing.get("tokenizer_path")
    if not tokenizer_path:
        raise typer.BadParameter("tokenizer path must be provided or set in manifest.packing")
    out_dir = out_dir or Path(cfg.outputs.get("packed_dir", "out/packed"))
    ensure_dir(out_dir)

    chars_per_token = chars_per_token or cfg.chars_per_token

    if isinstance(eos_token, OptionInfo):
        eos_token = None
    if isinstance(eos_id, OptionInfo):
        eos_id = None
    if eos_token is None:
        if str(tokenizer_path).endswith(".model"):
            eos_token = "<|eos|>"
        elif str(tokenizer_path).endswith(".json"):
            eos_token = "</s>"

    resolved_root = _resolve_data_root(data_root)
    tokenizer = load_tokenizer(tokenizer_path, eos_token=eos_token, eos_id=eos_id)
    logger.info(
        "Pack start: seq_len=%s num_seqs=%s tokenizer=%s log_every=%s shuffle_buffer_docs=%s repeat_sources=%s",
        seq_len,
        num_seqs,
        tokenizer.name,
        log_every,
        shuffle_buffer_docs,
        repeat_sources,
    )
    if raw_domain_weights:
        logger.info("Pack raw domain weights: %s", raw_domain_weights)
    logger.info("Pack normalized domain weights: %s", cfg.domain_weights)

    domain_weights = cfg.domain_weights
    source_weights = cfg.source_weights

    iterators = {}
    for domain, sources in cfg.source_paths.items():
        for source, paths in sources.items():
            key = f"{domain}:{source}"
            expanded = expand_source_paths(paths, data_root=resolved_root)
            if not expanded:
                raise RuntimeError(f"No input files matched for {key}: {paths}")
            source_seed = seed + (int(stable_hash(key), 16) % (2**31 - 1))
            iterators[key] = _iter_docs_for_source(
                expanded,
                shuffle_buffer_docs=shuffle_buffer_docs,
                seed=source_seed,
                repeat=repeat_sources,
                source_key=key,
            )

    sampler = TokenAwareSampler(domain_weights, source_weights, seed=seed)
    builder = PackBuilder(seq_len, tokenizer)

    bin_path = out_dir / "input_ids.mmap"
    meta_path = out_dir / "meta.json"
    index_path = out_dir / "index.jsonl.zst"

    mmap = np.memmap(bin_path, dtype=np.int32, mode="w+", shape=(num_seqs, seq_len))
    pack_manifest = PackManifest(seq_len=seq_len, tokenizer_path=tokenizer.name, seed=seed)

    seq_idx = 0
    docs_seen = 0
    last_logged = 0
    last_time = time.time()
    start_time = last_time
    if created_at is None:
        created_at = default_created_at()

    index_cm = None
    index_writer = None
    if emit_index:
        index_cm = open_zst_writer(index_path)
        index_writer = index_cm.__enter__()

    try:
        while seq_idx < num_seqs:
            domain = sampler.choose_domain()
            source = sampler.choose_source(domain)
            key = f"{domain}:{source}"
            try:
                doc = next(iterators[key])
            except StopIteration:
                raise RuntimeError(f"Source exhausted: {key}")
            text = doc.get("text", "")
            doc_id = doc.get("doc_id") or doc.get("id") or ""
            est_tokens = estimate_tokens(text, chars_per_token=chars_per_token)
            sampler.update(domain, source, est_tokens)
            pack_manifest.add(domain, source, est_tokens)
            docs_seen += 1
            if docs_seen % log_every == 0:
                logger.info("Pack doc progress: docs_seen=%s sequences=%s", docs_seen, seq_idx)
            for seq in builder.add_doc(doc_id, text):
                if seq_idx >= num_seqs:
                    break
                mmap[seq_idx, :] = np.asarray(seq.tokens, dtype=np.int32)
                if index_writer is not None:
                    segments = [s for s in seq.segments if s.get("doc_id")]
                    entry = {
                        "row": seq_idx,
                        "segments": segments,
                        "domain": doc.get("domain") or domain,
                        "source": doc.get("source") or source,
                    }
                    if len(segments) == 1:
                        entry.update(
                            {
                                "doc_id": segments[0]["doc_id"],
                                "start_token": segments[0]["start"],
                                "end_token": segments[0]["end"],
                            }
                        )
                    index_writer.write((json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
                seq_idx += 1
                if seq_idx - last_logged >= log_every:
                    now = time.time()
                    elapsed = max(1e-9, now - start_time)
                    rate = seq_idx / elapsed
                    pct = (seq_idx / max(1, num_seqs)) * 100.0
                    eta_sec = (num_seqs - seq_idx) / max(rate, 1e-9)
                    logger.info(
                        "Pack progress: sequences=%s/%s pct~=%.1f%% docs_seen=%s rate=%.2f seq/s ETA~=%.0fs",
                        seq_idx,
                        num_seqs,
                        pct,
                        docs_seen,
                        rate,
                        eta_sec,
                    )
                    last_logged = seq_idx
        tail = builder.finalize()
        if tail is not None and seq_idx < num_seqs:
            mmap[seq_idx, :] = np.asarray(tail.tokens, dtype=np.int32)
            if index_writer is not None:
                segments = [s for s in tail.segments if s.get("doc_id")]
                entry = {"row": seq_idx, "segments": segments, "domain": domain, "source": source}
                if len(segments) == 1:
                    entry.update(
                        {
                            "doc_id": segments[0]["doc_id"],
                            "start_token": segments[0]["start"],
                            "end_token": segments[0]["end"],
                        }
                    )
                index_writer.write((json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
            seq_idx += 1
    finally:
        if index_cm is not None:
            index_cm.__exit__(None, None, None)

    mmap.flush()
    meta_payload = {
        "seq_len": seq_len,
        "num_seqs": int(num_seqs),
        "n_seqs": num_seqs,
        "dtype": "int32",
        "tokenizer_path": tokenizer.name,
        "created_at": created_at,
        "seed": seed,
        "chars_per_token": chars_per_token,
        "counts": {
            "docs_seen": docs_seen,
            "counts_by_domain": pack_manifest.counts_by_domain,
            "counts_by_source": pack_manifest.counts_by_source,
            "est_tokens_by_domain": pack_manifest.est_tokens_by_domain,
            "est_tokens_by_source": pack_manifest.est_tokens_by_source,
        },
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8")
    pack_manifest.write(out_dir / "pack_manifest.json")
    logger.info("Pack done: sequences=%s out=%s", num_seqs, out_dir)


@app.command("report")
def report_cmd(
    input: List[str] = typer.Option(..., "--input"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    logger = setup_logger()
    inputs = resolve_inputs(input)
    logger.info("Report start: inputs=%s", len(inputs))
    aggregate = {
        "docs": 0,
        "bytes": 0,
        "est_tokens": 0,
        "drops": {},
        "by_domain": {},
        "by_source": {},
        "by_domain_tokens": {},
        "by_source_tokens": {},
        "dedup_drops": {},
        "top_sources_by_tokens": [],
        "worst_examples": [],
    }

    worst_limit = 5
    worst_samples = []

    for path in inputs:
        path_obj = Path(path)
        if path_obj.suffix in {".jsonl", ".zst"} or path_obj.name.endswith(".jsonl.zst"):
            for record in read_jsonl(path_obj):
                if record.get("keep") is False:
                    reason = record.get("reason", "dropped")
                    aggregate["dedup_drops"].setdefault(reason, 0)
                    aggregate["dedup_drops"][reason] += 1
                    score = record.get("score")
                    sample = {
                        "doc_id": record.get("doc_id"),
                        "source": record.get("source"),
                        "domain": record.get("domain"),
                        "reason": reason,
                        "score": score,
                        "cluster_id": record.get("cluster_id"),
                    }
                    worst_samples.append(sample)
            continue

        payload = json.loads(path_obj.read_text(encoding="utf-8"))
        counts = payload.get("counts", {})
        aggregate["docs"] += counts.get("docs", 0)
        aggregate["bytes"] += counts.get("bytes", 0)
        aggregate["est_tokens"] += counts.get("est_tokens", 0)
        domain = payload.get("domain")
        source = payload.get("source")
        if domain:
            aggregate["by_domain"].setdefault(domain, 0)
            aggregate["by_domain"][domain] += counts.get("docs", 0)
            aggregate["by_domain_tokens"].setdefault(domain, 0)
            aggregate["by_domain_tokens"][domain] += counts.get("est_tokens", 0)
        if source:
            aggregate["by_source"].setdefault(source, 0)
            aggregate["by_source"][source] += counts.get("docs", 0)
            aggregate["by_source_tokens"].setdefault(source, 0)
            aggregate["by_source_tokens"][source] += counts.get("est_tokens", 0)
        for reason, count in payload.get("drops", {}).items():
            aggregate["drops"].setdefault(reason, 0)
            aggregate["drops"][reason] += count

    top_sources = sorted(
        aggregate["by_source_tokens"].items(), key=lambda item: item[1], reverse=True
    )
    aggregate["top_sources_by_tokens"] = [{"source": s, "est_tokens": t} for s, t in top_sources[:10]]

    if worst_samples:
        def score_key(item):
            score = item.get("score")
            return score if isinstance(score, (int, float)) else float("inf")

        worst_samples.sort(key=score_key)
        aggregate["worst_examples"] = worst_samples[:worst_limit]

    out.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    logger.info("Report written: %s", out)


@app.command("smoke")
def smoke_cmd(
    out_dir: Path = typer.Option(Path("out"), "--out-dir"),
    seed: int = typer.Option(1234, "--seed"),
) -> None:
    logger = setup_logger()
    initialize_runtime(logger)
    log_every = get_log_every()
    logger.info("Smoke start: out_dir=%s log_every=%s", out_dir, log_every)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    raw_dir = out_dir / "raw"
    norm_dir = out_dir / "normalized"
    dedup_dir = out_dir / "dedup"
    packed_dir = out_dir / "packed"
    ensure_dir(raw_dir)

    raw_path = raw_dir / "local.jsonl"
    sample_text = (
        "This is a tiny sample document for packing. "
        "It repeats to pass the minimum length filter. "
        "This is a tiny sample document for packing. "
        "It repeats to pass the minimum length filter. "
        "This is a tiny sample document for packing. "
        "It repeats to pass the minimum length filter."
    )
    raw_path.write_text(
        "\n".join(
            [
                json.dumps({"text": sample_text}),
                json.dumps({"text": sample_text + " Extra sentence for variety."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info("Normalize -> %s", norm_dir)
    normalize_files(
        inputs=[raw_path],
        out_dir=norm_dir / "filtered_web" / "local",
        domain="filtered_web",
        source="local",
        seed=seed,
        shard_size=2,
        license_tag=None,
        use_language_heuristic=False,
        chars_per_token=4.0,
        logger=logger,
        log_every=log_every,
    )

    logger.info("Dedup -> %s", dedup_dir)
    dedup_cmd(
        input=[str(norm_dir / "filtered_web" / "local" / "*.jsonl.zst")],
        out_dir=dedup_dir,
        mode="exact",
        scope="global",
        simhash_threshold=3,
        hash_bits=64,
        seed=seed,
    )

    tokenizer_path = out_dir / "tokenizer.json"
    _build_smoke_tokenizer(tokenizer_path)

    manifest_path = out_dir / "smoke_manifest.yaml"
    manifest_path.write_text(
        """seed: 1\noutputs:\n  normalized_dir: {norm_dir}\n  dedup_dir: {dedup_dir}\n  packed_dir: {packed_dir}\npacking:\n  seq_len: 16\n  num_seqs: 4\n  tokenizer_path: {tokenizer}\ndomains:\n  filtered_web:\n    weight: 1.0\n    sources:\n      local:\n        weight: 1.0\n        type: local_jsonl\n        params:\n          paths:\n            - {dedup_file}\n""".format(
            norm_dir=str(norm_dir),
            dedup_dir=str(dedup_dir),
            packed_dir=str(packed_dir),
            tokenizer=str(tokenizer_path),
            dedup_file=str(dedup_dir / "deduped.jsonl.zst"),
        ),
        encoding="utf-8",
    )

    logger.info("Pack -> %s", packed_dir)
    pack_cmd(
        manifest=manifest_path,
        out_dir=packed_dir,
        seq_len=16,
        num_seqs=4,
        tokenizer_path=str(tokenizer_path),
        emit_index=True,
        created_at="2025-01-01T00:00:00Z",
        seed=seed,
        chars_per_token=4.0,
    )

    report_cmd(
        input=[str(norm_dir / "filtered_web" / "local" / "*.manifest.json")],
        out=out_dir / "report.json",
    )
    logger.info("Smoke pipeline complete: %s", out_dir)


def _build_smoke_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {
        "<eos>": 0,
        "This": 1,
        "is": 2,
        "a": 3,
        "tiny": 4,
        "sample": 5,
        "document": 6,
        "for": 7,
        "packing.": 8,
        "Another": 9,
        "short": 10,
        "doc": 11,
        "the": 12,
        "pipeline": 13,
        "smoke": 14,
        "test.": 15,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<eos>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


if __name__ == "__main__":
    app()

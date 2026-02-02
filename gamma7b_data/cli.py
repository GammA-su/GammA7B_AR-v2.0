import json
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

from .config import load_manifest
from .dedup import Deduper, SimhashClusterer, exact_hash
from .hf_stream import HF_SOURCE_MAP, stream_hf_to_normalized
from .manifest import PackManifest
from .normalize import normalize_files
from .packing import PackBuilder, default_created_at, load_tokenizer
from .sampling import TokenAwareSampler
from .schema import NormalizedDocument
from .stackexchange import ingest_stackexchange
from .utils import (
    ensure_dir,
    estimate_tokens,
    get_log_every,
    initialize_runtime,
    iter_files,
    open_zst_writer,
    quality_score,
    read_jsonl,
    resolve_inputs,
    setup_logger,
)
from .wikipedia import ingest_wikipedia

app = typer.Typer(add_completion=False)


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
    filter: Optional[str] = typer.Option(None, "--filter"),
) -> None:
    from datasets import get_dataset_config_names

    configs = get_dataset_config_names(dataset)
    if filter:
        needle = filter.lower()
        configs = [cfg for cfg in configs if needle in cfg.lower()]
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
    scope: str = typer.Option("global", "--scope"),
    simhash_threshold: int = typer.Option(3, "--simhash-threshold"),
    hash_bits: int = typer.Option(64, "--hash-bits"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    _ = seed
    logger = setup_logger()
    log_every = get_log_every()
    logger.info("Dedup start: inputs=%s log_every=%s", len(input), log_every)
    inputs = resolve_inputs(input)
    ensure_dir(out_dir)
    out_path = out_dir / "deduped.jsonl.zst"
    log_path = out_dir / "decisions.jsonl.zst"

    use_near = mode in {"near", "both"}
    use_exact = mode in {"exact", "both"}
    if mode not in {"near", "both", "exact"}:
        raise typer.BadParameter("mode must be exact, near, or both")
    if scope not in {"global", "per-source"}:
        raise typer.BadParameter("scope must be global or per-source")

    deduper_global = Deduper(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
    deduper_by_source = {}

    kept = 0
    dropped = 0
    processed = 0
    if use_near:
        logger.info("Dedup mode near/both: computing best-in-cluster candidates")
        cluster_best: Dict[str, str] = {}
        cluster_best_score: Dict[str, float] = {}
        cluster_counts: Dict[str, int] = {}
        exact_seen: Dict[str, str] = {}
        clusterers: Dict[str, SimhashClusterer] = {}

        for path in iter_files(inputs):
            for record in read_jsonl(path):
                doc = NormalizedDocument(**record)
                if use_exact:
                    content_hash = exact_hash(doc.text)
                    if content_hash in exact_seen:
                        continue
                    exact_seen[content_hash] = doc.doc_id
                if scope == "per-source":
                    clusterer = clusterers.setdefault(
                        doc.source, SimhashClusterer(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
                    )
                    cluster_key_prefix = f"{doc.source}:"
                else:
                    clusterer = clusterers.setdefault(
                        "__global__", SimhashClusterer(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
                    )
                    cluster_key_prefix = ""
                match = clusterer.assign(doc.doc_id, doc.text)
                cluster_key = f"{cluster_key_prefix}{match.cluster_id}"
                score = quality_score(doc.text, doc.meta)
                cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
                best_score = cluster_best_score.get(cluster_key)
                if best_score is None or score > best_score:
                    cluster_best_score[cluster_key] = score
                    cluster_best[cluster_key] = doc.doc_id

        exact_seen = {}
        clusterers = {}
        with open_zst_writer(out_path) as out_writer, open_zst_writer(log_path) as log_writer:
            for path in iter_files(inputs):
                logger.info("Dedup reading: %s", path)
                file_processed = 0
                file_kept = 0
                file_dropped = 0
                for record in read_jsonl(path):
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
                            clusterer = clusterers.setdefault(
                                doc.source, SimhashClusterer(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
                            )
                            cluster_key_prefix = f"{doc.source}:"
                        else:
                            clusterer = clusterers.setdefault(
                                "__global__", SimhashClusterer(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
                            )
                            cluster_key_prefix = ""
                        match = clusterer.assign(doc.doc_id, doc.text)
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
    else:
        with open_zst_writer(out_path) as out_writer, open_zst_writer(log_path) as log_writer:
            for path in iter_files(inputs):
                logger.info("Dedup reading: %s", path)
                file_processed = 0
                file_kept = 0
                file_dropped = 0
                for record in read_jsonl(path):
                    doc = NormalizedDocument(**record)
                    if scope == "per-source":
                        deduper = deduper_by_source.setdefault(
                            doc.source, Deduper(simhash_threshold=simhash_threshold, hash_bits=hash_bits)
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


def _iter_docs_for_source(paths: List[str]):
    for path in resolve_inputs(paths):
        for record in read_jsonl(path):
            yield record


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

    tokenizer = load_tokenizer(tokenizer_path, eos_token=eos_token, eos_id=eos_id)
    logger.info(
        "Pack start: seq_len=%s num_seqs=%s tokenizer=%s log_every=%s",
        seq_len,
        num_seqs,
        tokenizer.name,
        log_every,
    )

    domain_weights = cfg.domain_weights
    source_weights = cfg.source_weights

    iterators = {}
    for domain, sources in cfg.source_paths.items():
        for source, paths in sources.items():
            key = f"{domain}:{source}"
            iterators[key] = _iter_docs_for_source(paths)

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
                    entry = {"row": seq_idx, "segments": segments}
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
                    logger.info("Pack progress: sequences=%s docs_seen=%s", seq_idx, docs_seen)
                    last_logged = seq_idx
        tail = builder.finalize()
        if tail is not None and seq_idx < num_seqs:
            mmap[seq_idx, :] = np.asarray(tail.tokens, dtype=np.int32)
            if index_writer is not None:
                segments = [s for s in tail.segments if s.get("doc_id")]
                entry = {"row": seq_idx, "segments": segments}
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

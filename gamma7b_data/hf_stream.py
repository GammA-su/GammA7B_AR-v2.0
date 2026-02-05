from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import itertools
import time

from datasets import load_dataset
from tqdm import tqdm

from .hf_configs import filter_hf_configs
from .schema import NormalizedDocument
from .utils import open_zst_writer, stable_hash


HF_SOURCE_MAP: Dict[str, Dict] = {
    "fineweb_edu_default": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config_name": "default",
        "split": "train",
        "text_field": "text",
        "id_field": "id",
        "meta_fields": ["url", "score", "language_score"],
        "domain": "filtered_web",
    },
    "fineweb2": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config_name": "eng_Latn",
        "split": "train",
        "text_field": "text",
        "id_field": "id",
        "meta_fields": ["url", "date", "timestamp", "language", "language_score", "minhash_cluster_size"],
        "domain": "filtered_web",
    },
    "the-stack-v2": {
        "dataset": "bigcode/the-stack-v2",
        "split": "train",
        "text_field": "content",
        "id_field": "id",
        "meta_fields": ["path", "language", "repo_name"],
        "domain": "code_docs",
    },
    "starcoder2data-extras": {
        "dataset": "bigcode/starcoder2data-extras",
        "split": "train",
        "text_field": "content",
        "id_field": "id",
        "meta_fields": ["path", "language", "repo_name"],
        "domain": "code_docs",
    },
    "open-web-math": {
        "dataset": "open-web-math/open-web-math",
        "split": "train",
        "text_field": "text",
        "id_field": "id",
        "meta_fields": ["source"],
        "domain": "math",
    },
    "arxiv-abstracts-2021": {
        "dataset": "gfissore/arxiv-abstracts-2021",
        "split": "train",
        "text_field": "abstract",
        "id_field": "id",
        "meta_fields": ["title", "categories", "authors", "update_date"],
        "domain": "paper_teasers",
    },
    "openstax-text": {
        "dataset": "crumb/openstax-text",
        "split": "train",
        "text_field": "text",
        "id_field": "id",
        "meta_fields": ["title", "book", "chapter"],
        "domain": "books_longform",
    },
    "open-text-books": {
        "dataset": "izumi-lab/open-text-books",
        "split": "train",
        "text_field": "text",
        "id_field": "id",
        "meta_fields": ["title", "url"],
        "domain": "books_longform",
    },
    "gutenberg_clean_en": {
        "dataset": "nikolina-p/gutenberg_clean_en",
        "split": "train",
        "text_field": "text",
        "id_field": "id",
        "meta_fields": ["title", "author"],
        "domain": "books_longform",
    },
}


def _iter_hf_rows_with_retries(
    dataset_name: str,
    split: str,
    config_name: Optional[str],
    revision: Optional[str],
    limit: Optional[int],
    max_retries: int,
    retry_wait: float,
    logger=None,
) -> Iterator[Tuple[int, Dict]]:
    idx = 0
    attempt = 0
    while True:
        try:
            if config_name:
                ds = load_dataset(dataset_name, config_name, split=split, streaming=True, revision=revision)
            else:
                ds = load_dataset(dataset_name, split=split, streaming=True, revision=revision)
            for row in itertools.islice(ds, idx, None):
                yield idx, row
                idx += 1
                if limit is not None and idx >= limit:
                    return
            return
        except ValueError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            if logger:
                logger.warning(
                    "HF stream error (attempt %s/%s): %s; retrying in %.1fs (resume idx=%s)",
                    attempt,
                    max_retries,
                    exc,
                    retry_wait,
                    idx,
                )
            time.sleep(max(0.0, float(retry_wait)))


def stream_hf_dataset_to_normalized(
    dataset_name: str,
    out_path: Path,
    domain: str,
    source: str,
    limit: Optional[int] = None,
    split: Optional[str] = None,
    config_name: Optional[str] = None,
    text_field: Optional[str] = None,
    id_field: Optional[str] = None,
    meta_fields: Optional[List[str]] = None,
    license_tag: Optional[str] = None,
    revision: Optional[str] = None,
    max_retries: int = 5,
    retry_wait: float = 1.0,
    logger=None,
    log_every: int = 300,
) -> None:
    split = split or "train"
    text_field = text_field or "text"
    id_field = id_field or "id"
    meta_fields = meta_fields or []

    try:
        if config_name:
            load_dataset(dataset_name, config_name, split=split, streaming=True, revision=revision)
        else:
            load_dataset(dataset_name, split=split, streaming=True, revision=revision)
    except ValueError as exc:
        if config_name is None:
            hint = ""
            suggestion = f"python -m gamma7b_data.cli hf-configs --dataset {dataset_name}"
            if "starcoder2data-extras" in dataset_name:
                suggestion = f"{suggestion} --contains documentation"
            else:
                suggestion = f"{suggestion} --lang eng --script Latn"
            try:
                from datasets import get_dataset_config_names

                configs = get_dataset_config_names(dataset_name)
                eng_configs = filter_hf_configs(configs, lang="eng")
                if eng_configs:
                    sample = eng_configs[:5]
                    hint = f" Available eng configs: {', '.join(sample)}."
                elif configs:
                    sample = configs[:5]
                    hint = f" Example configs: {', '.join(sample)}."
            except Exception:
                hint = ""
            raise ValueError(
                f"Dataset '{dataset_name}' requires a config_name. "
                f"Run `{suggestion}` to list available configs."
                f"{hint}"
            ) from exc
        raise

    if logger:
        logger.info(
            "HF stream config: dataset=%s config=%s split=%s text_field=%s id_field=%s",
            dataset_name,
            config_name,
            split,
            text_field,
            id_field,
        )
    total = 0
    with open_zst_writer(out_path) as writer:
        rows = _iter_hf_rows_with_retries(
            dataset_name=dataset_name,
            split=split,
            config_name=config_name,
            revision=revision,
            limit=limit,
            max_retries=max_retries,
            retry_wait=retry_wait,
            logger=logger,
        )
        for idx, row in tqdm(rows, desc=f"stream {source}", unit="docs"):
            text = row.get(text_field) or ""
            doc_id = row.get(id_field) or stable_hash(f"{source}:{idx}:{text}")
            meta = {k: row.get(k) for k in meta_fields if k in row}
            created_at = row.get("timestamp") or row.get("date")
            doc = NormalizedDocument(
                text=text,
                source=source,
                domain=domain,
                doc_id=str(doc_id),
                license_tag=license_tag,
                created_at=created_at,
                meta=meta or None,
            )
            writer.write((json_dumps(doc.to_json()) + "\n").encode("utf-8"))
            total += 1
            if logger and total % log_every == 0:
                logger.info("HF stream progress %s: docs=%s", source, total)
    if logger:
        logger.info("HF stream done %s: docs=%s out=%s", source, total, out_path)


def stream_hf_to_normalized(
    source_name: str,
    out_path: Path,
    domain: Optional[str] = None,
    source: Optional[str] = None,
    limit: Optional[int] = None,
    split: Optional[str] = None,
    config_name: Optional[str] = None,
    text_field: Optional[str] = None,
    id_field: Optional[str] = None,
    meta_fields: Optional[List[str]] = None,
    license_tag: Optional[str] = None,
    max_retries: int = 5,
    retry_wait: float = 1.0,
    logger=None,
    log_every: int = 300,
) -> None:
    cfg = HF_SOURCE_MAP.get(source_name)
    if cfg is None:
        raise ValueError(f"Unknown HF source: {source_name}")
    dataset_name = cfg["dataset"]
    config_name = config_name or cfg.get("config_name")
    split = split or cfg.get("split", "train")
    text_field = text_field or cfg.get("text_field", "text")
    id_field = id_field or cfg.get("id_field", "id")
    meta_fields = meta_fields or cfg.get("meta_fields", [])
    domain = domain or cfg.get("domain", "filtered_web")
    source = source or source_name
    revision = cfg.get("revision")

    stream_hf_dataset_to_normalized(
        dataset_name=dataset_name,
        out_path=out_path,
        domain=domain,
        source=source,
        limit=limit,
        split=split,
        config_name=config_name,
        text_field=text_field,
        id_field=id_field,
        meta_fields=meta_fields,
        license_tag=license_tag,
        revision=revision,
        max_retries=max_retries,
        retry_wait=retry_wait,
        logger=logger,
        log_every=log_every,
    )


def json_dumps(payload: Dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)

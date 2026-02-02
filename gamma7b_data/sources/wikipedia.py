import bz2
import gzip
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional

from ..utils import open_zst_writer


def open_maybe_compressed(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, "rb")
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def strip_wikitext(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"''+", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_pages(path: Path, limit: Optional[int] = None) -> Iterator[dict]:
    with open_maybe_compressed(path) as fh:
        context = ET.iterparse(fh, events=("end",))
        count = 0
        for _, elem in context:
            if elem.tag.endswith("page"):
                ns = elem.findtext("./ns")
                if ns != "0":
                    elem.clear()
                    continue
                title = elem.findtext("./title") or ""
                page_id = elem.findtext("./id") or ""
                rev = elem.find("./revision")
                text = ""
                timestamp = ""
                if rev is not None:
                    text = rev.findtext("./text") or ""
                    timestamp = rev.findtext("./timestamp") or ""
                yield {
                    "id": page_id,
                    "title": title,
                    "timestamp": timestamp,
                    "text": strip_wikitext(text),
                }
                count += 1
                if limit is not None and count >= limit:
                    break
                elem.clear()


def ingest_wikipedia(path: Path, out_path: Path, limit: Optional[int] = None, logger=None, log_every: int = 300) -> None:
    if logger:
        logger.info("Wikipedia XML ingest config: input=%s limit=%s", path, limit)
    with open_zst_writer(out_path) as writer:
        count = 0
        for page in iter_pages(path, limit=limit):
            writer.write((json.dumps(page, ensure_ascii=False) + "\n").encode("utf-8"))
            count += 1
            if logger and count % log_every == 0:
                logger.info("Wikipedia XML ingest progress: pages=%s", count)
    if logger:
        logger.info("Wikipedia XML ingest done: pages=%s out=%s", count, out_path)

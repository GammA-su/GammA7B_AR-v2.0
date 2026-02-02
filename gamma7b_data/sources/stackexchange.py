import json
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from ..utils import open_zst_writer


def strip_html(text: str) -> str:
    text = re.sub(r"<pre><code>(.*?)</code></pre>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<code>(.*?)</code>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_posts(posts_path: Path) -> Iterator[dict]:
    try:
        context = ET.iterparse(posts_path, events=("end",))
        for _, elem in context:
            if elem.tag.endswith("row"):
                yield elem.attrib
                elem.clear()
        return
    except ET.ParseError:
        pass

    text = posts_path.read_text(encoding="utf-8", errors="ignore")
    pos = 0
    while True:
        start = text.find("<row", pos)
        if start == -1:
            break
        i = start + 4
        in_quote = None
        while i < len(text):
            ch = text[i]
            if ch in ("\"", "'"):
                if in_quote is None:
                    in_quote = ch
                elif in_quote == ch:
                    in_quote = None
            elif ch == ">" and in_quote is None:
                tag_body = text[start + 4 : i]
                attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag_body))
                if attrs:
                    yield attrs
                pos = i + 1
                break
            i += 1
        else:
            break


def resolve_posts_path(path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if path.is_dir():
        candidate = path / "Posts.xml"
        if not candidate.exists():
            raise FileNotFoundError(f"Posts.xml not found in {path}")
        return candidate, None
    if path.suffix == ".7z":
        try:
            import py7zr
        except ImportError as exc:
            raise ImportError("py7zr is required to extract .7z StackExchange archives.") from exc
        tmpdir = tempfile.TemporaryDirectory()
        with py7zr.SevenZipFile(path, mode="r") as archive:
            archive.extractall(path=tmpdir.name)
        candidate = Path(tmpdir.name) / "Posts.xml"
        if not candidate.exists():
            # Try any nested path
            found = list(Path(tmpdir.name).rglob("Posts.xml"))
            if not found:
                raise FileNotFoundError("Posts.xml not found in archive")
            candidate = found[0]
        return candidate, tmpdir
    if path.suffix == ".xml":
        return path, None
    raise ValueError(f"Unsupported StackExchange input: {path}")


def ingest_stackexchange(path: Path, out_path: Path, limit: Optional[int] = None, logger=None, log_every: int = 300) -> None:
    posts_path, tmpdir = resolve_posts_path(path)
    if logger:
        logger.info("StackExchange raw ingest config: posts=%s limit=%s", posts_path, limit)
    questions: Dict[str, dict] = {}
    best_answers: Dict[str, Tuple[int, dict]] = {}
    pending_answers: Dict[str, Tuple[int, dict]] = {}

    count = 0
    for row in iter_posts(posts_path):
        post_type = row.get("PostTypeId")
        if post_type == "1":
            qid = row.get("Id")
            if not qid:
                continue
            question = {
                "id": qid,
                "title": row.get("Title", ""),
                "body": strip_html(row.get("Body", "")),
                "tags": row.get("Tags", ""),
                "creation_date": row.get("CreationDate", ""),
                "accepted_answer_id": row.get("AcceptedAnswerId"),
                "score": int(row.get("Score", "0")),
            }
            questions[qid] = question
            if qid in pending_answers:
                best_answers[qid] = pending_answers.pop(qid)
        elif post_type == "2":
            parent = row.get("ParentId")
            if not parent:
                continue
            answer = {
                "id": row.get("Id"),
                "body": strip_html(row.get("Body", "")),
                "score": int(row.get("Score", "0")),
            }
            score = answer["score"]
            if parent in questions:
                current = best_answers.get(parent)
                if current is None or score > current[0]:
                    best_answers[parent] = (score, answer)
            else:
                pending = pending_answers.get(parent)
                if pending is None or score > pending[0]:
                    pending_answers[parent] = (score, answer)
        if limit is not None and len(questions) >= limit:
            break
        if logger and len(questions) and len(questions) % log_every == 0:
            logger.info("StackExchange raw parse progress: questions=%s", len(questions))

    with open_zst_writer(out_path) as writer:
        for qid, question in questions.items():
            answer_tuple = best_answers.get(qid)
            answer = answer_tuple[1] if answer_tuple else {"body": ""}
            record = {
                "id": qid,
                "question": question,
                "answer": answer,
            }
            writer.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            count += 1
            if logger and count % log_every == 0:
                logger.info("StackExchange raw ingest progress: docs=%s", count)
            if limit is not None and count >= limit:
                break
    if logger:
        logger.info("StackExchange raw ingest done: docs=%s out=%s", count, out_path)
    if tmpdir is not None:
        tmpdir.cleanup()

from pathlib import Path

from gamma7b_data.stackexchange import ingest_stackexchange
from gamma7b_data.utils import read_jsonl


def test_stackexchange_parser(tmp_path: Path):
    xml = """<?xml version='1.0' encoding='utf-8'?>
<posts>
  <row Id="1" PostTypeId="1" Title="Test Q" Body="<p>Question body</p>" Tags="&lt;python&gt;" CreationDate="2020-01-01" Score="1" />
  <row Id="2" PostTypeId="2" ParentId="1" Body="<p>Answer body</p>" Score="5" />
</posts>
"""
    posts_path = tmp_path / "Posts.xml"
    posts_path.write_text(xml, encoding="utf-8")

    out_path = tmp_path / "out.jsonl.zst"
    ingest_stackexchange(posts_path, out_path, limit=10)

    records = list(read_jsonl(out_path))
    assert len(records) == 1
    record = records[0]
    assert record["source"] == "stackexchange"
    assert record["domain"] == "forums_qa"
    assert record["doc_id"]
    assert "Question body" in record["text"]
    assert "Answer body" in record["text"]

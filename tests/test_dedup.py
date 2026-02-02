from gamma7b_data.dedup import Deduper
from gamma7b_data.schema import NormalizedDocument
from gamma7b_data.cli import dedup_cmd
from gamma7b_data.utils import read_jsonl


def test_exact_and_near_dedup():
    deduper = Deduper(simhash_threshold=3, hash_bits=64)
    doc1 = NormalizedDocument(text="hello world", source="s", domain="d", doc_id="1")
    doc2 = NormalizedDocument(text="hello world", source="s", domain="d", doc_id="2")
    doc3 = NormalizedDocument(text="Hello world", source="s", domain="d", doc_id="3")

    d1 = deduper.check(doc1, use_near=True)
    d2 = deduper.check(doc2, use_near=True)
    d3 = deduper.check(doc3, use_near=True)

    assert d1.keep is True
    assert d2.keep is False
    assert d2.reason == "exact_dup"
    assert d3.keep is False
    assert d3.reason == "near_dup"


def test_dedup_keeps_best_in_cluster(tmp_path):
    records = [
        {
            "text": "Short doc. " * 5,
            "source": "s",
            "domain": "d",
            "doc_id": "doc_short",
            "license_tag": None,
            "created_at": None,
            "meta": {"url": "https://example.com/a"},
        },
        {
            "text": "Longer doc with more content. " * 50,
            "source": "s",
            "domain": "d",
            "doc_id": "doc_long",
            "license_tag": None,
            "created_at": None,
            "meta": {"url": "https://example.com/b"},
        },
    ]
    input_path = tmp_path / "in.jsonl"
    import json

    with input_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    out_dir = tmp_path / "out"
    dedup_cmd(
        input=[str(input_path)],
        out_dir=out_dir,
        mode="near",
        scope="global",
        simhash_threshold=64,
        hash_bits=64,
        seed=0,
    )

    kept = list(read_jsonl(out_dir / "deduped.jsonl.zst"))
    assert len(kept) == 1
    assert kept[0]["doc_id"] == "doc_long"

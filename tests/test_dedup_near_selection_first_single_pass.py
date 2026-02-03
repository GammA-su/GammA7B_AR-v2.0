import json

from gamma7b_data.cli import dedup_cmd
from gamma7b_data.utils import read_jsonl


def test_dedup_near_selection_first_keeps_first(tmp_path):
    records = [
        {
            "text": "Near duplicate doc. " * 10,
            "source": "s",
            "domain": "d",
            "doc_id": "doc_first",
            "license_tag": None,
            "created_at": None,
            "meta": {"url": "https://example.com/a"},
        },
        {
            "text": "Near duplicate doc. " * 9 + "extra",
            "source": "s",
            "domain": "d",
            "doc_id": "doc_second",
            "license_tag": None,
            "created_at": None,
            "meta": {"url": "https://example.com/b"},
        },
    ]
    input_path = tmp_path / "in.jsonl"
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
        near_selection="first",
    )

    kept = list(read_jsonl(out_dir / "deduped.jsonl.zst"))
    assert len(kept) == 1
    assert kept[0]["doc_id"] == "doc_first"

from gamma7b_data.cli import dedup_cmd
from gamma7b_data.utils import read_jsonl


def test_near_dup_with_punctuation_and_whitespace(tmp_path):
    records = [
        {
            "text": "Hello, world! This is   a test. Don't panic.",
            "source": "s",
            "domain": "d",
            "doc_id": "doc_a",
            "license_tag": None,
            "created_at": None,
            "meta": {"url": "https://example.com/a"},
        },
        {
            "text": "Hello world this is a test dont   panic",
            "source": "s",
            "domain": "d",
            "doc_id": "doc_b",
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
        simhash_threshold=3,
        hash_bits=64,
        seed=0,
    )

    kept = list(read_jsonl(out_dir / "deduped.jsonl.zst"))
    assert len(kept) == 1

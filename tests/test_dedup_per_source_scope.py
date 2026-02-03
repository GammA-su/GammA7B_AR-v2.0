import json

from gamma7b_data.cli import dedup_cmd
from gamma7b_data.utils import open_zst_writer, read_jsonl


def _write_zst(path, records):
    with open_zst_writer(path) as fh:
        for record in records:
            fh.write((json.dumps(record) + "\n").encode("utf-8"))


def test_dedup_per_source_scope(tmp_path):
    records = [
        {
            "text": "Same text",
            "source": "source_a",
            "domain": "domain_a",
            "doc_id": "doc_a",
            "license_tag": None,
            "created_at": None,
            "meta": None,
        },
        {
            "text": "Same text",
            "source": "source_b",
            "domain": "domain_b",
            "id": "legacy_b",
            "license_tag": None,
            "created_at": None,
            "meta": None,
        },
    ]
    input_path = tmp_path / "in.jsonl.zst"
    _write_zst(input_path, records)

    out_global = tmp_path / "out_global"
    dedup_cmd(
        input=[str(input_path)],
        out_dir=out_global,
        mode="exact",
        scope="global",
        simhash_threshold=3,
        hash_bits=64,
        seed=0,
    )
    kept_global = list(read_jsonl(out_global / "deduped.jsonl.zst"))
    assert len(kept_global) == 1

    out_per_source = tmp_path / "out_per_source"
    dedup_cmd(
        input=[str(input_path)],
        out_dir=out_per_source,
        mode="exact",
        scope="per-source",
        simhash_threshold=3,
        hash_bits=64,
        seed=0,
    )
    kept_per_source = list(read_jsonl(out_per_source / "deduped.jsonl.zst"))
    assert len(kept_per_source) == 2

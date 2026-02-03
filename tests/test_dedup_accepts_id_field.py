import json

from gamma7b_data.cli import dedup_cmd
from gamma7b_data.utils import open_zst_writer, read_jsonl


def test_dedup_accepts_id_field(tmp_path):
    in_path = tmp_path / "in.jsonl.zst"
    with open_zst_writer(in_path) as fh:
        record = {
            "text": "hello",
            "source": "s",
            "domain": "d",
            "id": "legacy-id",
            "license_tag": None,
            "created_at": None,
            "meta": None,
        }
        fh.write((json.dumps(record) + "\n").encode("utf-8"))

    out_dir = tmp_path / "out"
    dedup_cmd(
        input=[str(in_path)],
        out_dir=out_dir,
        mode="exact",
        scope="global",
        simhash_threshold=3,
        hash_bits=64,
        seed=0,
    )

    kept = list(read_jsonl(out_dir / "deduped.jsonl.zst"))
    assert len(kept) == 1
    assert kept[0]["doc_id"] == "legacy-id"

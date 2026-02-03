import json
from pathlib import Path

from gamma7b_data.cli import dedup_cmd
from gamma7b_data.utils import open_zst_writer, read_jsonl


def test_dedup_discovers_jsonlzst(tmp_path: Path):
    data_path = tmp_path / "x.jsonl.zst"
    with open_zst_writer(data_path) as fh:
        fh.write(
            (
                json.dumps(
                    {
                        "text": "a",
                        "source": "s",
                        "domain": "d",
                        "doc_id": "1",
                        "license_tag": None,
                        "created_at": None,
                        "meta": None,
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        fh.write(
            (
                json.dumps(
                    {
                        "text": "b",
                        "source": "s",
                        "domain": "d",
                        "doc_id": "2",
                        "license_tag": None,
                        "created_at": None,
                        "meta": None,
                    }
                )
                + "\n"
            ).encode("utf-8")
        )

    manifest = tmp_path / "x.jsonl.manifest.json"
    manifest.write_text(json.dumps({"shard_path": str(data_path)}), encoding="utf-8")

    out_dir = tmp_path / "out"
    dedup_cmd(
        input=[str(tmp_path)],
        out_dir=out_dir,
        mode="exact",
        scope="per-source",
        simhash_threshold=3,
        hash_bits=64,
        seed=0,
    )

    kept = list(read_jsonl(out_dir / "deduped.jsonl.zst"))
    assert len(kept) == 2

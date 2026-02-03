import json
from pathlib import Path

from gamma7b_data.cli import dedup_cmd
from gamma7b_data.utils import open_zst_writer, read_jsonl


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_dedup_ignores_manifest_json(tmp_path: Path):
    a_jsonl = tmp_path / "a.jsonl"
    _write_jsonl(
        a_jsonl,
        [
            {
                "text": "hello",
                "source": "s",
                "domain": "d",
                "doc_id": "1",
                "license_tag": None,
                "created_at": None,
                "meta": None,
            }
        ],
    )

    b_jsonl_zst = tmp_path / "b.jsonl.zst"
    with open_zst_writer(b_jsonl_zst) as fh:
        fh.write(
            (
                json.dumps(
                    {
                        "text": "world",
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

    manifest = tmp_path / "c.jsonl.manifest.json"
    manifest.write_text(json.dumps({"foo": "bar"}, indent=2), encoding="utf-8")

    out_dir = tmp_path / "out"
    dedup_cmd(
        input=[str(tmp_path / "*")],
        out_dir=out_dir,
        mode="exact",
        scope="global",
        simhash_threshold=3,
        hash_bits=64,
        seed=0,
    )

    kept = list(read_jsonl(out_dir / "deduped.jsonl.zst"))
    assert len(kept) == 2

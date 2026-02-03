import json
import os

from gamma7b_data.cli import dedup_cmd
from gamma7b_data.utils import open_zst_writer, read_jsonl


def _write_zst(path, records):
    with open_zst_writer(path) as fh:
        for record in records:
            fh.write((json.dumps(record) + "\n").encode("utf-8"))


def test_dedup_resume(tmp_path):
    inputs = []
    records = [
        {
            "text": "Near duplicate doc. " * 10,
            "source": "s",
            "domain": "d",
            "doc_id": "doc_a",
            "license_tag": None,
            "created_at": None,
            "meta": None,
        },
        {
            "text": "Near duplicate doc. " * 9 + "extra",
            "source": "s",
            "domain": "d",
            "doc_id": "doc_b",
            "license_tag": None,
            "created_at": None,
            "meta": None,
        },
        {
            "text": "Another doc.",
            "source": "s",
            "domain": "d",
            "doc_id": "doc_c",
            "license_tag": None,
            "created_at": None,
            "meta": None,
        },
    ]
    for idx in range(3):
        path = tmp_path / f"in_{idx}.jsonl.zst"
        _write_zst(path, [records[idx]])
        inputs.append(str(path))

    out_expected = tmp_path / "out_expected"
    dedup_cmd(
        input=inputs,
        out_dir=out_expected,
        mode="near",
        scope="global",
        simhash_threshold=64,
        hash_bits=64,
        seed=0,
        near_selection="best",
    )
    expected = list(read_jsonl(out_expected / "deduped.jsonl.zst"))

    out_resume = tmp_path / "out_resume"
    os.environ["GAMMA7B_DEDUP_STOP_AFTER_FILES"] = "2"
    try:
        dedup_cmd(
            input=inputs,
            out_dir=out_resume,
            mode="near",
            scope="global",
            simhash_threshold=64,
            hash_bits=64,
            seed=0,
            near_selection="best",
        )
    except SystemExit:
        pass
    finally:
        os.environ.pop("GAMMA7B_DEDUP_STOP_AFTER_FILES", None)

    assert (out_resume / "dedup_state.pkl").exists()

    dedup_cmd(
        input=inputs,
        out_dir=out_resume,
        mode="near",
        scope="global",
        simhash_threshold=64,
        hash_bits=64,
        seed=0,
        near_selection="best",
        resume=True,
    )
    resumed = list(read_jsonl(out_resume / "deduped.jsonl.zst"))
    assert resumed == expected

import io
import json

from gamma7b_data.repair_jsonl import repair_jsonl_zst
from gamma7b_data.utils import open_zst_reader, open_zst_writer


def test_repair_jsonl_zst_basic(tmp_path):
    in_path = tmp_path / "broken.jsonl.zst"
    with open_zst_writer(in_path) as fh:
        fh.write(b'{"text":"hello\\nworld"}\n')
        fh.write(b'{"text":"broken\nline"}\n')
        fh.write(b'{"text":"ok"}\n')
    out_path = tmp_path / "fixed.jsonl.zst"
    result = repair_jsonl_zst(in_path, out_path, inplace=False, backup=False)

    assert result["records_ok"] == 3
    with open_zst_reader(out_path) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        lines = [json.loads(line) for line in text_stream if line]
    assert lines[1]["text"] == "broken\nline"


def test_repair_jsonl_zst_inplace_backup(tmp_path):
    in_path = tmp_path / "broken.jsonl.zst"
    with open_zst_writer(in_path) as fh:
        fh.write(b'{"text":"a\nb"}\n')
    result = repair_jsonl_zst(in_path, in_path, inplace=True, backup=True)
    assert result["backup_path"] is not None
    assert (in_path.with_suffix(in_path.suffix + ".bak")).exists()

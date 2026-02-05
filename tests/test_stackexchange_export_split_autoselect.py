from pathlib import Path
import runpy
import sys

from gamma7b_data.utils import read_jsonl


def test_stackexchange_export_split_autoselect(tmp_path: Path):
    out_dir = tmp_path / "out"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "41_hf_export_stackexchange.py"

    argv = [
        str(script_path),
        "--split",
        "auto",
        "--out-dir",
        str(out_dir),
        "--max-docs",
        "5",
        "--part-size",
        "5",
        "--zstd-level",
        "1",
        "--zstd-threads",
        "0",
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv

    part_path = out_dir / "part_00000.jsonl.zst"
    assert part_path.exists()
    rows = list(read_jsonl(part_path))
    assert len(rows) >= 1

import json
from pathlib import Path

from gamma7b_data.cli import manifest_audit_cmd
from gamma7b_data.utils import open_zst_writer


def _write_manifest(path: Path, glob_pattern: str) -> None:
    payload = f"""
seed: 0
outputs:
  normalized_dir: out/normalized
domains:
  reference:
    weight: 1.0
    sources:
      local_ref:
        weight: 1.0
        type: local_dir
        params:
          paths:
            - {glob_pattern}
"""
    path.write_text(payload.strip() + "\n", encoding="utf-8")


def test_manifest_audit_probe_error_message(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "doc1.jsonl.zst"
    with open_zst_writer(file_path) as fh:
        fh.write(b"{not-json}\n")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, str(data_dir / "*.jsonl.zst"))

    out_path = tmp_path / "report.json"
    try:
        manifest_audit_cmd(manifest=manifest_path, strict=True, out=out_path)
    except Exception as exc:
        msg = str(exc)
        assert "Probe errors detected" in msg
        assert "local_ref" in msg
        assert out_path.exists()
        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert report["probe_errors"]
    else:
        raise AssertionError("Expected strict manifest audit to fail with probe errors message")

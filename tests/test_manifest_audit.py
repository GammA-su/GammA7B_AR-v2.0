import json
from pathlib import Path

from gamma7b_data.cli import manifest_audit_cmd
from gamma7b_data.utils import open_zst_writer


def _write_manifest(path: Path, glob_pattern: str, weight: float = 1.0) -> None:
    payload = f"""
seed: 0
outputs:
  normalized_dir: out/normalized
domains:
  reference:
    weight: 1.0
    sources:
      local_ref:
        weight: {weight}
        type: local_dir
        params:
          paths:
            - {glob_pattern}
"""
    path.write_text(payload.strip() + "\n", encoding="utf-8")


def test_manifest_audit_passes_when_files_exist(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "doc1.jsonl.zst"
    with open_zst_writer(file_path) as fh:
        fh.write(b'{"text": "hello"}\n')
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, str(data_dir / "*.jsonl.zst"))

    out_path = tmp_path / "report.json"
    manifest_audit_cmd(manifest=manifest_path, strict=True, out=out_path)
    assert out_path.exists()


def test_manifest_audit_fails_on_missing_files(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, str(tmp_path / "missing" / "*.jsonl.zst"))

    try:
        manifest_audit_cmd(manifest=manifest_path, strict=True, out=None)
    except Exception as exc:
        assert "Missing local source" in str(exc)
    else:
        raise AssertionError("Expected manifest audit to fail in strict mode")


def test_manifest_audit_fails_on_placeholder(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "doc1.jsonl.zst"
    with open_zst_writer(file_path) as fh:
        fh.write(b'{"text": "placeholder content", "license_tag": "placeholder"}\n')
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, str(data_dir / "*.jsonl.zst"))

    try:
        manifest_audit_cmd(manifest=manifest_path, strict=True, out=None)
    except Exception as exc:
        assert "Placeholder records detected" in str(exc)
    else:
        raise AssertionError("Expected manifest audit to fail on placeholder records")


def test_manifest_audit_probe_error_strict(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "doc1.jsonl.zst"
    with open_zst_writer(file_path) as fh:
        fh.write(b"{not-json}\n")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, str(data_dir / "*.jsonl.zst"))

    out_path = tmp_path / "report.json"
    manifest_audit_cmd(manifest=manifest_path, strict=False, out=out_path)
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["probe_errors"]

    try:
        manifest_audit_cmd(manifest=manifest_path, strict=True, out=None)
    except Exception as exc:
        assert "Probe errors detected" in str(exc)
    else:
        raise AssertionError("Expected manifest audit to fail on probe errors in strict mode")


def test_manifest_audit_probe_line_too_long(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "doc1.jsonl.zst"
    long_text = "A" * (8 * 1024 * 1024 + 10)
    with open_zst_writer(file_path) as fh:
        fh.write((f'{{"text":"{long_text}"}}\\n').encode("utf-8"))
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, str(data_dir / "*.jsonl.zst"))

    out_path = tmp_path / "report.json"
    manifest_audit_cmd(manifest=manifest_path, strict=True, out=out_path)
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["probe_errors"] == []

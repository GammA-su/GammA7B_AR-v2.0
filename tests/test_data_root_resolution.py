import json
from pathlib import Path

from gamma7b_data.cli import manifest_audit_cmd
from gamma7b_data.utils import open_zst_writer


def _write_manifest(path: Path) -> None:
    payload = """
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
            - data/normalized/reference/local_ref/*.jsonl.zst
"""
    path.write_text(payload.strip() + "\n", encoding="utf-8")


def test_data_root_resolves_relative_paths(tmp_path):
    data_dir = tmp_path / "data" / "normalized" / "reference" / "local_ref"
    data_dir.mkdir(parents=True)
    shard_path = data_dir / "part_00000.jsonl.zst"
    with open_zst_writer(shard_path) as fh:
        fh.write(b'{"text": "hello"}\n')

    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path)

    out_path = tmp_path / "audit.json"
    manifest_audit_cmd(manifest=manifest_path, strict=True, out=out_path, data_root=tmp_path)
    report = json.loads(out_path.read_text(encoding="utf-8"))

    sources = report["sources"]
    assert sources[0]["n_files"] == 1

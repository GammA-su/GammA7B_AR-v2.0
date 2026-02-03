import json
from pathlib import Path

from gamma7b_data.cli import mix_report_cmd
from gamma7b_data.utils import open_zst_writer


def _write_manifest(path: Path, s1_glob: str, s2_glob: str) -> None:
    payload = f"""
seed: 0
outputs:
  normalized_dir: out/normalized
domains:
  d1:
    weight: 0.6
    sources:
      s1:
        weight: 1.0
        type: local_dir
        params:
          paths:
            - {s1_glob}
  d2:
    weight: 0.4
    sources:
      s2:
        weight: 1.0
        type: local_dir
        params:
          paths:
            - {s2_glob}
"""
    path.write_text(payload.strip() + "\n", encoding="utf-8")


def _write_manifest_with_hf(path: Path, s1_glob: str) -> None:
    payload = f"""
seed: 0
outputs:
  normalized_dir: out/normalized
domains:
  d1:
    weight: 0.6
    sources:
      s1:
        weight: 1.0
        type: local_dir
        params:
          paths:
            - {s1_glob}
  d2:
    weight: 0.4
    sources:
      hf_src:
        weight: 1.0
        type: hf_stream
        params:
          dataset: dummy/hf
          split: train
          text_field: text
"""
    path.write_text(payload.strip() + "\n", encoding="utf-8")


def test_mix_report_local_stats(tmp_path):
    s1_dir = tmp_path / "s1"
    s2_dir = tmp_path / "s2"
    s1_dir.mkdir()
    s2_dir.mkdir()

    s1_file = s1_dir / "part_00000.jsonl.zst"
    with open_zst_writer(s1_file) as fh:
        fh.write(b'{"text": "hello"}\n')
        fh.write(b'{"text": "world"}\n')

    s2_file = s2_dir / "part_00000.jsonl.zst"
    with open_zst_writer(s2_file):
        pass

    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, str(s1_dir / "*.jsonl.zst"), str(s2_dir / "*.jsonl.zst"))

    out_path = tmp_path / "report.json"
    mix_report_cmd(manifest=manifest_path, out=out_path, max_docs_per_source=20000)
    report = json.loads(out_path.read_text(encoding="utf-8"))

    sources = {f"{s['domain']}/{s['source']}": s for s in report["sources"]}
    assert sources["d1/s1"]["target_weight_domain"] == 0.6
    assert sources["d1/s1"]["target_weight_source"] == 1.0
    assert sources["d1/s1"]["n_docs"] == 2
    assert sources["d2/s2"]["n_docs"] == 0
    assert str(s2_file) in report["empty_zst_files"]


def test_mix_report_hf_stream_sampling(monkeypatch, tmp_path):
    s1_dir = tmp_path / "s1"
    s1_dir.mkdir()

    s1_file = s1_dir / "part_00000.jsonl.zst"
    with open_zst_writer(s1_file) as fh:
        fh.write(b'{"text": "hello"}\n')

    def fake_load_dataset(dataset, config_name=None, split=None, streaming=None, revision=None):
        assert dataset == "dummy/hf"
        assert split == "train"
        assert streaming is True
        return iter([{"text": "aaa"}, {"text": "bbbb"}])

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest_with_hf(manifest_path, str(s1_dir / "*.jsonl.zst"))

    out_path = tmp_path / "report.json"
    mix_report_cmd(
        manifest=manifest_path,
        out=out_path,
        max_docs_per_source=20000,
        sample_hf_stream_docs=2,
        max_sample_chars_per_doc=20000,
    )
    report = json.loads(out_path.read_text(encoding="utf-8"))
    sources = {f"{s['domain']}/{s['source']}": s for s in report["sources"]}
    assert sources["d2/hf_src"]["n_docs_sampled"] == 2
    assert "domain_shares_estimated_all_sources" in report


def test_mix_report_hf_stream_sampling_error(monkeypatch, tmp_path):
    s1_dir = tmp_path / "s1"
    s1_dir.mkdir()

    s1_file = s1_dir / "part_00000.jsonl.zst"
    with open_zst_writer(s1_file) as fh:
        fh.write(b'{"text": "hello"}\n')

    def fake_load_dataset(dataset, config_name=None, split=None, streaming=None, revision=None):
        raise RuntimeError("boom")

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest_with_hf(manifest_path, str(s1_dir / "*.jsonl.zst"))

    out_path = tmp_path / "report.json"
    mix_report_cmd(
        manifest=manifest_path,
        out=out_path,
        max_docs_per_source=20000,
        sample_hf_stream_docs=2,
        max_sample_chars_per_doc=20000,
    )
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["sample_errors"]
    sources = {f"{s['domain']}/{s['source']}": s for s in report["sources"]}
    assert sources["d2/hf_src"]["sample_error"] == "boom"

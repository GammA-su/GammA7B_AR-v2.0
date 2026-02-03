import io
import json

from gamma7b_data.utils import open_zst_reader


def test_hf_to_local_normalized_truncation_and_config_alias(monkeypatch, tmp_path):
    calls = {}

    def fake_load_dataset(dataset, config_name=None, split=None, streaming=None):
        calls["dataset"] = dataset
        calls["config_name"] = config_name
        calls["split"] = split
        calls["streaming"] = streaming
        if split == "validation":
            raise ValueError("split not found")
        return iter([{"text": "abcdef", "id": "doc1"}])

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(datasets, "get_dataset_split_names", lambda dataset, config_name=None: ["train", "test"])

    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "hf_to_local_normalized.py"
    spec = importlib.util.spec_from_file_location("hf_to_local_normalized", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    out_path = tmp_path / "out.jsonl.zst"
    module.main(
        [
            "--dataset",
            "dummy/ds",
            "--config",
            "documentation",
            "--split",
            "validation",
            "--out-path",
            str(out_path),
            "--max-chars",
            "3",
        ]
    )

    assert calls["dataset"] == "dummy/ds"
    assert calls["config_name"] == "documentation"
    assert calls["split"] == "train"
    assert calls["streaming"] is True

    with open_zst_reader(out_path) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        line = text_stream.readline()
    payload = json.loads(line)
    assert payload["text"] == "abc"

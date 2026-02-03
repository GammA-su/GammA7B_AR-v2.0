from gamma7b_data.hf_stream import stream_hf_dataset_to_normalized


def test_hf_stream_passes_config_name(monkeypatch, tmp_path):
    calls = {}

    def fake_load_dataset(dataset, config_name=None, split=None, streaming=None, revision=None):
        calls["dataset"] = dataset
        calls["config_name"] = config_name
        calls["split"] = split
        calls["streaming"] = streaming
        calls["revision"] = revision
        return []

    import gamma7b_data.hf_stream as hf_stream

    monkeypatch.setattr(hf_stream, "load_dataset", fake_load_dataset)

    out_path = tmp_path / "out.jsonl.zst"
    stream_hf_dataset_to_normalized(
        dataset_name="bigcode/starcoder2data-extras",
        config_name="documentation",
        split="train",
        out_path=out_path,
        domain="code_docs",
        source="starcoder2data-extras",
    )

    assert calls["dataset"] == "bigcode/starcoder2data-extras"
    assert calls["config_name"] == "documentation"
    assert calls["split"] == "train"
    assert calls["streaming"] is True

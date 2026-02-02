from gamma7b_data.cli import hf_configs_cmd


def test_hf_configs_filter(monkeypatch, capsys):
    import datasets

    def fake_configs(_dataset):
        return ["eng_Latn", "fra_Latn", "deu_Latn"]

    monkeypatch.setattr(datasets, "get_dataset_config_names", fake_configs)

    hf_configs_cmd(dataset="dummy/ds", filter="eng")
    out = capsys.readouterr().out.strip().splitlines()

    assert out[0] == "eng_Latn"
    assert out[-1] == "Total configs: 1"

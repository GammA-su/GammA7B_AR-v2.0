import sys
import types

from gamma7b_data.hf_configs import filter_hf_configs, get_hf_dataset_config_names


def test_hf_configs_readme_fallback_filters_lang_script(monkeypatch, tmp_path):
    readme = """---
configs:
  - config_name: eng_Latn
  - config_name: fra_Latn
  - config_name: eng_Cyrl
---
"""
    readme_path = tmp_path / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    fake_datasets = types.SimpleNamespace(
        get_dataset_config_names=lambda _dataset: (_ for _ in ()).throw(RuntimeError("datasets unavailable"))
    )
    fake_hf_hub = types.SimpleNamespace(
        hf_hub_download=lambda **_kwargs: str(readme_path),
        list_repo_files=lambda **_kwargs: [],
    )

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)

    configs, note = get_hf_dataset_config_names("dummy/dataset")
    assert "README fallback" in (note or "")
    assert configs == ["eng_Latn", "fra_Latn", "eng_Cyrl"]
    assert filter_hf_configs(configs, lang="eng", script="Latn") == ["eng_Latn"]


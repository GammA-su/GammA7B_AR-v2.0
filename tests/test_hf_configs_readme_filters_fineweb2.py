from gamma7b_data.hf_configs import filter_hf_configs


def test_hf_configs_readme_filters_fineweb2_lang_script():
    configs = ["eng_Latn", "fra_Latn", "eng_Cyrl", "deu_Latn"]
    assert filter_hf_configs(configs, lang="eng", script="Latn") == ["eng_Latn"]

from gamma7b_data.hf_configs import filter_hf_configs


def test_hf_configs_filter_lang_prefix():
    configs = ["asm_Beng", "fra_Latn", "eng_Latn", "eng_Cyrl"]
    assert filter_hf_configs(configs, lang="eng") == ["eng_Latn", "eng_Cyrl"]


def test_hf_configs_filter_lang_script():
    configs = ["asm_Beng", "fra_Latn", "eng_Latn", "eng_Cyrl"]
    assert filter_hf_configs(configs, lang="eng", script="Latn") == ["eng_Latn"]


def test_hf_configs_filter_contains_matches_beng():
    configs = ["asm_Beng", "fra_Latn", "eng_Latn", "eng_Cyrl"]
    assert filter_hf_configs(configs, contains="eng") == [
        "asm_Beng",
        "eng_Latn",
        "eng_Cyrl",
    ]

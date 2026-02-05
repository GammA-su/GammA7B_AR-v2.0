import re
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import yaml


def _extract_front_matter(readme_text: str) -> str:
    lines = readme_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[1:idx])
    return ""


def _parse_configs_from_readme(readme_text: str) -> List[str]:
    front_matter = _extract_front_matter(readme_text)
    configs: List[str] = []
    if front_matter:
        try:
            payload = yaml.safe_load(front_matter) or {}
        except Exception:
            payload = {}
        cfg_list = payload.get("configs") if isinstance(payload, dict) else None
        if isinstance(cfg_list, list):
            for item in cfg_list:
                if not isinstance(item, dict):
                    continue
                value = item.get("config_name") or item.get("name")
                if value:
                    configs.append(str(value))
    if configs:
        return list(dict.fromkeys(configs))

    pattern = re.compile(r"^\s*-\s*config_name\s*:\s*['\"]?([^'\"\n#]+)")
    for line in readme_text.splitlines():
        match = pattern.search(line)
        if match:
            configs.append(match.group(1).strip())
    return list(dict.fromkeys(configs))


def _parse_configs_from_repo_files(paths: Iterable[str]) -> List[str]:
    configs: List[str] = []
    for path in paths:
        parts = path.split("/", 2)
        if len(parts) < 3:
            continue
        if parts[0] != "data":
            continue
        config_name = parts[1]
        if config_name and "." in config_name:
            configs.append(config_name)
    return list(dict.fromkeys(configs))


def get_hf_dataset_config_names(dataset: str) -> Tuple[List[str], Optional[str]]:
    datasets_err: Optional[str] = None
    try:
        from datasets import get_dataset_config_names

        configs = [str(cfg) for cfg in get_dataset_config_names(dataset)]
        if configs:
            return configs, None
    except Exception as exc:
        datasets_err = str(exc)

    try:
        from huggingface_hub import hf_hub_download

        readme_path = hf_hub_download(repo_id=dataset, repo_type="dataset", filename="README.md")
        readme_text = Path(readme_path).read_text(encoding="utf-8", errors="ignore")
        configs = _parse_configs_from_readme(readme_text)
        if configs:
            if datasets_err:
                return configs, f"datasets lookup failed, using README fallback: {datasets_err}"
            return configs, "using README fallback"
    except Exception as exc:
        if datasets_err:
            readme_err = str(exc)
        else:
            return [], f"README fallback failed: {exc}"

    try:
        from huggingface_hub import list_repo_files

        repo_files = list_repo_files(repo_id=dataset, repo_type="dataset")
        configs = _parse_configs_from_repo_files(repo_files)
        if configs:
            if datasets_err:
                return configs, f"datasets lookup failed, using repo-files fallback: {datasets_err}"
            return configs, "using repo-files fallback"
    except Exception as exc:
        if datasets_err:
            if "readme_err" in locals():
                return [], (
                    f"datasets lookup failed: {datasets_err}; "
                    f"README fallback failed: {readme_err}; "
                    f"repo-files fallback failed: {exc}"
                )
            return [], f"datasets lookup failed: {datasets_err}; repo-files fallback failed: {exc}"
        return [], f"repo-files fallback failed: {exc}"

    if datasets_err:
        return [], f"datasets lookup failed: {datasets_err}"
    return [], None


def filter_hf_configs(
    configs: Iterable[str],
    lang: Optional[str] = None,
    script: Optional[str] = None,
    contains: Optional[str] = None,
) -> List[str]:
    def _lang_aliases(value: str) -> Set[str]:
        aliases = {value}
        # Common HF shorthand for English can be "en" while CLI examples use "eng".
        if value == "eng":
            aliases.add("en")
        elif value == "en":
            aliases.add("eng")
        return aliases

    def _split_lang_script(cfg: str) -> Optional[Tuple[str, str]]:
        if not re.match(r"^[a-z]{3}_[A-Za-z]{4}$", cfg):
            return None
        parts = cfg.split("_", 1)
        return parts[0], parts[1]

    configs_list = list(configs)
    if lang:
        filtered: List[str] = []
        lang_set = _lang_aliases(lang)
        for cfg in configs_list:
            parsed = _split_lang_script(cfg)
            if parsed is not None:
                cfg_lang, cfg_script = parsed
                if cfg_lang not in lang_set:
                    continue
                if script is not None and cfg_script != script:
                    continue
                filtered.append(cfg)
                continue

            # Legacy fallback for non FineWeb-2 style names.
            parts = cfg.split("_", 1)
            if parts[0] not in lang_set:
                continue
            if script is not None:
                if len(parts) < 2 or parts[1] != script:
                    continue
            filtered.append(cfg)
        return filtered
    if contains:
        needle = contains.lower()
        return [cfg for cfg in configs_list if needle in cfg.lower()]
    return configs_list

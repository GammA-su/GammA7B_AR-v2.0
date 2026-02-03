from typing import Iterable, List, Optional


def filter_hf_configs(
    configs: Iterable[str],
    lang: Optional[str] = None,
    script: Optional[str] = None,
    contains: Optional[str] = None,
) -> List[str]:
    configs_list = list(configs)
    if lang:
        filtered: List[str] = []
        for cfg in configs_list:
            parts = cfg.split("_", 1)
            if parts[0] != lang:
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

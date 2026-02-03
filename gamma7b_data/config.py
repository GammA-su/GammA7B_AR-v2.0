from dataclasses import dataclass
from pathlib import Path
import glob
import os
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SourceSpec:
    name: str
    weight: float
    type: str
    params: Dict[str, Any]
    paths: List[str]


@dataclass
class DomainSpec:
    name: str
    weight: float
    sources: Dict[str, SourceSpec]


@dataclass
class PipelineConfig:
    seed: int
    outputs: Dict[str, str]
    dedup: Dict[str, Any]
    packing: Dict[str, Any]
    chars_per_token: float
    domains: Dict[str, DomainSpec]

    @property
    def domain_weights(self) -> Dict[str, float]:
        return {name: dom.weight for name, dom in self.domains.items()}

    @property
    def source_weights(self) -> Dict[str, Dict[str, float]]:
        return {name: {s.name: s.weight for s in dom.sources.values()} for name, dom in self.domains.items()}

    @property
    def source_paths(self) -> Dict[str, Dict[str, List[str]]]:
        return {name: {s.name: s.paths for s in dom.sources.values()} for name, dom in self.domains.items()}


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def _validate_weights(weights: Dict[str, float], label: str) -> None:
    total = sum(weights.values())
    if not weights:
        raise ValueError(f"Missing weights for {label}")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Weights for {label} must sum to 1.0 (got {total:.6f})")


def load_manifest(path: Path) -> PipelineConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    seed = int(payload.get("seed", 0))
    outputs = payload.get("outputs", {})
    dedup_cfg = payload.get("dedup", {})
    packing_cfg = payload.get("packing", {})
    chars_per_token = float(payload.get("chars_per_token", packing_cfg.get("chars_per_token", 4.0)))

    if "normalized_dir" not in outputs:
        outputs["normalized_dir"] = "out/normalized"
    if "dedup_dir" not in outputs:
        outputs["dedup_dir"] = "out/dedup"
    if "packed_dir" not in outputs:
        outputs["packed_dir"] = "out/packed"

    domains_payload = payload.get("domains", {})
    if not domains_payload:
        raise ValueError("Manifest missing 'domains'")

    raw_domain_weights: Dict[str, float] = {}
    raw_sources: Dict[str, Dict[str, SourceSpec]] = {}

    for domain, cfg in domains_payload.items():
        raw_domain_weights[domain] = float(cfg.get("weight", 0.0))
        sources_payload = cfg.get("sources", {})
        if not sources_payload:
            raise ValueError(f"Domain '{domain}' has no sources")
        source_weights: Dict[str, float] = {}
        sources: Dict[str, SourceSpec] = {}
        for source, scfg in sources_payload.items():
            s_weight = float(scfg.get("weight", 0.0))
            s_type = scfg.get("type", "local_dir")
            params = scfg.get("params", {}) or {}
            paths = params.get("paths", [])
            if not paths and "paths" in scfg:
                paths = scfg.get("paths", [])
            if isinstance(paths, str):
                paths = [paths]
            if not paths:
                normalized_dir = outputs.get("normalized_dir", "out/normalized")
                pattern = f"{normalized_dir}/{domain}/{source}/*.jsonl.zst"
                paths = [pattern]
            source_weights[source] = s_weight
            sources[source] = SourceSpec(
                name=source,
                weight=s_weight,
                type=s_type,
                params=params,
                paths=list(paths),
            )
        _validate_weights(source_weights, f"sources for domain '{domain}'")
        source_weights = _normalize(source_weights)
        for source in sources.values():
            source.weight = source_weights[source.name]
        raw_sources[domain] = sources

    _validate_weights(raw_domain_weights, "domains")
    domain_weights = _normalize(raw_domain_weights)

    domains: Dict[str, DomainSpec] = {}
    for domain, sources in raw_sources.items():
        domains[domain] = DomainSpec(name=domain, weight=domain_weights[domain], sources=sources)

    return PipelineConfig(
        seed=seed,
        outputs=outputs,
        dedup=dedup_cfg,
        packing=packing_cfg,
        chars_per_token=chars_per_token,
        domains=domains,
    )


def resolve_paths(paths: List[str], data_root: Optional[str] = None) -> List[str]:
    if not data_root:
        return list(paths)
    resolved: List[str] = []
    for path in paths:
        if os.path.isabs(path):
            resolved.append(path)
        else:
            resolved.append(str(Path(data_root) / path))
    return resolved


def expand_source_paths(paths: List[str], data_root: Optional[str] = None) -> List[str]:
    expanded: List[str] = []
    for pattern in resolve_paths(paths, data_root=data_root):
        expanded.extend(glob.glob(pattern))
    return sorted({path for path in expanded})

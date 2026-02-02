import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from .utils import bytes_len


@dataclass
class ShardManifest:
    shard_path: str
    domain: str
    source: str
    seed: int
    counts: Dict[str, int] = field(default_factory=lambda: {"docs": 0, "bytes": 0, "est_tokens": 0})
    drops: Dict[str, int] = field(default_factory=dict)

    def add_doc(self, text: str, est_tokens: int) -> None:
        self.counts["docs"] += 1
        self.counts["bytes"] += bytes_len(text)
        self.counts["est_tokens"] += est_tokens

    def add_drop(self, reason: str) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1

    def write(self, path: Path) -> None:
        payload = {
            "shard_path": self.shard_path,
            "domain": self.domain,
            "source": self.source,
            "seed": self.seed,
            "counts": self.counts,
            "drops": self.drops,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class PackManifest:
    seq_len: int
    tokenizer_path: str
    seed: int
    counts_by_domain: Dict[str, int] = field(default_factory=dict)
    counts_by_source: Dict[str, int] = field(default_factory=dict)
    est_tokens_by_domain: Dict[str, int] = field(default_factory=dict)
    est_tokens_by_source: Dict[str, int] = field(default_factory=dict)

    def add(self, domain: str, source: str, est_tokens: int) -> None:
        self.counts_by_domain[domain] = self.counts_by_domain.get(domain, 0) + 1
        self.counts_by_source[source] = self.counts_by_source.get(source, 0) + 1
        self.est_tokens_by_domain[domain] = self.est_tokens_by_domain.get(domain, 0) + est_tokens
        self.est_tokens_by_source[source] = self.est_tokens_by_source.get(source, 0) + est_tokens

    def write(self, path: Path) -> None:
        payload = {
            "seq_len": self.seq_len,
            "tokenizer_path": self.tokenizer_path,
            "seed": self.seed,
            "counts_by_domain": self.counts_by_domain,
            "counts_by_source": self.counts_by_source,
            "est_tokens_by_domain": self.est_tokens_by_domain,
            "est_tokens_by_source": self.est_tokens_by_source,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List

from .utils import estimate_tokens


@dataclass
class SourceIterator:
    name: str
    domain: str
    iterator: Iterator[dict]

    def next_doc(self) -> dict:
        return next(self.iterator)


class TokenAwareSampler:
    def __init__(self, domain_weights: Dict[str, float], source_weights: Dict[str, Dict[str, float]], seed: int = 0):
        self.domain_weights = domain_weights
        self.source_weights = source_weights
        self.rng = random.Random(seed)
        self.total_tokens = 0
        self.domain_tokens = {d: 0 for d in domain_weights}
        self.source_tokens = {(d, src): 0 for d in source_weights for src in source_weights[d]}

    def _choose(self, weights: Dict[str, float], current: Dict[str, int], total: int) -> str:
        if total <= 0:
            items = sorted(weights.items())
            r = self.rng.random() * sum(w for _, w in items)
            upto = 0.0
            for key, w in items:
                upto += w
                if upto >= r:
                    return key
            return items[-1][0]
        deficits = []
        for key, weight in weights.items():
            target = weight * total
            deficit = target - current.get(key, 0)
            deficits.append((deficit, key))
        deficits.sort(reverse=True)
        top_deficit = deficits[0][0]
        top = [key for deficit, key in deficits if deficit == top_deficit]
        if len(top) == 1:
            return top[0]
        tied_weights = {k: weights[k] for k in top}
        items = sorted(tied_weights.items())
        r = self.rng.random() * sum(w for _, w in items)
        upto = 0.0
        for key, w in items:
            upto += w
            if upto >= r:
                return key
        return items[-1][0]

    def choose_domain(self) -> str:
        return self._choose(self.domain_weights, self.domain_tokens, self.total_tokens)

    def choose_source(self, domain: str) -> str:
        weights = self.source_weights[domain]
        current = {k: self.source_tokens.get((domain, k), 0) for k in weights}
        total = sum(current.values())
        return self._choose(weights, current, total)

    def update(self, domain: str, source: str, est_tokens: int) -> None:
        self.total_tokens += est_tokens
        self.domain_tokens[domain] = self.domain_tokens.get(domain, 0) + est_tokens
        key = (domain, source)
        self.source_tokens[key] = self.source_tokens.get(key, 0) + est_tokens


def sample_documents(
    sampler: TokenAwareSampler,
    sources: Dict[str, SourceIterator],
    max_docs: int,
    chars_per_token: float = 4.0,
) -> Iterator[dict]:
    count = 0
    while count < max_docs:
        domain = sampler.choose_domain()
        source = sampler.choose_source(domain)
        key = f"{domain}:{source}"
        doc = sources[key].next_doc()
        est_tokens = estimate_tokens(doc["text"], chars_per_token=chars_per_token)
        sampler.update(domain, source, est_tokens)
        count += 1
        yield doc

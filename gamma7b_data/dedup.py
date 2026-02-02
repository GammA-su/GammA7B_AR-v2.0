import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .schema import NormalizedDocument
from .utils import stable_hash


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def exact_hash(text: str) -> str:
    return stable_hash(normalize_text(text))


def _hash_to_int(text: str, hash_bits: int) -> int:
    digest_size = max(1, (hash_bits + 7) // 8)
    payload = text.encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(payload, digest_size=digest_size).digest()
    value = int.from_bytes(digest, "little")
    if hash_bits >= digest_size * 8:
        return value
    mask = (1 << hash_bits) - 1
    return value & mask


def simhash(text: str, hash_bits: int = 64) -> int:
    tokens = text.lower().split()
    if not tokens:
        return 0
    shingles: List[str] = []
    if len(tokens) < 3:
        shingles = tokens
    else:
        for i in range(len(tokens) - 2):
            shingles.append(" ".join(tokens[i : i + 3]))
    v = [0] * hash_bits
    for shingle in shingles:
        h = _hash_to_int(shingle, hash_bits)
        for i in range(hash_bits):
            bit = 1 << i
            v[i] += 1 if h & bit else -1
    fingerprint = 0
    for i, val in enumerate(v):
        if val > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def simhash_bands(hash_bits: int, bands: int = 4) -> List[Tuple[int, int]]:
    band_size = hash_bits // bands
    return [(i * band_size, (i + 1) * band_size) for i in range(bands)]


def band_key(fingerprint: int, start: int, end: int) -> int:
    mask = (1 << (end - start)) - 1
    return (fingerprint >> start) & mask


@dataclass
class DedupDecision:
    keep: bool
    reason: str
    dup_of: Optional[str] = None
    score: Optional[int] = None
    meta: Optional[Dict[str, object]] = None


@dataclass
class NearDupMatch:
    cluster_id: str
    match_id: Optional[str]
    distance: Optional[int]
    fingerprint: int


class SimhashClusterer:
    def __init__(self, simhash_threshold: int = 3, hash_bits: int = 64):
        self.simhash_threshold = simhash_threshold
        self.hash_bits = hash_bits
        self.simhash_index: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
        self.band_ranges = simhash_bands(hash_bits)
        self.doc_cluster: Dict[str, str] = {}

    def assign(self, doc_id: str, text: str) -> NearDupMatch:
        fp = simhash(text, self.hash_bits)
        match_id = None
        match_dist = None
        cluster_id = None
        for start, end in self.band_ranges:
            key = (start, band_key(fp, start, end))
            for candidate_id, candidate_fp in self.simhash_index.get(key, []):
                dist = hamming_distance(fp, candidate_fp)
                if dist <= self.simhash_threshold:
                    match_id = candidate_id
                    match_dist = dist
                    cluster_id = self.doc_cluster.get(candidate_id, candidate_id)
                    break
            if cluster_id is not None:
                break
        if cluster_id is None:
            cluster_id = doc_id
        for start, end in self.band_ranges:
            key = (start, band_key(fp, start, end))
            self.simhash_index.setdefault(key, []).append((doc_id, fp))
        self.doc_cluster[doc_id] = cluster_id
        return NearDupMatch(cluster_id=cluster_id, match_id=match_id, distance=match_dist, fingerprint=fp)


class Deduper:
    def __init__(self, simhash_threshold: int = 3, hash_bits: int = 64):
        self.simhash_threshold = simhash_threshold
        self.hash_bits = hash_bits
        self.exact_seen: Dict[str, str] = {}
        self.simhash_index: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
        self.band_ranges = simhash_bands(hash_bits)

    def check(self, doc: NormalizedDocument, use_near: bool = True) -> DedupDecision:
        content_hash = exact_hash(doc.text)
        if content_hash in self.exact_seen:
            return DedupDecision(
                False,
                "exact_dup",
                self.exact_seen[content_hash],
                score=0,
                meta={"exact_hash": content_hash},
            )
        if use_near:
            fp = simhash(doc.text, self.hash_bits)
            for start, end in self.band_ranges:
                key = (start, band_key(fp, start, end))
                for match_id, match_fp in self.simhash_index.get(key, []):
                    dist = hamming_distance(fp, match_fp)
                    if dist <= self.simhash_threshold:
                        return DedupDecision(
                            False,
                            "near_dup",
                            match_id,
                            score=dist,
                            meta={"simhash": fp},
                        )
            for start, end in self.band_ranges:
                key = (start, band_key(fp, start, end))
                self.simhash_index.setdefault(key, []).append((doc.doc_id, fp))
        self.exact_seen[content_hash] = doc.doc_id
        return DedupDecision(True, "keep", None, score=None, meta={"exact_hash": content_hash})

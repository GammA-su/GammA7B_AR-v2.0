import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .schema import NormalizedDocument
from .utils import stable_hash


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def exact_hash(text: str) -> str:
    return stable_hash(normalize_text(text))


def normalize_for_simhash(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    out: List[str] = []
    length = len(normalized)
    for i, ch in enumerate(normalized):
        if ch.isspace():
            out.append(" ")
            continue
        if ch in {"'", "’"}:
            prev = normalized[i - 1] if i > 0 else ""
            nxt = normalized[i + 1] if i + 1 < length else ""
            if prev.isalnum() and nxt.isalnum():
                continue
            out.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            out.append(" ")
        else:
            out.append(ch)
    collapsed = re.sub(r"\s+", " ", "".join(out)).strip()
    return collapsed


def _char_ngrams(text: str, size: int = 5) -> List[str]:
    if len(text) < size:
        return [text] if text else []
    return [text[i : i + size] for i in range(len(text) - size + 1)]


def _hash_to_int(text: str, hash_bits: int) -> int:
    digest_size = max(1, (hash_bits + 7) // 8)
    payload = text.encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(payload, digest_size=digest_size).digest()
    value = int.from_bytes(digest, "little")
    if hash_bits >= digest_size * 8:
        return value
    mask = (1 << hash_bits) - 1
    return value & mask


def simhash_from_normalized(normalized: str, hash_bits: int = 64) -> int:
    if not normalized:
        return 0
    tokens = normalized.split()
    shingles: List[str] = []
    if len(normalized) < 400:
        compact = normalized.replace(" ", "")
        shingles.extend(_char_ngrams(compact, size=5))
        shingles.extend(tokens)
    elif len(tokens) < 3:
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


def simhash(text: str, hash_bits: int = 64) -> int:
    normalized = normalize_for_simhash(text)
    return simhash_from_normalized(normalized, hash_bits=hash_bits)


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
        self.fingerprints: List[Tuple[str, int]] = []
        self.band_ranges = simhash_bands(hash_bits)
        self.doc_cluster: Dict[str, str] = {}

    def assign(self, doc_id: str, text: str) -> NearDupMatch:
        normalized = normalize_for_simhash(text)
        fp = simhash_from_normalized(normalized, self.hash_bits)
        return self.assign_fp(doc_id, fp, normalized_len=len(normalized))

    def assign_fp(self, doc_id: str, fp: int, normalized_len: int = 0) -> NearDupMatch:
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
        if cluster_id is None and (normalized_len < 400 or self.simhash_threshold >= self.hash_bits):
            best_id = None
            best_dist = None
            for candidate_id, candidate_fp in self.fingerprints:
                dist = hamming_distance(fp, candidate_fp)
                if dist <= self.simhash_threshold:
                    if best_dist is None or dist < best_dist or (
                        dist == best_dist and str(candidate_id) < str(best_id)
                    ):
                        best_id = candidate_id
                        best_dist = dist
            if best_id is not None:
                match_id = best_id
                match_dist = best_dist
                cluster_id = self.doc_cluster.get(best_id, best_id)
        if cluster_id is None:
            cluster_id = doc_id
        for start, end in self.band_ranges:
            key = (start, band_key(fp, start, end))
            self.simhash_index.setdefault(key, []).append((doc_id, fp))
        self.fingerprints.append((doc_id, fp))
        self.doc_cluster[doc_id] = cluster_id
        return NearDupMatch(cluster_id=cluster_id, match_id=match_id, distance=match_dist, fingerprint=fp)


class Deduper:
    def __init__(self, simhash_threshold: int = 3, hash_bits: int = 64):
        self.simhash_threshold = simhash_threshold
        self.hash_bits = hash_bits
        self.exact_seen: Dict[str, str] = {}
        self.simhash_index: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
        self.fingerprints: List[Tuple[str, int]] = []
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
            normalized = normalize_for_simhash(doc.text)
            fp = simhash_from_normalized(normalized, self.hash_bits)
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
            if len(normalized) < 400 or self.simhash_threshold >= self.hash_bits:
                best_id = None
                best_dist = None
                for match_id, match_fp in self.fingerprints:
                    dist = hamming_distance(fp, match_fp)
                    if dist <= self.simhash_threshold:
                        if best_dist is None or dist < best_dist or (dist == best_dist and match_id < best_id):
                            best_id = match_id
                            best_dist = dist
                if best_id is not None:
                    return DedupDecision(
                        False,
                        "near_dup",
                        best_id,
                        score=best_dist,
                        meta={"simhash": fp},
                    )
            for start, end in self.band_ranges:
                key = (start, band_key(fp, start, end))
                self.simhash_index.setdefault(key, []).append((doc.doc_id, fp))
            self.fingerprints.append((doc.doc_id, fp))
        self.exact_seen[content_hash] = doc.doc_id
        return DedupDecision(True, "keep", None, score=None, meta={"exact_hash": content_hash})

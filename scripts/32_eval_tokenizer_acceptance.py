#!/usr/bin/env python3
import argparse
import io
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import zstandard as zstd

try:
    import sentencepiece as spm
except Exception as exc:  # pragma: no cover
    spm = None


IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
SCI_RE = re.compile(r"\b\d+(?:\.\d+)?e[+-]?\d+\b", re.IGNORECASE)


@dataclass
class DomainStats:
    total_chars: int = 0
    total_tokens: int = 0
    total_pieces: int = 0
    byte_fallback_pieces: int = 0
    ident_total: int = 0
    ident_le2: int = 0
    sci_total: int = 0
    sci_le2: int = 0


def _is_byte_fallback(piece: str) -> bool:
    return piece.startswith("<0x") and piece.endswith(">")


def _sample_by_domain(dedup_zst: str, per_domain: int, seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[str]] = defaultdict(list)
    dctx = zstd.ZstdDecompressor()
    with open(dedup_zst, "rb") as f, dctx.stream_reader(f) as r:
        t = io.TextIOWrapper(r, encoding="utf-8")
        for line in t:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            dom = (obj.get("domain") or "unknown").strip()
            txt = obj.get("text") or ""
            if not isinstance(txt, str) or not txt:
                continue
            bucket = buckets[dom]
            if len(bucket) < per_domain:
                bucket.append(txt)
            else:
                # reservoir sampling per domain
                j = rng.randint(0, len(bucket))
                if j < per_domain:
                    bucket[j] = txt
    return buckets


def _eval_domain(sp, texts: List[str]) -> DomainStats:
    stats = DomainStats()
    for txt in texts:
        stats.total_chars += len(txt)
        ids = sp.EncodeAsIds(txt)
        stats.total_tokens += len(ids)
        for i in ids:
            piece = sp.IdToPiece(i)
            stats.total_pieces += 1
            if _is_byte_fallback(piece):
                stats.byte_fallback_pieces += 1

        idents = IDENT_RE.findall(txt)
        for ident in idents[:50]:
            stats.ident_total += 1
            if len(sp.EncodeAsIds(ident)) <= 2:
                stats.ident_le2 += 1

        scis = SCI_RE.findall(txt)
        for sci in scis[:20]:
            stats.sci_total += 1
            if len(sp.EncodeAsIds(sci)) <= 2:
                stats.sci_le2 += 1
    return stats


def run_eval(
    tokenizer_model: str,
    dedup_zst: str,
    sample_lines_per_domain: int,
    seed: int,
    max_token_per_char: float,
    max_byte_fallback_rate: float,
    min_ident_le2_rate: float,
    min_sci_le2_rate: float,
) -> Tuple[Dict[str, Dict[str, float]], bool]:
    if spm is None:
        raise RuntimeError("sentencepiece not installed")
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model)

    samples = _sample_by_domain(dedup_zst, sample_lines_per_domain, seed)
    report: Dict[str, Dict[str, float]] = {}
    ok = True

    for dom, texts in sorted(samples.items()):
        stats = _eval_domain(sp, texts)
        token_per_char = stats.total_tokens / max(1, stats.total_chars)
        byte_fallback_rate = stats.byte_fallback_pieces / max(1, stats.total_pieces)
        ident_rate = stats.ident_le2 / max(1, stats.ident_total)
        sci_rate = stats.sci_le2 / max(1, stats.sci_total)

        report[dom] = {
            "token_per_char": token_per_char,
            "byte_fallback_rate": byte_fallback_rate,
            "ident_le2_rate": ident_rate,
            "sci_le2_rate": sci_rate,
            "samples": len(texts),
        }

        if token_per_char > max_token_per_char:
            ok = False
        if byte_fallback_rate > max_byte_fallback_rate:
            ok = False
        if stats.ident_total > 0 and ident_rate < min_ident_le2_rate:
            ok = False
        if stats.sci_total > 0 and sci_rate < min_sci_le2_rate:
            ok = False

    return report, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-model", required=True)
    ap.add_argument("--dedup-zst", required=True)
    ap.add_argument("--sample-lines-per-domain", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-token-per-char", type=float, default=0.6)
    ap.add_argument("--max-byte-fallback-rate", type=float, default=0.08)
    ap.add_argument("--min-ident-le2-rate", type=float, default=0.7)
    ap.add_argument("--min-sci-le2-rate", type=float, default=0.6)
    args = ap.parse_args()

    report, ok = run_eval(
        tokenizer_model=args.tokenizer_model,
        dedup_zst=args.dedup_zst,
        sample_lines_per_domain=args.sample_lines_per_domain,
        seed=args.seed,
        max_token_per_char=args.max_token_per_char,
        max_byte_fallback_rate=args.max_byte_fallback_rate,
        min_ident_le2_rate=args.min_ident_le2_rate,
        min_sci_le2_rate=args.min_sci_le2_rate,
    )

    print("== tokenizer acceptance report ==")
    for dom, stats in report.items():
        print(
            f"{dom:20s} token/char={stats['token_per_char']:.3f} "
            f"byte_fallback={stats['byte_fallback_rate']:.3f} "
            f"ident<=2={stats['ident_le2_rate']:.3f} "
            f"sci<=2={stats['sci_le2_rate']:.3f} "
            f"samples={stats['samples']}"
        )

    if not ok:
        print("FAIL: acceptance thresholds not met", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

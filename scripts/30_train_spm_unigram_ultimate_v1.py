#!/usr/bin/env python3
import argparse
import io
import json
import random
import tempfile
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

try:
    import orjson  # type: ignore
except Exception:
    orjson = None
from pathlib import Path

import sentencepiece as spm
import zstandard as zstd


def _parse_line(line: str):
    if not line.strip():
        return None
    try:
        if orjson is not None:
            obj = orjson.loads(line)
        else:
            obj = json.loads(line)
    except Exception:
        return None
    txt = obj.get("text")
    dom = obj.get("domain") or "unknown"
    if isinstance(txt, str) and txt:
        return dom, txt
    return None


def stream_dedup_texts(dedup_zst: str, workers: int = 16, batch_size: int = 4096):
    dctx = zstd.ZstdDecompressor()
    with open(dedup_zst, "rb") as f, dctx.stream_reader(f) as r:
        t = io.TextIOWrapper(r, encoding="utf-8", errors="replace")
        if workers <= 1:
            for line in t:
                item = _parse_line(line)
                if item is not None:
                    yield item
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            batch = []
            for line in t:
                batch.append(line)
                if len(batch) >= batch_size:
                    for item in ex.map(_parse_line, batch):
                        if item is not None:
                            yield item
                    batch = []
            if batch:
                for item in ex.map(_parse_line, batch):
                    if item is not None:
                        yield item


def number_zoo_lines(n: int, seed: int):
    rng = random.Random(seed)
    for _ in range(n):
        a = rng.randint(0, 999999)
        b = rng.randint(1, 999999)
        sci = f"{rng.randint(0,99)}.{rng.randint(0,999999)}e{rng.choice(['+','-'])}{rng.randint(0,99)}"
        hx = "0x" + "".join(rng.choice("0123456789ABCDEF") for _ in range(rng.randint(4, 16)))
        bn = "0b" + "".join(rng.choice("01") for _ in range(rng.randint(8, 32)))
        dec = f"{rng.randint(0,9999)}.{rng.randint(0,999999)}"
        ver = f"v{rng.randint(0,20)}.{rng.randint(0,20)}.{rng.randint(0,20)}"
        date = f"{rng.randint(1990,2035):04d}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        ops = " == != <= >= -> :: += ** // ... "
        yield f"{a}/{b} {sci} {hx} {bn} {dec} {ver} {date}{ops}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup-zst", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=44032)
    ap.add_argument("--target-bytes", type=int, default=400_000_000)
    ap.add_argument("--number-zoo-frac", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boost-code", type=float, default=1.2)
    ap.add_argument("--boost-math", type=float, default=1.2)
    ap.add_argument("--include-domain-tokens", action="store_true")
    ap.add_argument(
        "--min-bytes-per-domain",
        type=int,
        default=2_000_000,
        help="Floor bytes to collect per domain before proportional fill.",
    )
    ap.add_argument(
        "--allow-missing-domains",
        action="store_true",
        help="If set, do not fail when a domain contributes 0 bytes.",
    )
    ap.add_argument(
        "--spm-normalization-rule",
        default="identity",
        help="SentencePiece normalization_rule_name. Use 'identity' when we pre-normalize text.",
    )
    ap.add_argument(
        "--spm-normalization-rule-tsv",
        default="",
        help="Optional path to a SentencePiece normalization rule TSV (overrides rule name).",
    )
    ap.add_argument("--spm-minloglevel", type=int, default=1, help="SentencePiece minloglevel (higher = quieter).")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument(
        "--single-pass",
        action="store_true",
        help="Faster: fill floors and budgets in one pass (may fail if domains appear late).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain_aliases = {
        "code": "code_docs",
        "the_stack_v2": "code_docs",
        "the-stack": "code_docs",
        "stack": "code_docs",
        "wiki": "reference",
        "wikipedia": "reference",
        "encyclopedia": "reference",
        "books": "books_longform",
        "longform": "books_longform",
        "web": "filtered_web",
        "filteredweb": "filtered_web",
        "papers": "paper_teasers",
        "paper": "paper_teasers",
        "paper_preview": "paper_teasers",
        "tutorials": "tutorials_notebooks",
        "notebooks": "tutorials_notebooks",
    }

    base = {
        "filtered_web": 0.30,
        "books_longform": 0.18,
        "reference": 0.08,
        "forums_qa": 0.10,
        "code_docs": 0.18,
        "math": 0.12,
        "tutorials_notebooks": 0.02,
        "paper_teasers": 0.02,
    }

    boosted = dict(base)
    boosted["code_docs"] *= args.boost_code
    boosted["math"] *= args.boost_math
    total_weight = sum(boosted.values())
    boosted = {k: v / total_weight for k, v in boosted.items()}

    budgets = {k: int(args.target_bytes * v) for k, v in boosted.items()}
    print("TARGET_BYTES:", args.target_bytes)
    print("BOOSTED_DOMAIN_WEIGHTS:", boosted)
    print("DOMAIN_BYTE_BUDGETS:", budgets)

    tmpdir = Path(tempfile.mkdtemp(prefix="spm_tok_corpus_"))
    corpus_path = tmpdir / "corpus.txt"
    print("CORPUS_TMP:", corpus_path)

    written = {k: 0 for k in budgets}
    total_written = 0
    seen = 0
    log_every = 1000
    raw_dom_counter = Counter()

    def canon(dom: str) -> str:
        dom = (dom or "").strip()
        raw_dom_counter[dom] += 1
        return domain_aliases.get(dom, dom)

    with corpus_path.open("wb") as w:
        floors_left = set(budgets.keys())
        floor_hit = {k: False for k in budgets}
        for dom_raw, txt in stream_dedup_texts(args.dedup_zst, workers=args.workers, batch_size=args.batch_size):
            seen += 1
            dom = canon(dom_raw)
            if dom not in budgets:
                if dom.startswith("books"):
                    dom2 = "books_longform"
                elif dom.startswith("paper"):
                    dom2 = "paper_teasers"
                else:
                    dom2 = "filtered_web"
            else:
                dom2 = dom

            if written[dom2] >= budgets[dom2]:
                continue

            txt = unicodedata.normalize("NFC", txt)
            if not txt.endswith("\n"):
                txt = txt + "\n"

            payload = txt.encode("utf-8", errors="replace")
            b = len(payload)
            if written[dom2] + b > budgets[dom2]:
                b_allow = budgets[dom2] - written[dom2]
                if b_allow <= 0:
                    continue
                cut = max(1, int(len(payload) * (b_allow / max(1, b))))
                payload = payload[:cut]
                b = len(payload)

            w.write(payload)
            written[dom2] += b
            total_written += b

            if written[dom2] >= args.min_bytes_per_domain:
                if not floor_hit[dom2]:
                    floor_hit[dom2] = True
                    print(f"FLOOR_REACHED {dom2} bytes={written[dom2]}")
                floors_left.discard(dom2)

            if seen % log_every == 0:
                pct = (total_written / max(1, args.target_bytes)) * 100.0
                print(f"PROGRESS seen={seen} bytes={total_written} ({pct:.1f}%)")
                print("DOMAIN_BYTES:", written)

            if all(written[k] >= budgets[k] for k in budgets):
                print("BUDGETS SATISFIED: stopping early")
                break

        if not args.single_pass and floors_left:
            print("PHASE_B: filling remaining budgets")
            for dom_raw, txt in stream_dedup_texts(args.dedup_zst, workers=args.workers, batch_size=args.batch_size):
                dom = canon(dom_raw)
                if dom not in budgets:
                    if dom.startswith("books"):
                        dom2 = "books_longform"
                    elif dom.startswith("paper"):
                        dom2 = "paper_teasers"
                    else:
                        dom2 = "filtered_web"
                else:
                    dom2 = dom

                if written[dom2] >= budgets[dom2]:
                    continue

                txt = unicodedata.normalize("NFC", txt)
                if not txt.endswith("\n"):
                    txt = txt + "\n"

                payload = txt.encode("utf-8", errors="replace")
                b = len(payload)
                if written[dom2] + b > budgets[dom2]:
                    b_allow = budgets[dom2] - written[dom2]
                    if b_allow <= 0:
                        continue
                    cut = max(1, int(len(payload) * (b_allow / max(1, b))))
                    payload = payload[:cut]
                    b = len(payload)
                w.write(payload)
                written[dom2] += b
                total_written += b
                if all(written[k] >= budgets[k] for k in budgets):
                    print("BUDGETS SATISFIED: stopping early")
                    break

        zoo_target = int(max(0, args.number_zoo_frac) * max(1, total_written))
        zoo_written = 0
        print("NUMBER_ZOO_TARGET_BYTES:", zoo_target)
        for line in number_zoo_lines(2_000_000, seed=args.seed):
            payload = line.encode("utf-8")
            b = len(payload)
            if zoo_written + b > zoo_target:
                break
            w.write(payload)
            zoo_written += b
            total_written += b
    print("NUMBER_ZOO_WRITTEN_BYTES:", zoo_written)

    zero = [d for d, v in written.items() if v == 0]
    if zero:
        msg = (
            f"Tokenizer corpus has 0 bytes for domains: {zero}\n"
            f"Top raw domain strings: {raw_dom_counter.most_common(20)}\n"
            "Your dedup file likely lacks these domains or uses different labels.\n"
            "Fix by regenerating dedup to include all normalized sources or extend domain aliases."
        )
        if args.allow_missing_domains:
            print("WARNING:", msg)
        else:
            raise SystemExit(msg)

    bos_piece = "<|bos|>"
    eos_piece = "<|eos|>"
    pad_piece = "<|pad|>"
    unk_piece = "<|unk|>"

    user_syms = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
    if args.include_domain_tokens:
        user_syms += ["<|code|>", "<|math|>", "<|paper|>", "<|text|>"]

    model_prefix = str(out_dir / "tokenizer")

    print("SPM_TRAIN_START")
    spm_kwargs = dict(
        input=str(corpus_path),
        model_prefix=model_prefix,
        model_type="unigram",
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        max_sentence_length=16384,
        split_by_unicode_script=True,
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece=unk_piece,
        bos_piece=bos_piece,
        eos_piece=eos_piece,
        pad_piece=pad_piece,
        user_defined_symbols=",".join(user_syms),
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        seed_sentencepiece_size=1_000_000,
        minloglevel=args.spm_minloglevel,
    )
    if args.spm_normalization_rule_tsv:
        spm_kwargs["normalization_rule_tsv"] = args.spm_normalization_rule_tsv
    else:
        spm_kwargs["normalization_rule_name"] = args.spm_normalization_rule
    spm.SentencePieceTrainer.Train(**spm_kwargs)

    print("SPM_TRAIN_DONE")
    print("WROTE:", out_dir / "tokenizer.model")
    print("WROTE:", out_dir / "tokenizer.vocab")
    print("CORPUS_BYTES:", total_written)
    print("DOMAIN_BYTES:", written)
    print("USER_DEFINED_SYMBOLS:", user_syms)


if __name__ == "__main__":
    main()

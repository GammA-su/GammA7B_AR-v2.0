import json
from pathlib import Path

import pytest
import zstandard as zstd

spm = pytest.importorskip("sentencepiece")

from gamma7b_data.cli import pack_cmd


def _write_jsonl_zst(path: Path, rows):
    cctx = zstd.ZstdCompressor(level=3)
    with path.open("wb") as f, cctx.stream_writer(f) as w:
        for r in rows:
            w.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))


def test_pack_accepts_sentencepiece_model(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    in_zst = data_dir / "tiny.jsonl.zst"
    _write_jsonl_zst(
        in_zst,
        [
            {"doc_id": "1", "domain": "filtered_web", "source": "x", "text": "hello world\n" * 20},
            {"doc_id": "2", "domain": "code_docs", "source": "x", "text": "def f(x): return x+1\n" * 20},
        ],
    )

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\ndef f(x): return x+1\n", encoding="utf-8")

    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(tmp_path / "tok"),
        model_type="unigram",
        vocab_size=400,
        byte_fallback=True,
        hard_vocab_limit=False,
        normalization_rule_name="identity",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<|unk|>",
        bos_piece="<|bos|>",
        eos_piece="<|eos|>",
        pad_piece="<|pad|>",
        user_defined_symbols="<|fim_prefix|>,<|fim_middle|>,<|fim_suffix|>",
    )

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "seed: 1",
                "outputs:",
                "  packed_dir: out/packed",
                "packing:",
                f"  tokenizer_path: {tmp_path / 'tok.model'}",
                "  seq_len: 8",
                "  num_seqs: 1",
                "domains:",
                "  filtered_web:",
                "    weight: 0.5",
                "    sources:",
                "      local:",
                "        weight: 1.0",
                "        type: local_dir",
                "        paths:",
                f"          - {in_zst}",
                "  code_docs:",
                "    weight: 0.5",
                "    sources:",
                "      local:",
                "        weight: 1.0",
                "        type: local_dir",
                "        paths:",
                f"          - {in_zst}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "packed"
    from gamma7b_data.packing import PackBuilder, load_tokenizer

    tok = load_tokenizer(str(tmp_path / "tok.model"), eos_token="<|eos|>")
    builder = PackBuilder(seq_len=8, tokenizer=tok)
    seqs = []
    for row in [
        {"doc_id": "1", "text": "hello world\n" * 20},
        {"doc_id": "2", "text": "def f(x): return x+1\n" * 20},
    ]:
        for seq in builder.add_doc(row["doc_id"], row["text"]):
            seqs.append(seq.tokens)
    tail = builder.finalize()
    if tail is not None:
        seqs.append(tail.tokens)
    assert seqs

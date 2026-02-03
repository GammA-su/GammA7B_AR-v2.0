import importlib.util
import json
from pathlib import Path

import pytest
import zstandard as zstd

spm = pytest.importorskip("sentencepiece")


def _load_eval_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "32_eval_tokenizer_acceptance.py"
    spec = importlib.util.spec_from_file_location("eval_acceptance", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl_zst(path: Path, rows):
    cctx = zstd.ZstdCompressor(level=3)
    with path.open("wb") as f, cctx.stream_writer(f) as w:
        for r in rows:
            w.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))


def test_tokenizer_acceptance_eval_smoke(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nvar_name = 1.23e-10\n", encoding="utf-8")

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
    )

    dedup = tmp_path / "dedup.jsonl.zst"
    _write_jsonl_zst(
        dedup,
        [
            {"domain": "filtered_web", "text": "hello world"},
            {"domain": "code_docs", "text": "var_name = 1.23e-10"},
        ],
    )

    mod = _load_eval_module()
    report, ok = mod.run_eval(
        tokenizer_model=str(tmp_path / "tok.model"),
        dedup_zst=str(dedup),
        sample_lines_per_domain=10,
        seed=0,
        max_token_per_char=2.0,
        max_byte_fallback_rate=1.0,
        min_ident_le2_rate=0.0,
        min_sci_le2_rate=0.0,
    )
    assert ok is True
    assert "filtered_web" in report
    assert "code_docs" in report

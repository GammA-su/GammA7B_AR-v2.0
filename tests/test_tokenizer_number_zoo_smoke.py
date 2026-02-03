from pathlib import Path

import pytest

spm = pytest.importorskip("sentencepiece")


def test_number_zoo_smoke(tmp_path: Path):
    txt = tmp_path / "zoo.txt"
    txt.write_text(
        "1.23e-10\n0xDEADBEEF\n2026-02-03\nv1.2.3\n0b101010\n",
        encoding="utf-8",
    )
    spm.SentencePieceTrainer.Train(
        input=str(txt),
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
    sp = spm.SentencePieceProcessor()
    sp.Load(str(tmp_path / "tok.model"))
    for s in ["1.23e-10", "0xDEADBEEF", "2026-02-03"]:
        ids = sp.EncodeAsIds(s)
        assert len(ids) <= 12

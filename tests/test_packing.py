from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from gamma7b_data.packing import PackBuilder, load_tokenizer


def test_packing_memmap_shape(tmp_path: Path):
    vocab = {"<eos>": 0, "hello": 1, "world": 2, "again": 3}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<eos>"))
    tokenizer.pre_tokenizer = Whitespace()
    tok_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tok_path))

    tok = load_tokenizer(str(tok_path))
    builder = PackBuilder(seq_len=4, tokenizer=tok)

    seqs = []
    for seq in builder.add_doc("doc1", "hello world"):
        seqs.append(seq.tokens)
    tail = builder.finalize()
    if tail is not None:
        seqs.append(tail.tokens)

    mmap_path = tmp_path / "input_ids.mmap"
    mmap = np.memmap(mmap_path, dtype=np.int32, mode="w+", shape=(len(seqs), 4))
    for i, seq in enumerate(seqs):
        mmap[i, :] = np.asarray(seq, dtype=np.int32)
    mmap.flush()

    reloaded = np.memmap(mmap_path, dtype=np.int32, mode="r", shape=(len(seqs), 4))
    assert reloaded.shape == (len(seqs), 4)

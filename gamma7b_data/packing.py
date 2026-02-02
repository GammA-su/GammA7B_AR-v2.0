import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from tokenizers import Tokenizer


class HFTokenizer:
    def __init__(self, tokenizer_path: Path, eos_token: Optional[str] = None, eos_id: Optional[int] = None):
        self.tokenizer_path = str(tokenizer_path)
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        if eos_id is None:
            if eos_token is None:
                eos_token = "<eos>"
            eos_id = self.tokenizer.token_to_id(eos_token)
            if eos_id is None:
                for fallback in ["</s>", "<|endoftext|>"]:
                    eos_id = self.tokenizer.token_to_id(fallback)
                    if eos_id is not None:
                        break
        if eos_id is None:
            raise ValueError("Unable to resolve eos_id from tokenizer")
        self._eos_id = int(eos_id)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def name(self) -> str:
        return self.tokenizer_path


def load_tokenizer(tokenizer_path: str, eos_token: Optional[str] = None, eos_id: Optional[int] = None) -> HFTokenizer:
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    return HFTokenizer(path, eos_token=eos_token, eos_id=eos_id)


@dataclass
class PackedSequence:
    tokens: List[int]
    segments: List[Dict[str, Optional[int]]]


class PackBuilder:
    def __init__(self, seq_len: int, tokenizer: HFTokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.buffer: List[int] = []
        self.buffer_segments: List[Dict[str, Optional[int]]] = []

    def _append_segment(self, doc_id: Optional[str], start: int, end: int) -> None:
        if end <= start:
            return
        self.buffer_segments.append({"doc_id": doc_id, "start": start, "end": end})

    def _emit_from_buffer(self) -> Optional[PackedSequence]:
        if len(self.buffer) < self.seq_len:
            return None
        seq_tokens = self.buffer[: self.seq_len]
        self.buffer = self.buffer[self.seq_len :]

        segments: List[Dict[str, Optional[int]]] = []
        remaining = self.seq_len
        while remaining > 0 and self.buffer_segments:
            seg = self.buffer_segments[0]
            seg_len = seg["end"] - seg["start"]
            if seg_len <= remaining:
                segments.append(seg)
                remaining -= seg_len
                self.buffer_segments.pop(0)
            else:
                segments.append({"doc_id": seg["doc_id"], "start": seg["start"], "end": seg["start"] + remaining})
                seg["start"] += remaining
                remaining = 0
        return PackedSequence(tokens=seq_tokens, segments=segments)

    def _pad_and_flush(self) -> Optional[PackedSequence]:
        if not self.buffer:
            return None
        if len(self.buffer) < self.seq_len:
            pad = self.seq_len - len(self.buffer)
            self.buffer.extend([self.tokenizer.eos_id] * pad)
            self._append_segment(None, 0, pad)
        seq = PackedSequence(tokens=self.buffer[: self.seq_len], segments=self.buffer_segments)
        self.buffer = []
        self.buffer_segments = []
        return seq

    def add_doc(self, doc_id: str, text: str) -> List[PackedSequence]:
        tokens = self.tokenizer.encode(text)
        eos = self.tokenizer.eos_id
        output: List[PackedSequence] = []
        if len(tokens) >= self.seq_len:
            # Flush any buffered short docs first (pad as needed).
            seq = self._pad_and_flush()
            if seq is not None:
                output.append(seq)
            # Take contiguous spans from the long doc.
            idx = 0
            while idx + self.seq_len <= len(tokens):
                chunk = tokens[idx : idx + self.seq_len]
                output.append(
                    PackedSequence(tokens=chunk, segments=[{"doc_id": doc_id, "start": idx, "end": idx + self.seq_len}])
                )
                idx += self.seq_len
            tail = tokens[idx:]
            if tail:
                self.buffer = tail
                self._append_segment(doc_id, idx, idx + len(tail))
        else:
            if self.buffer:
                self.buffer.append(eos)
                self._append_segment(None, 0, 1)
            self.buffer.extend(tokens)
            self._append_segment(doc_id, 0, len(tokens))

        while True:
            seq = self._emit_from_buffer()
            if seq is None:
                break
            output.append(seq)
        return output

    def finalize(self) -> Optional[PackedSequence]:
        return self._pad_and_flush()


def write_memmap(path: Path, sequences: Iterable[List[int]], num_sequences: int, seq_len: int) -> None:
    mmap = np.memmap(path, dtype=np.int32, mode="w+", shape=(num_sequences, seq_len))
    idx = 0
    for seq in sequences:
        if idx >= num_sequences:
            break
        mmap[idx, :] = np.asarray(seq, dtype=np.int32)
        idx += 1
    mmap.flush()


def default_created_at() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


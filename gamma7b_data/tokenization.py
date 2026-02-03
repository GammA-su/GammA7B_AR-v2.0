from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass(frozen=True)
class TokenizerSpec:
    kind: str  # "tokenizers_json" | "sentencepiece"
    path: Path
    eos_token: str
    bos_token: str
    pad_token: str
    unk_token: str


class TokenizerWrapper:
    def __init__(self, spec: TokenizerSpec, eos_id: Optional[int] = None):
        self.spec = spec
        if spec.kind == "tokenizers_json":
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(spec.path))
            self._tok = tok
            self._eos_id = tok.token_to_id(spec.eos_token)
            if self._eos_id is None:
                for fallback in ["<eos>", "<|endoftext|>", "</s>"]:
                    self._eos_id = tok.token_to_id(fallback)
                    if self._eos_id is not None:
                        break
            self._bos_id = tok.token_to_id(spec.bos_token)
            self._pad_id = tok.token_to_id(spec.pad_token)
            self._unk_id = tok.token_to_id(spec.unk_token)
        elif spec.kind == "sentencepiece":
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.Load(str(spec.path))
            self._tok = sp
            self._eos_id = sp.PieceToId(spec.eos_token)
            self._bos_id = sp.PieceToId(spec.bos_token)
            self._pad_id = sp.PieceToId(spec.pad_token)
            self._unk_id = sp.PieceToId(spec.unk_token)
        else:
            raise ValueError(f"unknown tokenizer kind: {spec.kind}")

        if eos_id is not None:
            self._eos_id = int(eos_id)
        if self._eos_id is None or self._eos_id < 0:
            raise ValueError(f"eos token not found: {spec.eos_token}")
        if self._bos_id is None:
            self._bos_id = -1
        if self._pad_id is None:
            self._pad_id = -1
        if self._unk_id is None:
            self._unk_id = -1

    @property
    def eos_id(self) -> int:
        return int(self._eos_id)

    @property
    def bos_id(self) -> int:
        return int(self._bos_id)

    @property
    def pad_id(self) -> int:
        return int(self._pad_id)

    @property
    def unk_id(self) -> int:
        return int(self._unk_id)

    @property
    def name(self) -> str:
        return str(self.spec.path)

    def encode(self, text: str) -> List[int]:
        if self.spec.kind == "tokenizers_json":
            enc = self._tok.encode(text)
            return list(enc.ids)
        return list(self._tok.EncodeAsIds(text))


def load_tokenizer(
    path: Union[str, Path],
    eos_token: Optional[str] = None,
    bos_token: Optional[str] = None,
    pad_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    eos_id: Optional[int] = None,
) -> TokenizerWrapper:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix == ".json":
        return TokenizerWrapper(
            TokenizerSpec(
                kind="tokenizers_json",
                path=p,
                eos_token=eos_token or "</s>",
                bos_token=bos_token or "<s>",
                pad_token=pad_token or "<pad>",
                unk_token=unk_token or "<unk>",
            ),
            eos_id=eos_id,
        )
    if p.suffix == ".model":
        return TokenizerWrapper(
            TokenizerSpec(
                kind="sentencepiece",
                path=p,
                eos_token=eos_token or "<|eos|>",
                bos_token=bos_token or "<|bos|>",
                pad_token=pad_token or "<|pad|>",
                unk_token=unk_token or "<|unk|>",
            ),
            eos_id=eos_id,
        )
    raise ValueError(f"unsupported tokenizer file: {p} (expected .json or .model)")

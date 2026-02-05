import hashlib
import json
import logging
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import zstandard as zstd

_RUNTIME_INITIALIZED = False
_FAISS_STATE = {"enabled": False, "device": None, "num_gpus": 0, "error": None}


def setup_logger(name: str = "gamma7b_data", verbose: bool = False, use_rich: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    handler: logging.Handler
    if use_rich:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, show_path=False)
            handler.setFormatter(logging.Formatter("%(message)s"))
        except Exception:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def make_rng(seed: int) -> random.Random:
    return random.Random(seed)


def make_numpy_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def stable_hash(text: str, digest_size: int = 8) -> str:
    payload = text.encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(payload, digest_size=digest_size).hexdigest()
    return digest


def get_log_every(default: int = 300) -> int:
    try:
        return max(1, int(os.getenv("GAMMA7B_LOG_EVERY", str(default))))
    except ValueError:
        return default


def set_default_threads(num_threads: int = 16, logger: Optional[logging.Logger] = None) -> None:
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(key, str(num_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if logger:
        logger.info("cpu threads: %s", os.environ.get("OMP_NUM_THREADS", str(num_threads)))


def _try_faiss_gpu(device_id: int) -> Optional[Exception]:
    try:
        import faiss  # type: ignore

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = device_id
        index = faiss.GpuIndexFlatL2(res, 1, cfg)
        xb = np.zeros((1, 1), dtype="float32")
        index.add(xb)
        index.search(xb, 1)
        return None
    except Exception as exc:
        return exc


def enable_faiss_gpu(device_id: int = 1, logger: Optional[logging.Logger] = None, fallback_device: int = 0) -> Dict:
    global _FAISS_STATE
    if _FAISS_STATE["enabled"] or _FAISS_STATE["error"] is not None or _FAISS_STATE["num_gpus"] > 0:
        if logger:
            logger.info(
                "faiss gpu check: requested gpu number %s enabled=%s (active_gpu=%s, num_gpus=%s)",
                device_id,
                _FAISS_STATE["enabled"],
                _FAISS_STATE["device"],
                _FAISS_STATE["num_gpus"],
            )
        return _FAISS_STATE
    try:
        import faiss  # type: ignore

        num_gpus = int(getattr(faiss, "get_num_gpus", lambda: 0)())
    except Exception as exc:
        _FAISS_STATE["error"] = str(exc)
        if logger:
            logger.info(
                "faiss gpu check: requested gpu number %s enabled=False (import failed: %s)",
                device_id,
                _FAISS_STATE["error"],
            )
        return _FAISS_STATE

    _FAISS_STATE["num_gpus"] = num_gpus
    if num_gpus <= 0:
        if logger:
            logger.info("faiss gpu check: requested gpu number %s enabled=False (num_gpus=0)", device_id)
        return _FAISS_STATE

    err = _try_faiss_gpu(device_id)
    if err is None:
        _FAISS_STATE.update({"enabled": True, "device": device_id, "error": None})
        if logger:
            logger.info(
                "faiss gpu enabled gpu number %s (num_gpus=%s)",
                device_id,
                num_gpus,
            )
        return _FAISS_STATE

    if fallback_device is not None and fallback_device != device_id and num_gpus > fallback_device:
        err = _try_faiss_gpu(fallback_device)
        if err is None:
            _FAISS_STATE.update({"enabled": True, "device": fallback_device, "error": None})
            if logger:
                logger.info(
                    "faiss gpu enabled gpu number %s (fallback from gpu=%s, num_gpus=%s)",
                    fallback_device,
                    device_id,
                    num_gpus,
                )
            return _FAISS_STATE

    _FAISS_STATE["error"] = str(err)
    if logger:
        logger.info(
            "faiss gpu check: requested gpu number %s enabled=False (num_gpus=%s, error=%s)",
            device_id,
            num_gpus,
            _FAISS_STATE["error"],
        )
    return _FAISS_STATE


def initialize_runtime(
    logger: Optional[logging.Logger] = None,
    cpu_threads: int = 16,
    faiss_gpu_device: int = 1,
    enable_faiss: bool = True,
) -> None:
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return
    _RUNTIME_INITIALIZED = True
    set_default_threads(cpu_threads, logger=logger)
    if enable_faiss and faiss_gpu_device > 0:
        enable_faiss_gpu(device_id=faiss_gpu_device, logger=logger)
    elif logger:
        if enable_faiss:
            logger.info("faiss gpu check skipped (faiss_gpu_device=%s)", faiss_gpu_device)
        else:
            logger.info("faiss gpu check skipped for this command")
    if logger:
        logger.info("log cadence: every %s records", get_log_every())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@contextmanager
def open_zst_reader(path: Path):
    with path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            yield reader


@contextmanager
def open_zst_writer(path: Path, level: Optional[int] = None, threads: Optional[int] = None):
    if level is None:
        try:
            level = int(os.getenv("GAMMA7B_ZSTD_LEVEL", "3"))
        except ValueError:
            level = 3
    with path.open("wb") as fh:
        if threads is not None and threads > 0:
            cctx = zstd.ZstdCompressor(level=level, threads=threads)
        else:
            cctx = zstd.ZstdCompressor(level=level)
        with cctx.stream_writer(fh) as writer:
            yield writer


def _is_zst(path: Path) -> bool:
    return path.suffix == ".zst" or path.name.endswith(".jsonl.zst")


try:
    import orjson  # type: ignore
except Exception:
    orjson = None


def _json_loads(line: str) -> Dict:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def read_jsonl(path: Path) -> Iterator[Dict]:
    if _is_zst(path):
        import io

        with open_zst_reader(path) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                if not line:
                    continue
                yield _json_loads(line)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line:
                    continue
                yield _json_loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    if _is_zst(path):
        with open_zst_writer(path) as writer:
            for row in rows:
                writer.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    else:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_files(paths: List[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*.jsonl*")):
                yield child
        else:
            yield path


def resolve_inputs(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        item = os.path.expanduser(item)
        if any(ch in item for ch in ["*", "?", "["]):
            import glob

            out.extend([Path(p) for p in sorted(glob.glob(item, recursive=True))])
        else:
            out.append(Path(item))
    return out


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    if not text:
        return 1
    return max(1, int(round(len(text) / max(chars_per_token, 1e-6))))


def bytes_len(text: str) -> int:
    return len(text.encode("utf-8"))


def stable_shard_name(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:05d}.jsonl.zst"


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ch.isprintable())
    return printable / max(1, len(text))


def repetition_3gram_ratio(text: str, max_tokens: int = 4000) -> float:
    tokens = text.split()
    if len(tokens) < 3:
        return 0.0
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    total = len(tokens) - 2
    grams = {}
    for i in range(total):
        gram = (tokens[i], tokens[i + 1], tokens[i + 2])
        grams[gram] = grams.get(gram, 0) + 1
    unique = len(grams)
    if total <= 0:
        return 0.0
    return max(0.0, 1.0 - (unique / total))


def url_quality_penalty(url: Optional[str]) -> float:
    if not url:
        return 0.0
    lowered = url.lower()
    low_signal = ["pinterest.com", "facebook.com", "instagram.com", "tiktok.com", "twitter.com", "x.com"]
    for domain in low_signal:
        if domain in lowered:
            return -0.5
    return 0.0


def quality_score(text: str, meta: Optional[Dict[str, object]] = None) -> float:
    if not text:
        return -10.0
    char_len = len(text)
    length_score = min(char_len / 1000.0, 5.0)
    if char_len < 200:
        length_score -= 2.0
    if char_len > 20000:
        length_score -= min((char_len - 20000) / 20000.0, 2.0)

    pr = printable_ratio(text)
    printable_score = (pr - 0.85) * 2.0

    rep = repetition_3gram_ratio(text)
    repetition_penalty = rep * 2.0

    url = None
    if isinstance(meta, dict):
        url_val = meta.get("url")
        if isinstance(url_val, str):
            url = url_val
    url_penalty = url_quality_penalty(url)

    return length_score + printable_score - repetition_penalty + url_penalty

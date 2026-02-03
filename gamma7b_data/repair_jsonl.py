import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from .utils import open_zst_reader, open_zst_writer


class RepairError(RuntimeError):
    pass


def _replace_inplace(in_path: Path, tmp_path: Path, backup: bool) -> Optional[Path]:
    backup_path = None
    if backup:
        backup_path = in_path.with_suffix(in_path.suffix + ".bak")
        shutil.copy2(in_path, backup_path)
    os.replace(tmp_path, in_path)
    return backup_path


def repair_jsonl_zst(
    in_path: Path,
    out_path: Path,
    *,
    inplace: bool = False,
    backup: bool = True,
    zstd_level: int = 3,
    max_bad: int = 1000,
    validate: bool = True,
    log_every: int = 0,
    logger=None,
    threads: Optional[int] = None,
) -> Dict[str, object]:
    if inplace:
        out_path = in_path.with_name(in_path.name + ".tmp")

    records_ok = 0
    records_bad = 0
    wrote_bytes = 0
    record_buf = bytearray()
    in_str = False
    esc = False
    last_was_cr = False

    def flush_record():
        nonlocal records_ok, records_bad, wrote_bytes, record_buf
        if not record_buf:
            return
        if validate:
            try:
                text = record_buf.decode("utf-8")
                obj = json_loads(text)
            except Exception:
                records_bad += 1
                if records_bad > max_bad:
                    raise RepairError(f"Too many bad records (> {max_bad}).")
            else:
                payload = json_dumps(obj) + "\n"
                wrote = payload.encode("utf-8")
                writer.write(wrote)
                wrote_bytes += len(wrote)
                records_ok += 1
        else:
            record_buf.append(10)
            writer.write(record_buf)
            wrote_bytes += len(record_buf)
            records_ok += 1
        record_buf = bytearray()
        if logger and log_every and (records_ok + records_bad) % log_every == 0:
            logger.info(
                "Repair progress %s: ok=%s bad=%s bytes=%s",
                in_path,
                records_ok,
                records_bad,
                wrote_bytes,
            )

    in_path = Path(in_path)
    out_path = Path(out_path)
    json_loads = json.loads
    json_dumps = lambda obj: json.dumps(obj, ensure_ascii=False)
    try:
        import orjson  # type: ignore

        json_loads = orjson.loads
        json_dumps = lambda obj: orjson.dumps(obj).decode("utf-8")
    except Exception:
        pass

    with open_zst_reader(in_path) as reader, open_zst_writer(
        out_path, level=zstd_level, threads=threads
    ) as writer:
        while True:
            chunk = reader.read(65536)
            if not chunk:
                break
            for b in chunk:
                if last_was_cr:
                    last_was_cr = False
                    if b == 10:  # \n after \r
                        continue
                if b == 10:  # \n
                    if in_str:
                        record_buf.extend(b"\\n")
                    else:
                        flush_record()
                    continue
                if b == 13:  # \r
                    if in_str:
                        record_buf.extend(b"\\r")
                    else:
                        flush_record()
                        last_was_cr = True
                    continue
                record_buf.append(b)
                if in_str:
                    if esc:
                        esc = False
                    elif b == 92:  # backslash
                        esc = True
                    elif b == 34:  # quote
                        in_str = False
                else:
                    if b == 34:  # quote
                        in_str = True
                        esc = False
        flush_record()

    backup_path = None
    if inplace:
        backup_path = _replace_inplace(in_path, out_path, backup=backup)
        out_path = in_path

    return {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "records_ok": records_ok,
        "records_bad": records_bad,
        "wrote_bytes": wrote_bytes,
        "backup_path": str(backup_path) if backup_path else None,
    }

import json
import os
import subprocess
from pathlib import Path


def test_pack_meta_num_seqs_matches_index():
    pack_out = os.environ.get("PACK_SMOKE")
    if not pack_out or not Path(pack_out).exists():
        return

    meta_path = Path(pack_out) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert isinstance(meta.get("seq_len"), int)
    num_seqs = meta.get("num_seqs")
    if num_seqs is None:
        # Backward compat: older packs may only have n_seqs.
        num_seqs = meta.get("n_seqs")
        if not isinstance(num_seqs, int):
            return
    assert isinstance(num_seqs, int), f"num_seqs must be int, got {meta.get('num_seqs')}"

    cmd = f"zstdcat '{pack_out}/index.jsonl.zst' | wc -l"
    n = int(subprocess.check_output(cmd, shell=True, text=True).strip())
    assert num_seqs == n, f"meta.num_seqs={num_seqs} != index rows={n}"

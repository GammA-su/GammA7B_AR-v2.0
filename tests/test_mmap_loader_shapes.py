import os
import importlib.util

def test_open_pack_reads_meta():
    if importlib.util.find_spec("torch") is None:
        return
    pack = os.environ.get("PACK_SMOKE")
    if not pack:
        return
    from gamma7b_train.data_mmap import open_pack

    p = open_pack(pack, use_index_domains=False)
    assert p.seq_len > 0
    assert p.num_seqs > 0
    assert os.path.exists(p.mmap_path)

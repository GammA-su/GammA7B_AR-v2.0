# GammA7B_AR-v2.0 — Stage A Data Pipeline

Deterministic, license-aware pretraining data pipeline for Stage A next-token training. Produces normalized shards, deduped corpora, and packed fixed-length token sequences suitable for pretraining.

## Design choices (short)
- Determinism: seeded sampler + stable input order + explicit manifests.
- Minimal deps: `datasets`, `tokenizers`, `zstandard`, `numpy`, `pyyaml`, `typer`.
- Offline-ready: once raw/normalized shards exist, no network is required.
- Document-aware packing: contiguous spans from a doc before mixing.

## Normalized schema
```
{
  "text": str,
  "source": str,
  "domain": str,
  "doc_id": str,
  "license_tag": str | null,
  "created_at": str | null,
  "meta": dict | null
}
```

## Stage A manifest
Edit `stageA_manifest.yaml` (or `configs/stageA_manifest.yaml`) with your paths + tokenizer JSON. We validate weights and default normalized paths based on `outputs.normalized_dir`.

## CLI
All commands are deterministic given `--seed`.

```
python -m gamma7b_data.cli hf-configs --dataset HuggingFaceFW/fineweb-2 --lang eng --script Latn
python -m gamma7b_data.cli hf-stream --source fineweb2 --out-dir data/normalized
python -m gamma7b_data.cli hf-stream --source fineweb2 --config-name eng_Latn --out-dir data/normalized
python -m gamma7b_data.cli ingest-manifest --manifest configs/stageA_manifest.yaml --out-dir out/stageA_raw
python -m gamma7b_data.cli ingest-wikipedia --input /path/to/wikiextractor --out-dir data/normalized
python -m gamma7b_data.cli ingest-stackexchange --input /path/to/Posts.xml --out-dir data/normalized

python -m gamma7b_data.cli normalize \
  --input data/raw/my_source/*.jsonl.zst \
  --out-dir data/normalized \
  --domain filtered_web \
  --source fineweb2

python -m gamma7b_data.cli dedup \
  --input data/normalized/**/**/*.jsonl.zst \
  --out-dir data/dedup \
  --mode both --scope global

python -m gamma7b_data.cli pack \
  --manifest stageA_manifest.yaml \
  --out-dir data/packed \
  --num-seqs 1000

python -m gamma7b_data.cli report \
  --input data/normalized/**/**/*.manifest.json \
  --out data/report.json
```
> WARNING: Substring filtering (e.g. `--contains eng` / `--filter eng`) can match "Beng" (e.g. `asm_Beng`). Prefer `--lang eng` for language-aware matching.

Schema probe (streaming one row to confirm `text_field`/`id_field`):
```
uv run python -c "from datasets import load_dataset; row=next(iter(load_dataset('teven/stackexchange', split='train', streaming=True))); print(sorted(row.keys()))"
```

## Recommended web backbone
- Use FineWeb-Edu for English web data.
- Use FineWeb-2 when you need multilingual expansion.

Canonical Stage-A ingestion:
```
uv run python -m gamma7b_data.cli ingest-manifest --manifest configs/stageA_manifest.yaml --out-dir out/stageA_raw
```

Fast dedup (per-source + single-pass near):
```
uv run python -m gamma7b_data.cli dedup \
  --input data/normalized/**/**/*.jsonl.zst \
  --out-dir data/dedup \
  --mode both \
  --scope per-source \
  --near-selection first
```

Global flags:
- `--cpu-threads 16` (default)
- `--faiss-gpu 0` (default; 0 disables FAISS GPU probing)
- `--verbose`
- `--log-every 300` (default progress cadence)

Notes:
- `--config-name` applies to all `--source` entries in that command.

## Packed output format
`pack` writes:
- `input_ids.mmap`: `int32` memmap, shape `(n_seqs, seq_len)`
- `meta.json`: `{seq_len, n_seqs, tokenizer_path, created_at, counts}`
- `index.jsonl.zst` (optional): mapping from row → doc segments

This is ready to memory-map in your trainer. Example (PyTorch):
```
import numpy as np
mmap = np.memmap("data/packed/input_ids.mmap", dtype=np.int32, mode="r")
```

## UV-first setup (Python 3.10 required for faiss-gpu)
```
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```
If you need StackExchange `.7z` ingestion:
```
uv pip install -e ".[stackexchange]"
```

## Fresh machine (no data) - Stage A end-to-end
```bash
export WORKSPACE=/workspace
export HF_HOME="$WORKSPACE/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$WORKSPACE/.cache"
export TMPDIR="$WORKSPACE/tmp"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

cd /workspace/GammA7B_AR-v2.0
uv sync --extra dev

# Pull all HF sources in configs/stageA_manifest.yaml
# (includes wikipedia + forums_qa via hf_stream sources).
uv run python -m gamma7b_data.cli ingest-manifest \
  --manifest configs/stageA_manifest.yaml \
  --out-dir "$WORKSPACE/out/stageA_raw"

# If you also have extra local raw JSONL/ZST shards, normalize them:
# uv run python -m gamma7b_data.cli normalize --input "$WORKSPACE/data/raw/**/*.jsonl.zst" --out-dir "$WORKSPACE/data/normalized" --domain filtered_web --source local_extra

uv run python -m gamma7b_data.cli dedup \
  --input "$WORKSPACE/data/normalized/**/**/*.jsonl.zst" \
  --out-dir "$WORKSPACE/data/dedup" \
  --mode both --scope global

uv run python scripts/32_split_dedup_by_domain.py \
  --dedup-zst "$WORKSPACE/data/dedup/deduped.jsonl.zst" \
  --out-dir "$WORKSPACE/data/dedup/stageA_latest_split_by_domain"

uv run python scripts/33_make_split_manifest.py \
  --out configs/stageA_manifest_dedup_split.yaml \
  --split-rel data/dedup/stageA_latest_split_by_domain

uv run python -m gamma7b_data.cli pack \
  --manifest configs/stageA_manifest_dedup_split.yaml \
  --out-dir "$WORKSPACE/data/packed" \
  --num-seqs 1000
```

## Run tests
```
uv run pytest -q
```

## Smoke pipeline (tiny end-to-end)
```
uv run python -m gamma7b_data.cli smoke
```
Outputs (under `out/`):
- `out/normalized/**/*.jsonl.zst`
- `out/dedup/deduped.jsonl.zst` + `out/dedup/decisions.jsonl.zst`
- `out/packed/input_ids.mmap` + `out/packed/meta.json`
- `out/report.json`

## Notes
- HF streaming is the only online step. Everything else is offline once raw/normalized shards are present.
- Filters are heuristic CPU-only (ASCII ratio + boilerplate + length).
- Dedup logs decisions with reason codes for auditability.
- StackExchange `.7z` archives require `uv pip install -e ".[stackexchange]"`.
- Default runtime uses 16 CPU threads and skips FAISS GPU probing unless `--faiss-gpu` is set (see `--cpu-threads` and `--faiss-gpu` flags).
- Zstandard compression defaults to level 3 for speed (override with `GAMMA7B_ZSTD_LEVEL`).
- `faiss-gpu` requires a CUDA-capable environment; if unavailable, logs will show GPU enablement failed and the pipeline continues on CPU.
- Control progress logging cadence with `GAMMA7B_LOG_EVERY` (default: 300 records).
- FineWeb-2 requires an explicit config (default `eng_Latn`); override with `--config-name` or list with `hf-configs`.

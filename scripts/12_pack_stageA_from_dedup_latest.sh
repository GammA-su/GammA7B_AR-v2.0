#!/usr/bin/env bash
set -euo pipefail

source ./env.local.sh

: "${GAMMA7B_DATA_ROOT:?set GAMMA7B_DATA_ROOT}"
: "${GAMMA7B_OUT_ROOT:?set GAMMA7B_OUT_ROOT}"
: "${TOKENIZER_JSON:?set TOKENIZER_JSON to your tokenizer.json path}"

DEDUP_LATEST_DIR="$GAMMA7B_DATA_ROOT/dedup/stageA_latest"
DEDUP_FILE="$DEDUP_LATEST_DIR/deduped.jsonl.zst"

if [[ ! -f "$DEDUP_FILE" ]]; then
  echo "Missing: $DEDUP_FILE"
  echo "Did you create the stageA_latest symlink?"
  exit 1
fi

echo "[check] zstd integrity..."
zstd -t "$DEDUP_FILE"

echo "[check] first record parses..."
uv run python - <<'PY'
import io, json, zstandard as zstd, pathlib
p = pathlib.Path("$DEDUP_FILE")
dctx = zstd.ZstdDecompressor()
with p.open("rb") as f, dctx.stream_reader(f) as r:
    t = io.TextIOWrapper(r, encoding="utf-8")
    obj = json.loads(t.readline())
print("OK keys:", sorted(obj.keys())[:20])
PY

PACK_OUT="$GAMMA7B_OUT_ROOT/packed_stageA_seq4096"
mkdir -p "$PACK_OUT"

echo "[pack] -> $PACK_OUT"
uv run python -m gamma7b_data.cli pack \
  --manifest configs/stageA_manifest_dedup_latest.yaml \
  --out-dir "$PACK_OUT" \
  --seq-len 4096 \
  --tokenizer "$TOKENIZER_JSON" \
  --emit-index

echo "[smoke]"
uv run python -m gamma7b_data.cli smoke \
  --manifest configs/stageA_manifest_dedup_latest.yaml \
  --data-root "$GAMMA7B_DATA_ROOT" \
  --out-dir "$PACK_OUT"

echo "[report]"
uv run python -m gamma7b_data.cli report \
  --manifest configs/stageA_manifest_dedup_latest.yaml \
  --data-root "$GAMMA7B_DATA_ROOT" \
  --out-dir "$PACK_OUT"

echo "DONE: packed output at $PACK_OUT"

#!/usr/bin/env bash
set -euo pipefail

source ./env.local.sh || true

# Adjust if your HDD normalized lives elsewhere
SSD_ROOT="${SSD_ROOT:-/home/alpahos/satassd/gamma_data_root}"
HDD_ROOT="${HDD_ROOT:-/home/alpahos/8to/gamma_data/data_root/data}"

SSD_NORM="$SSD_ROOT/normalized"
HDD_NORM="$HDD_ROOT/normalized"

need_domains=(books_longform code_docs filtered_web math paper_teasers forums_qa reference tutorials_notebooks)

echo "SSD_NORM=$SSD_NORM"
echo "HDD_NORM=$HDD_NORM"

echo
echo "== domains present on SSD =="
for d in "${need_domains[@]}"; do
  if [[ -d "$SSD_NORM/$d" ]]; then echo "  [OK] $d"; else echo "  [--] $d"; fi
done

echo
echo "== syncing missing domains HDD -> SSD (rsync) =="
mkdir -p "$SSD_NORM"
for d in "${need_domains[@]}"; do
  if [[ -d "$SSD_NORM/$d" ]]; then
    continue
  fi
  if [[ ! -d "$HDD_NORM/$d" ]]; then
    echo "  [WARN] missing on HDD too: $HDD_NORM/$d"
    continue
  fi
  echo "  rsync $d ..."
  rsync -a --info=progress2 "$HDD_NORM/$d/" "$SSD_NORM/$d/"
done

echo
echo "== shard counts on SSD normalized =="
find "$SSD_NORM" -type f -name 'part_*.jsonl.zst' | wc -l
du -sh "$SSD_NORM" || true

echo
echo "== run dedup (exact-only, per-source scope) =="
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$SSD_ROOT/dedup/stageA_global_full_${TS}"
mkdir -p "$OUT_DIR"

uv run python -m gamma7b_data.cli dedup \
  --input "$SSD_NORM" \
  --out-dir "$OUT_DIR" \
  --mode exact \
  --scope per-source \
  --seed 0

echo
echo "== update stageA_latest symlink =="
rm -f "$SSD_ROOT/dedup/stageA_latest"
ln -s "$OUT_DIR" "$SSD_ROOT/dedup/stageA_latest"
echo "stageA_latest -> $(readlink -f "$SSD_ROOT/dedup/stageA_latest")"

echo
echo "== sanity: histogram on new dedup =="
DEDUP_ZST="$SSD_ROOT/dedup/stageA_latest/deduped.jsonl.zst"
uv run python scripts/31_domain_histogram_dedup.py --dedup-zst "$DEDUP_ZST" --log-every 200000 | head -n 120

echo
echo "Done."

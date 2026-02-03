#!/usr/bin/env bash
set -euo pipefail

# Links SSD split folder into HDD data_root so manifest "data/..." paths resolve.

SSD_ROOT="${SSD_ROOT:-/home/alpahos/satassd/gamma_data_root}"
HDD_DATA_ROOT="${HDD_DATA_ROOT:-/home/alpahos/8to/gamma_data/data_root/data}"
SPLIT_DIR_SSD="${SPLIT_DIR_SSD:-$SSD_ROOT/dedup/stageA_latest_split_by_domain}"

[[ -d "$SPLIT_DIR_SSD" ]] || { echo "ERR: SPLIT_DIR_SSD not found: $SPLIT_DIR_SSD"; exit 2; }

mkdir -p "$HDD_DATA_ROOT/dedup"
rm -f "$HDD_DATA_ROOT/dedup/stageA_latest_split_by_domain"
ln -s "$SPLIT_DIR_SSD" "$HDD_DATA_ROOT/dedup/stageA_latest_split_by_domain"

echo "OK: linked:"
ls -lah "$HDD_DATA_ROOT/dedup/stageA_latest_split_by_domain" | sed -n '1,80p'
